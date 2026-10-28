// SPL 单流计算核。预测、评分、更新；不截断状态或主评分强度。
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

template<class T> T read(std::istream &f) {
    T x{}; if(!f.read(reinterpret_cast<char*>(&x),sizeof(x))) throw std::runtime_error("输入截断"); return x;
}
template<class T> void write(std::ostream &f,T x) { f.write(reinterpret_cast<const char*>(&x),sizeof(x)); }
const double INF=std::numeric_limits<double>::infinity();
double ladd(double a,double b) {
    if(a==-INF) return b;
    if(b==-INF) return a;
    return std::max(a,b)+std::log1p(std::exp(-std::abs(a-b)));
}
// Wright omega 的正实分支，Newton 解 w+log(w)=z，避免构造 exp(z)。
double omega(double z) {
    if(z < -2.) {
        const long double r=std::exp(static_cast<long double>(z));
        long double w=r;
        for(int j=0;j<6;++j) {
            const long double e=std::exp(w), d=(w-r/e)/(1+w);
            w-=d;
            if(std::abs(d)<=2e-19L*std::max(w,1e-300L)) break;
        }
        return w;
    }
    double w=z>1 ? z-std::log(z)+std::log(z)/z : std::exp(z)/(1+std::exp(z));
    for(int j=0;j<15;++j) {
        const double d=(w+std::log(w)-z)*w/(1+w);
        const double next=w-d;
        if(next<=0) w*=.5; else w=next;
        if(std::abs(d)<=2e-16*std::max(w,1.)) break;
    }
    return w;
}
void update(double &theta,double &V,int count,double v,double dt,bool implicit) {
    if(!implicit) {
        const double lr=std::log(v)+theta+std::log(dt);
        double gain,decrement;
        if(lr>=0) { const double e=std::exp(-lr); gain=v*e/(1+e); decrement=1/(1+e); }
        else { const double e=std::exp(lr); gain=v/(1+e); decrement=e/(1+e); }
        theta=theta+gain*count-decrement; V=gain;
    } else {
        const double lv=std::log(v*dt), z=lv+theta+v*count;
        const double w=omega(z);
        theta=z < -2 ? theta+v*count-w : std::log(w)-lv;
        V=v/(1+w);
    }
}
// 正的指数项和有限线性项分离；大损失不转成 double。
struct Mass {
    long double small=0; double big=-INF;
    void add(double logx) {
        if(logx<600 && big==-INF) { small+=std::exp(static_cast<long double>(logx)); return; }
        if(big==-INF) { big=small>0 ? std::log(small) : -INF; small=0; }
        if(logx>big-40) big=ladd(big,logx);
    }
    double logvalue() const { return big==-INF ? (small>0 ? std::log(small) : -INF) : big; }
};
std::pair<double,double> subtract(double a,double b) {
    if(a==b) return {0,0};
    const double sign=a>b ? 1. : -1.;
    const double hi=std::max(a,b),lo=std::min(a,b);
    return {sign,(hi+std::log(-std::expm1(lo-hi)))/std::log(10.)};
}
std::pair<double,double> total(long double finite,const Mass &mass,bool excess) {
    if(mass.big==-INF) {
        const long double value=excess ? finite+mass.small : finite-mass.small;
        return value==0 ? std::make_pair(0.,0.) : std::make_pair(value>0?1.:-1.,static_cast<double>(std::log10(std::abs(value))));
    }
    const double lm=mass.logvalue();
    if(excess) {
        if(finite>=0) return {1,ladd(lm,finite>0?std::log(finite):-INF)/std::log(10.)};
        return subtract(lm,std::log(-finite));
    }
    if(finite<=0) return {-1,ladd(lm,finite<0?std::log(-finite):-INF)/std::log(10.)};
    return subtract(std::log(finite),lm);
}
int gaussian(std::istream &in,std::ostream &out) {
    const int n=read<std::int32_t>(in),nc=read<std::int32_t>(in),start=read<std::int32_t>(in),keep=read<std::int32_t>(in),wake=read<std::int32_t>(in);
    const double dt=read<double>(in),t0=read<double>(in),v0=read<double>(in),maxtruth=read<double>(in);
    if(n<=0 || n>100000000 || nc<=0 || nc>323 || start<0 || start>=n || dt<=0) throw std::runtime_error("维度无效");
    std::vector<std::int32_t> counts(n);
    std::vector<double> truth(n),lf(n);
    for(auto &x:counts) {x=read<std::int32_t>(in);if(x<0)throw std::runtime_error("负计数");}
    for(auto &x:truth) x=read<double>(in);
    for(int k=0;k<n;++k) lf[k]=std::lgamma(counts[k]+1.);
    for(int j=0;j<nc;++j) {
        const int kind=read<std::int32_t>(in);
        const double qc=read<double>(in),cap=read<double>(in),mu=read<double>(in),kap=read<double>(in);
        const double q=qc*dt,rho=std::exp(-kap*dt),ld=std::log(dt);
            double theta=t0,V=v0,maxlog=-INF,first=0,clipped=0;
        long double finite=0,oracle=0;
        int streak=0; double recovery=-1;
        Mass mass;
        std::vector<double> trace(keep ? static_cast<std::size_t>(n)*10:0);
        for(int k=0;k<n;++k) {
            const int c=counts[k];
            double tp=theta,v=V+q;
            if(kind==4) {tp=mu+rho*(theta-mu);v=rho*rho*V+q;}
            if(kind==2 || kind==3) v=std::min(v,cap);
            const double lm=tp+ld;
            const double linear=c*lm-lf[k];
            const double score=linear-std::exp(lm);
            if(k>=start) {
                finite+=linear;mass.add(lm);
                if(truth[k]>=0) {
                    if(truth[k]==0 && c!=0) throw std::runtime_error("真零存在事件");
                    oracle+= c==0 ? -truth[k]-lf[k] : c*std::log(truth[k])-truth[k]-lf[k];
                }
                maxlog=std::max(maxlog,tp);
                if(maxtruth>0) { const double lclip=std::min(lm,std::log(1e4*maxtruth*dt)); clipped+=c*lclip-std::exp(lclip)-lf[k]; }
            }
            theta=tp;update(theta,V,c,v,dt,kind==1 || kind==3 || kind==4);
            if(k==wake) first=theta;
            if(k>=wake && wake>=0 && truth[k]>0) {
                streak=std::abs(tp-std::log(truth[k]/dt))<=std::log(1.5) ? streak+1 : 0;
                const int dwell=static_cast<int>(std::ceil(1./dt));
                if(streak>=dwell && recovery<0) recovery=(k-wake-dwell+1)*dt;
            }
            if(keep) {
                const auto sl=total(linear,Mass{std::exp(lm<600?lm:-INF),lm>=600?lm:-INF},false);
                double vals[]={tp,v,std::exp(tp),score,theta,V,tp+v/2,sl.first,sl.second,lm};
                for(int h=0;h<10;++h) trace[static_cast<std::size_t>(k)*10+h]=vals[h];
            }
            if(!std::isfinite(theta) || !std::isfinite(V) || V<0) throw std::runtime_error("非有限状态");
        }
        const auto sc=total(finite,mass,false),ex=total(oracle-finite,mass,true);
        const double vals[]={sc.first,sc.second,ex.first,ex.second,maxlog,first,recovery,clipped,theta,V,static_cast<double>(oracle),static_cast<double>(finite)};
        for(double x:vals) write(out,x);
        if(keep) out.write(reinterpret_cast<const char*>(trace.data()),static_cast<std::streamsize>(trace.size()*sizeof(double)));
    }
    return 0;
}
int grid(std::istream &in,std::ostream &out) {
    const int n=read<std::int32_t>(in),start=read<std::int32_t>(in),keep=read<std::int32_t>(in);
    const double dt=read<double>(in),q=read<double>(in),t0=read<double>(in),v0=read<double>(in),lo=read<double>(in),hi=read<double>(in),h=read<double>(in),ns=read<double>(in);
    const int size=static_cast<int>(std::llround((hi-lo)/h))+1,half=static_cast<int>(std::ceil(ns*std::sqrt(q)/h));
    if(n<1 || size<3 || half>=size || start<0 || start>=n) throw std::runtime_error("网格参数无效");
    std::vector<int> counts(n);for(auto &c:counts)c=read<std::int32_t>(in);
    std::vector<double> x(size),rate(size),p(size),pred(size),post(size),kernel(2*half+1);
    double norm=0;
    for(int i=0;i<size;++i) {x[i]=lo+i*h;rate[i]=std::exp(x[i]);p[i]=std::exp(-(x[i]-t0)*(x[i]-t0)/(2*v0));norm+=p[i];}
    for(auto &z:p)z/=norm;
    norm=0;
    for(int a=-half;a<=half;++a){kernel[a+half]=std::exp(-a*h*a*h/(2*q));norm+=kernel[a+half];}
    for(auto &z:kernel)z/=norm;
    double sumscore=0,maxedge=0,loss_sum=0,maxloss=0,postmean=0,postvar=0;
    std::vector<double> trace(keep ? static_cast<std::size_t>(n)*8:0);
    for(int k=0;k<n;++k) {
        norm=0;
        // 与 numpy.convolve 相同的直接卷积，不使用 FFT。
        for(int i=0;i<size;++i) {
            double s=0;
            const int a0=std::max(-half,i-size+1),a1=std::min(half,i);
            for(int a=a0;a<=a1;++a)s+=p[i-a]*kernel[a+half];
            pred[i]=s;norm+=s;
        }
        const double lost=std::max(0.,1-norm);loss_sum+=lost;maxloss=std::max(maxloss,lost);
        double m=0,tm=0,edge=0;
        for(int i=0;i<size;++i){pred[i]/=norm;m+=pred[i]*rate[i];tm+=pred[i]*x[i];if(x[i]<lo+5)edge+=pred[i];}
        maxedge=std::max(maxedge,edge);
        double tv=0,mx=-INF;
        for(int i=0;i<size;++i) {tv+=pred[i]*(x[i]-tm)*(x[i]-tm);post[i]=pred[i]>0?std::log(pred[i])+counts[k]*(x[i]+std::log(dt))-dt*rate[i]-std::lgamma(counts[k]+1.): -INF;mx=std::max(mx,post[i]);}
        norm=0;for(auto &z:post){z=std::exp(z-mx);norm+=z;}
        const double score=mx+std::log(norm);
        if(k>=start)sumscore+=score;
        postmean=0;double pt=0;
        for(int i=0;i<size;++i){p[i]=post[i]/norm;postmean+=p[i]*rate[i];pt+=p[i]*x[i];}
        postvar=0;for(int i=0;i<size;++i)postvar+=p[i]*(x[i]-pt)*(x[i]-pt);
        if(keep){double vals[]={m,score,tm,tv,postmean,pt,postvar,edge};for(int j=0;j<8;++j)trace[static_cast<std::size_t>(k)*8+j]=vals[j];}
    }
    const double vals[]={sumscore,postmean,postvar,maxedge,maxloss,loss_sum,static_cast<double>(size),static_cast<double>(half)};
    for(double z:vals)write(out,z);
    if(keep)out.write(reinterpret_cast<const char*>(trace.data()),static_cast<std::streamsize>(trace.size()*sizeof(double)));
    return 0;
}
int main(int argc,char **argv) {
    try {
        if(argc!=3)throw std::runtime_error("需要输入与输出文件");
        std::ifstream in(argv[1],std::ios::binary);std::ofstream out(argv[2],std::ios::binary);
        const int magic=read<std::int32_t>(in);
        if(magic==202610211)gaussian(in,out);
        else if(magic==202610212)grid(in,out);
        else throw std::runtime_error("协议标识无效");
        if(!out)throw std::runtime_error("输出失败");
    }catch(const std::exception &e){std::cerr<<e.what();return 1;}
    return 0;
}
