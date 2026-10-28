"""Excess loss contributed by the silent bins themselves (true rate 0) for the
implicit filter, from the stationary E state at rate Lam, bin width dt."""
import math
from scipy.special import lambertw

def silence_loss(Lam, dt, qc, L):
    q = qc * dt
    V = (-q + math.sqrt(q * q + 4 * q / (Lam * dt))) / 2  # root of V^2+qV=q/(Lam dt)
    th = math.log(Lam); tot = 0.0
    for _ in range(int(round(L / dt))):
        v = V + q
        tot += math.exp(th) * dt          # r = lam_hat*dt - 0 - 0*log(...)
        # implicit update with C=0: th+ = th- - W(v dt e^{th-})
        th = th - lambertw(v * dt * math.exp(th)).real
        V = v / (1 + v * math.exp(th) * dt)
    return tot

for qc in (0.02, 0.2):
    a, b = silence_loss(1000, 1.0, qc, 60), silence_loss(1000, 1.0, qc, 1200)
    print(f'qc={qc}: silence loss L=60 {a:.3f}, L=1200 {b:.3f}, increase {b - a:.3f} nats')
print('2m log 20 =', 2 * 1000 * math.log(20))
