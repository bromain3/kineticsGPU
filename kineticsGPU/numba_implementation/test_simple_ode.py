"""
test_simple_ode.py
==================
Test standalone de l'intégrateur Backward Euler GMRES sur Van der Pol.
Aucun import du projet kineticsGPU.
Comparaison avec scipy.integrate.solve_ivp (BDF).
"""

import numpy as np
import math
import time
from numba import cuda
from scipy.integrate import solve_ivp
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

MU  = 10000.0   # stiffness
RST = 4       # restart GMRES (suffisant pour N=2)

# =============================================================================
#  Kernel device : utilitaires GMRES inline
# =============================================================================

@cuda.jit(device=True, inline=True)
def _norm(a, n):
    s = 0.0
    for i in range(n): s += a[i]*a[i]
    return math.sqrt(s)

@cuda.jit(device=True, inline=True)
def _norm_v(V, o, n):
    s = 0.0
    for i in range(n): v = V[o+i]; s += v*v
    return math.sqrt(s)

@cuda.jit(device=True, inline=True)
def _dot_v(a, oa, b, ob, n):
    s = 0.0
    for i in range(n): s += a[oa+i]*b[ob+i]
    return s

@cuda.jit(device=True, inline=True)
def _axpy_v(alpha, x, ox, y, oy, n):
    for i in range(n): y[oy+i] += alpha*x[ox+i]

@cuda.jit(device=True, inline=True)
def _scale_v(alpha, a, o, n):
    for i in range(n): a[o+i] *= alpha

@cuda.jit(device=True, inline=True)
def _zero_v(a, o, n):
    for i in range(n): a[o+i] = 0.0

@cuda.jit(device=True, inline=True)
def _givens(v1, v2):
    if v2 == 0.0: return 1.0, 0.0
    if math.fabs(v2) > math.fabs(v1):
        t = -v1/v2; sn = 1.0/math.sqrt(1.0+t*t); cs = t*sn
    else:
        t = -v2/v1; cs = 1.0/math.sqrt(1.0+t*t); sn = t*cs
    return cs, sn

@cuda.jit(device=True, inline=False)
def _backsubst(H, e1, y, rst, n):
    for i in range(n-1, -1, -1):
        val = e1[i]
        for j in range(i+1, n): val -= H[j*(rst+1)+i]*y[j]
        d = H[i*(rst+1)+i]
        if math.fabs(d) < 1e-300: d = 1e-300
        y[i] = val/d

@cuda.jit(device=True, inline=False)
def _vdp_rhs(y0, y1, out):
    out[0] = y1
    out[1] = MU*(1.0 - y0*y0)*y1 - y0

@cuda.jit(device=True, inline=False)
def _vdp_matvec(V, vo, h, y, jvp_tmp, out):
    """w = v + h*J@v  pour Van der Pol."""
    vy0 = V[vo]; vy1 = V[vo+1]
    jvp_tmp[0] = vy1
    jvp_tmp[1] = (-2.0*MU*y[0]*y[1] - 1.0)*vy0 + MU*(1.0 - y[0]*y[0])*vy1
    out[0] = vy0 + h*jvp_tmp[0]
    out[1] = vy1 + h*jvp_tmp[1]

@cuda.jit(device=True, inline=False)
def _gmres_vdp(b, x, h, y, V, H, cs, sn, e1, yv, tmp, jvp_tmp, tol, maxiter):
    N = 2
    b_norm = _norm(b, N)
    if b_norm < 1e-300: x[0] = 0.0; x[1] = 0.0; return
    abs_tol = tol*b_norm

    for _outer in range(maxiter):
        V[0] = x[0]; V[1] = x[1]
        _zero_v(tmp, 0, N)
        _vdp_matvec(V, 0, h, y, jvp_tmp, tmp)
        V[0] = b[0]-tmp[0]; V[1] = b[1]-tmp[1]

        beta = _norm_v(V, 0, N)
        if beta < abs_tol: return
        _scale_v(1.0/beta, V, 0, N)
        for i in range(RST+1): e1[i] = 0.0
        e1[0] = beta

        j_conv = RST
        for j in range(RST):
            vj = j*N
            _zero_v(tmp, 0, N)
            _vdp_matvec(V, vj, h, y, jvp_tmp, tmp)

            for i in range(j+1):
                vi = i*N
                h_ij = _dot_v(tmp, 0, V, vi, N)
                H[j*(RST+1)+i] = h_ij
                _axpy_v(-h_ij, V, vi, tmp, 0, N)

            h_jp1j = _norm_v(tmp, 0, N)
            H[j*(RST+1)+j+1] = h_jp1j
            vj1 = (j+1)*N
            if h_jp1j > 1e-300:
                inv_h = 1.0/h_jp1j
                for k in range(N): V[vj1+k] = tmp[k]*inv_h
            else:
                _zero_v(V, vj1, N)

            for i in range(j):
                ci = cs[i]; si = sn[i]
                h0 = H[j*(RST+1)+i]; h1 = H[j*(RST+1)+i+1]
                H[j*(RST+1)+i]   =  ci*h0 + si*h1
                H[j*(RST+1)+i+1] = -si*h0 + ci*h1

            h0 = H[j*(RST+1)+j]; h1 = H[j*(RST+1)+j+1]
            cs_j, sn_j = _givens(h0, h1)
            cs[j] = cs_j; sn[j] = sn_j
            H[j*(RST+1)+j]   =  cs_j*h0 + sn_j*h1
            H[j*(RST+1)+j+1] = -sn_j*h0 + cs_j*h1
            e1[j+1] = -sn_j*e1[j]; e1[j] = cs_j*e1[j]

            if math.fabs(e1[j+1]) < abs_tol: j_conv = j+1; break
        else:
            j_conv = RST

        for i in range(j_conv): yv[i] = 0.0
        _backsubst(H, e1, yv, RST, j_conv)
        for i in range(j_conv):
            vi = i*N; yi = yv[i]
            for k in range(N): x[k] += yi*V[vi+k]

        V[0] = x[0]; V[1] = x[1]
        _zero_v(tmp, 0, N)
        _vdp_matvec(V, 0, h, y, jvp_tmp, tmp)
        res2 = (b[0]-tmp[0])**2 + (b[1]-tmp[1])**2
        if math.sqrt(res2) < abs_tol: return


# =============================================================================
#  Kernel Backward Euler Van der Pol
# =============================================================================

NEWT_TOL  = 1e-10
NEWT_ITER = 30
GMRS_TOL  = 1e-10
GMRS_IT   = 10

@cuda.jit
def vdp_be_kernel(Y_in, Y_out, h,
                  gV, gH, gCs, gSn, gE1, gYv, gTmp, gJvp, gSol, gF, gRhs):
    cell = cuda.grid(1)
    if cell >= Y_in.shape[0]: return

    yn = Y_in[cell]; yk = Y_out[cell]
    V = gV[cell]; H = gH[cell]; cs = gCs[cell]; sn = gSn[cell]
    e1 = gE1[cell]; yv = gYv[cell]; tmp = gTmp[cell]; jvp = gJvp[cell]
    sol = gSol[cell]; F = gF[cell]; rhs = gRhs[cell]

    _vdp_rhs(yn[0], yn[1], rhs)
    yk[0] = yn[0] + h*rhs[0]; yk[1] = yn[1] + h*rhs[1]

    for _nit in range(NEWT_ITER):
        _vdp_rhs(yk[0], yk[1], rhs)
        F[0] = yk[0] - yn[0] - h*rhs[0]
        F[1] = yk[1] - yn[1] - h*rhs[1]
        if math.sqrt(F[0]*F[0] + F[1]*F[1]) < NEWT_TOL: break

        sol[0] = 0.0; sol[1] = 0.0
        F[0] = -F[0]; F[1] = -F[1]
        _gmres_vdp(F, sol, h, yk, V, H, cs, sn, e1, yv, tmp, jvp,
                   GMRS_TOL, GMRS_IT)
        yk[0] += sol[0]; yk[1] += sol[1]


# =============================================================================
#  Main
# =============================================================================

def main():
    print("=" * 50)
    print(f"Backward Euler GMRES — Van der Pol (mu={MU})")
    print("=" * 50)

    NCELLS = 1
    T_END  = 2.0
    H      = 5e-4
    N      = 2

    # Référence scipy
    def vdp(t, y):
        return [y[1], MU*(1 - y[0]**2)*y[1] - y[0]]

    t0  = time.perf_counter()
    ref = solve_ivp(vdp, [0, T_END], [2.0, 0.0],
                    method='BDF', rtol=1e-12, atol=1e-14, dense_output=True)
    print(f"\nScipy BDF : {(time.perf_counter()-t0)*1e3:.1f} ms")
    print(f"  y_final = {ref.y[:,-1]}")

    # Buffers GPU
    Y0 = np.zeros((NCELLS, N), dtype=np.float64); Y0[:, 0] = 2.0
    def mk(s): return cuda.device_array(s, dtype=np.float64)
    gV  = mk((NCELLS, (RST+1)*N)); gH  = mk((NCELLS, RST*(RST+1)))
    gCs = mk((NCELLS, RST));       gSn = mk((NCELLS, RST))
    gE1 = mk((NCELLS, RST+1));     gYv = mk((NCELLS, RST))
    gTmp= mk((NCELLS, N));         gJvp= mk((NCELLS, N))
    gSol= mk((NCELLS, N));         gF  = mk((NCELLS, N))
    gRhs= mk((NCELLS, N))

    tpb = 128; blk = (NCELLS+tpb-1)//tpb

    # JIT warm-up
    Yin = cuda.to_device(Y0); Yout = cuda.device_array_like(Yin)
    print("\nCompilation JIT...")
    t0 = time.perf_counter()
    vdp_be_kernel[(blk,),(tpb,)](Yin, Yout, np.float64(H),
        gV,gH,gCs,gSn,gE1,gYv,gTmp,gJvp,gSol,gF,gRhs)
    cuda.synchronize()
    print(f"  {(time.perf_counter()-t0)*1e3:.1f} ms (compile + 1 pas)")

    # Intégration
    Yin = cuda.to_device(Y0); Yout = cuda.device_array_like(Yin)
    n_steps = int(math.ceil(T_END / H))
    t = 0.0
    t0 = time.perf_counter()
    for _ in range(n_steps):
        h_eff = min(H, T_END - t)
        if h_eff < 1e-15: break
        vdp_be_kernel[(blk,),(tpb,)](Yin, Yout, np.float64(h_eff),
            gV,gH,gCs,gSn,gE1,gYv,gTmp,gJvp,gSol,gF,gRhs)
        Yin, Yout = Yout, Yin
        t += h_eff
    cuda.synchronize()
    t_gpu = time.perf_counter() - t0

    Yf    = Yin.copy_to_host()
    y_gpu = Yf[0]
    y_ref = ref.sol(T_END)
    err   = np.abs(y_gpu - y_ref) / (np.abs(y_ref) + 1e-15)

    print(f"\nGPU BE ({NCELLS} réacteurs) : {t_gpu*1e3:.1f} ms ({n_steps} pas)")
    print(f"  y_gpu   = {y_gpu}")
    print(f"  y_ref   = {y_ref}")
    print(f"  err rel = {err}")

    # Tracé
    tf = np.linspace(0, T_END, 1000)
    yf = ref.sol(tf)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(tf, yf[0], 'b-', lw=1.5, label='scipy BDF')
    axes[0].plot(T_END, y_gpu[0], 'ro', ms=9, label='GPU BE')
    axes[0].set(xlabel='t', ylabel='y₀', title='Van der Pol — y₀(t)'); axes[0].legend(); axes[0].grid(alpha=.3)
    axes[1].plot(yf[0], yf[1], 'b-', lw=1.5, label='scipy BDF')
    axes[1].plot(y_gpu[0], y_gpu[1], 'ro', ms=9, label='GPU BE')
    axes[1].set(xlabel='y₀', ylabel='y₁', title='Portrait de phase'); axes[1].legend(); axes[1].grid(alpha=.3)
    plt.tight_layout()
    plt.savefig('test_simple_ode.png', dpi=150)
    print("Figure : test_simple_ode.png")


if __name__ == '__main__':
    main()