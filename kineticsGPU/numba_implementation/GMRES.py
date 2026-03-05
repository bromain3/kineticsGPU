"""
gmres_numba.py
==============
GMRES redémarré avec rotations de Givens — device function Numba.

Contraintes :
  - 1 thread CUDA = 1 système
  - Zéro mémoire locale ou partagée : tous les buffers en mémoire globale

La matvec est inlinée dans le code appelant (Numba ne supporte pas les
pointeurs de fonctions device). Ce module fournit les briques GMRES
(Krylov, Arnoldi, Givens, solve triangulaire) et la fonction principale
`gmres_device` qui appelle une matvec définie ailleurs via un mécanisme
de callback simulé par héritage de device function.

En pratique, pour un cas concret, on réimplémente `gmres_solve_XXX`
qui appelle `_gmres_core` avec la matvec spécifique inline.

Interface de `_gmres_core` :
    _gmres_core(b, x, N, restart, tol, maxiter,
                V, H, cs, sn, e1, y, tmp,
                <matvec inline via appelant>)

Mais comme Numba ne supporte pas les callbacks, on exporte seulement
les utilitaires vectoriels et Arnoldi, et la logique complète est dans
backward_euler_numba.py.
"""

from numba import cuda
import math


# =============================================================================
#  Opérations vectorielles (tableaux 1D en mémoire globale)
# =============================================================================

@cuda.jit(device=True, inline=True)
def norm1d(a, n):
    s = 0.0
    for i in range(n):
        s += a[i] * a[i]
    return math.sqrt(s)

@cuda.jit(device=True, inline=True)
def norm_off(a, offset, n):
    """||a[offset:offset+n]||"""
    s = 0.0
    for i in range(n):
        v = a[offset + i]
        s += v * v
    return math.sqrt(s)

@cuda.jit(device=True, inline=True)
def dot_off(a, oa, b, ob, n):
    """a[oa:oa+n] · b[ob:ob+n]"""
    s = 0.0
    for i in range(n):
        s += a[oa + i] * b[ob + i]
    return s

@cuda.jit(device=True, inline=True)
def axpy_off(alpha, x, ox, y, oy, n):
    """y[oy:oy+n] += alpha * x[ox:ox+n]"""
    for i in range(n):
        y[oy + i] += alpha * x[ox + i]

@cuda.jit(device=True, inline=True)
def scale_off(alpha, a, offset, n):
    for i in range(n):
        a[offset + i] *= alpha

@cuda.jit(device=True, inline=True)
def zero_off(a, offset, n):
    for i in range(n):
        a[offset + i] = 0.0


# =============================================================================
#  Rotation de Givens
# =============================================================================

@cuda.jit(device=True, inline=True)
def givens_rotation(v1, v2):
    """(cs, sn) tels que [cs sn; -sn cs]^T [v1; v2] = [r; 0]."""
    if v2 == 0.0:
        return 1.0, 0.0
    if math.fabs(v2) > math.fabs(v1):
        t  = -v1 / v2
        sn = 1.0 / math.sqrt(1.0 + t * t)
        cs = t * sn
    else:
        t  = -v2 / v1
        cs = 1.0 / math.sqrt(1.0 + t * t)
        sn = t * cs
    return cs, sn


# =============================================================================
#  Résolution triangulaire supérieure
#  H : column-major dans H_flat  →  H[i,j] = H_flat[j*(restart+1)+i]
# =============================================================================

@cuda.jit(device=True, inline=False)
def back_substitution(H_flat, e1, y, restart, n):
    for i in range(n - 1, -1, -1):
        val = e1[i]
        for j in range(i + 1, n):
            val -= H_flat[j * (restart + 1) + i] * y[j]
        d = H_flat[i * (restart + 1) + i]
        if math.fabs(d) < 1e-300:
            d = 1e-300
        y[i] = val / d