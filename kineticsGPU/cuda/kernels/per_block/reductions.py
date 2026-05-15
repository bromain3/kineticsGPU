"""
kernels/per_block/reductions.py — Helpers de réduction parallèle (per-block).

Fonctions device coopératives utilisant cuda.syncthreads().
Toutes supposent que les threads d'un même bloc coopèrent.

Fonctions exportées :
    _block_reduce_sum(val, scratch, tid, n_threads)
        → résultat dans scratch[0], thread 0 seulement
    _block_dot(a, b, scratch, tid, n, n_threads)
        → résultat dans scratch[0]
    _block_norm2(a, scratch, tid, n, n_threads)
        → ||a||² dans scratch[0]
    _block_norm2_inplace_sub(rhs, dx, Jv_TC, h, scratch, out, tid, n, n_threads)
        → calcule r = rhs - (dx - h*Jv_TC), écrit dans out, ||r||² dans scratch[0]
"""

import math
from numba import cuda, float64


@cuda.jit(device=True, inline=True)
def _block_reduce_sum(val, scratch, tid, n_threads):
    """
    Réduction par somme à l'échelle du bloc.

    scratch : tableau partagé de longueur >= n_threads (float64)
    val     : contribution locale du thread tid
    Résultat dans scratch[0] (seul thread 0 peut le lire en sécurité).
    """
    scratch[tid] = val
    cuda.syncthreads()
    stride = n_threads >> 1
    while stride > 0:
        if tid < stride:
            scratch[tid] += scratch[tid + stride]
        cuda.syncthreads()
        stride >>= 1


@cuda.jit(device=True, inline=True)
def _block_dot(a, b, scratch, tid, n, n_threads):
    """
    Produit scalaire parallèle de a[0:n] et b[0:n].

    scratch : tableau partagé float64 de longueur >= n_threads.
    Résultat dans scratch[0].
    """
    acc = float64(0.0)
    i   = tid
    while i < n:
        acc += a[i] * b[i]
        i   += n_threads
    _block_reduce_sum(acc, scratch, tid, n_threads)


@cuda.jit(device=True, inline=True)
def _block_norm2(a, scratch, tid, n, n_threads):
    """
    Norme au carré parallèle ||a[0:n]||².

    Résultat dans scratch[0].
    """
    acc = float64(0.0)
    i   = tid
    while i < n:
        acc += a[i] * a[i]
        i   += n_threads
    _block_reduce_sum(acc, scratch, tid, n_threads)


@cuda.jit(device=True, inline=True)
def _block_norm2_inplace_sub(rhs, dx, Jv_TC, h, scratch, out, tid, n, n_threads):
    """
    Calcule le résidu r = rhs - (dx - h*Jv_TC), l'écrit dans out,
    et met ||r||² dans scratch[0].

    Utilisé pour la vérification du résidu vrai en fin de cycle GMRES.
    """
    acc = float64(0.0)
    i   = tid
    while i < n:
        ri     = rhs[i] - (dx[i] - h * Jv_TC[i])
        out[i] = ri
        acc   += ri * ri
        i     += n_threads
    _block_reduce_sum(acc, scratch, tid, n_threads)