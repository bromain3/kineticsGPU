"""
comparison_reactor.py

Intègre 1 réacteur 0D adiabatique isochore avec le kernel GPU Numba,
compare à Cantera, affiche les stats matérielles et les graphiques.

Usage :
    python comparison_reactor.py
"""

import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from numba import cuda
import cantera as ct

import kineticsGPU.numba_implementation.thermo as thermo
from kineticsGPU.reactionParser import parseReactions
from kineticsGPU.numba_implementation.yaml_to_GPU_array import (
    build_elementary_gpu_arrays,
    build_threebody_gpu_arrays,
    build_falloff_gpu_arrays,
)
from kineticsGPU.numba_implementation.integrator import (
    launch_integrate,
    THREADS_PER_BLOCK,
    MAX_RESTART, MAX_NSP,
    NEWTON_MAXITER, NEWTON_TOL, GMRES_TOL,
    _V_SIZE, _H_SIZE, _CS_SIZE, _SN_SIZE, _BETA_SIZE, _N_STATE,
    gmres_buffer_memory_mb, print_memory_layout,
)

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
YAML_FILE    = '../gri30.yaml'
T0           = 1700.0
P0           = ct.one_atm
PHI          = 1.0
T_END        = 3e-4       # s
DT           = 1e-7       # s
N_STEPS      = int(T_END / DT)
SPECIES_PLOT = ['CH4', 'O2', 'CO', 'CO2', 'H2O', 'H', 'OH', 'H2']
OUT_PNG      = 'reactor_comparison.png'

# ─────────────────────────────────────────────────────────────────────────────
#  1. STATS MATÉRIELLES GPU
# ─────────────────────────────────────────────────────────────────────────────

def print_gpu_stats(nsp):
    print("\n" + "═"*70)
    print("  GPU HARDWARE DIAGNOSTICS")
    print("═"*70)

    device = cuda.get_current_device()
    attr   = device.attributes          # dict, pas callable

    name = device.name
    if isinstance(name, bytes):
        name = name.decode()
    cc = device.compute_capability

    print(f"  Device name              : {name}")
    print(f"  Compute capability       : {cc[0]}.{cc[1]}")

    def _a(key, fallback=0):
        return attr.get(key, fallback)

    smem_total = _a('MAX_SHARED_MEMORY_PER_BLOCK')
    regs_block = _a('MAX_REGISTERS_PER_BLOCK')
    regs_sm    = _a('MAX_REGISTERS_PER_MULTIPROCESSOR')
    warps_sm   = _a('MAX_WARPS_PER_MULTIPROCESSOR')
    threads_sm = _a('MAX_THREADS_PER_MULTIPROCESSOR')
    threads_bl = _a('MAX_THREADS_PER_BLOCK')
    sm_count   = _a('MULTIPROCESSOR_COUNT')
    l2_cache   = _a('L2_CACHE_SIZE')
    mem_bw     = _a('GLOBAL_MEMORY_BUS_WIDTH')
    clock_mhz  = _a('CLOCK_RATE') // 1000
    mem_mhz    = _a('MEMORY_CLOCK_RATE') // 1000

    print(f"  SMs                      : {sm_count}")
    print(f"  Max threads/block        : {threads_bl}")
    print(f"  Max threads/SM           : {threads_sm}")
    print(f"  Max warps/SM             : {warps_sm}")
    print(f"  Registers/block          : {regs_block:,}")
    print(f"  Registers/SM             : {regs_sm:,}")
    print(f"  Shared memory/block      : {smem_total/1024:.1f} KB")
    print(f"  L2 cache                 : {l2_cache/1024:.0f} KB")
    print(f"  GPU clock                : {clock_mhz} MHz")
    print(f"  Memory clock             : {mem_mhz} MHz")
    print(f"  Memory bus width         : {mem_bw} bit")

    # ── Intégrateur : mémoire globale (plus de shared) ────────────────────
    n_state = nsp + 1
    print(f"\n  ── Intégrateur (nsp={nsp}, n_state={n_state}) ──")
    print(f"  Architecture mémoire     : GLOBALE (shared supprimée)")
    print(f"  MAX_RESTART              : {MAX_RESTART}")
    print(f"  MAX_NSP (compilé)        : {MAX_NSP}")
    print(f"  THREADS_PER_BLOCK        : {THREADS_PER_BLOCK}  (sans contrainte shared ✓)")
    print(f"  NEWTON_MAXITER           : {NEWTON_MAXITER}")
    print(f"  NEWTON_TOL               : {NEWTON_TOL:.0e}")
    print(f"  GMRES_TOL                : {GMRES_TOL:.0e}")

    print_memory_layout(1, nsp)

    # ── Registres locaux (estimation) ─────────────────────────────────────
    local_arrays = 6    # xk, x0, F_loc, dx_loc, rhs_loc, jvp_loc
    est_regs_per_thread = local_arrays * MAX_NSP * 2 + 80
    est_regs_per_block  = est_regs_per_thread * THREADS_PER_BLOCK
    print(f"\n  ── Pression registres (estimation) ──")
    print(f"  Local arrays ({local_arrays}×{MAX_NSP} doubles) : ~{local_arrays*MAX_NSP*2} regs/thread")
    print(f"  Scalaires divers         : ~80 regs/thread")
    print(f"  Total estimé/thread      : ~{est_regs_per_thread} regs")
    print(f"  Total estimé/bloc        : ~{est_regs_per_block:,} regs  (limit: {regs_block:,})")
    if regs_block > 0 and est_regs_per_block > regs_block:
        print(f"  ⚠ Risque de register spilling → mémoire locale")
    elif regs_block > 0:
        print(f"  Register pressure        : OK ✓  ({est_regs_per_block/regs_block*100:.0f}% du budget)")

    print("═"*70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
#  2. RÉFÉRENCE CANTERA
# ─────────────────────────────────────────────────────────────────────────────

def run_cantera():
    print("Running Cantera reference...")
    gas = ct.Solution(YAML_FILE)
    gas.set_equivalence_ratio(PHI, 'CH4', 'O2:1.0, N2:3.76')
    gas.TP = T0, P0

    reactor = ct.IdealGasReactor(gas, energy='on')
    sim     = ct.ReactorNet([reactor])

    times, T_arr, X_arr, P_arr = [0.0], [T0], [gas.X.copy()], [P0]

    t0 = time.perf_counter()
    while sim.time < T_END:
        sim.step()
        times.append(sim.time)
        T_arr.append(reactor.T)
        X_arr.append(reactor.thermo.X.copy())
        P_arr.append(reactor.thermo.P)
    elapsed = time.perf_counter() - t0

    print(f"  Cantera : {elapsed*1000:.2f} ms  ({len(times)} pas)")
    return (np.array(times), np.array(T_arr), np.array(X_arr),
            np.array(P_arr), list(gas.species_names), elapsed)


# ─────────────────────────────────────────────────────────────────────────────
#  3. INTÉGRATION GPU
# ─────────────────────────────────────────────────────────────────────────────

def run_gpu():
    print("Setting up GPU integrator...")

    lecteur   = thermo.yamlParser(YAML_FILE)
    th_object = lecteur.populateVectorizedThermoObject()
    nsp       = len(th_object.species)

    reactions   = parseReactions(YAML_FILE)
    elem_arrays = build_elementary_gpu_arrays(reactions, th_object)
    tb_arrays   = build_threebody_gpu_arrays(reactions, th_object)
    fo_arrays   = build_falloff_gpu_arrays(reactions, th_object)

    # Conditions initiales
    gas = ct.Solution(YAML_FILE)
    gas.set_equivalence_ratio(PHI, 'CH4', 'O2:1.0, N2:3.76')
    gas.TP = T0, P0
    C0 = gas.concentrations * 1e3   # kmol/m³ → mol/m³

    TC_np = np.zeros((1, nsp + 1), dtype=np.float64)
    TC_np[0, 0]  = T0
    TC_np[0, 1:] = C0

    TC_d          = cuda.to_device(TC_np)
    tmp_T         = cuda.to_device(np.array([T0], dtype=np.float64))
    tmp_C         = cuda.to_device(C0[np.newaxis, :].astype(np.float64))
    tmp_vT        = cuda.to_device(np.zeros(1, dtype=np.float64))
    tmp_vC        = cuda.to_device(np.zeros((1, nsp), dtype=np.float64))
    tmp_omega     = cuda.device_array((1, nsp), dtype=np.float64)
    tmp_jvp_omega = cuda.device_array((1, nsp), dtype=np.float64)

    # Warmup / compilation JIT
    print("  Compilation JIT (1er appel)...")
    t_jit = time.perf_counter()
    TC_out = launch_integrate(
        TC_d, 0.0, DT, 1,
        th_object.coef_low, th_object.coef_high, th_object.Tmid, nsp,
        elem_arrays, tb_arrays, fo_arrays,
        tmp_omega, tmp_jvp_omega,
        tmp_T, tmp_C, tmp_vT, tmp_vC)
    cuda.synchronize()
    t_jit = time.perf_counter() - t_jit
    print(f"  Compilation + 1 pas : {t_jit*1000:.0f} ms")

    # Réinitialisation
    cuda.to_device(TC_np, to=TC_d)

    SAVE_EVERY = max(1, N_STEPS // 500)
    t_save, T_gpu, C_gpu = [0.0], [T0], [C0.copy()]
    t_kernel = 0.0

    print(f"  Intégration : {N_STEPS} pas, sauvegarde tous les {SAVE_EVERY}...")
    t_wall = time.perf_counter()

    for step in range(0, N_STEPS, SAVE_EVERY):
        steps_this = min(SAVE_EVERY, N_STEPS - step)
        t0k = time.perf_counter()
        TC_d = launch_integrate(
            TC_d, step * DT, DT, steps_this,
            th_object.coef_low, th_object.coef_high, th_object.Tmid, nsp,
            elem_arrays, tb_arrays, fo_arrays,
            tmp_omega, tmp_jvp_omega,
            tmp_T, tmp_C, tmp_vT, tmp_vC)
        cuda.synchronize()
        t_kernel += time.perf_counter() - t0k

        TC_h = TC_d.copy_to_host()
        t_save.append((step + steps_this) * DT)
        T_gpu.append(TC_h[0, 0])
        C_gpu.append(TC_h[0, 1:].copy())

    t_wall = time.perf_counter() - t_wall
    print(f"  Wall time total  : {t_wall*1000:.2f} ms")
    print(f"  Kernel only      : {t_kernel*1000:.2f} ms")

    T_gpu = np.array(T_gpu)
    C_gpu = np.array(C_gpu)
    C_tot = C_gpu.sum(axis=1, keepdims=True)
    X_gpu = C_gpu / np.where(C_tot > 0, C_tot, 1.0)

    return (np.array(t_save), T_gpu, X_gpu,
            list(th_object.species), t_wall, t_kernel, t_jit, nsp)


# ─────────────────────────────────────────────────────────────────────────────
#  4. GRAPHIQUES
# ─────────────────────────────────────────────────────────────────────────────

def make_plots(times_can, T_can, X_can, species_can,
               times_gpu, T_gpu, X_gpu, species_gpu,
               elapsed_can, elapsed_gpu, t_kernel, t_jit, nsp):

    fig = plt.figure(figsize=(18, 14), facecolor='#0d1117')
    fig.suptitle(
        'GPU Reactor vs Cantera — GRI-Mech 3.0   CH₄/Air   φ=1   T₀=1700 K',
        fontsize=13, color='white', fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.44, wspace=0.36,
                           left=0.07, right=0.97, top=0.93, bottom=0.07)

    DARK   = '#0d1117'
    PANEL  = '#161b22'
    GRID_C = '#30363d'
    TEXT   = '#e6edf3'
    MUTED  = '#8b949e'
    COLORS = ['#58a6ff', '#f78166', '#3fb950', '#d2a8ff',
              '#ffa657', '#79c0ff', '#56d364', '#ff7b72']

    def _ax(gs_loc):
        ax = fig.add_subplot(gs_loc)
        ax.set_facecolor(PANEL)
        ax.grid(True, color=GRID_C, linewidth=0.5)
        ax.tick_params(colors=MUTED, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID_C)
        return ax

    LK = dict(fontsize=8, color=MUTED)
    TK = dict(fontsize=9, color=TEXT, fontweight='bold', pad=5)
    LEG = dict(fontsize=7, facecolor=PANEL, labelcolor=TEXT,
               edgecolor=GRID_C, loc='best')

    # ── Température ──────────────────────────────────────────────────────
    ax_T = _ax(gs[0, :2])
    ax_T.plot(times_can * 1e3, T_can, color=COLORS[1], lw=2.0,
              label='Cantera', zorder=3)
    ax_T.plot(times_gpu * 1e3, T_gpu, color=COLORS[0], lw=1.5,
              ls='--', label='GPU Numba', zorder=4)
    ax_T.set_xlabel('Temps [ms]', **LK)
    ax_T.set_ylabel('Température [K]', **LK)
    ax_T.set_title('Température', **TK)
    ax_T.legend(**LEG)

    # ── Stats ─────────────────────────────────────────────────────────────
    ax_s = fig.add_subplot(gs[0, 2])
    ax_s.set_facecolor(PANEL)
    ax_s.axis('off')
    speedup_wall   = elapsed_can / elapsed_gpu if elapsed_gpu > 0 else 0
    speedup_kernel = elapsed_can / t_kernel    if t_kernel    > 0 else 0
    gmres_mem_mb   = gmres_buffer_memory_mb(1)

    rows = [
        ('Cantera',        f'{elapsed_can*1000:.1f} ms'),
        ('GPU wall',       f'{elapsed_gpu*1000:.1f} ms'),
        ('GPU kernel',     f'{t_kernel*1000:.1f} ms'),
        ('JIT compile',    f'{t_jit*1000:.0f} ms'),
        ('Speedup wall',   f'{speedup_wall:.1f}×'),
        ('Speedup kernel', f'{speedup_kernel:.1f}×'),
        ('', ''),
        ('DT',             f'{DT:.1e} s'),
        ('N_STEPS',        f'{N_STEPS:,}'),
        ('nsp',            f'{nsp}'),
        ('MAX_RESTART',    f'{MAX_RESTART}'),
        ('Newton maxiter', f'{NEWTON_MAXITER}'),
        ('Newton tol',     f'{NEWTON_TOL:.0e}'),
        ('GMRES tol',      f'{GMRES_TOL:.0e}'),
        ('GMRES mem (×1)', f'{gmres_mem_mb:.3f} MB'),
        ('Mém. arch.',     'Globale'),
    ]
    y = 0.98
    for label, val in rows:
        if not label:
            y -= 0.035
            continue
        color = TEXT
        if '×' in val:
            try:
                v = float(val.replace('×', ''))
                color = '#3fb950' if v >= 1 else '#f78166'
            except Exception:
                pass
        ax_s.text(0.03, y, label + ' :', transform=ax_s.transAxes,
                  fontsize=7, color=MUTED, va='top')
        ax_s.text(0.58, y, val, transform=ax_s.transAxes,
                  fontsize=7, color=color, va='top', fontweight='bold')
        y -= 0.058
    ax_s.set_title('Run Statistics', **TK)

    # ── Erreur T ──────────────────────────────────────────────────────────
    ax_e = _ax(gs[1, :2])
    T_interp = np.interp(times_gpu, times_can, T_can)
    err_T = np.abs(T_gpu - T_interp) / (np.abs(T_interp) + 1.0)
    ax_e.semilogy(times_gpu * 1e3, err_T + 1e-16, color='#d2a8ff', lw=1.2)
    ax_e.axhline(1e-3, color=COLORS[1], ls=':', lw=1.0, label='0.1 %')
    ax_e.axhline(1e-6, color=COLORS[2], ls=':', lw=1.0, label='1 ppm')
    ax_e.set_xlabel('Temps [ms]', **LK)
    ax_e.set_ylabel('|T_gpu − T_can| / T_can', **LK)
    ax_e.set_title('Erreur relative température', **TK)
    ax_e.legend(**LEG)

    # ── Espèces (6 panneaux) ──────────────────────────────────────────────
    positions = [(1, 2), (2, 0), (2, 1), (2, 2), (1, 2), (2, 2)]
    sp_positions = [
        (1, 2), (2, 0), (2, 1), (2, 2),   # 4 premières
    ]
    # Layout : row1 col2 + row2 cols 0,1,2 = 4 cases
    layout = [(1, 2), (2, 0), (2, 1), (2, 2)]
    species_panel = SPECIES_PLOT[:4]

    for sp_idx, (sp, (row, col)) in enumerate(zip(species_panel, layout)):
        ax = _ax(gs[row, col])
        c  = COLORS[sp_idx % len(COLORS)]

        if sp in species_can:
            ic = species_can.index(sp)
            ax.plot(times_can * 1e3, X_can[:, ic], color=c, lw=1.8,
                    label='Cantera', zorder=3)
        if sp in species_gpu:
            ig = species_gpu.index(sp)
            ax.plot(times_gpu * 1e3, X_gpu[:, ig], color='white', lw=1.2,
                    ls='--', alpha=0.85, label='GPU', zorder=4)

        ax.set_xlabel('Temps [ms]', **LK)
        ax.set_ylabel('Fraction molaire', **LK)
        ax.set_title(sp, **TK)
        ax.legend(**LEG)

    out = OUT_PNG
    plt.savefig(out, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"\n  Graphique → {out}")
    plt.close()
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  5. MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    print("\n" + "═"*70)
    print("  REACTOR 0D — GPU Numba vs Cantera")
    print("═"*70)
    print(f"  Mécanisme : {YAML_FILE}")
    print(f"  T0={T0} K   P0={P0/ct.one_atm:.1f} atm   phi={PHI}")
    print(f"  t_end={T_END:.1e} s   dt={DT:.1e} s   steps={N_STEPS:,}")

    # Lire nsp pour les stats
    _lect = thermo.yamlParser(YAML_FILE)
    _th   = _lect.populateVectorizedThermoObject()
    _nsp  = len(_th.species)

    print_gpu_stats(_nsp)

    # Cantera
    (times_can, T_can, X_can, P_can,
     species_can, elapsed_can) = run_cantera()

    # GPU
    (times_gpu, T_gpu, X_gpu, species_gpu,
     elapsed_gpu, t_kernel, t_jit, nsp) = run_gpu()

    # ── Résumé accuracy ──────────────────────────────────────────────────
    print("\n" + "─"*70)
    print("  ACCURACY SUMMARY")
    print("─"*70)

    T_interp = np.interp(times_gpu, times_can, T_can)
    err_T_abs = np.abs(T_gpu - T_interp).max()
    err_T_rel = (np.abs(T_gpu - T_interp) / (T_interp + 1.0)).max()
    print(f"  Température  max abs error : {err_T_abs:.4f} K")
    print(f"  Température  max rel error : {err_T_rel*100:.4f} %")

    for sp in SPECIES_PLOT[:6]:
        if sp in species_can and sp in species_gpu:
            ic = species_can.index(sp)
            ig = species_gpu.index(sp)
            Xc = np.interp(times_gpu, times_can, X_can[:, ic])
            Xg = X_gpu[:, ig]
            scale = max(Xc.max(), 1e-12)
            err = np.abs(Xg - Xc).max() / scale
            print(f"  {sp:<6s}  max rel error (vs max X) : {err*100:.4f} %")

    speedup_wall   = elapsed_can / elapsed_gpu
    speedup_kernel = elapsed_can / t_kernel
    print(f"\n  Speedup (wall)   : {speedup_wall:.1f}×")
    print(f"  Speedup (kernel) : {speedup_kernel:.1f}×")
    print("─"*70)

    # Graphiques
    make_plots(
        times_can, T_can, X_can, species_can,
        times_gpu, T_gpu, X_gpu, species_gpu,
        elapsed_can, elapsed_gpu, t_kernel, t_jit, nsp)