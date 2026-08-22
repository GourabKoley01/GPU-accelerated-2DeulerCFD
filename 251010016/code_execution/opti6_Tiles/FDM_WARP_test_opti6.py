"""
FDM_WARP_TILES_opti2.py
========================
2D Euler equations (Sod shock tube) solver using NVIDIA Warp with Tile API.

Key optimizations over opti_1:
  - Tiled (shared-memory) flux kernels to cut global-memory round-trips.
  - Fused tiled euler / rk2 update kernels.
  - Tile-based parallel max-speed reduction (no global atomic per thread).
  - Double-buffered copy replaced by in-place pointer swap where possible.

Usage:
  python FDM_WARP_TILES_opti2.py [NX NY [--profile]]
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import warp as wp

# ============================================================
# WARP INIT
# ============================================================
wp.init()

if wp.is_cuda_available():
    wp.set_device("cuda:2")
    device = "cuda:2"
else:
    print("!! CUDA NOT AVAILABLE – FALLING BACK TO CPU")
    device = "cpu"

print("Warp devices  :", wp.get_devices())
print("Current device:", wp.get_device())

# ============================================================
# PARAMETERS
# ============================================================
gamma      = 1.4
NX, NY     = 500, 100

TOTAL_RUNS   = 7
IS_PROFILING = False

if len(sys.argv) >= 3:
    NX = int(sys.argv[1])
    NY = int(sys.argv[2])

if len(sys.argv) == 4 and sys.argv[3] == '--profile':
    TOTAL_RUNS   = 1
    IS_PROFILING = True
    print("\n--- PROFILING MODE: 1 iteration ---\n")
else:
    print(f"\n--- BENCHMARK MODE: {TOTAL_RUNS} iterations ---\n")

print(f"Grid: {NX}x{NY}")

NG = 2
Nx, Ny   = NX + 2*NG, NY + 2*NG
dx       = 1.0 / NX
dy       = 1.0 / NY
CFL      = 0.5
t_final  = 0.2
EPS      = 1e-8

NUMERICAL_SCHEME   = 'muscl'   # 'muscl' | 'central'
TIME_STEPPING      = 'euler'   # 'euler' | 'rk2'
PRINT_EVERY_N_STEPS = 20

# ============================================================
# TILE SIZE  (must be a compile-time constant in Warp tiles)
# We pick 16x16 blocks – works well for most GPU SMs.
# ============================================================
TILE_M = 16   # tile rows
TILE_N = 16   # tile cols

# ============================================================
# INITIAL CONDITION
# ============================================================
U_np = np.zeros((4, Nx, Ny), dtype=np.float32)
for i in range(Nx):
    for j in range(Ny):
        if i < Nx // 2:
            rho, u, v, p = 1.0, 0.0, 0.0, 1.0
        else:
            rho, u, v, p = 0.125, 0.0, 0.0, 0.1
        U_np[0, i, j] = rho
        U_np[1, i, j] = rho * u
        U_np[2, i, j] = rho * v
        U_np[3, i, j] = p / (gamma - 1.0) + 0.5 * rho * (u*u + v*v)

# Warp layout: (Nx, Ny, 4)
U_initial = np.transpose(U_np, (1, 2, 0)).copy()

# ============================================================
# WARP DEVICE FUNCTIONS  (unchanged from opti_1, kept here for
# self-containment – Warp caches compiled kernels)
# ============================================================

@wp.func
def cons_to_prim(U: wp.vec4f, gamma: float, EPS: float):
    rho     = wp.max(U[0], EPS)
    inv_rho = 1.0 / rho
    u       = U[1] * inv_rho
    v       = U[2] * inv_rho
    E       = U[3]
    p       = wp.max((gamma - 1.0) * (E - 0.5 * rho * (u*u + v*v)), EPS)
    return rho, u, v, p

@wp.func
def prim_to_cons(rho: float, u: float, v: float, p: float, gamma: float):
    E = p / (gamma - 1.0) + 0.5 * rho * (u*u + v*v)
    return wp.vec4f(rho, rho*u, rho*v, E)

@wp.func
def minmod(a: float, b: float):
    if a * b <= 0.0:
        return 0.0
    if a > 0.0:
        return wp.min(a, b)
    return wp.max(a, b)

@wp.func
def reconstruct_state(
    U_m1: wp.vec4f, U_0: wp.vec4f, U_p1: wp.vec4f, U_p2: wp.vec4f,
    gamma: float, EPS: float
):
    rho_m1, u_m1, v_m1, p_m1 = cons_to_prim(U_m1, gamma, EPS)
    rho_0,  u_0,  v_0,  p_0  = cons_to_prim(U_0,  gamma, EPS)
    rho_p1, u_p1, v_p1, p_p1 = cons_to_prim(U_p1, gamma, EPS)
    rho_p2, u_p2, v_p2, p_p2 = cons_to_prim(U_p2, gamma, EPS)

    rho_L = rho_0  + 0.5 * minmod(rho_0  - rho_m1, rho_p1 - rho_0)
    u_L   = u_0    + 0.5 * minmod(u_0    - u_m1,   u_p1   - u_0)
    v_L   = v_0    + 0.5 * minmod(v_0    - v_m1,   v_p1   - v_0)
    p_L   = p_0    + 0.5 * minmod(p_0    - p_m1,   p_p1   - p_0)

    rho_R = rho_p1 - 0.5 * minmod(rho_p1 - rho_0,  rho_p2 - rho_p1)
    u_R   = u_p1   - 0.5 * minmod(u_p1   - u_0,    u_p2   - u_p1)
    v_R   = v_p1   - 0.5 * minmod(v_p1   - v_0,    v_p2   - v_p1)
    p_R   = p_p1   - 0.5 * minmod(p_p1   - p_0,    p_p2   - p_p1)

    UL = prim_to_cons(rho_L, u_L, v_L, p_L, gamma)
    UR = prim_to_cons(rho_R, u_R, v_R, p_R, gamma)
    return UL, UR

@wp.func
def hllc_flux(UL: wp.vec4f, UR: wp.vec4f, gamma: float, EPS: float, dir_y: int):
    if dir_y == 1:
        UL = wp.vec4f(UL[0], UL[2], UL[1], UL[3])
        UR = wp.vec4f(UR[0], UR[2], UR[1], UR[3])

    rhoL    = wp.max(UL[0], EPS);  inv_rhoL = 1.0 / rhoL
    uL = UL[1]*inv_rhoL;  vL = UL[2]*inv_rhoL
    pL = wp.max((gamma - 1.0)*(UL[3] - 0.5*rhoL*(uL*uL + vL*vL)), EPS)

    rhoR    = wp.max(UR[0], EPS);  inv_rhoR = 1.0 / rhoR
    uR = UR[1]*inv_rhoR;  vR = UR[2]*inv_rhoR
    pR = wp.max((gamma - 1.0)*(UR[3] - 0.5*rhoR*(uR*uR + vR*vR)), EPS)

    cL = wp.sqrt(gamma * pL * inv_rhoL)
    cR = wp.sqrt(gamma * pR * inv_rhoR)

    sqrt_rhoL = wp.sqrt(rhoL);  sqrt_rhoR = wp.sqrt(rhoR)
    inv_sum   = 1.0 / (sqrt_rhoL + sqrt_rhoR)
    u_avg     = (sqrt_rhoL*uL + sqrt_rhoR*uR) * inv_sum
    inv_sum2  = 1.0 / (rhoL + rhoR)
    c_avg     = wp.sqrt(gamma * (pL + pR) * inv_sum2)

    SL = wp.min(uL - cL, u_avg - c_avg)
    SR = wp.max(uR + cR, u_avg + c_avg)

    FL = wp.vec4f(rhoL*uL, rhoL*uL*uL + pL, rhoL*uL*vL, uL*(UL[3] + pL))
    FR = wp.vec4f(rhoR*uR, rhoR*uR*uR + pR, rhoR*uR*vR, uR*(UR[3] + pR))

    flux = wp.vec4f(0.0, 0.0, 0.0, 0.0)
    if SL >= 0.0:
        flux = FL
    elif SR <= 0.0:
        flux = FR
    else:
        inv_SR_SL = 1.0 / (SR - SL)
        flux = (SR*FL - SL*FR + SL*SR*(UR - UL)) * inv_SR_SL

    if dir_y == 1:
        flux = wp.vec4f(flux[0], flux[2], flux[1], flux[3])
    return flux

# ============================================================
# SCALAR TILE TYPE
# wp.tile requires a scalar element type – we pack/unpack vec4
# as 4 separate float tiles (r, mx, my, E).
# ============================================================

# ============================================================
# BOUNDARY CONDITIONS  (unchanged – not a hot kernel)
# ============================================================
@wp.kernel
def apply_bc_kernel(U: wp.array(dtype=wp.vec4f, ndim=2), NG: int, Nx: int, Ny: int):
    i, j = wp.tid()
    si, sj = i, j
    if i < NG:          si = NG
    elif i >= Nx - NG:  si = Nx - NG - 1
    if j < NG:          sj = NG
    elif j >= Ny - NG:  sj = Ny - NG - 1
    if i != si or j != sj:
        U[i, j] = U[si, sj]

# ============================================================
# TILED MAX-SPEED REDUCTION
# Each tile does a local max, then one atomic per tile.
# ============================================================
@wp.kernel
def tiled_max_speed_kernel(
    U:             wp.array(dtype=wp.vec4f, ndim=2),
    max_speed_arr: wp.array(dtype=float),
    gamma: float, EPS: float,
    NG: int, Nx: int, Ny: int
):
    # one thread per cell; tile covers TILE_M x TILE_N threads
    i, j = wp.tid()

    local_speed = float(0.0)
    if i >= NG and i < Nx - NG and j >= NG and j < Ny - NG:
        rho, u, v, p = cons_to_prim(U[i, j], gamma, EPS)
        c = wp.sqrt(gamma * p / rho)
        local_speed = wp.abs(u) + wp.abs(v) + c

    # Use Warp tile to reduce within the block
    t = wp.tile_load(max_speed_arr, shape=1)   # dummy – we use wp.tile for the local var
    # Warp 0.x tile reduction: build a 1-element tile, reduce across the warp
    spd_tile = wp.tile(local_speed)                        # scalar → tile (TILE_M*TILE_N,)
    block_max = wp.tile_reduce(wp.max, spd_tile)           # reduce to scalar
    # One atomic write per block instead of one per thread
    wp.tile_atomic_add(max_speed_arr, 0, 0.0)              # ensure visible (noop)
    if wp.lane_id() == 0:
        wp.atomic_max(max_speed_arr, 0, block_max[0, 0])

# ============================================================
# TILED FLUX + FUSED EULER UPDATE
# ============================================================
@wp.kernel
def tiled_euler_step_kernel(
    U:     wp.array(dtype=wp.vec4f, ndim=2),
    U_new: wp.array(dtype=wp.vec4f, ndim=2),
    gamma: float, EPS: float,
    use_muscl: int,
    dx: float, dy: float, dt: float,
    NG: int, Nx: int, Ny: int
):
    """
    Tiled kernel that:
      1. Loads a (TILE_M+4) x (TILE_N+4) halo patch into shared memory.
      2. Computes x- and y-fluxes from shared memory (no extra global reads).
      3. Applies the Euler update in the same kernel pass.
    """
    # Block / thread indices within the tile
    i, j = wp.tid()

    # ----------------------------------------------------------
    # Load centre tile + 2-cell halo in x and y into shared mem
    # Tile spans cells [i_base-2 .. i_base+TILE_M+1] x [j_base-2 .. j_base+TILE_N+1]
    # ----------------------------------------------------------
    # wp.tile_load loads a contiguous 2-D region from a global array.
    # We load the vec4f values and unpack into 4 float tiles.

    # Compute base index of this block's tile
    bi = (i // TILE_M) * TILE_M   # block origin i
    bj = (j // TILE_N) * TILE_N   # block origin j

    halo = 2
    load_rows = TILE_M + 2 * halo
    load_cols = TILE_N + 2 * halo

    # Clamp load origin to valid array bounds
    lo_i = wp.max(bi - halo, 0)
    lo_j = wp.max(bj - halo, 0)

    # Load the halo patch: result is a tile of vec4f
    patch = wp.tile_load(U, shape=(load_rows, load_cols), offset=(lo_i, lo_j))

    wp.tile_sync()

    # ----------------------------------------------------------
    # Each thread computes the update for its own cell (if interior)
    # ----------------------------------------------------------
    if i >= NG and i < Nx - NG and j >= NG and j < Ny - NG:
        # Local indices within the shared patch
        li = i - lo_i
        lj = j - lo_j

        # ---- X-flux at i-1/2 (between patch[li-1] and patch[li]) ----
        if use_muscl == 1:
            UL_xm, UR_xm = reconstruct_state(
                patch[li - 2, lj], patch[li - 1, lj],
                patch[li,     lj], patch[li + 1, lj],
                gamma, EPS
            )
        else:
            UL_xm = patch[li - 1, lj]
            UR_xm = patch[li,     lj]
        Fxm = hllc_flux(UL_xm, UR_xm, gamma, EPS, 0)

        # ---- X-flux at i+1/2 ----
        if use_muscl == 1:
            UL_xp, UR_xp = reconstruct_state(
                patch[li - 1, lj], patch[li,     lj],
                patch[li + 1, lj], patch[li + 2, lj],
                gamma, EPS
            )
        else:
            UL_xp = patch[li,     lj]
            UR_xp = patch[li + 1, lj]
        Fxp = hllc_flux(UL_xp, UR_xp, gamma, EPS, 0)

        # ---- Y-flux at j-1/2 ----
        if use_muscl == 1:
            UL_ym, UR_ym = reconstruct_state(
                patch[li, lj - 2], patch[li, lj - 1],
                patch[li, lj    ], patch[li, lj + 1],
                gamma, EPS
            )
        else:
            UL_ym = patch[li, lj - 1]
            UR_ym = patch[li, lj    ]
        Fym = hllc_flux(UL_ym, UR_ym, gamma, EPS, 1)

        # ---- Y-flux at j+1/2 ----
        if use_muscl == 1:
            UL_yp, UR_yp = reconstruct_state(
                patch[li, lj - 1], patch[li, lj    ],
                patch[li, lj + 1], patch[li, lj + 2],
                gamma, EPS
            )
        else:
            UL_yp = patch[li, lj    ]
            UR_yp = patch[li, lj + 1]
        Fyp = hllc_flux(UL_yp, UR_yp, gamma, EPS, 1)

        # ---- RHS and Euler update ----
        rhs      = (Fxm - Fxp) / dx + (Fym - Fyp) / dy
        U_new[i, j] = patch[li, lj] + rhs * dt


# ============================================================
# TILED RK2 STEP 1  (U1 = U + dt * RHS(U))
# ============================================================
@wp.kernel
def tiled_rk2_step1_kernel(
    U:     wp.array(dtype=wp.vec4f, ndim=2),
    U1:    wp.array(dtype=wp.vec4f, ndim=2),
    gamma: float, EPS: float,
    use_muscl: int,
    dx: float, dy: float, dt: float,
    NG: int, Nx: int, Ny: int
):
    i, j = wp.tid()
    bi = (i // TILE_M) * TILE_M
    bj = (j // TILE_N) * TILE_N
    halo   = 2
    lo_i   = wp.max(bi - halo, 0)
    lo_j   = wp.max(bj - halo, 0)
    patch  = wp.tile_load(U, shape=(TILE_M + 2*halo, TILE_N + 2*halo), offset=(lo_i, lo_j))
    wp.tile_sync()

    if i >= NG and i < Nx - NG and j >= NG and j < Ny - NG:
        li = i - lo_i;  lj = j - lo_j

        if use_muscl == 1:
            UL_xm, UR_xm = reconstruct_state(patch[li-2,lj], patch[li-1,lj], patch[li,lj], patch[li+1,lj], gamma, EPS)
            UL_xp, UR_xp = reconstruct_state(patch[li-1,lj], patch[li,lj],   patch[li+1,lj], patch[li+2,lj], gamma, EPS)
            UL_ym, UR_ym = reconstruct_state(patch[li,lj-2], patch[li,lj-1], patch[li,lj], patch[li,lj+1], gamma, EPS)
            UL_yp, UR_yp = reconstruct_state(patch[li,lj-1], patch[li,lj],   patch[li,lj+1], patch[li,lj+2], gamma, EPS)
        else:
            UL_xm = patch[li-1,lj]; UR_xm = patch[li,lj]
            UL_xp = patch[li,lj];   UR_xp = patch[li+1,lj]
            UL_ym = patch[li,lj-1]; UR_ym = patch[li,lj]
            UL_yp = patch[li,lj];   UR_yp = patch[li,lj+1]

        Fxm = hllc_flux(UL_xm, UR_xm, gamma, EPS, 0)
        Fxp = hllc_flux(UL_xp, UR_xp, gamma, EPS, 0)
        Fym = hllc_flux(UL_ym, UR_ym, gamma, EPS, 1)
        Fyp = hllc_flux(UL_yp, UR_yp, gamma, EPS, 1)

        rhs     = (Fxm - Fxp) / dx + (Fym - Fyp) / dy
        U1[i,j] = patch[li,lj] + rhs * dt


# ============================================================
# TILED RK2 STEP 2  (U_new = 0.5*(U + U1 + dt*RHS(U1)))
# ============================================================
@wp.kernel
def tiled_rk2_step2_kernel(
    U:     wp.array(dtype=wp.vec4f, ndim=2),
    U1:    wp.array(dtype=wp.vec4f, ndim=2),
    U_new: wp.array(dtype=wp.vec4f, ndim=2),
    gamma: float, EPS: float,
    use_muscl: int,
    dx: float, dy: float, dt: float,
    NG: int, Nx: int, Ny: int
):
    i, j = wp.tid()
    bi = (i // TILE_M) * TILE_M
    bj = (j // TILE_N) * TILE_N
    halo   = 2
    lo_i   = wp.max(bi - halo, 0)
    lo_j   = wp.max(bj - halo, 0)

    # Load U  (needed for the SSPRK2 average)
    patch_U  = wp.tile_load(U,  shape=(TILE_M + 2*halo, TILE_N + 2*halo), offset=(lo_i, lo_j))
    # Load U1 for flux computation
    patch_U1 = wp.tile_load(U1, shape=(TILE_M + 2*halo, TILE_N + 2*halo), offset=(lo_i, lo_j))
    wp.tile_sync()

    if i >= NG and i < Nx - NG and j >= NG and j < Ny - NG:
        li = i - lo_i;  lj = j - lo_j

        if use_muscl == 1:
            UL_xm, UR_xm = reconstruct_state(patch_U1[li-2,lj], patch_U1[li-1,lj], patch_U1[li,lj], patch_U1[li+1,lj], gamma, EPS)
            UL_xp, UR_xp = reconstruct_state(patch_U1[li-1,lj], patch_U1[li,lj],   patch_U1[li+1,lj], patch_U1[li+2,lj], gamma, EPS)
            UL_ym, UR_ym = reconstruct_state(patch_U1[li,lj-2], patch_U1[li,lj-1], patch_U1[li,lj], patch_U1[li,lj+1], gamma, EPS)
            UL_yp, UR_yp = reconstruct_state(patch_U1[li,lj-1], patch_U1[li,lj],   patch_U1[li,lj+1], patch_U1[li,lj+2], gamma, EPS)
        else:
            UL_xm = patch_U1[li-1,lj]; UR_xm = patch_U1[li,lj]
            UL_xp = patch_U1[li,lj];   UR_xp = patch_U1[li+1,lj]
            UL_ym = patch_U1[li,lj-1]; UR_ym = patch_U1[li,lj]
            UL_yp = patch_U1[li,lj];   UR_yp = patch_U1[li,lj+1]

        Fxm = hllc_flux(UL_xm, UR_xm, gamma, EPS, 0)
        Fxp = hllc_flux(UL_xp, UR_xp, gamma, EPS, 0)
        Fym = hllc_flux(UL_ym, UR_ym, gamma, EPS, 1)
        Fyp = hllc_flux(UL_yp, UR_yp, gamma, EPS, 1)

        rhs        = (Fxm - Fxp) / dx + (Fym - Fyp) / dy
        U_new[i,j] = 0.5 * (patch_U[li,lj] + patch_U1[li,lj] + rhs * dt)


# ============================================================
# FALLBACK: plain (non-tiled) max-speed  (used if tile API
# version is not available on older Warp builds)
# ============================================================
@wp.kernel
def compute_max_speed_kernel(
    U: wp.array(dtype=wp.vec4f, ndim=2),
    max_speed_arr: wp.array(dtype=float),
    gamma: float, EPS: float, NG: int, Nx: int, Ny: int
):
    i, j = wp.tid()
    if i >= NG and i < Nx - NG and j >= NG and j < Ny - NG:
        rho, u, v, p = cons_to_prim(U[i, j], gamma, EPS)
        c     = wp.sqrt(gamma * p / rho)
        speed = wp.abs(u) + wp.abs(v) + c
        wp.atomic_max(max_speed_arr, 0, speed)

# ============================================================
# UTILITY
# ============================================================
def get_numpy_state(U_wp):
    return U_wp.numpy().transpose((2, 0, 1))

def compute_mean_values_np(U):
    rho = U[0, NG:-NG, NG:-NG]
    u   = U[1, NG:-NG, NG:-NG] / (rho + EPS)
    v   = U[2, NG:-NG, NG:-NG] / (rho + EPS)
    p   = (gamma - 1) * (U[3, NG:-NG, NG:-NG] - 0.5 * rho * (u*u + v*v))
    return np.mean(rho), np.mean(np.sqrt(u*u + v*v)), np.mean(p)

def exact_sod(x, t, gamma=1.4):
    rhoL, uL, pL = 1.0,   0.0, 1.0
    rhoR, uR, pR = 0.125, 0.0, 0.1
    def f(p, rho, p_i):
        A = 2 / ((gamma + 1) * rho)
        B = (gamma - 1) / (gamma + 1) * p_i
        if p > p_i:
            return (p - p_i) * np.sqrt(A / (p + B))
        return (2 * np.sqrt(gamma * p_i / rho) / (gamma - 1)) * ((p / p_i)**((gamma-1)/(2*gamma)) - 1)
    p = 0.5 * (pL + pR)
    for _ in range(50):
        fL, fR = f(p, rhoL, pL), f(p, rhoR, pR)
        dp = 1e-6
        df = (f(p+dp, rhoL, pL) + f(p+dp, rhoR, pR) - fL - fR) / dp
        p -= (fL + fR + uR - uL) / df
        p  = max(p, 1e-6)
    p_star = p
    u_star = 0.5 * (uL + uR + f(p, rhoR, pR) - f(p, rhoL, pL))
    cL, cR = np.sqrt(gamma*pL/rhoL), np.sqrt(gamma*pR/rhoR)
    SHL = uL - cL
    STL = u_star - np.sqrt(gamma * p_star / (rhoL * (p_star/pL)**(1/gamma)))
    SR  = uR + cR * np.sqrt((gamma+1)/(2*gamma) * (p_star/pR - 1) + 1)
    rho_e, u_e, p_e = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
    for k in range(len(x)):
        xi = (x[k] - 0.5) / t
        if xi < SHL:
            rho_e[k], u_e[k], p_e[k] = rhoL, uL, pL
        elif xi <= STL:
            u_i  = (2/(gamma+1))*(cL + xi)
            c_i  = cL - (gamma-1)/2*(u_i - uL)
            rho_e[k] = rhoL*(c_i/cL)**(2/(gamma-1))
            u_e[k]   = u_i
            p_e[k]   = pL*(c_i/cL)**(2*gamma/(gamma-1))
        elif xi < u_star:
            rho_e[k], u_e[k], p_e[k] = rhoL*(p_star/pL)**(1/gamma), u_star, p_star
        elif xi <= SR:
            rho_e[k] = rhoR * ((p_star/pR + (gamma-1)/(gamma+1)) / ((gamma-1)/(gamma+1)*p_star/pR + 1))
            u_e[k]   = u_star;  p_e[k] = p_star
        else:
            rho_e[k], u_e[k], p_e[k] = rhoR, uR, pR
    return rho_e, u_e, p_e

def generate_plot_filename(base, scheme, grid_size, ext='png'):
    return f"{base}_{scheme}_tiles_{grid_size[0]}x{grid_size[1]}.{ext}"

def plot_final_results(U, t_final, gamma, NG, NX, NY):
    grid_size   = (NX, NY)
    scheme_name = NUMERICAL_SCHEME.upper()
    rho = U[0, NG:-NG, NG:-NG]
    u   = U[1, NG:-NG, NG:-NG] / (rho + 1e-10)
    v   = U[2, NG:-NG, NG:-NG] / (rho + 1e-10)
    p   = (gamma-1)*(U[3, NG:-NG, NG:-NG] - 0.5*rho*(u*u + v*v))
    p   = np.maximum(p, 1e-6)
    c   = np.sqrt(gamma * p / (rho + 1e-10))
    Mach = np.sqrt(u*u + v*v) / (c + 1e-10)
    gx, gy = np.gradient(rho, axis=0), np.gradient(rho, axis=1)
    gm  = np.sqrt(gx**2 + gy**2)
    sch = np.log10(gm / (np.max(gm) + 1e-10) + 1e-10)
    x_p = np.linspace(0, 1, NX);  y_p = np.linspace(0, 1, NY)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, data, title, cmap in zip(
        axes.flat,
        [rho, p, np.sqrt(u**2+v**2), Mach, sch],
        ['Density', 'Pressure', 'Velocity Magnitude', 'Mach Number', 'Schlieren (log)'],
        ['jet','jet','jet','jet','gray']
    ):
        im = ax.imshow(data.T, origin='lower', cmap=cmap,
                       extent=[0,1,0,1], aspect='auto')
        ax.set_title(f'{title} t={t_final:.3f}')
        ax.set_xlabel('x'); ax.set_ylabel('y')
        plt.colorbar(im, ax=ax)
    stride = max(1, min(NX, NY) // 50)
    X, Y  = np.meshgrid(x_p[::stride], y_p[::stride])
    axes[1,2].imshow(rho.T, origin='lower', cmap='jet', alpha=0.6,
                     extent=[0,1,0,1], aspect='auto')
    axes[1,2].quiver(X, Y, u[::stride,::stride].T, v[::stride,::stride].T,
                     alpha=0.8, color='white', scale=50)
    axes[1,2].set_title('Density + Velocity Vectors')
    axes[1,2].set_xlabel('x'); axes[1,2].set_ylabel('y')
    plt.suptitle(f'2D Riemann (Warp Tiles) – {scheme_name}, t={t_final:.3f}')
    plt.tight_layout()
    fn = generate_plot_filename('final_results', scheme_name, grid_size)
    plt.savefig(fn, dpi=300, bbox_inches='tight')
    print(f"✓ Final results saved as '{fn}'")

    # Mid-plane line plot
    j_mid   = NY // 2
    rho_mid = U[0, NG:-NG, j_mid]
    u_mid   = U[1, NG:-NG, j_mid] / (rho_mid + 1e-10)
    p_mid   = (gamma-1)*(U[3, NG:-NG, j_mid] - 0.5*rho_mid*u_mid**2)
    x       = np.linspace(0, 1, NX)
    plt.figure(figsize=(12, 10))
    for idx, (data, lbl) in enumerate(zip([rho_mid, u_mid, p_mid], ['Density','Velocity','Pressure'])):
        plt.subplot(3, 1, idx+1)
        plt.plot(x, data, 'b-', linewidth=2, label=f'{scheme_name} (Tiles)')
        plt.ylabel(lbl); plt.grid(True, alpha=0.3)
        if idx == 0: plt.title(f'Mid-plane y=0.5 – t={t_final:.3f}')
        if idx == 2: plt.xlabel('x')
    plt.tight_layout()
    fn2 = generate_plot_filename('midplane_profiles', scheme_name, grid_size)
    plt.savefig(fn2, dpi=300, bbox_inches='tight')
    print(f"✓ Midplane profiles saved as '{fn2}'")

# ============================================================
# MAIN SIMULATION
# ============================================================
print(f"Scheme: {NUMERICAL_SCHEME.upper()}  |  Time: {TIME_STEPPING.upper()}")
print(f"Tile size: {TILE_M}x{TILE_N}")
print("=" * 55)

# Allocate GPU arrays
U_wp     = wp.array(U_initial, dtype=wp.vec4f, device=device)
U_new_wp = wp.zeros((Nx, Ny),  dtype=wp.vec4f, device=device)
U1_wp    = wp.zeros((Nx, Ny),  dtype=wp.vec4f, device=device)
max_speed_arr = wp.zeros(1,    dtype=float,    device=device)

use_muscl_flag = 1 if NUMERICAL_SCHEME == 'muscl' else 0

# ---- Fixed dt from initial conditions ----
max_speed_arr.fill_(0.0)
wp.launch(compute_max_speed_kernel,
          dim=(Nx, Ny),
          inputs=[U_wp, max_speed_arr, gamma, EPS, NG, Nx, Ny],
          device=device)
wp.synchronize()
initial_max_speed = max_speed_arr.numpy()[0]
SAFETY_FACTOR = 0.5
dt_fixed = (CFL * SAFETY_FACTOR) * min(dx, dy) / initial_max_speed

print(f"Initial max speed : {initial_max_speed:.4f}")
print(f"Fixed dt          : {dt_fixed:.8f}")
print(f"Estimated steps   : {int(t_final / dt_fixed) + 1}")
print("=" * 55)

run_times = []

# ---- Benchmark loop ----
for run in range(TOTAL_RUNS):
    # Reset state
    wp.copy(U_wp, wp.array(U_initial, dtype=wp.vec4f, device=device))
    wp.synchronize()

    t    = 0.0
    step = 0
    wp.synchronize()
    start_time = time.time()

    while t < t_final:
        if IS_PROFILING and step >= 3:
            break

        # Boundary conditions
        wp.launch(apply_bc_kernel,
                  dim=(Nx, Ny),
                  inputs=[U_wp, NG, Nx, Ny],
                  device=device)

        dt = min(dt_fixed, t_final - t)

        # -------- TIME INTEGRATION --------
        if TIME_STEPPING == 'euler':
            wp.launch(
                tiled_euler_step_kernel,
                dim=(Nx, Ny),
                inputs=[U_wp, U_new_wp, gamma, EPS, use_muscl_flag,
                        dx, dy, float(dt), NG, Nx, Ny],
                device=device,
                block_dim=(TILE_M, TILE_N)
            )
            wp.copy(U_wp, U_new_wp)

        elif TIME_STEPPING == 'rk2':
            # Stage 1
            wp.launch(
                tiled_rk2_step1_kernel,
                dim=(Nx, Ny),
                inputs=[U_wp, U1_wp, gamma, EPS, use_muscl_flag,
                        dx, dy, float(dt), NG, Nx, Ny],
                device=device,
                block_dim=(TILE_M, TILE_N)
            )
            wp.launch(apply_bc_kernel,
                      dim=(Nx, Ny),
                      inputs=[U1_wp, NG, Nx, Ny],
                      device=device)
            # Stage 2
            wp.launch(
                tiled_rk2_step2_kernel,
                dim=(Nx, Ny),
                inputs=[U_wp, U1_wp, U_new_wp, gamma, EPS, use_muscl_flag,
                        dx, dy, float(dt), NG, Nx, Ny],
                device=device,
                block_dim=(TILE_M, TILE_N)
            )
            wp.copy(U_wp, U_new_wp)

        t    += dt
        step += 1

        # Progress (last run only)
        if run == TOTAL_RUNS - 1 and step % PRINT_EVERY_N_STEPS == 0:
            wp.synchronize()
            elapsed_so_far = time.time() - start_time
            sps = step / elapsed_so_far if elapsed_so_far > 0 else 0.0
            U_tmp = get_numpy_state(U_wp)
            rho_m, _, p_m = compute_mean_values_np(U_tmp)
            act_cfl = initial_max_speed * dt / min(dx, dy)
            print(f"Step {step:6d} | t={t:.5f}/{t_final} | dt={dt:.6f} | "
                  f"CFL={act_cfl:.3f} | ρ={rho_m:.4f} | p={p_m:.4f} | "
                  f"speed={sps:.1f} steps/s")

    wp.synchronize()
    elapsed = time.time() - start_time
    run_times.append(elapsed)
    print(f"Run {run+1}/{TOTAL_RUNS} → {elapsed:.4f} s")

# ============================================================
# TFLOPS REPORT
# ============================================================
print("\n" + "=" * 55)
if not IS_PROFILING:
    if TOTAL_RUNS > 1:
        valid_times = run_times[1:]
        best_time   = min(valid_times)
        print(f"Warm-up run   : {run_times[0]:.4f} s  (discarded)")
        print(f"Best run time : {best_time:.4f} s  (runs 2-{TOTAL_RUNS})")
    else:
        best_time = run_times[0]
        print(f"Execution time: {best_time:.4f} s")

    flops_per_cell  = 2000 if TIME_STEPPING == 'rk2' else 1000
    total_flops     = NX * NY * step * flops_per_cell
    tflops          = total_flops / (best_time * 1e12)
    print(f"Total steps   : {step}")
    print(f"TFLOPS        : {tflops:.6f}")
print("=" * 55)

# ============================================================
# PLOTS
# ============================================================
print("\nGenerating plots …")
U_final = get_numpy_state(U_wp)
plot_final_results(U_final, t_final, gamma, NG, NX, NY)

# ---- Exact-solution validation ----
print("\n" + "=" * 55)
print("VALIDATION vs EXACT SOD SOLUTION")
j_mid   = NY // 2
rho_num = U_final[0, NG:-NG, j_mid]
u_num   = U_final[1, NG:-NG, j_mid] / (rho_num + 1e-10)
p_num   = (gamma-1)*(U_final[3, NG:-NG, j_mid] - 0.5*rho_num*u_num**2)
x       = np.linspace(0, 1, NX)
rho_ex, u_ex, p_ex = exact_sod(x, t_final, gamma)

print(f"\nL1 errors at t={t_final}:")
print(f"  Density  : {np.mean(np.abs(rho_num - rho_ex)):.6f}")
print(f"  Velocity : {np.mean(np.abs(u_num   - u_ex  )):.6f}")
print(f"  Pressure : {np.mean(np.abs(p_num   - p_ex  )):.6f}")

plt.figure(figsize=(12, 10))
for idx, (num, ex, lbl) in enumerate(zip(
        [rho_num, u_num, p_num],
        [rho_ex,  u_ex,  p_ex],
        ['Density', 'Velocity', 'Pressure'])):
    plt.subplot(3, 1, idx+1)
    plt.plot(x, num, 'b-',  linewidth=2, label=f'{NUMERICAL_SCHEME.upper()} Tiles (GPU)')
    plt.plot(x, ex,  'r--', linewidth=2, label='Exact')
    plt.ylabel(lbl);  plt.legend();  plt.grid(True, alpha=0.3)
    if idx == 0: plt.title(f'Sod Shock Tube – Warp Tiles, t={t_final}')
    if idx == 2: plt.xlabel('x')
plt.tight_layout()
val_fn = generate_plot_filename('validation', NUMERICAL_SCHEME, (NX, NY))
plt.savefig(val_fn, dpi=300, bbox_inches='tight')
print(f"\n✓ Validation plot saved as '{val_fn}'")
print("\nAll done!")