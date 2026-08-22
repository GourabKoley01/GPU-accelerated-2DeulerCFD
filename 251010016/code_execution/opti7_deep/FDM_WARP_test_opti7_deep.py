#%%writefile FDM_WARP_test_opti7.py
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import warp as wp

# ============================================
# WARP INITIALIZATION & OPTIMIZATIONS
# ============================================
wp.config.verify_fp = False    # Disable NaN/Inf checking
wp.config.fast_math = True     # Enable fast math
wp.init()

if wp.is_cuda_available():
    wp.set_device("cuda:0")     # Adjust to your GPU
    device = "cuda:0"
else:
    device = "cpu"

print("Warp devices:", wp.get_devices())
print("Current device:", wp.get_device())

# ============================================
# PARAMETERS
# ============================================
gamma = 1.4
NX, NY = 500, 100
TOTAL_RUNS = 7
IS_PROFILING = False

if len(sys.argv) >= 3:
    NX = int(sys.argv[1])
    NY = int(sys.argv[2])

if len(sys.argv) == 4 and sys.argv[3] == '--profile':
    TOTAL_RUNS = 1
    IS_PROFILING = True
    print("\n--- PROFILING MODE ACTIVE ---\n")
else:
    print(f"\n--- BENCHMARK MODE: {TOTAL_RUNS} runs ---\n")

print(f"Grid: {NX}x{NY}")

NG = 2
Nx, Ny = NX + 2*NG, NY + 2*NG
dx = 1.0 / NX
dy = 1.0 / NY

CFL = 0.5
t_final = 0.2
EPS = 1e-8

NUMERICAL_SCHEME = 'muscl'   # always MUSCL
TIME_STEPPING = 'rk3'        # third‑order SSP
PRINT_EVERY_N_STEPS = 20

# ============================================
# INITIAL CONDITION (SoA NumPy)
# ============================================
rho_np = np.zeros((Nx, Ny), dtype=np.float32)
rhou_np = np.zeros((Nx, Ny), dtype=np.float32)
rhov_np = np.zeros((Nx, Ny), dtype=np.float32)
E_np = np.zeros((Nx, Ny), dtype=np.float32)

for i in range(Nx):
    for j in range(Ny):
        if i < Nx//2:
            rho, u, v, p = 1.0, 0.0, 0.0, 1.0
        else:
            rho, u, v, p = 0.125, 0.0, 0.0, 0.1
        rho_np[i,j] = rho
        rhou_np[i,j] = rho * u
        rhov_np[i,j] = rho * v
        E_np[i,j] = p/(gamma-1.0) + 0.5 * rho * (u*u + v*v)

# ============================================
# WARP KERNELS (REGISTER‑REDUCED FUSION)
# ============================================

@wp.func
def cons_to_prim(U: wp.vec4f, gamma: float, EPS: float):
    rho = wp.max(U[0], EPS)
    inv_rho = 1.0 / rho
    u = U[1] * inv_rho
    v = U[2] * inv_rho
    p = wp.max((gamma - 1.0) * (U[3] - 0.5 * rho * (u*u + v*v)), EPS)
    return rho, u, v, p

@wp.func
def prim_to_cons(rho: float, u: float, v: float, p: float, gamma: float):
    E = p / (gamma - 1.0) + 0.5 * rho * (u*u + v*v)
    return wp.vec4f(rho, rho*u, rho*v, E)

@wp.func
def minmod(a: float, b: float):
    if a * b <= 0.0:
        return 0.0
    return wp.min(a, b) if a > 0.0 else wp.max(a, b)

@wp.func
def reconstruct_state(U_m1: wp.vec4f, U_0: wp.vec4f, U_p1: wp.vec4f, U_p2: wp.vec4f,
                      gamma: float, EPS: float):
    rho_m1, u_m1, v_m1, p_m1 = cons_to_prim(U_m1, gamma, EPS)
    rho_0,  u_0,  v_0,  p_0  = cons_to_prim(U_0,  gamma, EPS)
    rho_p1, u_p1, v_p1, p_p1 = cons_to_prim(U_p1, gamma, EPS)
    rho_p2, u_p2, v_p2, p_p2 = cons_to_prim(U_p2, gamma, EPS)

    # Left state
    rho_L = rho_0 + 0.5 * minmod(rho_0 - rho_m1, rho_p1 - rho_0)
    u_L   = u_0   + 0.5 * minmod(u_0   - u_m1,   u_p1   - u_0)
    v_L   = v_0   + 0.5 * minmod(v_0   - v_m1,   v_p1   - v_0)
    p_L   = p_0   + 0.5 * minmod(p_0   - p_m1,   p_p1   - p_0)

    # Right state
    rho_R = rho_p1 - 0.5 * minmod(rho_p1 - rho_0, rho_p2 - rho_p1)
    u_R   = u_p1   - 0.5 * minmod(u_p1   - u_0,   u_p2   - u_p1)
    v_R   = v_p1   - 0.5 * minmod(v_p1   - v_0,   v_p2   - v_p1)
    p_R   = p_p1   - 0.5 * minmod(p_p1   - p_0,   p_p2   - p_p1)

    return prim_to_cons(rho_L, u_L, v_L, p_L, gamma), prim_to_cons(rho_R, u_R, v_R, p_R, gamma)

@wp.func
def hllc_flux(UL: wp.vec4f, UR: wp.vec4f, gamma: float, EPS: float, dir_y: int):
    if dir_y == 1:
        UL = wp.vec4f(UL[0], UL[2], UL[1], UL[3])
        UR = wp.vec4f(UR[0], UR[2], UR[1], UR[3])

    rhoL = wp.max(UL[0], EPS)
    inv_rhoL = 1.0 / rhoL
    uL = UL[1] * inv_rhoL
    vL = UL[2] * inv_rhoL
    pL = wp.max((gamma - 1.0) * (UL[3] - 0.5 * rhoL * (uL*uL + vL*vL)), EPS)

    rhoR = wp.max(UR[0], EPS)
    inv_rhoR = 1.0 / rhoR
    uR = UR[1] * inv_rhoR
    vR = UR[2] * inv_rhoR
    pR = wp.max((gamma - 1.0) * (UR[3] - 0.5 * rhoR * (uR*uR + vR*vR)), EPS)

    cL = wp.sqrt(gamma * pL * inv_rhoL)
    cR = wp.sqrt(gamma * pR * inv_rhoR)

    sqrt_rhoL = wp.sqrt(rhoL)
    sqrt_rhoR = wp.sqrt(rhoR)
    u_avg = (sqrt_rhoL * uL + sqrt_rhoR * uR) / (sqrt_rhoL + sqrt_rhoR)

    inv_sum_rho = 1.0 / (rhoL + rhoR)
    c_avg = wp.sqrt(gamma * (pL + pR) * inv_sum_rho)

    SL = wp.min(uL - cL, u_avg - c_avg)
    SR = wp.max(uR + cR, u_avg + c_avg)

    FL = wp.vec4f(rhoL * uL, rhoL * uL * uL + pL, rhoL * uL * vL, uL * (UL[3] + pL))
    FR = wp.vec4f(rhoR * uR, rhoR * uR * uR + pR, rhoR * uR * vR, uR * (UR[3] + pR))

    if SL >= 0.0:
        flux = FL
    elif SR <= 0.0:
        flux = FR
    else:
        inv_SR_SL = 1.0 / (SR - SL)
        flux = (SR * FL - SL * FR + SL * SR * (UR - UL)) * inv_SR_SL

    if dir_y == 1:
        flux = wp.vec4f(flux[0], flux[2], flux[1], flux[3])
    return flux

# -------------------- Helper to load a single cell --------------------
@wp.func
def load_U(rho: wp.array(dtype=wp.float32, ndim=2),
           rhou: wp.array(dtype=wp.float32, ndim=2),
           rhov: wp.array(dtype=wp.float32, ndim=2),
           E: wp.array(dtype=wp.float32, ndim=2),
           i: int, j: int):
    return wp.vec4f(rho[i,j], rhou[i,j], rhov[i,j], E[i,j])

# -------------------- Boundary conditions (SoA) --------------------
@wp.kernel
def apply_bc_kernel(rho: wp.array(dtype=wp.float32, ndim=2),
                    rhou: wp.array(dtype=wp.float32, ndim=2),
                    rhov: wp.array(dtype=wp.float32, ndim=2),
                    E: wp.array(dtype=wp.float32, ndim=2),
                    NG: int, Nx: int, Ny: int):
    i, j = wp.tid()
    src_i, src_j = i, j
    if i < NG: src_i = NG
    elif i >= Nx - NG: src_i = Nx - NG - 1
    if j < NG: src_j = NG
    elif j >= Ny - NG: src_j = Ny - NG - 1
    if i != src_i or j != src_j:
        rho[i,j] = rho[src_i, src_j]
        rhou[i,j] = rhou[src_i, src_j]
        rhov[i,j] = rhov[src_i, src_j]
        E[i,j] = E[src_i, src_j]

# -------------------- Compute max speed (for dt) --------------------
@wp.kernel
def compute_max_speed_kernel(rho: wp.array(dtype=wp.float32, ndim=2),
                             rhou: wp.array(dtype=wp.float32, ndim=2),
                             rhov: wp.array(dtype=wp.float32, ndim=2),
                             E: wp.array(dtype=wp.float32, ndim=2),
                             max_speed_arr: wp.array(dtype=float),
                             gamma: float, EPS: float,
                             NG: int, Nx: int, Ny: int):
    i, j = wp.tid()
    if i >= NG and i < Nx - NG and j >= NG and j < Ny - NG:
        U = load_U(rho, rhou, rhov, E, i, j)
        rho_p, u_p, v_p, p_p = cons_to_prim(U, gamma, EPS)
        c = wp.sqrt(gamma * p_p / rho_p)
        speed = wp.abs(u_p) + wp.abs(v_p) + c
        wp.atomic_max(max_speed_arr, 0, speed)

# ---------- EXTREME FUSED RK3 KERNELS (REGISTER‑REDUCED) ----------
# Stage 1: U1 = U + dt * L(U)
@wp.kernel
def rk3_stage1_kernel(
    rho: wp.array(dtype=wp.float32, ndim=2), rhou: wp.array(dtype=wp.float32, ndim=2),
    rhov: wp.array(dtype=wp.float32, ndim=2), E: wp.array(dtype=wp.float32, ndim=2),
    rho1: wp.array(dtype=wp.float32, ndim=2), rhou1: wp.array(dtype=wp.float32, ndim=2),
    rhov1: wp.array(dtype=wp.float32, ndim=2), E1: wp.array(dtype=wp.float32, ndim=2),
    gamma: float, EPS: float, dx: float, dy: float, dt: float,
    NG: int, Nx: int, Ny: int):
    i, j = wp.tid()
    if i >= NG and i < Nx - NG and j >= NG and j < Ny - NG:
        # Load the 5x5 stencil for this cell (only needed indices)
        U_im2 = load_U(rho, rhou, rhov, E, i-2, j)
        U_im1 = load_U(rho, rhou, rhov, E, i-1, j)
        U_i   = load_U(rho, rhou, rhov, E, i,   j)
        U_ip1 = load_U(rho, rhou, rhov, E, i+1, j)
        U_ip2 = load_U(rho, rhou, rhov, E, i+2, j)
        U_jm2 = load_U(rho, rhou, rhov, E, i,   j-2)
        U_jm1 = load_U(rho, rhou, rhov, E, i,   j-1)
        U_jp1 = load_U(rho, rhou, rhov, E, i,   j+1)
        U_jp2 = load_U(rho, rhou, rhov, E, i,   j+2)

        # ---- X direction fluxes ----
        UL_x1, UR_x1 = reconstruct_state(U_im2, U_im1, U_i, U_ip1, gamma, EPS)
        Fx_left = hllc_flux(UL_x1, UR_x1, gamma, EPS, 0)
        UL_x2, UR_x2 = reconstruct_state(U_im1, U_i, U_ip1, U_ip2, gamma, EPS)
        Fx_right = hllc_flux(UL_x2, UR_x2, gamma, EPS, 0)
        rhs_x = (Fx_left - Fx_right) / dx

        # ---- Y direction fluxes ----
        UL_y1, UR_y1 = reconstruct_state(U_jm2, U_jm1, U_i, U_jp1, gamma, EPS)
        Fy_bottom = hllc_flux(UL_y1, UR_y1, gamma, EPS, 1)
        UL_y2, UR_y2 = reconstruct_state(U_jm1, U_i, U_jp1, U_jp2, gamma, EPS)
        Fy_top = hllc_flux(UL_y2, UR_y2, gamma, EPS, 1)
        rhs_y = (Fy_bottom - Fy_top) / dy

        local_rhs = rhs_x + rhs_y
        rho1[i,j]  = U_i[0] + local_rhs[0] * dt
        rhou1[i,j] = U_i[1] + local_rhs[1] * dt
        rhov1[i,j] = U_i[2] + local_rhs[2] * dt
        E1[i,j]    = U_i[3] + local_rhs[3] * dt

# Stage 2: U2 = 0.75*U + 0.25*(U1 + dt*L(U1))
@wp.kernel
def rk3_stage2_kernel(
    rho_n: wp.array(dtype=wp.float32, ndim=2), rhou_n: wp.array(dtype=wp.float32, ndim=2),
    rhov_n: wp.array(dtype=wp.float32, ndim=2), E_n: wp.array(dtype=wp.float32, ndim=2),
    rho1: wp.array(dtype=wp.float32, ndim=2), rhou1: wp.array(dtype=wp.float32, ndim=2),
    rhov1: wp.array(dtype=wp.float32, ndim=2), E1: wp.array(dtype=wp.float32, ndim=2),
    rho2: wp.array(dtype=wp.float32, ndim=2), rhou2: wp.array(dtype=wp.float32, ndim=2),
    rhov2: wp.array(dtype=wp.float32, ndim=2), E2: wp.array(dtype=wp.float32, ndim=2),
    gamma: float, EPS: float, dx: float, dy: float, dt: float,
    NG: int, Nx: int, Ny: int):
    i, j = wp.tid()
    if i >= NG and i < Nx - NG and j >= NG and j < Ny - NG:
        U1_im2 = load_U(rho1, rhou1, rhov1, E1, i-2, j)
        U1_im1 = load_U(rho1, rhou1, rhov1, E1, i-1, j)
        U1_i   = load_U(rho1, rhou1, rhov1, E1, i,   j)
        U1_ip1 = load_U(rho1, rhou1, rhov1, E1, i+1, j)
        U1_ip2 = load_U(rho1, rhou1, rhov1, E1, i+2, j)
        U1_jm2 = load_U(rho1, rhou1, rhov1, E1, i,   j-2)
        U1_jm1 = load_U(rho1, rhou1, rhov1, E1, i,   j-1)
        U1_jp1 = load_U(rho1, rhou1, rhov1, E1, i,   j+1)
        U1_jp2 = load_U(rho1, rhou1, rhov1, E1, i,   j+2)

        # X fluxes on U1
        UL_x1, UR_x1 = reconstruct_state(U1_im2, U1_im1, U1_i, U1_ip1, gamma, EPS)
        Fx_left = hllc_flux(UL_x1, UR_x1, gamma, EPS, 0)
        UL_x2, UR_x2 = reconstruct_state(U1_im1, U1_i, U1_ip1, U1_ip2, gamma, EPS)
        Fx_right = hllc_flux(UL_x2, UR_x2, gamma, EPS, 0)
        rhs_x = (Fx_left - Fx_right) / dx

        # Y fluxes on U1
        UL_y1, UR_y1 = reconstruct_state(U1_jm2, U1_jm1, U1_i, U1_jp1, gamma, EPS)
        Fy_bottom = hllc_flux(UL_y1, UR_y1, gamma, EPS, 1)
        UL_y2, UR_y2 = reconstruct_state(U1_jm1, U1_i, U1_jp1, U1_jp2, gamma, EPS)
        Fy_top = hllc_flux(UL_y2, UR_y2, gamma, EPS, 1)
        rhs_y = (Fy_bottom - Fy_top) / dy

        local_rhs1 = rhs_x + rhs_y
        Un_i = load_U(rho_n, rhou_n, rhov_n, E_n, i, j)

        rho2[i,j]  = 0.75 * Un_i[0] + 0.25 * (U1_i[0] + local_rhs1[0] * dt)
        rhou2[i,j] = 0.75 * Un_i[1] + 0.25 * (U1_i[1] + local_rhs1[1] * dt)
        rhov2[i,j] = 0.75 * Un_i[2] + 0.25 * (U1_i[2] + local_rhs1[2] * dt)
        E2[i,j]    = 0.75 * Un_i[3] + 0.25 * (U1_i[3] + local_rhs1[3] * dt)

# Stage 3: U_new = 1/3*U + 2/3*(U2 + dt*L(U2))
@wp.kernel
def rk3_stage3_kernel(
    rho_n: wp.array(dtype=wp.float32, ndim=2), rhou_n: wp.array(dtype=wp.float32, ndim=2),
    rhov_n: wp.array(dtype=wp.float32, ndim=2), E_n: wp.array(dtype=wp.float32, ndim=2),
    rho2: wp.array(dtype=wp.float32, ndim=2), rhou2: wp.array(dtype=wp.float32, ndim=2),
    rhov2: wp.array(dtype=wp.float32, ndim=2), E2: wp.array(dtype=wp.float32, ndim=2),
    rho_new: wp.array(dtype=wp.float32, ndim=2), rhou_new: wp.array(dtype=wp.float32, ndim=2),
    rhov_new: wp.array(dtype=wp.float32, ndim=2), E_new: wp.array(dtype=wp.float32, ndim=2),
    gamma: float, EPS: float, dx: float, dy: float, dt: float,
    NG: int, Nx: int, Ny: int):
    i, j = wp.tid()
    if i >= NG and i < Nx - NG and j >= NG and j < Ny - NG:
        U2_im2 = load_U(rho2, rhou2, rhov2, E2, i-2, j)
        U2_im1 = load_U(rho2, rhou2, rhov2, E2, i-1, j)
        U2_i   = load_U(rho2, rhou2, rhov2, E2, i,   j)
        U2_ip1 = load_U(rho2, rhou2, rhov2, E2, i+1, j)
        U2_ip2 = load_U(rho2, rhou2, rhov2, E2, i+2, j)
        U2_jm2 = load_U(rho2, rhou2, rhov2, E2, i,   j-2)
        U2_jm1 = load_U(rho2, rhou2, rhov2, E2, i,   j-1)
        U2_jp1 = load_U(rho2, rhou2, rhov2, E2, i,   j+1)
        U2_jp2 = load_U(rho2, rhou2, rhov2, E2, i,   j+2)

        # X fluxes on U2
        UL_x1, UR_x1 = reconstruct_state(U2_im2, U2_im1, U2_i, U2_ip1, gamma, EPS)
        Fx_left = hllc_flux(UL_x1, UR_x1, gamma, EPS, 0)
        UL_x2, UR_x2 = reconstruct_state(U2_im1, U2_i, U2_ip1, U2_ip2, gamma, EPS)
        Fx_right = hllc_flux(UL_x2, UR_x2, gamma, EPS, 0)
        rhs_x = (Fx_left - Fx_right) / dx

        # Y fluxes on U2
        UL_y1, UR_y1 = reconstruct_state(U2_jm2, U2_jm1, U2_i, U2_jp1, gamma, EPS)
        Fy_bottom = hllc_flux(UL_y1, UR_y1, gamma, EPS, 1)
        UL_y2, UR_y2 = reconstruct_state(U2_jm1, U2_i, U2_jp1, U2_jp2, gamma, EPS)
        Fy_top = hllc_flux(UL_y2, UR_y2, gamma, EPS, 1)
        rhs_y = (Fy_bottom - Fy_top) / dy

        local_rhs2 = rhs_x + rhs_y
        Un_i = load_U(rho_n, rhou_n, rhov_n, E_n, i, j)

        rho_new[i,j]  = (1.0/3.0) * Un_i[0] + (2.0/3.0) * (U2_i[0] + local_rhs2[0] * dt)
        rhou_new[i,j] = (1.0/3.0) * Un_i[1] + (2.0/3.0) * (U2_i[1] + local_rhs2[1] * dt)
        rhov_new[i,j] = (1.0/3.0) * Un_i[2] + (2.0/3.0) * (U2_i[2] + local_rhs2[2] * dt)
        E_new[i,j]    = (1.0/3.0) * Un_i[3] + (2.0/3.0) * (U2_i[3] + local_rhs2[3] * dt)

# ============================================
# UTILITY FUNCTIONS (unchanged from opti6)
# ============================================
def get_numpy_state(rho_wp, rhou_wp, rhov_wp, E_wp):
    return np.stack([rho_wp.numpy(), rhou_wp.numpy(), rhov_wp.numpy(), E_wp.numpy()], axis=0)

def exact_sod(x, t, gamma=1.4):
    rhoL, uL, pL = 1.0, 0.0, 1.0
    rhoR, uR, pR = 0.125, 0.0, 0.1
    def f(p, rho, p_i):
        A = 2 / ((gamma + 1) * rho)
        B = (gamma - 1) / (gamma + 1) * p_i
        if p > p_i:
            return (p - p_i) * np.sqrt(A / (p + B))
        else:
            return (2 * np.sqrt(gamma * p_i / rho) / (gamma - 1)) * ((p / p_i)**((gamma - 1)/(2*gamma)) - 1)
    p = 0.5 * (pL + pR)
    for _ in range(50):
        fL, fR = f(p, rhoL, pL), f(p, rhoR, pR)
        func = fL + fR + uR - uL
        dp = 1e-6
        df = (f(p+dp, rhoL, pL) + f(p+dp, rhoR, pR) - fL - fR) / dp
        p -= func / df
        p = max(p, 1e-6)
    p_star, u_star = p, 0.5 * (uL + uR + f(p, rhoR, pR) - f(p, rhoL, pL))
    cL, cR = np.sqrt(gamma * pL / rhoL), np.sqrt(gamma * pR / rhoR)
    SHL = uL - cL
    STL = u_star - np.sqrt(gamma * p_star / (rhoL * (p_star / pL)**(1/gamma)))
    SR = uR + cR * np.sqrt((gamma + 1)/(2*gamma) * (p_star/pR - 1) + 1)
    rho, u, p_out = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
    for i in range(len(x)):
        xi = (x[i] - 0.5) / t
        if xi < SHL:
            rho[i], u[i], p_out[i] = rhoL, uL, pL
        elif xi <= STL:
            u_i = (2/(gamma+1)) * (cL + xi)
            c_i = cL - (gamma-1)/2 * (u_i - uL)
            rho[i], u[i], p_out[i] = rhoL * (c_i/cL)**(2/(gamma-1)), u_i, pL * (c_i/cL)**(2*gamma/(gamma-1))
        elif xi < u_star:
            rho[i], u[i], p_out[i] = rhoL * (p_star/pL)**(1/gamma), u_star, p_star
        elif xi <= SR:
            rho[i], u[i], p_out[i] = rhoR * ((p_star/pR + (gamma-1)/(gamma+1)) / ((gamma-1)/(gamma+1) * p_star/pR + 1)), u_star, p_star
        else:
            rho[i], u[i], p_out[i] = rhoR, uR, pR
    return rho, u, p_out

def generate_plot_filename(base_name, scheme, grid_size, ext='png'):
    return f"{base_name}_{scheme}_opti7_{grid_size[0]}x{grid_size[1]}.{ext}"

def plot_final_results(U, t_final, gamma, NG, NX, NY):
    grid_size = (NX, NY)
    scheme_name = NUMERICAL_SCHEME.upper()
    rho = U[0, NG:-NG, NG:-NG]
    u = U[1, NG:-NG, NG:-NG] / (rho + 1e-10)
    v = U[2, NG:-NG, NG:-NG] / (rho + 1e-10)
    p = (gamma-1)*(U[3, NG:-NG, NG:-NG] - 0.5*rho*(u*u + v*v))
    p = np.maximum(p, 1e-6)
    c = np.sqrt(gamma * p / (rho + 1e-10))
    Mach = np.sqrt(u*u + v*v) / (c + 1e-10)
    grad_x = np.gradient(rho, axis=0)
    grad_y = np.gradient(rho, axis=1)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    schlieren = np.log10(grad_mag / (np.max(grad_mag) + 1e-10) + 1e-10)
    x_phys = np.linspace(0, 1, NX)
    y_phys = np.linspace(0, 1, NY)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    im1 = axes[0,0].imshow(rho.T, origin='lower', cmap='jet', extent=[0, 1, 0, 1], aspect='auto')
    axes[0,0].set_title(f'Density at t={t_final:.3f}')
    axes[0,0].set_xlabel('x'); axes[0,0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0,0])
    im2 = axes[0,1].imshow(p.T, origin='lower', cmap='jet', extent=[0, 1, 0, 1], aspect='auto')
    axes[0,1].set_title(f'Pressure at t={t_final:.3f}')
    axes[0,1].set_xlabel('x'); axes[0,1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[0,1])
    vel_mag = np.sqrt(u**2 + v**2)
    im3 = axes[0,2].imshow(vel_mag.T, origin='lower', cmap='jet', extent=[0, 1, 0, 1], aspect='auto')
    axes[0,2].set_title(f'Velocity Magnitude at t={t_final:.3f}')
    axes[0,2].set_xlabel('x'); axes[0,2].set_ylabel('y')
    plt.colorbar(im3, ax=axes[0,2])
    im4 = axes[1,0].imshow(Mach.T, origin='lower', cmap='jet', extent=[0, 1, 0, 1], aspect='auto')
    axes[1,0].set_title(f'Mach Number at t={t_final:.3f}')
    axes[1,0].set_xlabel('x'); axes[1,0].set_ylabel('y')
    plt.colorbar(im4, ax=axes[1,0])
    im5 = axes[1,1].imshow(schlieren.T, origin='lower', cmap='gray', extent=[0, 1, 0, 1], aspect='auto')
    axes[1,1].set_title('Schlieren (log scale)')
    axes[1,1].set_xlabel('x'); axes[1,1].set_ylabel('y')
    plt.colorbar(im5, ax=axes[1,1])
    stride = max(1, min(NX, NY) // 50)
    X, Y = np.meshgrid(x_phys[::stride], y_phys[::stride])
    U_ds = u[::stride, ::stride]
    V_ds = v[::stride, ::stride]
    axes[1,2].imshow(rho.T, origin='lower', cmap='jet', alpha=0.6, extent=[0, 1, 0, 1], aspect='auto')
    axes[1,2].quiver(X, Y, U_ds.T, V_ds.T, alpha=0.8, color='white', scale=50)
    axes[1,2].set_title('Density with Velocity Vectors')
    axes[1,2].set_xlabel('x'); axes[1,2].set_ylabel('y')
    plt.suptitle(f'2D Riemann Problem - {scheme_name} (opti7), t={t_final:.3f}')
    plt.tight_layout()
    filename = generate_plot_filename('final_results', scheme_name, grid_size)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ {filename}")
    j_mid = NY // 2
    rho_mid = U[0, NG:-NG, j_mid]
    u_mid = U[1, NG:-NG, j_mid] / (rho_mid + 1e-10)
    p_mid = (gamma-1)*(U[3, NG:-NG, j_mid] - 0.5*rho_mid*u_mid**2)
    x = np.linspace(0, 1, NX)
    plt.figure(figsize=(12, 10))
    plt.subplot(3, 1, 1)
    plt.plot(x, rho_mid, 'b-', linewidth=2, label=f'{scheme_name} (GPU)')
    plt.ylabel('Density'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.title(f'Mid-plane (y=0.5) at t={t_final:.3f}')
    plt.subplot(3, 1, 2)
    plt.plot(x, u_mid, 'b-', linewidth=2)
    plt.ylabel('Velocity'); plt.grid(True, alpha=0.3)
    plt.subplot(3, 1, 3)
    plt.plot(x, p_mid, 'b-', linewidth=2)
    plt.ylabel('Pressure'); plt.xlabel('x'); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    midplane_filename = generate_plot_filename('midplane_profiles', scheme_name, grid_size)
    plt.savefig(midplane_filename, dpi=300, bbox_inches='tight')
    print(f"✓ {midplane_filename}")

# ============================================
# MAIN SIMULATION LOOP
# ============================================
print(f"Grid: {NX}×{NY}")
print(f"Scheme: {NUMERICAL_SCHEME.upper()} (GPU - {device})")
print(f"Time stepping: {TIME_STEPPING.upper()}")
print("="*50)

# Allocate SoA arrays
rho_n = wp.array(rho_np, dtype=wp.float32, device=device)
rhou_n = wp.array(rhou_np, dtype=wp.float32, device=device)
rhov_n = wp.array(rhov_np, dtype=wp.float32, device=device)
E_n = wp.array(E_np, dtype=wp.float32, device=device)

rho_new = wp.zeros((Nx, Ny), dtype=wp.float32, device=device)
rhou_new = wp.zeros((Nx, Ny), dtype=wp.float32, device=device)
rhov_new = wp.zeros((Nx, Ny), dtype=wp.float32, device=device)
E_new = wp.zeros((Nx, Ny), dtype=wp.float32, device=device)

rho_1 = wp.zeros((Nx, Ny), dtype=wp.float32, device=device)
rhou_1 = wp.zeros((Nx, Ny), dtype=wp.float32, device=device)
rhov_1 = wp.zeros((Nx, Ny), dtype=wp.float32, device=device)
E_1 = wp.zeros((Nx, Ny), dtype=wp.float32, device=device)

rho_2 = wp.zeros((Nx, Ny), dtype=wp.float32, device=device)
rhou_2 = wp.zeros((Nx, Ny), dtype=wp.float32, device=device)
rhov_2 = wp.zeros((Nx, Ny), dtype=wp.float32, device=device)
E_2 = wp.zeros((Nx, Ny), dtype=wp.float32, device=device)

max_speed_arr = wp.zeros(1, dtype=float, device=device)

# Compute fixed dt
max_speed_arr.fill_(0.0)
wp.launch(compute_max_speed_kernel, dim=(Nx, Ny),
          inputs=[rho_n, rhou_n, rhov_n, E_n, max_speed_arr, gamma, EPS, NG, Nx, Ny],
          device=device)
wp.synchronize()
initial_max_speed = max_speed_arr.numpy()[0]
SAFETY_FACTOR = 0.5
dt_fixed = (CFL * SAFETY_FACTOR) * min(dx, dy) / initial_max_speed
print(f"Initial max speed: {initial_max_speed:.3f}")
print(f"Fixed dt = {dt_fixed:.6f}")
print(f"Estimated steps: {int(t_final/dt_fixed) + 1}")

run_times = []

# Benchmark loop
for run in range(TOTAL_RUNS):
    wp.copy(rho_n, wp.array(rho_np, dtype=wp.float32, device=device))
    wp.copy(rhou_n, wp.array(rhou_np, dtype=wp.float32, device=device))
    wp.copy(rhov_n, wp.array(rhov_np, dtype=wp.float32, device=device))
    wp.copy(E_n, wp.array(E_np, dtype=wp.float32, device=device))
    wp.synchronize()

    t = 0.0
    step = 0
    start_time = time.time()

    while t < t_final:
        if IS_PROFILING and step >= 3:
            break

        wp.launch(apply_bc_kernel, dim=(Nx, Ny),
                  inputs=[rho_n, rhou_n, rhov_n, E_n, NG, Nx, Ny], device=device)

        dt = dt_fixed
        if t + dt > t_final:
            dt = t_final - t

        # RK3 stages with reduced register usage and block_dim=128
        wp.launch(rk3_stage1_kernel, dim=(Nx, Ny), block_dim=128,
                  inputs=[rho_n, rhou_n, rhov_n, E_n,
                          rho_1, rhou_1, rhov_1, E_1,
                          gamma, EPS, dx, dy, float(dt), NG, Nx, Ny], device=device)
        wp.launch(apply_bc_kernel, dim=(Nx, Ny),
                  inputs=[rho_1, rhou_1, rhov_1, E_1, NG, Nx, Ny], device=device)

        wp.launch(rk3_stage2_kernel, dim=(Nx, Ny), block_dim=128,
                  inputs=[rho_n, rhou_n, rhov_n, E_n,
                          rho_1, rhou_1, rhov_1, E_1,
                          rho_2, rhou_2, rhov_2, E_2,
                          gamma, EPS, dx, dy, float(dt), NG, Nx, Ny], device=device)
        wp.launch(apply_bc_kernel, dim=(Nx, Ny),
                  inputs=[rho_2, rhou_2, rhov_2, E_2, NG, Nx, Ny], device=device)

        wp.launch(rk3_stage3_kernel, dim=(Nx, Ny), block_dim=128,
                  inputs=[rho_n, rhou_n, rhov_n, E_n,
                          rho_2, rhou_2, rhov_2, E_2,
                          rho_new, rhou_new, rhov_new, E_new,
                          gamma, EPS, dx, dy, float(dt), NG, Nx, Ny], device=device)

        # Update state
        wp.copy(rho_n, rho_new)
        wp.copy(rhou_n, rhou_new)
        wp.copy(rhov_n, rhov_new)
        wp.copy(E_n, E_new)

        t += dt
        step += 1

    end_time = time.time()
    elapsed = end_time - start_time
    run_times.append(elapsed)
    print(f"Run {run+1}/{TOTAL_RUNS} completed in {elapsed:.4f} s")

# TFLOPS reporting (using the same flop count as opti6 for comparison)
print("\n" + "="*50)
if not IS_PROFILING:
    if TOTAL_RUNS > 1:
        valid_times = run_times[1:]
        best_time = min(valid_times)
        print(f"Discarding warm-up: {run_times[0]:.4f} s")
        print(f"Best time (runs 2-{TOTAL_RUNS}): {best_time:.4f} s")
    else:
        best_time = run_times[0]
        print(f"Execution time: {best_time:.4f} s")

    flops_per_cell_per_step = 12000   # same as opti6 (RK3)
    total_cells = NX * NY
    total_flops = total_cells * step * flops_per_cell_per_step
    tflops = total_flops / (best_time * 1e12)
    print(f"Total steps: {step}")
    print(f"Achieved TFLOPS: {tflops:.6f} TFLOPS")
else:
    print("Profiling mode – ignoring timing.")
print("="*50)

# Post‑processing: plots and validation
print("\n" + "="*50)
print("CREATING FINAL PLOTS")
U_final = get_numpy_state(rho_n, rhou_n, rhov_n, E_n)
plot_final_results(U_final, t_final, gamma, NG, NX, NY)

print("\nVALIDATION WITH EXACT SOD SOLUTION")
j_mid = NY // 2
rho_num = U_final[0, NG:-NG, j_mid]
u_num = U_final[1, NG:-NG, j_mid] / (rho_num + 1e-10)
p_num = (gamma-1)*(U_final[3, NG:-NG, j_mid] - 0.5*rho_num*u_num**2)
x = np.linspace(0, 1, NX)
rho_ex, u_ex, p_ex = exact_sod(x, t_final, gamma)
print(f"L1 errors at t={t_final}:")
print(f"  Density:  {np.mean(np.abs(rho_num - rho_ex)):.6f}")
print(f"  Velocity: {np.mean(np.abs(u_num - u_ex)):.6f}")
print(f"  Pressure: {np.mean(np.abs(p_num - p_ex)):.6f}")

plt.figure(figsize=(12,10))
plt.subplot(3,1,1)
plt.plot(x, rho_num, 'b-', label=f'{NUMERICAL_SCHEME.upper()} (GPU)')
plt.plot(x, rho_ex, 'r--', label='Exact')
plt.ylabel('Density'); plt.legend(); plt.grid(True)
plt.title(f'Sod shock tube validation, t={t_final}')
plt.subplot(3,1,2)
plt.plot(x, u_num, 'b-'); plt.plot(x, u_ex, 'r--')
plt.ylabel('Velocity'); plt.grid(True)
plt.subplot(3,1,3)
plt.plot(x, p_num, 'b-'); plt.plot(x, p_ex, 'r--')
plt.ylabel('Pressure'); plt.xlabel('x'); plt.grid(True)
plt.tight_layout()
validation_filename = generate_plot_filename('validation', NUMERICAL_SCHEME, (NX, NY))
plt.savefig(validation_filename, dpi=300, bbox_inches='tight')
print(f"✓ {validation_filename}\nAll plots generated.")
