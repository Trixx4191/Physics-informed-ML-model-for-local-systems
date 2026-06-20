"""
Synthetic data generator.
Numerically solves the 1D Saint-Venant equations using a Lax-Friedrichs scheme
to produce ground-truth h(x,t) and u(x,t) for fluid presets (river examples).
This data is then sampled sparsely to simulate real sensor networks.
"""

import numpy as np


# ── Fluid (river) presets ───────────────────────────────────────────────────
RIVERS = {
    "volta": {
        "name": "Volta River (Ghana)",
        "length": 50_000,      # 50 km reach (m)
        "S0": 0.0008,          # gentle West-African gradient
        "n_manning": 0.035,    # sandy alluvial channel
        "h0": 3.5,             # base depth (m)
        "u0": 0.8,             # base velocity (m/s)
        "flood_amp": 1.8,      # flood wave amplitude (m)
        "flood_period": 43200, # 12-hour flood wave
    },
    "amazon": {
        "name": "Amazon River (Brazil)",
        "length": 200_000,     # 200 km reach
        "S0": 0.00003,         # extremely flat
        "n_manning": 0.040,    # vegetated banks
        "h0": 12.0,
        "u0": 1.2,
        "flood_amp": 5.0,
        "flood_period": 86400,
    },
    "rhine": {
        "name": "Rhine River (Germany)",
        "length": 80_000,
        "S0": 0.00025,
        "n_manning": 0.028,    # engineered channel
        "h0": 5.0,
        "u0": 1.5,
        "flood_amp": 2.5,
        "flood_period": 28800,
    },
    "dam": {
        "name": "Generic Dam / Reservoir",
        "length": 30_000,      # downstream reach
        "S0": 0.0005,
        "n_manning": 0.025,
        "h0": 4.0,
        "u0": 1.0,
        "flood_amp": 3.0,
        "flood_period": 21600,
        "reservoir_area": 1e6,  # m²
        "Cd": 0.611,
        "gate_width": 10.0,
    },
    "custom": {
        "name": "Custom River",
        "length": 50_000,
        "S0": 0.001,
        "n_manning": 0.030,
        "h0": 2.0,
        "u0": 1.0,
        "flood_amp": 1.0,
        "flood_period": 36000,
    },
}


def solve_saint_venant(river_key="volta", Nx=200, Nt=500, T=86400,
                       n_override=None, custom_params=None):
    """
    Lax-Friedrichs explicit finite difference solver for 1D Saint-Venant.
    Returns arrays: x (Nx,), t (Nt,), h (Nt,Nx), u (Nt,Nx)
    """
    cfg = RIVERS[river_key].copy()
    if custom_params:
        cfg.update(custom_params)
    if n_override is not None:
        cfg["n_manning"] = n_override

    L = cfg["length"]
    S0 = cfg["S0"]
    n = cfg["n_manning"]
    h0 = cfg["h0"]
    u0 = cfg["u0"]
    amp = cfg["flood_amp"]
    period = cfg["flood_period"]
    g = 9.81

    dx = L / Nx
    dt = T / Nt
    x = np.linspace(0, L, Nx)
    t = np.linspace(0, T, Nt)

    # CFL check
    c_max = np.sqrt(g * (h0 + amp)) + (u0 + 0.5)
    cfl = c_max * dt / dx
    if cfl > 0.9:
        # auto-adjust Nt to satisfy CFL
        Nt = int(T * c_max / (0.8 * dx)) + 1
        dt = T / Nt
        t = np.linspace(0, T, Nt)

    # Initial conditions — uniform flow
    h = np.full((Nt, Nx), h0)
    u = np.full((Nt, Nx), u0)
    h[0] = h0
    u[0] = u0

    def friction(h_arr, u_arr):
        return g * n**2 * u_arr * np.abs(u_arr) / (h_arr**(4/3) + 1e-8)

    for k in range(Nt - 1):
        hk = h[k].copy()
        uk = u[k].copy()
        qk = hk * uk

        # Upstream boundary: sinusoidal flood wave
        h_upstream = h0 + amp * np.sin(2 * np.pi * t[k] / period) * max(0, np.sin(np.pi * t[k] / T))
        u_upstream = u0 + 0.3 * np.sin(2 * np.pi * t[k] / period)

        # Lax-Friedrichs for continuity: h_{k+1,i} = 0.5(h_{i+1}+h_{i-1}) - dt/(2dx)*(q_{i+1}-q_{i-1})
        h_new = np.zeros(Nx)
        u_new = np.zeros(Nx)

        # Interior points
        for i in range(1, Nx - 1):
            # Continuity
            h_new[i] = 0.5 * (hk[i+1] + hk[i-1]) - (dt / (2*dx)) * (qk[i+1] - qk[i-1])
            h_new[i] = max(h_new[i], 0.05)  # wet/dry treatment

            # Momentum flux: F = u²h + 0.5·g·h²
            F = uk**2 * hk + 0.5 * g * hk**2
            q_lf = 0.5 * (qk[i+1] + qk[i-1]) - (dt / (2*dx)) * (F[i+1] - F[i-1])
            q_lf += dt * g * hk[i] * S0  # bed-slope source term (stable, non-stiff)

            # Friction is numerically STIFF: at small dt it's negligible, but
            # for the timesteps this explicit scheme actually uses, treating
            # it explicitly (using uk, the OLD velocity) overshoots by several
            # multiples of the discharge itself and flips sign every step —
            # an oscillating blowup, not a genuine flood wave. The standard
            # fix is a semi-implicit friction update: solve
            #   u_new = (q_lf/h_new) / (1 + dt*g*n^2*|u_old|/h_new^(7/3))
            # which damps toward equilibrium instead of overshooting it.
            denom = 1.0 + dt * g * (n**2) * abs(uk[i]) / (h_new[i]**(7.0/3.0) + 1e-8)
            u_new[i] = (q_lf / (h_new[i] + 1e-8)) / denom
            # physical velocity cap as a final safety net, not the primary
            # stabiliser (the semi-implicit step above should rarely hit it)
            u_new[i] = np.clip(u_new[i], -8.0, 8.0)

        # Boundaries
        h_new[0] = h_upstream
        u_new[0] = u_upstream
        h_new[-1] = h_new[-2]   # outflow (zero-gradient)
        u_new[-1] = u_new[-2]

        h[k+1] = h_new
        u[k+1] = u_new

    return x, t, h, u, cfg


def generate_sparse_observations(x, t, h, u, fraction=0.1, noise_std=0.02,
                                  random_seed=42):
    """
    Subsample ground truth to simulate a sparse sensor network.
    fraction: fraction of (x,t) pairs used as training data (0.05 – 1.0)
    noise_std: Gaussian noise level (fraction of signal std)
    Returns: dict with train/test tensors and normalisation stats
    """
    rng = np.random.RandomState(random_seed)
    Nt, Nx = h.shape
    N_total = Nt * Nx
    N_train = max(50, int(N_total * fraction))

    # All grid points
    xx, tt = np.meshgrid(x, t)
    X_all = xx.ravel()
    T_all = tt.ravel()
    H_all = h.ravel()
    U_all = u.ravel()

    # Random sparse sample
    idx = rng.choice(N_total, N_train, replace=False)
    X_train = X_all[idx]
    T_train = T_all[idx]
    H_train = H_all[idx] + rng.randn(N_train) * noise_std * H_all[idx].std()
    U_train = U_all[idx] + rng.randn(N_train) * noise_std * U_all[idx].std()

    # Normalise to [-1, 1]
    x_mean, x_std = X_all.mean(), X_all.std()
    t_mean, t_std = T_all.mean(), T_all.std()
    h_mean, h_std = H_all.mean(), H_all.std()
    u_mean, u_std = U_all.mean(), U_all.std()

    def norm_x(v): return (v - x_mean) / x_std
    def norm_t(v): return (v - t_mean) / t_std
    def norm_h(v): return (v - h_mean) / h_std
    def norm_u(v): return (v - u_mean) / u_std

    stats = dict(x_mean=x_mean, x_std=x_std, t_mean=t_mean, t_std=t_std,
                 h_mean=h_mean, h_std=h_std, u_mean=u_mean, u_std=u_std)

    train = {
        "xt": np.stack([norm_x(X_train), norm_t(T_train)], axis=1).astype(np.float32),
        "h":  norm_h(H_train).reshape(-1, 1).astype(np.float32),
        "u":  norm_u(U_train).reshape(-1, 1).astype(np.float32),
    }
    full = {
        "xt": np.stack([norm_x(X_all), norm_t(T_all)], axis=1).astype(np.float32),
        "h":  norm_h(H_all).reshape(-1, 1).astype(np.float32),
        "u":  norm_u(U_all).reshape(-1, 1).astype(np.float32),
        "h_raw": H_all, "u_raw": U_all,
        "x_raw": X_all, "t_raw": T_all,
    }
    return train, full, stats


def generate_dam_data(T=86400, Nt_res=300, Nx=100, Nt_reach=300,
                      gate_width=None, reservoir_area=None):
    """
    Generate synthetic dam data:
    - Reservoir level Z(t) from mass balance with variable inflow
    - Downstream reach h(x,t), u(x,t) driven by gate outflow
    """
    cfg = RIVERS["dam"].copy()
    if gate_width is not None:
        cfg["gate_width"] = gate_width
    if reservoir_area is not None:
        cfg["reservoir_area"] = reservoir_area
    g = 9.81
    A = cfg["reservoir_area"]
    Cd = cfg["Cd"]
    W = cfg["gate_width"]
    L = cfg["length"]

    t_res = np.linspace(0, T, Nt_res)
    dt = T / Nt_res

    # Variable inflow: base + storm pulse
    Q_in = 50 + 30 * np.sin(2 * np.pi * t_res / (T * 0.4)) * (t_res / T)

    # Euler integration of mass balance: A·dZ/dt = Q_in - Q_out
    Z = np.zeros(Nt_res)
    Z[0] = 8.0  # initial reservoir level (m)
    for k in range(Nt_res - 1):
        Q_out = Cd * W * np.sqrt(g) * max(Z[k], 0)**1.5
        dZ = (Q_in[k] - Q_out) / A
        Z[k+1] = max(Z[k] + dt * dZ, 0.1)

    Q_out_series = Cd * W * np.sqrt(g) * Z**1.5

    # Downstream reach driven by gate outflow as upstream BC
    x_reach, t_reach, h_reach, u_reach, _ = solve_saint_venant(
        "dam", Nx=Nx, Nt=Nt_reach, T=T)

    return {
        "t_res": t_res, "Z": Z, "Q_in": Q_in, "Q_out": Q_out_series,
        "x_reach": x_reach, "t_reach": t_reach,
        "h_reach": h_reach, "u_reach": u_reach,
        "cfg": cfg,
    }
