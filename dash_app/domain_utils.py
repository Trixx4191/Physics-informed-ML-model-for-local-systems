"""
dash_app/domain_utils.py

Shared helpers that bridge the 5 PDEModule domains to the Dash UI:
  - generate ground-truth data for each domain
  - build sparse training sets
  - train via the generic engine
  - build Plotly 3D figures (go.Surface) for visualisation

Keeping this separate from app.py keeps the callback file readable.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import plotly.graph_objects as go
from sklearn.metrics import r2_score

from core.pde_module import PINNEngine, train_pinn, train_pinn_with_extra_params, sample_collocation
from domains.fluids import FluidsPDE, InverseFluidsPDE
from domains.heat import HeatPDE, solve_heat_analytical
from domains.wave import WavePDE, solve_wave_analytical
from domains.gravity import GravityPDE, generate_orbit_data
from domains.elasticity import ElasticityPDE, solve_beam_analytical
from domains.dam import DamSystem, train_dam_system
from data.generator import solve_saint_venant, RIVERS, generate_dam_data


PINN_BLUE = "#534AB7"
TRUE_GRAY = "#5F5E5A"
SURFACE_COLORSCALE = "Viridis"


# ═══════════════════════════════════════════════════════════════════════════════
# Domain registry — single source of truth for the dropdown + dispatch logic
# ═══════════════════════════════════════════════════════════════════════════════

DOMAIN_INFO = {
    "fluids": {
        "label": "Fluids — river flow (Saint-Venant)",
        "equation": "dh/dt + d(uh)/dx = 0,   d(uh)/dt + d(u2h+0.5gh2)/dx + gh(Sf-S0)=0",
        "field_dims": 2,   # (x, t) grid -> surface
        "outputs": ["h", "u"],
    },
    "heat": {
        "label": "Heat diffusion",
        "equation": "dT/dt = alpha * d2T/dx2",
        "field_dims": 2,
        "outputs": ["T"],
    },
    "wave": {
        "label": "Wave propagation",
        "equation": "d2u/dt2 = c2 * d2u/dx2",
        "field_dims": 2,
        "outputs": ["u"],
    },
    "gravity": {
        "label": "Gravity — orbital mechanics",
        "equation": "d2x/dt2 = -GM x / r3,   d2y/dt2 = -GM y / r3",
        "field_dims": 1,   # trajectory, not a field
        "outputs": ["x", "y"],
    },
    "elasticity": {
        "label": "Elasticity — beam bending",
        "equation": "EI * d4v/dx4 = q(x)",
        "field_dims": 1,   # 1D curve, not a 2D field
        "outputs": ["v"],
    },
    "dam": {
        "label": "Dam & reservoir (coupled ODE + PDE)",
        "equation": "A*dZ/dt = Q_in(t) - Q_out(Z),   plus Saint-Venant in the downstream reach",
        "field_dims": 1,   # reservoir level over time; reach gets its own 3D surface
        "outputs": ["Z", "h", "u"],
    },
    "inverse_fluids": {
        "label": "Fluids — infer roughness (inverse)",
        "equation": "Saint-Venant, with n(x) LEARNED rather than supplied",
        "field_dims": 2,
        "outputs": ["h", "u", "n(x)"],
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Data generation per domain
# ═══════════════════════════════════════════════════════════════════════════════

def generate_domain_data(domain_key, river_key="volta", **kwargs):
    """
    Returns a dict with whatever's needed to train + plot that domain.
    Keys vary by domain but always include 'coords_full', 'targets_full'
    in normalised form for training, plus raw arrays for plotting.
    """
    if domain_key == "fluids":
        x, t, h, u, cfg = solve_saint_venant(river_key, Nx=50, Nt=80)
        Nt_r, Nx_r = h.shape
        xx, tt = np.meshgrid(x, t)
        return dict(x=x, t=t, h=h, u=u, cfg=cfg,
                    X=xx.ravel(), T=tt.ravel(),
                    H=h.ravel(), U=u.ravel(),
                    grid_shape=(Nt_r, Nx_r))

    if domain_key == "heat":
        alpha = kwargs.get("alpha", 0.05)
        x = np.linspace(0, 1, 50)
        t = np.linspace(0, 2, 50)
        T_field = solve_heat_analytical(x, t, alpha=alpha, T0=100.0)
        xx, tt = np.meshgrid(x, t)
        return dict(x=x, t=t, T=T_field, alpha=alpha,
                    X=xx.ravel(), Tt=tt.ravel(), Y=T_field.ravel(),
                    grid_shape=T_field.shape)

    if domain_key == "wave":
        c = kwargs.get("c", 1.0)
        mode = kwargs.get("mode", 1)
        x = np.linspace(0, 1, 50)
        t = np.linspace(0, 2, 50)
        U_field = solve_wave_analytical(x, t, c=c, mode=mode, amplitude=1.0)
        xx, tt = np.meshgrid(x, t)
        return dict(x=x, t=t, U=U_field, c=c, mode=mode,
                    X=xx.ravel(), Tt=tt.ravel(), Y=U_field.ravel(),
                    grid_shape=U_field.shape)

    if domain_key == "gravity":
        e = kwargs.get("eccentricity", 0.3)
        GM = kwargs.get("GM", 1.0)
        t, x_orb, y_orb, period = generate_orbit_data(
            GM=GM, a=1.0, e=e, n_points=300, n_periods=1)
        return dict(t=t, x=x_orb, y=y_orb, period=period, GM=GM, e=e)

    if domain_key == "elasticity":
        load_type = kwargs.get("load_type", "uniform")
        EI = kwargs.get("EI", 1.0)
        q0 = kwargs.get("q0", 1.0)
        x = np.linspace(0, 1, 100)
        v_true = solve_beam_analytical(x, L=1.0, EI=EI, q0=q0)
        return dict(x=x, v=v_true, EI=EI, q0=q0, load_type=load_type)

    if domain_key == "dam":
        dam_data = generate_dam_data(Nt_res=150, Nx=40, Nt_reach=150)
        return dam_data

    if domain_key == "inverse_fluids":
        true_n = kwargs.get("true_n", 0.035)
        x, t, h, u, cfg = solve_saint_venant(
            river_key, Nx=50, Nt=80, n_override=true_n)
        xx, tt = np.meshgrid(x, t)
        return dict(x=x, t=t, h=h, u=u, cfg=cfg, true_n=true_n,
                    X=xx.ravel(), T=tt.ravel(), H=h.ravel(), U=u.ravel(),
                    grid_shape=h.shape)

    raise ValueError(f"Unknown domain: {domain_key}")


# ═══════════════════════════════════════════════════════════════════════════════
# Training dispatch — builds the right PDEModule + sparse split + trains
# ═══════════════════════════════════════════════════════════════════════════════

def train_domain(domain_key, data, fraction=0.15, n_epochs=1500, lr=1e-3,
                 lambda_pde=0.1, hidden=64, depth=4, seed=0, callback=None):
    """
    Universal training dispatcher.
    Returns: model, history, stats (normalisation), extra (predictions etc.)
    """
    rng = np.random.RandomState(seed)

    if domain_key == "fluids":
        cfg = data["cfg"]
        pde = FluidsPDE(S0=cfg["S0"], n_manning=cfg["n_manning"])
        model = PINNEngine(pde, hidden=hidden, depth=depth)

        N = len(data["X"])
        n_train = max(30, int(N * fraction))
        idx = rng.choice(N, n_train, replace=False)

        x_m, x_s = data["X"].mean(), data["X"].std()
        t_m, t_s = data["T"].mean(), data["T"].std()
        h_m, h_s = data["H"].mean(), data["H"].std()
        u_m, u_s = data["U"].mean(), data["U"].std()

        coords = torch.tensor(np.stack([
            (data["X"][idx]-x_m)/x_s, (data["T"][idx]-t_m)/t_s], axis=1),
            dtype=torch.float32)
        targets = torch.tensor(np.stack([
            (data["H"][idx]-h_m)/h_s, (data["U"][idx]-u_m)/u_s], axis=1),
            dtype=torch.float32)

        model, hist = train_pinn(model, coords, targets, n_epochs=n_epochs,
                                 lr=lr, lambda_pde=lambda_pde, callback=callback)

        coords_full = torch.tensor(np.stack([
            (data["X"]-x_m)/x_s, (data["T"]-t_m)/t_s], axis=1), dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            pred = model(coords_full)
        h_pred = pred[:, 0].numpy() * h_s + h_m
        u_pred = pred[:, 1].numpy() * u_s + u_m
        r2 = r2_score(data["H"], h_pred)
        return model, hist, dict(x_m=x_m,x_s=x_s,t_m=t_m,t_s=t_s,h_m=h_m,h_s=h_s,u_m=u_m,u_s=u_s), \
               dict(h_pred=h_pred.reshape(data["grid_shape"]),
                    u_pred=u_pred.reshape(data["grid_shape"]), r2=r2, n_train=n_train)

    if domain_key == "heat":
        pde = HeatPDE(alpha=data["alpha"])
        model = PINNEngine(pde, hidden=hidden, depth=depth)

        N = len(data["X"])
        n_train = max(20, int(N * fraction))
        idx = rng.choice(N, n_train, replace=False)
        x_m, x_s = data["X"].mean(), data["X"].std()
        t_m, t_s = data["Tt"].mean(), data["Tt"].std()
        y_m, y_s = data["Y"].mean(), data["Y"].std() + 1e-8

        coords = torch.tensor(np.stack([
            (data["X"][idx]-x_m)/x_s, (data["Tt"][idx]-t_m)/t_s], axis=1),
            dtype=torch.float32)
        targets = torch.tensor(((data["Y"][idx]-y_m)/y_s).reshape(-1,1), dtype=torch.float32)

        model, hist = train_pinn(model, coords, targets, n_epochs=n_epochs,
                                 lr=lr, lambda_pde=lambda_pde, callback=callback)

        coords_full = torch.tensor(np.stack([
            (data["X"]-x_m)/x_s, (data["Tt"]-t_m)/t_s], axis=1), dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            pred = model(coords_full)
        T_pred = (pred[:,0].numpy() * y_s + y_m).reshape(data["grid_shape"])
        r2 = r2_score(data["Y"], T_pred.ravel())
        return model, hist, dict(x_m=x_m,x_s=x_s,t_m=t_m,t_s=t_s,y_m=y_m,y_s=y_s), \
               dict(T_pred=T_pred, r2=r2, n_train=n_train)

    if domain_key == "wave":
        pde = WavePDE(c=data["c"])
        model = PINNEngine(pde, hidden=hidden, depth=depth)

        N = len(data["X"])
        n_train = max(20, int(N * fraction))
        idx = rng.choice(N, n_train, replace=False)
        x_m, x_s = data["X"].mean(), data["X"].std()
        t_m, t_s = data["Tt"].mean(), data["Tt"].std()
        y_m, y_s = data["Y"].mean(), data["Y"].std() + 1e-8

        coords = torch.tensor(np.stack([
            (data["X"][idx]-x_m)/x_s, (data["Tt"][idx]-t_m)/t_s], axis=1),
            dtype=torch.float32)
        targets = torch.tensor(((data["Y"][idx]-y_m)/y_s).reshape(-1,1), dtype=torch.float32)

        model, hist = train_pinn(model, coords, targets, n_epochs=n_epochs,
                                 lr=lr, lambda_pde=lambda_pde, callback=callback)

        coords_full = torch.tensor(np.stack([
            (data["X"]-x_m)/x_s, (data["Tt"]-t_m)/t_s], axis=1), dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            pred = model(coords_full)
        U_pred = (pred[:,0].numpy() * y_s + y_m).reshape(data["grid_shape"])
        r2 = r2_score(data["Y"], U_pred.ravel())
        return model, hist, dict(x_m=x_m,x_s=x_s,t_m=t_m,t_s=t_s,y_m=y_m,y_s=y_s), \
               dict(U_pred=U_pred, r2=r2, n_train=n_train)

    if domain_key == "gravity":
        pde = GravityPDE(GM=data["GM"])
        model = PINNEngine(pde, hidden=hidden, depth=depth)

        N = len(data["t"])
        n_train = max(10, int(N * fraction))
        idx = rng.choice(N, n_train, replace=False)
        t_m, t_s = data["t"].mean(), data["t"].std()
        x_m, x_s = data["x"].mean(), data["x"].std()
        y_m, y_s = data["y"].mean(), data["y"].std()

        coords = torch.tensor(((data["t"][idx]-t_m)/t_s).reshape(-1,1), dtype=torch.float32)
        targets = torch.tensor(np.stack([
            (data["x"][idx]-x_m)/x_s, (data["y"][idx]-y_m)/y_s], axis=1), dtype=torch.float32)

        model, hist = train_pinn(model, coords, targets, n_epochs=n_epochs,
                                 lr=lr, lambda_pde=lambda_pde,
                                 colloc_ranges=[(-2,2)], callback=callback)

        coords_full = torch.tensor(((data["t"]-t_m)/t_s).reshape(-1,1), dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            pred = model(coords_full)
        x_pred = pred[:,0].numpy() * x_s + x_m
        y_pred = pred[:,1].numpy() * y_s + y_m
        r2x = r2_score(data["x"], x_pred)
        r2y = r2_score(data["y"], y_pred)
        return model, hist, dict(t_m=t_m,t_s=t_s,x_m=x_m,x_s=x_s,y_m=y_m,y_s=y_s), \
               dict(x_pred=x_pred, y_pred=y_pred, r2x=r2x, r2y=r2y, n_train=n_train)

    if domain_key == "elasticity":
        pde = ElasticityPDE(EI=data["EI"], q0=data["q0"], load_type=data["load_type"])
        model = PINNEngine(pde, hidden=hidden, depth=depth)

        N = len(data["x"])
        n_train = max(8, int(N * fraction))
        idx = rng.choice(N, n_train, replace=False)
        x_m, x_s = data["x"].mean(), data["x"].std()
        v_m, v_s = data["v"].mean(), data["v"].std() + 1e-8

        coords = torch.tensor(((data["x"][idx]-x_m)/x_s).reshape(-1,1), dtype=torch.float32)
        targets = torch.tensor(((data["v"][idx]-v_m)/v_s).reshape(-1,1), dtype=torch.float32)

        model, hist = train_pinn(model, coords, targets, n_epochs=n_epochs,
                                 lr=lr*0.5, lambda_pde=0.01,
                                 colloc_ranges=[(-2,2)], callback=callback)

        coords_full = torch.tensor(((data["x"]-x_m)/x_s).reshape(-1,1), dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            pred = model(coords_full)
        v_pred = pred[:,0].numpy() * v_s + v_m
        r2 = r2_score(data["v"], v_pred)
        return model, hist, dict(x_m=x_m,x_s=x_s,v_m=v_m,v_s=v_s), \
               dict(v_pred=v_pred, r2=r2, n_train=n_train)

    if domain_key == "dam":
        model, hist = train_dam_system(data, n_epochs=n_epochs, lr=lr,
                                       lambda_pde=lambda_pde, callback=callback)

        t_res = data["t_res"]
        t_norm = (t_res - t_res.mean()) / t_res.std()
        t_tensor = torch.tensor(t_norm.reshape(-1,1), dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            Z_pred_raw = model.reservoir_level(t_tensor).numpy().ravel()
        Z_true = data["Z"]
        Z_pred = ((Z_pred_raw - Z_pred_raw.mean())/(Z_pred_raw.std()+1e-8)
                 * Z_true.std() + Z_true.mean())
        r2_z = r2_score(Z_true, Z_pred)

        return model, hist, dict(), dict(Z_pred=Z_pred, r2_z=r2_z,
                                         n_train=len(t_res))

    if domain_key == "inverse_fluids":
        cfg = data["cfg"]
        pde = InverseFluidsPDE(S0=cfg["S0"])
        model = PINNEngine(pde, hidden=hidden, depth=depth)

        N = len(data["X"])
        n_train = max(30, int(N * fraction))
        idx = rng.choice(N, n_train, replace=False)
        x_m, x_s = data["X"].mean(), data["X"].std()
        t_m, t_s = data["T"].mean(), data["T"].std()
        h_m, h_s = data["H"].mean(), data["H"].std()
        u_m, u_s = data["U"].mean(), data["U"].std()

        coords = torch.tensor(np.stack([
            (data["X"][idx]-x_m)/x_s, (data["T"][idx]-t_m)/t_s], axis=1),
            dtype=torch.float32)
        targets = torch.tensor(np.stack([
            (data["H"][idx]-h_m)/h_s, (data["U"][idx]-u_m)/u_s], axis=1),
            dtype=torch.float32)

        def n_tracker():
            x_eval = torch.linspace(-1, 1, 100).unsqueeze(1)
            with torch.no_grad():
                return pde.get_n(x_eval).mean().item()

        model, hist = train_pinn_with_extra_params(
            model, coords, targets,
            extra_params=pde.n_net.parameters(),
            extra_penalty_fn=pde.smoothness_penalty,
            n_epochs=n_epochs, lr=lr, lambda_pde=lambda_pde, lambda_extra=0.005,
            history_extra_key="n_inferred", history_extra_fn=n_tracker,
            callback=callback)

        x_eval = torch.linspace(-1, 1, 200).unsqueeze(1)
        model.eval()
        with torch.no_grad():
            n_field = pde.get_n(x_eval).numpy().ravel()
        n_final = hist["n_inferred"][-1]
        true_n = data["true_n"]
        err_pct = abs(n_final - true_n) / true_n * 100

        return model, hist, dict(), dict(n_field=n_field, n_final=n_final,
                                         true_n=true_n, err_pct=err_pct,
                                         n_train=n_train)

    raise ValueError(f"Unknown domain: {domain_key}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plotly 3D figure builders
# ═══════════════════════════════════════════════════════════════════════════════

def make_3d_surface_pair(x, t, Z_true, Z_pred, x_label="x", t_label="t",
                         z_label="value", title_true="Ground truth",
                         title_pred="PINN prediction"):
    """Two side-by-side 3D surfaces sharing a camera, for comparison."""
    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=x, y=t, z=Z_true, colorscale=SURFACE_COLORSCALE,
        showscale=False, opacity=0.95, name=title_true,
        contours=dict(z=dict(show=True, usecolormap=True, project_z=True))
    ))
    fig.update_layout(
        scene=dict(
            xaxis_title=x_label, yaxis_title=t_label, zaxis_title=z_label,
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        title=title_true,
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig2 = go.Figure()
    fig2.add_trace(go.Surface(
        x=x, y=t, z=Z_pred, colorscale=SURFACE_COLORSCALE,
        showscale=True, opacity=0.95, name=title_pred,
        contours=dict(z=dict(show=True, usecolormap=True, project_z=True))
    ))
    fig2.update_layout(
        scene=dict(
            xaxis_title=x_label, yaxis_title=t_label, zaxis_title=z_label,
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        title=title_pred,
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig, fig2


def make_orbit_3d(x_true, y_true, x_pred=None, y_pred=None, train_idx=None):
    """3D-style orbit plot (z=0 plane, but rendered in a 3D scene so the
    user can tilt and see it as a literal orbital plane)."""
    fig = go.Figure()
    z_true = np.zeros_like(x_true)
    fig.add_trace(go.Scatter3d(
        x=x_true, y=y_true, z=z_true, mode="lines",
        line=dict(color=TRUE_GRAY, width=5), name="True orbit"
    ))
    if x_pred is not None:
        fig.add_trace(go.Scatter3d(
            x=x_pred, y=y_pred, z=np.zeros_like(x_pred), mode="lines",
            line=dict(color=PINN_BLUE, width=4, dash="dash"), name="PINN orbit"
        ))
    if train_idx is not None:
        fig.add_trace(go.Scatter3d(
            x=x_true[train_idx], y=y_true[train_idx], z=np.zeros(len(train_idx)),
            mode="markers", marker=dict(color="#D85A30", size=5),
            name="Training points (sparse)"
        ))
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0], mode="markers",
        marker=dict(color="#EF9F27", size=14, symbol="circle"),
        name="Central body"
    ))
    fig.update_layout(
        scene=dict(
            xaxis_title="x", yaxis_title="y", zaxis_title="",
            zaxis=dict(showticklabels=False, range=[-0.3, 0.3]),
            aspectmode="data",
            camera=dict(eye=dict(x=0.3, y=-0.3, z=1.8)),
        ),
        margin=dict(l=0, r=0, t=10, b=0),
        height=480,
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.05),
    )
    return fig


def make_beam_3d(x, v_true, v_pred=None, train_idx=None, width=0.15):
    """Render the beam as an actual 3D ribbon (extruded along a fake width
    axis) so deflection reads as a literal bending shape, not just a line."""
    fig = go.Figure()
    y_strip = np.array([-width/2, width/2])
    X, Y = np.meshgrid(x, y_strip)
    Z_true = np.tile(v_true, (2, 1))
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z_true, colorscale="Greys", showscale=False,
        opacity=0.9, name="True beam shape"
    ))
    if v_pred is not None:
        Z_pred = np.tile(v_pred, (2, 1))
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z_pred + 0.0005, colorscale=SURFACE_COLORSCALE,
            showscale=False, opacity=0.55, name="PINN prediction"
        ))
    if train_idx is not None:
        fig.add_trace(go.Scatter3d(
            x=x[train_idx], y=np.zeros(len(train_idx)), z=v_true[train_idx] + 0.001,
            mode="markers", marker=dict(color="#D85A30", size=5),
            name="Sensor points"
        ))
    fig.update_layout(
        scene=dict(
            xaxis_title="Position along beam", yaxis_title="",
            zaxis_title="Deflection",
            yaxis=dict(showticklabels=False),
            aspectmode="manual", aspectratio=dict(x=2, y=0.3, z=0.6),
            camera=dict(eye=dict(x=1.3, y=-1.8, z=0.9)),
        ),
        margin=dict(l=0, r=0, t=10, b=0),
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def make_loss_curve_fig(history):
    """Standard 2D loss curve — kept flat, 3D would add nothing here."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history["epoch"], y=history["loss_total"],
                             mode="lines", name="Total", line=dict(color=PINN_BLUE, width=2.5)))
    fig.add_trace(go.Scatter(x=history["epoch"], y=history["loss_data"],
                             mode="lines", name="Data", line=dict(color="#378ADD", width=1.5, dash="dash")))
    fig.add_trace(go.Scatter(x=history["epoch"], y=history["loss_pde"],
                             mode="lines", name="Physics", line=dict(color="#D85A30", width=1.5, dash="dot")))
    fig.update_layout(
        yaxis_type="log", height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.15),
        xaxis_title="Epoch", yaxis_title="Loss",
    )
    return fig


def make_dam_loss_curve_fig(history):
    """Dam's history dict has different keys (loss_reservoir, loss_reach
    instead of loss_data) since it's a composite two-network system, not
    a single PDEModule -- a separate figure builder is the honest choice
    rather than papering over the key mismatch."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history["epoch"], y=history["loss_total"],
                             mode="lines", name="Total", line=dict(color=PINN_BLUE, width=2.5)))
    fig.add_trace(go.Scatter(x=history["epoch"], y=history["loss_reservoir"],
                             mode="lines", name="Reservoir", line=dict(color="#378ADD", width=1.5, dash="dash")))
    fig.add_trace(go.Scatter(x=history["epoch"], y=history["loss_reach"],
                             mode="lines", name="Reach data", line=dict(color="#1D9E75", width=1.5, dash="dashdot")))
    fig.add_trace(go.Scatter(x=history["epoch"], y=history["loss_pde"],
                             mode="lines", name="Reach physics", line=dict(color="#D85A30", width=1.5, dash="dot")))
    fig.update_layout(
        yaxis_type="log", height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.15),
        xaxis_title="Epoch", yaxis_title="Loss",
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Dam-specific figure builders
# ═══════════════════════════════════════════════════════════════════════════════

def make_dam_reservoir_fig(t_res, Z_true, Z_pred, Q_in, Q_out):
    """2D is the right call here -- reservoir level over time is a single
    curve, not a field; a 3D surface would have nothing to show on the
    extra axis. Flat plots stay flat when that's genuinely the clearer view."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_res/3600, y=Z_true, mode="lines",
                             name="True Z(t)", line=dict(color=TRUE_GRAY, width=3)))
    fig.add_trace(go.Scatter(x=t_res/3600, y=Z_pred, mode="lines",
                             name="PINN Z(t)", line=dict(color=PINN_BLUE, width=2.5, dash="dash")))
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title="Reservoir level — prediction vs ground truth",
        xaxis_title="Time (hr)", yaxis_title="Z (m)",
        legend=dict(orientation="h", y=1.15),
    )
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=t_res/3600, y=Q_in, mode="lines",
                              name="Inflow Q_in", line=dict(color="#378ADD", width=2.5)))
    fig2.add_trace(go.Scatter(x=t_res/3600, y=Q_out, mode="lines",
                              name="Gate outflow Q_out", line=dict(color="#D85A30", width=2.5, dash="dash")))
    fig2.update_layout(
        height=320, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title="Reservoir mass balance: inflow vs gate outflow",
        xaxis_title="Time (hr)", yaxis_title="Discharge (m3/s)",
        legend=dict(orientation="h", y=1.15),
    )
    return fig, fig2


def make_dam_reach_surface(x_r, t_r, h_r):
    """3D surface for the downstream reach -- this genuinely benefits from
    3D since it's a real (x,t) field, unlike the single-curve reservoir."""
    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=x_r/1000, y=t_r/3600, z=h_r, colorscale=SURFACE_COLORSCALE,
        showscale=True, opacity=0.95,
        contours=dict(z=dict(show=True, usecolormap=True, project_z=True))
    ))
    fig.update_layout(
        scene=dict(xaxis_title="Distance below dam (km)", yaxis_title="Time (hr)",
                  zaxis_title="Depth h (m)",
                  camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9))),
        margin=dict(l=0, r=0, t=30, b=0), height=420,
        title="Downstream reach — ground truth h(x,t)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def make_n_field_fig(x_km, n_field, true_n):
    """Inferred Manning's roughness field -- 2D is correct here too, n(x)
    is a single curve along the river, not a 2-variable field."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_km, y=n_field, mode="lines",
                             name="Inferred n(x)", line=dict(color="#7B2D8B", width=3),
                             fill="tozeroy", fillcolor="rgba(123,45,139,0.12)"))
    fig.add_trace(go.Scatter(x=[x_km.min(), x_km.max()], y=[true_n, true_n],
                             mode="lines", name=f"True n={true_n:.3f}",
                             line=dict(color="black", width=2, dash="dash")))
    fig.update_layout(
        height=300, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title="Inferred Manning's roughness field",
        xaxis_title="Distance (km)", yaxis_title="Manning's n",
        yaxis_range=[0, 0.12],
        legend=dict(orientation="h", y=1.15),
    )
    return fig
