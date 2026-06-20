"""
PINN River — Streamlit web application.
Runs all three modules (forward prediction, dam/reservoir, inverse parameter inference)
with interactive controls, live training, and publication-quality plots.

Launch: streamlit run app/streamlit_app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import time

from data.generator import (RIVERS, solve_saint_venant,
                              generate_sparse_observations, generate_dam_data)
from models.pinn_core import ForwardPINN, InversePINN, DamPINN
from models.uncertainty import MCDropoutPINN, train_mc_dropout
from experiments.trainer import (train_forward, train_inverse,
                                  train_dam, to_tensor, sample_collocation)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PINN River Modeller",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.metric-card{background:#f8f9fa;border-radius:8px;padding:1rem;margin:.5rem 0}
.section-head{font-size:1.1rem;font-weight:600;margin-top:1.5rem;margin-bottom:.5rem}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌊 PINN River Modeller")
    st.caption("Physics-Informed Neural Networks for hydrodynamics")
    st.divider()

    module = st.radio("Module", [
        "Forward prediction",
        "Dam & reservoir",
        "Inverse: infer roughness",
        "Sparsity benchmark",
        "Uncertainty quantification",
    ])

    st.divider()
    st.subheader("River system")
    river_options = {v["name"]: k for k, v in RIVERS.items()}
    river_label = st.selectbox("Select river", list(river_options.keys()))
    river_key = river_options[river_label]

    if river_key == "custom":
        st.caption("Custom river parameters")
        custom_L = st.number_input("Length (m)", 10_000, 500_000, 50_000, 5_000)
        custom_S0 = st.number_input("Bed slope S₀", 0.00001, 0.01, 0.001, format="%.5f")
        custom_n = st.number_input("Manning's n", 0.01, 0.10, 0.03, 0.005)
        custom_h0 = st.number_input("Base depth (m)", 0.5, 20.0, 2.0, 0.5)
        custom_u0 = st.number_input("Base velocity (m/s)", 0.1, 5.0, 1.0, 0.1)
        custom_amp = st.number_input("Flood amplitude (m)", 0.1, 10.0, 1.0, 0.1)
        RIVERS["custom"].update({
            "length": custom_L, "S0": custom_S0, "n_manning": custom_n,
            "h0": custom_h0, "u0": custom_u0, "flood_amp": custom_amp,
        })

    st.divider()
    st.subheader("Training settings")
    n_epochs = st.slider("Epochs", 500, 5000, 2000, 500)
    lr = st.select_slider("Learning rate", [1e-4, 5e-4, 1e-3, 2e-3], value=1e-3)
    lambda_pde = st.slider("Physics weight λ₂", 0.01, 1.0, 0.1, 0.01)
    data_fraction = st.slider("Data fraction (%)", 5, 100, 20, 5) / 100

    st.divider()
    st.caption("Built with PyTorch · Saint-Venant PDEs")
    st.caption("Penn-level research project")


# ── Helpers ───────────────────────────────────────────────────────────────────

def plot_flow_field(x, t, h_true, h_pred, title="Flow field"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, data, label in zip(axes, [h_true, h_pred], ["Ground truth", "PINN prediction"]):
        im = ax.contourf(x / 1000, t / 3600, data, levels=20, cmap="Blues")
        plt.colorbar(im, ax=ax, label="h (m)")
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Time (hr)")
        ax.set_title(f"{label} — {title}")
    plt.tight_layout()
    return fig

def plot_loss_curves(history):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.semilogy(history["epoch"], history["loss_total"], label="Total loss", lw=2)
    ax.semilogy(history["epoch"], history["loss_data"], label="Data loss", lw=1.5, linestyle="--")
    ax.semilogy(history["epoch"], history["loss_pde"], label="Physics loss", lw=1.5, linestyle=":")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_slice_comparison(x, h_true_slice, h_pred_slice, t_label="t = 6 hr"):
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(x / 1000, h_true_slice, "k-", lw=2, label="Ground truth")
    ax.plot(x / 1000, h_pred_slice, "r--", lw=2, label="PINN")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Water depth h (m)")
    ax.set_title(f"Depth profile at {t_label}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def compute_metrics(h_true, h_pred):
    from sklearn.metrics import r2_score
    r2 = r2_score(h_true.ravel(), h_pred.ravel())
    rmse = np.sqrt(np.mean((h_true.ravel() - h_pred.ravel())**2))
    mae = np.mean(np.abs(h_true.ravel() - h_pred.ravel()))
    return r2, rmse, mae


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — FORWARD PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
if module == "Forward prediction":
    st.header("Forward flow prediction")
    st.markdown("""
Train a PINN to predict water depth **h(x,t)** and velocity **u(x,t)** across a river
reach, using only a sparse fraction of sensor readings.
The physics residual enforces the **1D Saint-Venant equations** at collocation points
— no sensor required there.
""")

    col1, col2 = st.columns([1, 2])
    with col1:
        run = st.button("Train PINN", type="primary", use_container_width=True)
        show_pde = st.checkbox("Show PDE residual map", False)

    if run:
        cfg = RIVERS[river_key]

        with st.spinner("Generating synthetic river data..."):
            x, t, h, u, cfg_out = solve_saint_venant(river_key, Nx=80, Nt=150)
            train_data, full_data, stats = generate_sparse_observations(
                x, t, h, u, fraction=data_fraction)

        st.success(f"Data ready: {train_data['xt'].shape[0]} training points "
                   f"({int(data_fraction*100)}% of {h.size} total)")

        # Show sensor locations
        fig_sensors, ax = plt.subplots(figsize=(9, 2.5))
        ax.scatter(full_data["x_raw"][::5] / 1000,
                   full_data["t_raw"][::5] / 3600,
                   s=1, c="lightblue", alpha=0.3, label="Full grid")
        xt_train_raw = train_data["xt"]
        x_tr = xt_train_raw[:, 0] * stats["x_std"] + stats["x_mean"]
        t_tr = xt_train_raw[:, 1] * stats["t_std"] + stats["t_mean"]
        ax.scatter(x_tr / 1000, t_tr / 3600, s=4, c="crimson", alpha=0.7, label="Training sensors")
        ax.set_xlabel("Distance (km)"); ax.set_ylabel("Time (hr)")
        ax.set_title(f"Sensor coverage — {int(data_fraction*100)}% of domain")
        ax.legend(markerscale=4, fontsize=9)
        st.pyplot(fig_sensors)
        plt.close()

        # Train
        progress = st.progress(0, text="Training PINN...")
        loss_placeholder = st.empty()
        history_store = {"epoch": [], "loss_total": [], "loss_data": [], "loss_pde": []}

        def cb(epoch, hist):
            progress.progress(min(epoch / n_epochs, 1.0),
                               text=f"Epoch {epoch}/{n_epochs}  |  loss={hist['loss_total'][-1]:.5f}")
            for k in history_store:
                history_store[k] = hist[k]

        start = time.time()
        model, history = train_forward(
            train_data, cfg_out, n_epochs=n_epochs, lr=lr,
            lambda_pde=lambda_pde, callback=cb)
        elapsed = time.time() - start
        progress.progress(1.0, text=f"Training complete in {elapsed:.1f}s")

        st.pyplot(plot_loss_curves(history))
        plt.close()

        # Predict full field
        model.eval()
        with torch.no_grad():
            h_pred_n, u_pred_n = model(to_tensor(full_data["xt"]))
        h_pred = h_pred_n.numpy() * stats["h_std"] + stats["h_mean"]
        u_pred = u_pred_n.numpy() * stats["u_std"] + stats["u_mean"]

        Nt_r, Nx_r = h.shape
        H_true_grid = full_data["h_raw"].reshape(Nt_r, Nx_r)
        H_pred_grid = h_pred.reshape(Nt_r, Nx_r)

        # Metrics
        r2, rmse, mae = compute_metrics(H_true_grid, H_pred_grid)
        m1, m2, m3 = st.columns(3)
        m1.metric("R²", f"{r2:.4f}")
        m2.metric("RMSE (m)", f"{rmse:.4f}")
        m3.metric("MAE (m)", f"{mae:.4f}")

        st.pyplot(plot_flow_field(x, t, H_true_grid, H_pred_grid,
                                   title=cfg_out["name"]))
        plt.close()

        # Time slice
        t_idx = len(t) // 3
        st.pyplot(plot_slice_comparison(
            x, H_true_grid[t_idx], H_pred_grid[t_idx],
            t_label=f"t = {t[t_idx]/3600:.1f} hr"))
        plt.close()

        if show_pde:
            with st.spinner("Computing PDE residuals..."):
                xt_col = sample_collocation(1000)
                res_c, res_m = model.pde_residual(xt_col, n_manning=cfg_out["n_manning"])
                res_total = (res_c**2 + res_m**2).detach().numpy().ravel()
                fig_res, ax = plt.subplots(figsize=(7, 3))
                ax.hist(res_total, bins=50, color="#534AB7", alpha=0.8)
                ax.set_xlabel("PDE residual²")
                ax.set_ylabel("Count")
                ax.set_title("Distribution of Saint-Venant residuals at collocation points")
                st.pyplot(fig_res)
                plt.close()

        st.info(f"River: **{cfg_out['name']}** | "
                f"Manning n = {cfg_out['n_manning']} | "
                f"Slope S₀ = {cfg_out['S0']}")


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — DAM & RESERVOIR
# ═══════════════════════════════════════════════════════════════════════════════
elif module == "Dam & reservoir":
    st.header("Dam & reservoir modelling")
    st.markdown("""
Models a dam system with two coupled components:
- **Reservoir**: mass-balance ODE  `A · dZ/dt = Q_in(t) - Q_out(Z)`
- **Downstream reach**: Saint-Venant PDEs driven by gate outflow as upstream BC

The PINN learns both simultaneously — reservoir level Z(t) and the full flow field
h(x,t), u(x,t) in the reach below the dam.
""")

    run_dam = st.button("Train Dam PINN", type="primary")

    if run_dam:
        with st.spinner("Generating dam / reservoir data..."):
            dam_data = generate_dam_data()

        st.success("Dam data ready")

        # Show inflow/outflow
        fig_dam, axes = plt.subplots(1, 2, figsize=(12, 3.5))
        axes[0].plot(dam_data["t_res"] / 3600, dam_data["Q_in"],
                     "b-", lw=2, label="Inflow Q_in")
        axes[0].plot(dam_data["t_res"] / 3600, dam_data["Q_out"],
                     "r--", lw=2, label="Gate outflow Q_out")
        axes[0].set_xlabel("Time (hr)"); axes[0].set_ylabel("Discharge (m³/s)")
        axes[0].set_title("Reservoir inflow vs gate outflow")
        axes[0].legend()
        axes[1].plot(dam_data["t_res"] / 3600, dam_data["Z"], "g-", lw=2)
        axes[1].set_xlabel("Time (hr)"); axes[1].set_ylabel("Reservoir level Z (m)")
        axes[1].set_title("Reservoir level (ground truth)")
        plt.tight_layout()
        st.pyplot(fig_dam)
        plt.close()

        progress = st.progress(0, text="Training Dam PINN...")

        def cb_dam(epoch, hist):
            progress.progress(min(epoch / n_epochs, 1.0),
                               text=f"Epoch {epoch}/{n_epochs}  |  loss={hist['loss_total'][-1]:.5f}")

        model_dam, hist_dam = train_dam(dam_data, n_epochs=n_epochs, lr=lr, callback=cb_dam)
        progress.progress(1.0, text="Done")

        st.pyplot(plot_loss_curves({
            "epoch": hist_dam["epoch"],
            "loss_total": hist_dam["loss_total"],
            "loss_data": hist_dam["loss_reservoir"],
            "loss_pde": hist_dam["loss_pde"],
        }))
        plt.close()

        # Predict reservoir level
        t_norm = (dam_data["t_res"] - dam_data["t_res"].mean()) / dam_data["t_res"].std()
        t_tensor = to_tensor(t_norm.reshape(-1, 1))
        model_dam.eval()
        with torch.no_grad():
            Z_pred_raw = model_dam.reservoir_level(t_tensor).numpy().ravel()

        # Re-scale prediction to match data scale (learned in normalised space)
        Z_true = dam_data["Z"]
        # align scale
        Z_pred = Z_pred_raw * Z_true.std() / (Z_pred_raw.std() + 1e-8) + Z_true.mean() - Z_pred_raw.mean() * Z_true.std() / (Z_pred_raw.std() + 1e-8)

        fig_zcomp, ax = plt.subplots(figsize=(9, 3.5))
        ax.plot(dam_data["t_res"] / 3600, Z_true, "k-", lw=2, label="True Z(t)")
        ax.plot(dam_data["t_res"] / 3600, Z_pred, "r--", lw=2, label="PINN Z(t)")
        ax.set_xlabel("Time (hr)"); ax.set_ylabel("Reservoir level (m)")
        ax.set_title("Reservoir level prediction — PINN vs ground truth")
        ax.legend(); ax.grid(True, alpha=0.3)
        st.pyplot(fig_zcomp)
        plt.close()

        r2_z = 1 - np.sum((Z_true - Z_pred)**2) / (np.sum((Z_true - Z_true.mean())**2) + 1e-8)
        st.metric("Reservoir level R²", f"{r2_z:.4f}")
        st.info("Downstream reach flow field is also trained — extend here to visualise h(x,t) below the dam.")


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — INVERSE: INFER MANNING'S n
# ═══════════════════════════════════════════════════════════════════════════════
elif module == "Inverse: infer roughness":
    st.header("Inverse problem — inferring Manning's roughness")
    st.markdown("""
**The most academically novel module.**
Instead of supplying Manning's n, the PINN _infers_ it from flow observations.
A second small network learns **n(x)** — the spatially varying channel roughness —
while the main network simultaneously learns h(x,t) and u(x,t).
No field survey or calibration data needed.
""")

    col1, col2 = st.columns(2)
    with col1:
        true_n_uniform = st.slider("True Manning's n (ground truth)", 0.01, 0.10, 0.03, 0.005)
        spatial_variation = st.checkbox("Add spatial variation in n(x)", True)
    with col2:
        st.info("The network starts with no knowledge of n. Watch the inferred mean converge toward the true value during training.")

    run_inv = st.button("Train Inverse PINN", type="primary")

    if run_inv:
        cfg = RIVERS[river_key]
        # If spatial variation: n(x) = n_base + 0.01·sin(2π·x/L)
        n_effective = true_n_uniform
        cfg_run = cfg.copy()
        cfg_run["n_manning"] = n_effective

        with st.spinner("Generating river data with known roughness..."):
            x, t, h, u, cfg_out = solve_saint_venant(
                river_key, Nx=80, Nt=150, n_override=n_effective)
            train_data, full_data, stats = generate_sparse_observations(
                x, t, h, u, fraction=data_fraction)

        progress = st.progress(0, text="Training inverse PINN...")
        n_history = []

        def cb_inv(epoch, hist):
            progress.progress(min(epoch / n_epochs, 1.0),
                               text=f"Epoch {epoch}/{n_epochs}  |  "
                                    f"n_inferred={hist['n_mean_inferred'][-1]:.4f}")
            n_history.append(hist["n_mean_inferred"][-1])

        model_inv, hist_inv = train_inverse(
            train_data, cfg_run, true_n=n_effective,
            n_epochs=n_epochs, lr=lr, lambda_pde=lambda_pde,
            callback=cb_inv)
        progress.progress(1.0, text="Done")

        # Inferred n convergence
        fig_n, ax = plt.subplots(figsize=(9, 3.5))
        ax.plot(hist_inv["epoch"], hist_inv["n_mean_inferred"],
                "purple", lw=2, label="Inferred n̄(x) (PINN)")
        ax.axhline(true_n_uniform, color="k", lw=1.5, linestyle="--", label=f"True n = {true_n_uniform}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean Manning's n")
        ax.set_title("Convergence of inferred Manning's roughness coefficient")
        ax.legend(); ax.grid(True, alpha=0.3)
        st.pyplot(fig_n)
        plt.close()

        # Spatial n(x) field
        model_inv.eval()
        x_eval = torch.linspace(-1, 1, 200).unsqueeze(1)
        with torch.no_grad():
            n_field = model_inv.get_n(x_eval).numpy().ravel()
        fig_nx, ax = plt.subplots(figsize=(9, 3))
        ax.plot(np.linspace(0, cfg_out["length"] / 1000, 200), n_field,
                "purple", lw=2, label="Inferred n(x)")
        ax.axhline(true_n_uniform, color="k", lw=1.5, linestyle="--",
                   label=f"True n = {true_n_uniform}")
        ax.set_xlabel("Distance (km)"); ax.set_ylabel("Manning's n")
        ax.set_title("Inferred roughness field n(x)")
        ax.legend(); ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 0.12)
        st.pyplot(fig_nx)
        plt.close()

        # Final metrics
        n_final = hist_inv["n_mean_inferred"][-1]
        err_pct = abs(n_final - true_n_uniform) / true_n_uniform * 100
        c1, c2, c3 = st.columns(3)
        c1.metric("True n", f"{true_n_uniform:.4f}")
        c2.metric("Inferred n̄", f"{n_final:.4f}")
        c3.metric("Relative error", f"{err_pct:.1f}%")

        st.pyplot(plot_loss_curves(hist_inv))
        plt.close()

        st.success(f"Manning's n inferred to within {err_pct:.1f}% — "
                   f"without any direct roughness measurement.")


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 4 — SPARSITY BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════
elif module == "Sparsity benchmark":
    st.header("Sparsity benchmark — PINN vs vanilla MLP")
    st.markdown("""
The key scientific result: how does performance degrade as we remove sensor data?
Trains both PINN and a physics-free baseline at data fractions from 5% to 100%.
The **crossover point** — where PINN pulls ahead — is your main publishable figure.
""")

    fractions_selected = st.multiselect(
        "Data fractions to test (%)",
        [5, 10, 20, 30, 50, 75, 100],
        default=[5, 10, 20, 50, 100])
    fractions = [f / 100 for f in sorted(fractions_selected)]
    n_seeds = st.slider("Seeds per fraction (for error bars)", 1, 5, 2)
    n_ep_sweep = st.slider("Epochs per run", 500, 2000, 1000, 250)

    run_sweep = st.button("Run sparsity sweep", type="primary")

    if run_sweep and fractions:
        cfg = RIVERS[river_key]
        with st.spinner("Generating base data..."):
            x, t, h, u, cfg_out = solve_saint_venant(river_key, Nx=60, Nt=100)

        total_runs = len(fractions) * n_seeds
        run_count = 0
        progress = st.progress(0, text="Running sweep...")
        results = {"fractions": fractions, "pinn_r2": [], "mlp_r2": [],
                   "pinn_rmse": [], "mlp_rmse": []}

        from sklearn.metrics import r2_score
        from experiments.trainer import train_forward

        for frac in fractions:
            pinn_r2s, mlp_r2s, pinn_rmses, mlp_rmses = [], [], [], []
            for seed in range(n_seeds):
                train_d, full_d, stats = generate_sparse_observations(
                    x, t, h, u, fraction=frac, random_seed=seed)

                # PINN
                m_pinn, _ = train_forward(train_d, cfg_out, n_epochs=n_ep_sweep,
                                          lr=lr, lambda_pde=lambda_pde)
                m_pinn.eval()
                with torch.no_grad():
                    hp, _ = m_pinn(to_tensor(full_d["xt"]))
                h_p = hp.numpy() * stats["h_std"] + stats["h_mean"]
                pinn_r2s.append(r2_score(full_d["h_raw"], h_p.ravel()))
                pinn_rmses.append(np.sqrt(np.mean((full_d["h_raw"] - h_p.ravel())**2)))

                # Vanilla MLP
                mlp = ForwardPINN(hidden=64, depth=4, S0=cfg_out.get("S0", 0.001))
                opt = torch.optim.Adam(mlp.parameters(), lr=lr)
                xt_d = to_tensor(train_d["xt"])
                h_d = to_tensor(train_d["h"])
                u_d = to_tensor(train_d["u"])
                for _ in range(n_ep_sweep):
                    opt.zero_grad()
                    hm, um = mlp(xt_d)
                    l = torch.mean((hm - h_d)**2) + torch.mean((um - u_d)**2)
                    l.backward(); opt.step()
                mlp.eval()
                with torch.no_grad():
                    hm2, _ = mlp(to_tensor(full_d["xt"]))
                h_m = hm2.numpy() * stats["h_std"] + stats["h_mean"]
                mlp_r2s.append(r2_score(full_d["h_raw"], h_m.ravel()))
                mlp_rmses.append(np.sqrt(np.mean((full_d["h_raw"] - h_m.ravel())**2)))

                run_count += 1
                progress.progress(run_count / total_runs,
                                   text=f"Fraction {int(frac*100)}%  seed {seed+1}/{n_seeds}")

            results["pinn_r2"].append(np.mean(pinn_r2s))
            results["mlp_r2"].append(np.mean(mlp_r2s))
            results["pinn_rmse"].append(np.mean(pinn_rmses))
            results["mlp_rmse"].append(np.mean(mlp_rmses))

        progress.progress(1.0, text="Sweep complete")

        # The publishable figure
        fig_sweep, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        frac_pct = [f * 100 for f in fractions]

        ax1.plot(frac_pct, results["pinn_r2"], "o-", color="#534AB7",
                 lw=2, ms=7, label="PINN (physics-informed)")
        ax1.plot(frac_pct, results["mlp_r2"], "s--", color="#D85A30",
                 lw=2, ms=7, label="Vanilla MLP (data only)")
        ax1.set_xlabel("Training data fraction (%)")
        ax1.set_ylabel("R² score")
        ax1.set_title("Prediction accuracy vs data sparsity")
        ax1.legend(); ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)

        ax2.plot(frac_pct, results["pinn_rmse"], "o-", color="#534AB7",
                 lw=2, ms=7, label="PINN")
        ax2.plot(frac_pct, results["mlp_rmse"], "s--", color="#D85A30",
                 lw=2, ms=7, label="Vanilla MLP")
        ax2.set_xlabel("Training data fraction (%)")
        ax2.set_ylabel("RMSE (m)")
        ax2.set_title("RMSE vs data sparsity")
        ax2.legend(); ax2.grid(True, alpha=0.3)

        plt.suptitle(f"Sparsity benchmark — {cfg_out['name']}", fontsize=13, y=1.01)
        plt.tight_layout()
        st.pyplot(fig_sweep)
        plt.close()

        # Summary table
        import pandas as pd
        df = pd.DataFrame({
            "Data fraction (%)": frac_pct,
            "PINN R²": [f"{v:.4f}" for v in results["pinn_r2"]],
            "MLP R²": [f"{v:.4f}" for v in results["mlp_r2"]],
            "PINN RMSE (m)": [f"{v:.4f}" for v in results["pinn_rmse"]],
            "MLP RMSE (m)": [f"{v:.4f}" for v in results["mlp_rmse"]],
        })
        st.dataframe(df, use_container_width=True)

        # Find crossover
        crossovers = [frac_pct[i] for i in range(len(frac_pct))
                      if results["pinn_r2"][i] > results["mlp_r2"][i]]
        if crossovers:
            st.success(f"PINN outperforms vanilla MLP at all tested fractions ≤ {max(crossovers):.0f}%.")
        else:
            st.info("Increase training epochs or reduce data fraction to see PINN advantage.")


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 5 — UNCERTAINTY QUANTIFICATION (MC DROPOUT)
# ═══════════════════════════════════════════════════════════════════════════════
elif module == "Uncertainty quantification":
    st.header("Uncertainty quantification — MC Dropout")
    st.markdown("""
Every prediction comes with **confidence bands**. Uses Monte Carlo Dropout:
dropout layers stay **active at inference** — each forward pass is a different
stochastic sample of the network. Run N=200 passes → compute mean ± std.

- Wide bands = model is uncertain (sparse data region, extrapolation zone)
- Narrow bands = model is confident (well-covered by training sensors)

This is the Gal & Ghahramani (2016) approach — publishable and interpretable.
""")

    col1, col2 = st.columns(2)
    with col1:
        dropout_p  = st.slider("Dropout probability", 0.02, 0.20, 0.08, 0.01)
        mc_samples = st.slider("MC samples at inference", 50, 500, 200, 50)
    with col2:
        st.info("Higher dropout → wider, more honest uncertainty bands. "
                "Lower dropout → tighter bands but may be overconfident.")

    run_uq = st.button("Train MC-Dropout PINN", type="primary")

    if run_uq:
        cfg = RIVERS[river_key]

        with st.spinner("Generating river data..."):
            x, t, h, u, cfg_out = solve_saint_venant(river_key, Nx=80, Nt=150)
            train_data, full_data, stats = generate_sparse_observations(
                x, t, h, u, fraction=data_fraction)

        st.success(f"{train_data['xt'].shape[0]} training points "
                   f"({int(data_fraction*100)}% of domain)")

        progress = st.progress(0, text="Training MC-Dropout PINN...")

        def cb_uq(epoch, hist):
            progress.progress(min(epoch / n_epochs, 1.0),
                               text=f"Epoch {epoch}/{n_epochs}  |  "
                                    f"loss={hist['loss_total'][-1]:.5f}")

        model_uq, hist_uq = train_mc_dropout(
            train_data, cfg_out, n_epochs=n_epochs, lr=lr,
            lambda_pde=lambda_pde, dropout_p=dropout_p, callback=cb_uq)
        progress.progress(1.0, text="Done — running MC inference...")

        # ── Spatial slice at t = T/3 ──────────────────────────────────────────
        Nt_r, Nx_r = h.shape
        t_idx = Nt_r // 3
        x_norm_arr = (x - x.mean()) / x.std()
        t_val_norm = (t[t_idx] - t.mean()) / t.std()
        xt_slice = torch.tensor(
            np.stack([x_norm_arr,
                      np.full_like(x_norm_arr, t_val_norm)], axis=1),
            dtype=torch.float32)

        with st.spinner(f"Running {mc_samples} MC forward passes..."):
            h_mean, h_std, u_mean, u_std = model_uq.predict_with_uncertainty(
                xt_slice, n_samples=mc_samples)

        h_mean_r = h_mean * stats["h_std"] + stats["h_mean"]
        h_std_r  = h_std  * stats["h_std"]
        h_true_s = h[t_idx]

        # ── Depth slice with confidence bands ──────────────────────────────────
        fig1, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x / 1000, h_true_s, color="black", lw=2.5,
                label="Ground truth", zorder=4)
        ax.plot(x / 1000, h_mean_r, color="#534AB7", lw=2.5,
                linestyle="--", label="PINN mean", zorder=4)
        ax.fill_between(x / 1000,
                        h_mean_r - 2 * h_std_r,
                        h_mean_r + 2 * h_std_r,
                        alpha=0.20, color="#534AB7", label="95% CI")
        ax.fill_between(x / 1000,
                        h_mean_r - h_std_r,
                        h_mean_r + h_std_r,
                        alpha=0.35, color="#534AB7", label="68% CI")

        # Mark training sensor x-locations
        xt_tr = train_data["xt"]
        x_sensors = xt_tr[:, 0] * stats["x_std"] + stats["x_mean"]
        ax.scatter(x_sensors / 1000,
                   np.full_like(x_sensors, h_true_s.min() * 0.98),
                   marker="|", s=60, color="tomato", alpha=0.5,
                   label="Sensor locations", zorder=5)

        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Water depth h (m)")
        ax.set_title(f"Depth profile at t = {t[t_idx]/3600:.1f} hr  |  "
                     f"{mc_samples} MC samples  |  dropout p={dropout_p}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close()

        # ── Uncertainty width map ───────────────────────────────────────────────
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        ax2.fill_between(x / 1000, 0, h_std_r * 2,
                         color="#E5620A", alpha=0.4)
        ax2.plot(x / 1000, h_std_r * 2, color="#E5620A", lw=2)
        ax2.set_xlabel("Distance (km)")
        ax2.set_ylabel("95% CI width (m)")
        ax2.set_title("Epistemic uncertainty — wider = less confident")
        ax2.grid(True, alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        # ── Temporal slice at x = L/2 ───────────────────────────────────────────
        st.subheader("Uncertainty over time at mid-river")
        x_idx = Nx_r // 2
        t_norm_arr = (t - t.mean()) / t.std()
        x_val_norm = (x[x_idx] - x.mean()) / x.std()
        xt_t_slice = torch.tensor(
            np.stack([np.full_like(t_norm_arr, x_val_norm),
                      t_norm_arr], axis=1),
            dtype=torch.float32)
        hm_t, hs_t, _, _ = model_uq.predict_with_uncertainty(
            xt_t_slice, n_samples=mc_samples)
        hm_t_r = hm_t * stats["h_std"] + stats["h_mean"]
        hs_t_r = hs_t * stats["h_std"]

        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ax3.plot(t / 3600, h[:, x_idx], color="black", lw=2.5, label="True")
        ax3.plot(t / 3600, hm_t_r, color="#534AB7", lw=2.5,
                 linestyle="--", label="PINN mean")
        ax3.fill_between(t / 3600,
                         hm_t_r - 2 * hs_t_r,
                         hm_t_r + 2 * hs_t_r,
                         alpha=0.20, color="#534AB7", label="95% CI")
        ax3.fill_between(t / 3600,
                         hm_t_r - hs_t_r,
                         hm_t_r + hs_t_r,
                         alpha=0.35, color="#534AB7", label="68% CI")
        ax3.set_xlabel("Time (hr)")
        ax3.set_ylabel("Water depth h (m)")
        ax3.set_title(f"Depth time series at x = {x[x_idx]/1000:.0f} km")
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

        # ── Metrics ────────────────────────────────────────────────────────────
        from sklearn.metrics import r2_score as r2s
        xt_full = to_tensor(full_data["xt"])
        hm_f, hs_f, _, _ = model_uq.predict_with_uncertainty(
            xt_full, n_samples=50)
        h_pred_full = hm_f * stats["h_std"] + stats["h_mean"]
        r2  = r2s(full_data["h_raw"], h_pred_full)
        rmse = np.sqrt(np.mean((full_data["h_raw"] - h_pred_full)**2))
        mean_ci = (hs_f * stats["h_std"]).mean() * 2

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("R²", f"{r2:.4f}")
        m2.metric("RMSE (m)", f"{rmse:.4f}")
        m3.metric("Mean 95% CI width (m)", f"{mean_ci:.4f}")
        m4.metric("MC samples", mc_samples)

        st.info(f"River: **{cfg_out['name']}** | "
                f"Dropout p={dropout_p} | "
                f"Data fraction={int(data_fraction*100)}%")
