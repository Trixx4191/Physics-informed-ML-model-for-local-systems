"""
run_experiment.py — standalone evaluation script.

Runs the full three-tier (fluids/dam) experiment and saves publication-quality
figures to `results/`. This script is fluids-focused (river presets) and is
kept as a reproducible paper experiment runner; other domains are accessible
via the Dash UI in `dash_app/`.

Usage examples (fluids-focused examples shown; script accepts domain/config args):
    python run_experiment.py --river volta --epochs 3000
    python run_experiment.py --river amazon --epochs 4000 --module all
    python run_experiment.py --river custom --S0 0.001 --n 0.03 --module forward
"""

import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

from data.generator import (RIVERS, solve_saint_venant,
                              generate_sparse_observations, generate_dam_data)
from experiments.trainer import (train_forward, train_inverse, train_dam,
                                  to_tensor, sample_collocation)
from models.uncertainty import train_mc_dropout
from sklearn.metrics import r2_score

os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/checkpoints", exist_ok=True)


# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

PINN_COLOR  = "#534AB7"
MLP_COLOR   = "#D85A30"
TRUE_COLOR  = "#111111"


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Forward prediction: flow field + slice
# ═══════════════════════════════════════════════════════════════════════════════

def run_forward(river_key, cfg, n_epochs, data_fraction=0.10, save=True):
    print(f"\n{'='*60}")
    print(f"MODULE 1: Forward prediction — {cfg['name']}")
    print(f"  epochs={n_epochs}  data_fraction={data_fraction*100:.0f}%")
    print(f"{'='*60}")

    x, t, h, u, cfg_out = solve_saint_venant(river_key, Nx=100, Nt=200)
    train_data, full_data, stats = generate_sparse_observations(
        x, t, h, u, fraction=data_fraction)

    print(f"  Grid: {h.shape}  |  Training points: {train_data['xt'].shape[0]}")

    t0 = time.time()
    model, history = train_forward(
        train_data, cfg_out, n_epochs=n_epochs,
        lambda_pde=0.1, n_colloc=3000)
    elapsed = time.time() - t0
    print(f"  Training done in {elapsed:.1f}s")

    torch.save(model.state_dict(),
               f"results/checkpoints/forward_{river_key}.pt")

    # Predict
    model.eval()
    with torch.no_grad():
        h_pred_n, u_pred_n = model(to_tensor(full_data["xt"]))
    h_pred = h_pred_n.numpy() * stats["h_std"] + stats["h_mean"]
    u_pred = u_pred_n.numpy() * stats["u_std"] + stats["u_mean"]

    Nt_r, Nx_r = h.shape
    H_true = full_data["h_raw"].reshape(Nt_r, Nx_r)
    H_pred = h_pred.reshape(Nt_r, Nx_r)

    r2   = r2_score(H_true.ravel(), H_pred.ravel())
    rmse = np.sqrt(np.mean((H_true - H_pred)**2))
    mae  = np.mean(np.abs(H_true - H_pred))
    print(f"  R²={r2:.4f}  RMSE={rmse:.4f}m  MAE={mae:.4f}m")

    if not save:
        return model, history, {"r2": r2, "rmse": rmse, "mae": mae}

    # ── Figure 1: 2×3 layout ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 9))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    # (0,0) Ground truth h
    ax00 = fig.add_subplot(gs[0, 0])
    im = ax00.contourf(x / 1000, t / 3600, H_true, levels=20, cmap="Blues")
    plt.colorbar(im, ax=ax00, label="h (m)")
    ax00.set_xlabel("Distance (km)"); ax00.set_ylabel("Time (hr)")
    ax00.set_title("Ground truth h(x,t)")

    # (0,1) PINN prediction h
    ax01 = fig.add_subplot(gs[0, 1])
    im2 = ax01.contourf(x / 1000, t / 3600, H_pred, levels=20, cmap="Blues")
    plt.colorbar(im2, ax=ax01, label="h (m)")
    ax01.set_xlabel("Distance (km)"); ax01.set_ylabel("Time (hr)")
    ax01.set_title(f"PINN prediction  (R²={r2:.3f})")

    # (0,2) Point-error map
    ax02 = fig.add_subplot(gs[0, 2])
    err = np.abs(H_true - H_pred)
    im3 = ax02.contourf(x / 1000, t / 3600, err, levels=15, cmap="Reds")
    plt.colorbar(im3, ax=ax02, label="|error| (m)")
    ax02.set_xlabel("Distance (km)"); ax02.set_ylabel("Time (hr)")
    ax02.set_title("Absolute error")

    # (1,0) Depth profile slice t=T/3
    ax10 = fig.add_subplot(gs[1, 0])
    t_idx = Nt_r // 3
    ax10.plot(x / 1000, H_true[t_idx], color=TRUE_COLOR, lw=2, label="True")
    ax10.plot(x / 1000, H_pred[t_idx], color=PINN_COLOR, lw=2,
              linestyle="--", label="PINN")
    ax10.set_xlabel("Distance (km)"); ax10.set_ylabel("h (m)")
    ax10.set_title(f"Depth profile at t = {t[t_idx]/3600:.1f} hr")
    ax10.legend(fontsize=9)

    # (1,1) Depth timeseries at x=L/2
    ax11 = fig.add_subplot(gs[1, 1])
    x_idx = Nx_r // 2
    ax11.plot(t / 3600, H_true[:, x_idx], color=TRUE_COLOR, lw=2, label="True")
    ax11.plot(t / 3600, H_pred[:, x_idx], color=PINN_COLOR, lw=2,
              linestyle="--", label="PINN")
    ax11.set_xlabel("Time (hr)"); ax11.set_ylabel("h (m)")
    ax11.set_title(f"Depth time series at x = {x[x_idx]/1000:.0f} km")
    ax11.legend(fontsize=9)

    # (1,2) Loss curves
    ax12 = fig.add_subplot(gs[1, 2])
    ax12.semilogy(history["epoch"], history["loss_total"],
                  color=PINN_COLOR, lw=2, label="Total")
    ax12.semilogy(history["epoch"], history["loss_data"],
                  color="steelblue", lw=1.5, linestyle="--", label="Data")
    ax12.semilogy(history["epoch"], history["loss_pde"],
                  color="tomato", lw=1.5, linestyle=":", label="Physics")
    ax12.set_xlabel("Epoch"); ax12.set_ylabel("Loss")
    ax12.set_title("Training loss curves")
    ax12.legend(fontsize=9)

    fig.suptitle(f"Forward PINN — {cfg_out['name']}  |  {int(data_fraction*100)}% data",
                 fontsize=14, y=1.01)

    path = f"results/figures/fig1_forward_{river_key}.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)

    return model, history, {"r2": r2, "rmse": rmse, "mae": mae}


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Sparsity sweep (the main publishable figure)
# ═══════════════════════════════════════════════════════════════════════════════

def run_sparsity_sweep(river_key, cfg, n_epochs_sweep=2000,
                       fractions=None, seeds=3, save=True):
    print(f"\n{'='*60}")
    print(f"MODULE 1b: Sparsity sweep — {cfg['name']}")
    print(f"{'='*60}")

    if fractions is None:
        fractions = [0.05, 0.10, 0.20, 0.50, 1.00]

    x, t, h, u, cfg_out = solve_saint_venant(river_key, Nx=60, Nt=100)

    pinn_r2, mlp_r2   = [], []
    pinn_rmse, mlp_rmse = [], []

    from models.pinn_core import ForwardPINN
    import torch.optim as optim

    for frac in fractions:
        pr2, mr2, pr, mr = [], [], [], []
        for seed in range(seeds):
            td, fd, st = generate_sparse_observations(
                x, t, h, u, fraction=frac, random_seed=seed)

            # PINN
            m, _ = train_forward(td, cfg_out, n_epochs=n_epochs_sweep,
                                  lambda_pde=0.1)
            m.eval()
            with torch.no_grad():
                hp, _ = m(to_tensor(fd["xt"]))
            h_p = hp.numpy() * st["h_std"] + st["h_mean"]
            pr2.append(r2_score(fd["h_raw"], h_p.ravel()))
            pr.append(np.sqrt(np.mean((fd["h_raw"] - h_p.ravel())**2)))

            # Vanilla MLP — identical architecture, physics loss removed
            mlp = ForwardPINN(hidden=64, depth=4, S0=cfg_out.get("S0", 0.001))
            opt = optim.Adam(mlp.parameters(), lr=1e-3)
            xt_d = to_tensor(td["xt"])
            h_d  = to_tensor(td["h"])
            u_d  = to_tensor(td["u"])
            for _ in range(n_epochs_sweep):
                opt.zero_grad()
                hm, um = mlp(xt_d)
                l = torch.mean((hm - h_d)**2) + torch.mean((um - u_d)**2)
                l.backward(); opt.step()
            mlp.eval()
            with torch.no_grad():
                hm2, _ = mlp(to_tensor(fd["xt"]))
            h_m = hm2.numpy() * st["h_std"] + st["h_mean"]
            mr2.append(r2_score(fd["h_raw"], h_m.ravel()))
            mr.append(np.sqrt(np.mean((fd["h_raw"] - h_m.ravel())**2)))

            print(f"  frac={frac*100:.0f}%  seed={seed}  "
                  f"PINN R²={pr2[-1]:.3f}  MLP R²={mr2[-1]:.3f}")

        pinn_r2.append(np.mean(pr2));  mlp_r2.append(np.mean(mr2))
        pinn_rmse.append(np.mean(pr)); mlp_rmse.append(np.mean(mr))

    frac_pct = [f * 100 for f in fractions]

    if not save:
        return {"fractions": frac_pct, "pinn_r2": pinn_r2, "mlp_r2": mlp_r2,
                "pinn_rmse": pinn_rmse, "mlp_rmse": mlp_rmse}

    # ── Figure 2: the central result ──────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(frac_pct, pinn_r2, "o-", color=PINN_COLOR, lw=2.5, ms=8,
             label="PINN (physics-informed)", zorder=3)
    ax1.plot(frac_pct, mlp_r2, "s--", color=MLP_COLOR, lw=2.5, ms=8,
             label="Vanilla MLP (data only)", zorder=3)
    ax1.fill_between(frac_pct, pinn_r2, mlp_r2,
                     where=[p > m for p, m in zip(pinn_r2, mlp_r2)],
                     alpha=0.12, color=PINN_COLOR, label="PINN advantage")
    ax1.set_xlabel("Training data fraction (%)")
    ax1.set_ylabel("R² score")
    ax1.set_title("Prediction accuracy vs data sparsity")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.25)
    ax1.set_ylim(bottom=max(0, min(min(pinn_r2), min(mlp_r2)) - 0.05))

    ax2.plot(frac_pct, pinn_rmse, "o-", color=PINN_COLOR, lw=2.5, ms=8,
             label="PINN")
    ax2.plot(frac_pct, mlp_rmse, "s--", color=MLP_COLOR, lw=2.5, ms=8,
             label="Vanilla MLP")
    ax2.set_xlabel("Training data fraction (%)")
    ax2.set_ylabel("RMSE (m)")
    ax2.set_title("RMSE vs data sparsity")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.25)

    fig.suptitle(f"Sparsity benchmark — {cfg_out['name']}  |  {seeds} seeds",
                 fontsize=13, y=1.01)
    plt.tight_layout()

    path = f"results/figures/fig2_sparsity_{river_key}.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)

    return {"fractions": frac_pct, "pinn_r2": pinn_r2, "mlp_r2": mlp_r2,
            "pinn_rmse": pinn_rmse, "mlp_rmse": mlp_rmse}


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Inverse problem: n(x) convergence
# ═══════════════════════════════════════════════════════════════════════════════

def run_inverse(river_key, cfg, n_epochs, true_n=None, data_fraction=0.15,
                save=True):
    print(f"\n{'='*60}")
    print(f"MODULE 3: Inverse — Manning's n inference — {cfg['name']}")
    print(f"{'='*60}")

    true_n = true_n or cfg.get("n_manning", 0.03)
    x, t, h, u, cfg_out = solve_saint_venant(
        river_key, Nx=80, Nt=150, n_override=true_n)
    train_data, full_data, stats = generate_sparse_observations(
        x, t, h, u, fraction=data_fraction)

    print(f"  True n = {true_n:.4f}  |  Training points: {train_data['xt'].shape[0]}")

    t0 = time.time()
    model, history = train_inverse(
        train_data, cfg_out, true_n=true_n,
        n_epochs=n_epochs, lambda_pde=0.1)
    elapsed = time.time() - t0
    print(f"  Training done in {elapsed:.1f}s")

    torch.save(model.state_dict(),
               f"results/checkpoints/inverse_{river_key}.pt")

    # Inferred n field
    x_eval = torch.linspace(-1, 1, 200).unsqueeze(1)
    model.eval()
    with torch.no_grad():
        n_field = model.get_n(x_eval).numpy().ravel()

    n_final = history["n_mean_inferred"][-1]
    err_pct = abs(n_final - true_n) / true_n * 100
    print(f"  Inferred n̄ = {n_final:.4f}  |  Error = {err_pct:.1f}%")

    if not save:
        return model, history, n_field, err_pct

    # ── Figure 3: 2×2 layout ──────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Inverse PINN — Manning's n inference — {cfg_out['name']}",
                 fontsize=13, y=1.01)

    # (0,0) n convergence over epochs
    ax = axes[0, 0]
    ax.plot(history["epoch"], history["n_mean_inferred"],
            color="#7B2D8B", lw=2.5, label="Inferred n̄(x)")
    ax.axhline(true_n, color=TRUE_COLOR, lw=1.5, linestyle="--",
               label=f"True n = {true_n:.4f}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Manning's n")
    ax.set_title("Convergence of inferred roughness")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)

    # (0,1) Inferred n(x) spatial field
    ax = axes[0, 1]
    x_km = np.linspace(0, cfg_out["length"] / 1000, 200)
    ax.plot(x_km, n_field, color="#7B2D8B", lw=2.5, label="Inferred n(x)")
    ax.axhline(true_n, color=TRUE_COLOR, lw=1.5, linestyle="--",
               label=f"True n = {true_n:.4f}")
    ax.fill_between(x_km, n_field, true_n,
                    alpha=0.15, color="#7B2D8B")
    ax.set_xlabel("Distance (km)"); ax.set_ylabel("Manning's n")
    ax.set_title(f"Inferred roughness field  (error = {err_pct:.1f}%)")
    ax.set_ylim(0, 0.12); ax.legend(fontsize=9); ax.grid(True, alpha=0.25)

    # (1,0) Flow prediction accuracy (using inferred n)
    with torch.no_grad():
        h_pred_n, _ = model(to_tensor(full_data["xt"]))
    h_pred = h_pred_n.numpy() * stats["h_std"] + stats["h_mean"]
    H_true = full_data["h_raw"]
    r2 = r2_score(H_true, h_pred.ravel())
    ax = axes[1, 0]
    ax.scatter(H_true[::20], h_pred.ravel()[::20],
               s=6, alpha=0.4, color=PINN_COLOR)
    lims = [min(H_true.min(), h_pred.min()), max(H_true.max(), h_pred.max())]
    ax.plot(lims, lims, "k--", lw=1.5)
    ax.set_xlabel("True h (m)"); ax.set_ylabel("Predicted h (m)")
    ax.set_title(f"Prediction parity plot  (R² = {r2:.3f})")
    ax.grid(True, alpha=0.25)

    # (1,1) Loss curves
    ax = axes[1, 1]
    ax.semilogy(history["epoch"], history["loss_total"],
                color=PINN_COLOR, lw=2, label="Total")
    ax.semilogy(history["epoch"], history["loss_data"],
                color="steelblue", lw=1.5, linestyle="--", label="Data")
    ax.semilogy(history["epoch"], history["loss_pde"],
                color="tomato", lw=1.5, linestyle=":", label="Physics")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training loss")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)

    plt.tight_layout()
    path = f"results/figures/fig3_inverse_{river_key}.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)

    return model, history, n_field, err_pct


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Uncertainty quantification
# ═══════════════════════════════════════════════════════════════════════════════

def run_uncertainty(river_key, cfg, n_epochs, data_fraction=0.10, save=True):
    print(f"\n{'='*60}")
    print(f"MODULE 4: Uncertainty quantification — {cfg['name']}")
    print(f"{'='*60}")

    x, t, h, u, cfg_out = solve_saint_venant(river_key, Nx=80, Nt=150)
    train_data, full_data, stats = generate_sparse_observations(
        x, t, h, u, fraction=data_fraction)

    t0 = time.time()
    model, history = train_mc_dropout(
        train_data, cfg_out, n_epochs=n_epochs, dropout_p=0.08)
    print(f"  Training done in {time.time()-t0:.1f}s")

    # Predict with uncertainty at a spatial slice (midpoint in time)
    Nt_r, Nx_r = h.shape
    t_idx = Nt_r // 2
    x_norm = (x - x.mean()) / x.std()
    t_norm_val = (t[t_idx] - t.mean()) / t.std()
    xt_slice = torch.tensor(
        np.stack([x_norm, np.full_like(x_norm, t_norm_val)], axis=1),
        dtype=torch.float32)

    h_mean, h_std, u_mean, u_std = model.predict_with_uncertainty(
        xt_slice, n_samples=200)

    h_mean_rescaled = h_mean * stats["h_std"] + stats["h_mean"]
    h_std_rescaled  = h_std  * stats["h_std"]

    h_true_slice = h[t_idx]

    if not save:
        return model, h_mean_rescaled, h_std_rescaled

    # ── Figure 4 ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(x / 1000, h_true_slice, color=TRUE_COLOR, lw=2.5,
            label="Ground truth", zorder=3)
    ax.plot(x / 1000, h_mean_rescaled, color=PINN_COLOR, lw=2.5,
            linestyle="--", label="PINN mean", zorder=3)
    ax.fill_between(x / 1000,
                    h_mean_rescaled - 2 * h_std_rescaled,
                    h_mean_rescaled + 2 * h_std_rescaled,
                    alpha=0.25, color=PINN_COLOR, label="95% CI")
    ax.fill_between(x / 1000,
                    h_mean_rescaled - h_std_rescaled,
                    h_mean_rescaled + h_std_rescaled,
                    alpha=0.35, color=PINN_COLOR, label="68% CI")
    ax.set_xlabel("Distance (km)"); ax.set_ylabel("Water depth h (m)")
    ax.set_title(f"Depth prediction with uncertainty  (t = {t[t_idx]/3600:.1f} hr)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)

    ax2 = axes[1]
    ax2.plot(x / 1000, h_std_rescaled * 2, color="#E5620A", lw=2.5)
    ax2.fill_between(x / 1000, 0, h_std_rescaled * 2, alpha=0.3, color="#E5620A")
    ax2.set_xlabel("Distance (km)"); ax2.set_ylabel("95% CI width (m)")
    ax2.set_title("Epistemic uncertainty — wider = less confident")
    ax2.grid(True, alpha=0.25)

    fig.suptitle(f"MC Dropout uncertainty — {cfg_out['name']}  |  "
                 f"{int(data_fraction*100)}% data  |  200 MC samples",
                 fontsize=12, y=1.01)
    plt.tight_layout()

    path = f"results/figures/fig4_uncertainty_{river_key}.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)

    return model, h_mean_rescaled, h_std_rescaled


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Dam & reservoir
# ═══════════════════════════════════════════════════════════════════════════════

def run_dam_experiment(n_epochs, save=True):
    print(f"\n{'='*60}")
    print(f"MODULE 2: Dam & reservoir")
    print(f"{'='*60}")

    dam_data = generate_dam_data()
    t0 = time.time()
    model, history = train_dam(dam_data, n_epochs=n_epochs)
    print(f"  Training done in {time.time()-t0:.1f}s")
    torch.save(model.state_dict(), "results/checkpoints/dam.pt")

    # Reconstruct Z prediction
    t_res = dam_data["t_res"]
    t_norm = (t_res - t_res.mean()) / t_res.std()
    t_tensor = to_tensor(t_norm.reshape(-1, 1))
    model.eval()
    with torch.no_grad():
        Z_raw = model.reservoir_level(t_tensor).numpy().ravel()
    Z_true = dam_data["Z"]
    # rescale prediction
    Z_pred = ((Z_raw - Z_raw.mean()) / (Z_raw.std() + 1e-8)) * Z_true.std() + Z_true.mean()

    r2_z = r2_score(Z_true, Z_pred)
    print(f"  Reservoir level R² = {r2_z:.4f}")

    if not save:
        return model, history, Z_pred, r2_z

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Dam & Reservoir PINN", fontsize=13, y=1.01)

    axes[0, 0].plot(t_res / 3600, dam_data["Q_in"], "b-", lw=2, label="Q_in")
    axes[0, 0].plot(t_res / 3600, dam_data["Q_out"], "r--", lw=2, label="Q_out (gate)")
    axes[0, 0].set_xlabel("Time (hr)"); axes[0, 0].set_ylabel("Discharge (m³/s)")
    axes[0, 0].set_title("Inflow vs gate outflow")
    axes[0, 0].legend(fontsize=9); axes[0, 0].grid(True, alpha=0.25)

    axes[0, 1].plot(t_res / 3600, Z_true, color=TRUE_COLOR, lw=2.5, label="True Z(t)")
    axes[0, 1].plot(t_res / 3600, Z_pred, color=PINN_COLOR, lw=2.5,
                    linestyle="--", label=f"PINN  (R²={r2_z:.3f})")
    axes[0, 1].set_xlabel("Time (hr)"); axes[0, 1].set_ylabel("Reservoir level Z (m)")
    axes[0, 1].set_title("Reservoir level prediction")
    axes[0, 1].legend(fontsize=9); axes[0, 1].grid(True, alpha=0.25)

    axes[1, 0].semilogy(history["epoch"], history["loss_total"],
                        color=PINN_COLOR, lw=2, label="Total")
    axes[1, 0].semilogy(history["epoch"], history["loss_reservoir"],
                        color="steelblue", lw=1.5, linestyle="--", label="Reservoir")
    axes[1, 0].semilogy(history["epoch"], history["loss_pde"],
                        color="tomato", lw=1.5, linestyle=":", label="Reach PDE")
    axes[1, 0].set_xlabel("Epoch"); axes[1, 0].set_ylabel("Loss")
    axes[1, 0].set_title("Training loss"); axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.25)

    # Downstream reach h at mid-time
    x_r = dam_data["x_reach"]
    t_r = dam_data["t_reach"]
    h_r = dam_data["h_reach"]
    t_mid = len(t_r) // 2
    axes[1, 1].plot(x_r / 1000, h_r[t_mid], color=TRUE_COLOR, lw=2,
                    label=f"True reach h at t={t_r[t_mid]/3600:.1f}hr")
    axes[1, 1].set_xlabel("Distance km"); axes[1, 1].set_ylabel("h (m)")
    axes[1, 1].set_title("Downstream reach (ground truth)")
    axes[1, 1].legend(fontsize=9); axes[1, 1].grid(True, alpha=0.25)

    plt.tight_layout()
    path = "results/figures/fig5_dam.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)

    return model, history, Z_pred, r2_z


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PINN Multi-Physics experiments")
    parser.add_argument("--river",   default="volta",
                        choices=list(RIVERS.keys()))
    parser.add_argument("--epochs",  type=int, default=3000)
    parser.add_argument("--module",  default="all",
                        choices=["all", "forward", "sparsity",
                                 "inverse", "dam", "uncertainty"])
    parser.add_argument("--fraction", type=float, default=0.10)
    parser.add_argument("--seeds",   type=int, default=3)
    args = parser.parse_args()

    cfg = RIVERS[args.river]
    print(f"\nPINN Multi-Physics Experiment")
    print(f"Preset: {cfg['name']}  |  Epochs: {args.epochs}  |  Module: {args.module}")

    if args.module in ("all", "forward"):
        run_forward(args.river, cfg, args.epochs, args.fraction)

    if args.module in ("all", "sparsity"):
        run_sparsity_sweep(args.river, cfg, n_epochs_sweep=min(args.epochs, 2000),
                           seeds=args.seeds)

    if args.module in ("all", "inverse"):
        run_inverse(args.river, cfg, args.epochs, data_fraction=args.fraction)

    if args.module in ("all", "dam"):
        run_dam_experiment(args.epochs)

    if args.module in ("all", "uncertainty"):
        run_uncertainty(args.river, cfg, args.epochs, args.fraction)

    print(f"\nAll figures saved to results/figures/")
    print(f"Checkpoints saved to results/checkpoints/")
