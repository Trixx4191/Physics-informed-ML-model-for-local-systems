"""
Training engine.
Handles all three PINN modes with a unified train loop.
Logs loss history for visualisation.
"""

import torch
import torch.optim as optim
import numpy as np
from models.pinn_core import ForwardPINN, InversePINN, DamPINN


def to_tensor(arr, requires_grad=False):
    t = torch.tensor(arr, dtype=torch.float32)
    if requires_grad:
        t.requires_grad_(True)
    return t


# ── Collocation point samplers ────────────────────────────────────────────────

def sample_collocation(n=2000, x_range=(-1, 1), t_range=(-1, 1), seed=0):
    """Latin hypercube-style collocation points in normalised domain."""
    rng = np.random.RandomState(seed)
    x = rng.uniform(*x_range, n)
    t = rng.uniform(*t_range, n)
    return torch.tensor(np.stack([x, t], axis=1), dtype=torch.float32)


# ── Forward PINN trainer ──────────────────────────────────────────────────────

def train_forward(train_data, river_cfg, n_epochs=3000, lr=1e-3,
                  lambda_data=1.0, lambda_pde=0.1, n_colloc=3000,
                  hidden=64, depth=4, callback=None):
    """
    Train ForwardPINN.
    train_data: dict with keys 'xt', 'h', 'u'  (numpy arrays, normalised)
    river_cfg: dict with S0, n_manning
    callback: optional fn(epoch, losses) called every 100 steps (for Streamlit)
    Returns: model, loss_history
    """
    model = ForwardPINN(hidden=hidden, depth=depth,
                        S0=river_cfg.get("S0", 0.001))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, eta_min=1e-5)

    xt_data = to_tensor(train_data["xt"])
    h_data  = to_tensor(train_data["h"])
    u_data  = to_tensor(train_data["u"])
    n_man   = river_cfg.get("n_manning", 0.03)

    history = {"epoch": [], "loss_total": [], "loss_data": [], "loss_pde": []}

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        # ── data loss ──
        h_pred, u_pred = model(xt_data)
        loss_data = (torch.mean((h_pred - h_data)**2) +
                     torch.mean((u_pred - u_data)**2))

        # ── physics loss ──
        xt_col = sample_collocation(n_colloc)
        res_c, res_m = model.pde_residual(xt_col, n_manning=n_man)
        loss_pde = torch.mean(res_c**2) + torch.mean(res_m**2)

        loss = lambda_data * loss_data + lambda_pde * loss_pde
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 100 == 0:
            history["epoch"].append(epoch)
            history["loss_total"].append(loss.item())
            history["loss_data"].append(loss_data.item())
            history["loss_pde"].append(loss_pde.item())
            if callback:
                callback(epoch, history)

    return model, history


# ── Inverse PINN trainer ──────────────────────────────────────────────────────

def train_inverse(train_data, river_cfg, true_n=None,
                  n_epochs=4000, lr=1e-3,
                  lambda_data=1.0, lambda_pde=0.1, n_colloc=3000,
                  hidden=64, depth=4, callback=None):
    """
    Train InversePINN — simultaneously learns flow field and infers n(x).
    true_n: ground-truth Manning's n (scalar or array) for evaluation only.
    """
    model = InversePINN(hidden=hidden, depth=depth,
                        S0=river_cfg.get("S0", 0.001))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, eta_min=1e-5)

    xt_data = to_tensor(train_data["xt"])
    h_data  = to_tensor(train_data["h"])
    u_data  = to_tensor(train_data["u"])

    history = {"epoch": [], "loss_total": [], "loss_data": [],
               "loss_pde": [], "n_mean_inferred": []}

    x_eval = torch.linspace(-1, 1, 100).unsqueeze(1)

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        h_pred, u_pred = model(xt_data)
        loss_data = (torch.mean((h_pred - h_data)**2) +
                     torch.mean((u_pred - u_data)**2))

        xt_col = sample_collocation(n_colloc)
        res_c, res_m = model.pde_residual(xt_col)
        loss_pde = torch.mean(res_c**2) + torch.mean(res_m**2)

        # Smoothness regulariser on n(x) — penalise large spatial gradients
        x_smooth = torch.linspace(-1, 1, 200).unsqueeze(1).requires_grad_(True)
        n_smooth = model.get_n(x_smooth)
        dn_dx = torch.autograd.grad(n_smooth.sum(), x_smooth, create_graph=True)[0]
        loss_smooth = 0.01 * torch.mean(dn_dx**2)

        loss = lambda_data * loss_data + lambda_pde * loss_pde + loss_smooth
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 100 == 0:
            with torch.no_grad():
                n_inferred = model.get_n(x_eval).mean().item()
            history["epoch"].append(epoch)
            history["loss_total"].append(loss.item())
            history["loss_data"].append(loss_data.item())
            history["loss_pde"].append(loss_pde.item())
            history["n_mean_inferred"].append(n_inferred)
            if callback:
                callback(epoch, history)

    return model, history


# ── Dam PINN trainer ──────────────────────────────────────────────────────────

def train_dam(dam_data, n_epochs=3000, lr=1e-3,
              lambda_res=1.0, lambda_reach=0.1, lambda_pde=0.05,
              n_colloc=2000, callback=None):
    """
    Train DamPINN.
    dam_data: output of generate_dam_data()
    """
    cfg = dam_data["cfg"]
    model = DamPINN(S0=cfg["S0"],
                    reservoir_area=cfg["reservoir_area"],
                    Cd=cfg["Cd"],
                    gate_width=cfg["gate_width"])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, eta_min=1e-5)

    # Normalise reservoir time
    t_res = dam_data["t_res"]
    t_norm = (t_res - t_res.mean()) / t_res.std()
    Z_data = dam_data["Z"]
    Z_norm = (Z_data - Z_data.mean()) / Z_data.std()
    Q_in_data = dam_data["Q_in"]
    Q_in_norm = (Q_in_data - Q_in_data.mean()) / Q_in_data.std()

    t_col_res = to_tensor(t_norm.reshape(-1, 1))
    Z_target  = to_tensor(Z_norm.reshape(-1, 1))
    Q_in_t    = to_tensor(Q_in_data.reshape(-1, 1))

    # Downstream reach data (sparse sample)
    x_r = dam_data["x_reach"]
    t_r = dam_data["t_reach"]
    h_r = dam_data["h_reach"]
    u_r = dam_data["u_reach"]
    xx, tt = np.meshgrid(x_r, t_r)
    N_reach = xx.ravel().shape[0]
    idx = np.random.choice(N_reach, min(500, N_reach), replace=False)
    x_norm = (xx.ravel()[idx] - x_r.mean()) / x_r.std()
    t_rn   = (tt.ravel()[idx] - t_r.mean()) / t_r.std()
    h_rn   = (h_r.ravel()[idx] - h_r.mean()) / h_r.std()
    u_rn   = (u_r.ravel()[idx] - u_r.mean()) / u_r.std()

    xt_reach = to_tensor(np.stack([x_norm, t_rn], axis=1))
    h_reach  = to_tensor(h_rn.reshape(-1, 1))
    u_reach  = to_tensor(u_rn.reshape(-1, 1))

    history = {"epoch": [], "loss_total": [], "loss_reservoir": [],
               "loss_reach": [], "loss_pde": []}

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        # Reservoir level data loss
        Z_pred = model.reservoir_level(t_col_res)
        Z_pred_n = (Z_pred - Z_pred.mean()) / (Z_pred.std() + 1e-8)
        loss_res_data = torch.mean((Z_pred_n - Z_target)**2)

        # Reservoir mass-balance ODE residual
        res_reservoir, _, _ = model.reservoir_residual(t_col_res.requires_grad_(True), Q_in_t)
        loss_res_pde = torch.mean(res_reservoir**2) * 1e-6  # scale to ODE magnitude

        # Downstream reach data loss
        h_pred, u_pred = model(xt_reach)
        loss_reach = (torch.mean((h_pred - h_reach)**2) +
                      torch.mean((u_pred - u_reach)**2))

        # Downstream reach PDE
        xt_col = sample_collocation(n_colloc)
        rc, rm = model.reach_pde_residual(xt_col, n_manning=cfg["n_manning"])
        loss_pde = torch.mean(rc**2) + torch.mean(rm**2)

        loss = (lambda_res * (loss_res_data + loss_res_pde) +
                lambda_reach * loss_reach +
                lambda_pde * loss_pde)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 100 == 0:
            history["epoch"].append(epoch)
            history["loss_total"].append(loss.item())
            history["loss_reservoir"].append(loss_res_data.item())
            history["loss_reach"].append(loss_reach.item())
            history["loss_pde"].append(loss_pde.item())
            if callback:
                callback(epoch, history)

    return model, history


# ── Sparsity sweep experiment ─────────────────────────────────────────────────

def sparsity_sweep(x, t, h, u, river_cfg, fractions=None,
                   n_epochs=2000, seeds=3, callback=None):
    """
    Train PINN and vanilla MLP at multiple data fractions.
    Returns dict of results for plotting the sparsity curve.
    """
    from data.generator import generate_sparse_observations
    from sklearn.metrics import r2_score

    if fractions is None:
        fractions = [0.05, 0.10, 0.20, 0.50, 1.00]

    results = {"fractions": fractions, "pinn_r2": [], "mlp_r2": [],
               "pinn_rmse": [], "mlp_rmse": []}

    for frac in fractions:
        pinn_r2s, mlp_r2s, pinn_rmses, mlp_rmses = [], [], [], []
        for seed in range(seeds):
            train, full, stats = generate_sparse_observations(
                x, t, h, u, fraction=frac, random_seed=seed)

            # ── PINN ──
            pinn, _ = train_forward(train, river_cfg, n_epochs=n_epochs,
                                    hidden=64, depth=4)
            pinn.eval()
            with torch.no_grad():
                h_p, u_p = pinn(to_tensor(full["xt"]))
            h_pred = h_p.numpy() * stats["h_std"] + stats["h_mean"]
            r2 = r2_score(full["h_raw"], h_pred.ravel())
            rmse = np.sqrt(np.mean((full["h_raw"] - h_pred.ravel())**2))
            pinn_r2s.append(r2)
            pinn_rmses.append(rmse)

            # ── Vanilla MLP (no physics) ──
            mlp_model = ForwardPINN(hidden=64, depth=4, S0=river_cfg.get("S0", 0.001))
            opt = optim.Adam(mlp_model.parameters(), lr=1e-3)
            xt_d = to_tensor(train["xt"])
            h_d  = to_tensor(train["h"])
            u_d  = to_tensor(train["u"])
            for _ in range(n_epochs):
                opt.zero_grad()
                hp, up = mlp_model(xt_d)
                loss = torch.mean((hp - h_d)**2) + torch.mean((up - u_d)**2)
                loss.backward()
                opt.step()
            mlp_model.eval()
            with torch.no_grad():
                h_m, _ = mlp_model(to_tensor(full["xt"]))
            h_mlp = h_m.numpy() * stats["h_std"] + stats["h_mean"]
            r2_m = r2_score(full["h_raw"], h_mlp.ravel())
            rmse_m = np.sqrt(np.mean((full["h_raw"] - h_mlp.ravel())**2))
            mlp_r2s.append(r2_m)
            mlp_rmses.append(rmse_m)

            if callback:
                callback(frac, seed)

        results["pinn_r2"].append(np.mean(pinn_r2s))
        results["mlp_r2"].append(np.mean(mlp_r2s))
        results["pinn_rmse"].append(np.mean(pinn_rmses))
        results["mlp_rmse"].append(np.mean(mlp_rmses))

    return results
