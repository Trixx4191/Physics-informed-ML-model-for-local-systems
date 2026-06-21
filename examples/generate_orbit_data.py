import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domains.gravity import GravityPDE, generate_orbit_data
from core.pde_module import PINNEngine, train_pinn


def main():
    np.random.seed(0)
    torch.manual_seed(0)

    n_points = 200
    n_epochs = 2500
    fraction = 0.18

    print(f"Generating orbit data with {n_points} points, {fraction*100:.0f}% training data, {n_epochs} epochs")

    # n_periods=1 is explicit and deliberate, not the function's default of 2.
    # Two periods over the same n_points halves the per-orbit sampling
    # density, which turns out to be a meaningfully harder reconstruction
    # problem from only 18% sparse data -- testing showed R2 dropping to
    # ~0.12 (worse than predicting the mean on x) even with more epochs
    # and collocation points. One period at this density is the scenario
    # that's actually been validated to reconstruct well (R2 ~ 0.85-0.87);
    # extending to multiple periods is a real, separate follow-up, not
    # something to silently default into.
    t, x, y, T_period = generate_orbit_data(GM=1.0, a=1.0, e=0.3, n_points=n_points, n_periods=1)
    coords = np.stack([t], axis=1).astype(np.float32)
    targets = np.stack([x, y], axis=1).astype(np.float32)

    t_mean, t_std = coords[:, 0].mean(), coords[:, 0].std()
    x_mean, x_std = targets[:, 0].mean(), targets[:, 0].std()
    y_mean, y_std = targets[:, 1].mean(), targets[:, 1].std()

    coords_norm = ((coords - t_mean) / (t_std + 1e-8)).astype(np.float32)
    targets_norm = np.stack([
        (targets[:, 0] - x_mean) / (x_std + 1e-8),
        (targets[:, 1] - y_mean) / (y_std + 1e-8),
    ], axis=1).astype(np.float32)

    # ── Sparse training split ────────────────────────────────────────────────
    # IMPORTANT: an earlier version of this script defined `fraction` but
    # never used it to subsample -- it trained on data_coords/data_targets
    # holding the FULL dataset (all n_points), so the printed "18% training
    # data" claim was false and the script was really doing a 100%-data
    # forward fit. That run also produced a surprisingly bad R2 (~0.15
    # mean), which traced back to a second bug below (missing collocation
    # range), not to sparsity. Both are fixed here: an actual sparse split
    # is drawn, and collocation points are sampled across the full time
    # extent the orbit covers.
    rng = np.random.RandomState(0)
    n_train = max(10, int(n_points * fraction))
    train_idx = np.sort(rng.choice(n_points, n_train, replace=False))

    data_coords = torch.tensor(coords_norm[train_idx], dtype=torch.float32)
    data_targets = torch.tensor(targets_norm[train_idx], dtype=torch.float32)

    # Full dataset, used only for evaluation (the model never trains on
    # the held-out 82%).
    full_coords = torch.tensor(coords_norm, dtype=torch.float32)

    # ── Collocation range ────────────────────────────────────────────────────
    # The default in train_pinn() is [(-1,1)], but normalised orbital time
    # can range slightly beyond that depending on t_mean/t_std, and the
    # physics residual needs to be checked across the orbit's full extent
    # for Newton's law to meaningfully constrain the unobserved 82%. A
    # narrower default under-constrains exactly the gap the PINN is meant
    # to fill, which is what produced the R2=-0.20 result before this fix.
    t_min_norm, t_max_norm = coords_norm.min(), coords_norm.max()
    pad = 0.25 * (t_max_norm - t_min_norm)
    colloc_range = [(t_min_norm - pad, t_max_norm + pad)]

    model = PINNEngine(GravityPDE(GM=1.0), hidden=64, depth=4)
    model, history = train_pinn(
        model,
        data_coords,
        data_targets,
        n_epochs=n_epochs,
        lr=1.5e-3,
        lambda_data=1.0,
        lambda_pde=0.15,
        n_colloc=1500,
        colloc_ranges=colloc_range,
    )

    model.eval()
    with torch.no_grad():
        pred = model(full_coords).numpy()

    x_pred = pred[:, 0] * x_std + x_mean
    y_pred = pred[:, 1] * y_std + y_mean

    # Evaluate against the FULL trajectory -- this is the actual claim
    # being made ("reconstructed the orbit from 18% of it"), so R2 must
    # be computed on all n_points, not just the training subset.
    r2x = r2_score(x, x_pred)
    r2y = r2_score(y, y_pred)
    r2 = float((r2x + r2y) / 2)

    output = {
        "t": t.tolist(),
        "x": x.tolist(),
        "y": y.tolist(),
        "x_pred": x_pred.tolist(),
        "y_pred": y_pred.tolist(),
        "train_idx": train_idx.tolist(),
        "n_train": int(n_train),
        "n_total": int(n_points),
        "r2x": float(r2x),
        "r2y": float(r2y),
        "r2": r2,
        "history": {
            "epoch": [int(e) for e in history["epoch"]],
            "loss_total": [float(v) for v in history["loss_total"]],
            "loss_data": [float(v) for v in history["loss_data"]],
            "loss_pde": [float(v) for v in history["loss_pde"]],
        },
    }

    with open("examples/orbit_data.json", "w") as f:
        json.dump(output, f)

    print(f"Saved examples/orbit_data.json")
    print(f"Trained on {n_train}/{n_points} points ({n_train/n_points*100:.0f}%)")
    print(f"R2 x={r2x:.4f}, R2 y={r2y:.4f}, mean R2={r2:.4f}")
    print("Note: this configuration (18% sparse data, single orbit) sits "
          "near a real convergence boundary -- repeat runs with different "
          "random seeds have shown R2_y varying roughly between 0.4 and "
          "0.9 while R2_x stays consistently strong (>0.85). This is "
          "genuine training variance worth reporting, not a bug to paper "
          "over; a fixed seed is set above for reproducibility of THIS "
          "particular run.")


if __name__ == "__main__":
    main()
