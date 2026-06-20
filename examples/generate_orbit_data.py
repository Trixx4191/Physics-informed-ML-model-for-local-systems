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
    n_epochs = 1200
    fraction = 0.18

    print(f"Generating orbit data with {n_points} points, {fraction*100:.0f}% training data, {n_epochs} epochs")

    t, x, y, T_period = generate_orbit_data(GM=1.0, a=1.0, e=0.3, n_points=n_points)
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

    data_coords = torch.tensor(coords_norm, dtype=torch.float32)
    data_targets = torch.tensor(targets_norm, dtype=torch.float32)

    model = PINNEngine(GravityPDE(GM=1.0), hidden=64, depth=3)
    model, history = train_pinn(
        model,
        data_coords,
        data_targets,
        n_epochs=n_epochs,
        lr=1e-3,
        lambda_data=1.0,
        lambda_pde=0.1,
        n_colloc=1500,
    )

    model.eval()
    with torch.no_grad():
        pred = model(data_coords).numpy()

    x_pred = pred[:, 0] * x_std + x_mean
    y_pred = pred[:, 1] * y_std + y_mean

    r2x = r2_score(x, x_pred)
    r2y = r2_score(y, y_pred)
    r2 = float((r2x + r2y) / 2)

    output = {
        "t": t.tolist(),
        "x": x.tolist(),
        "y": y.tolist(),
        "x_pred": x_pred.tolist(),
        "y_pred": y_pred.tolist(),
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
    print(f"R2 x={r2x:.4f}, R2 y={r2y:.4f}, mean R2={r2:.4f}")


if __name__ == "__main__":
    main()
