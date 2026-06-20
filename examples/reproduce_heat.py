"""Small reproducible example: train a tiny Heat PINN, save checkpoint and figure."""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from core.pde_module import PINNEngine, train_pinn
from domains.heat import HeatPDE, solve_heat_analytical


def ensure_dirs():
    os.makedirs("results/checkpoints", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)


def run(seed=0):
    np.random.seed(seed)
    x = np.linspace(0, 1, 32)
    t = np.linspace(0, 1, 32)
    xx, tt = np.meshgrid(x, t)
    T = solve_heat_analytical(x, t, alpha=0.01, L=1.0, T0=1.0)

    coords = np.stack([xx.ravel(), tt.ravel()], axis=1).astype(np.float32)
    targets = T.ravel().astype(np.float32)

    # normalise
    x_mean, x_std = coords[:, 0].mean(), coords[:, 0].std()
    t_mean, t_std = coords[:, 1].mean(), coords[:, 1].std()
    X = ((coords[:, 0] - x_mean) / (x_std + 1e-8)).reshape(-1, 1)
    Tn = ((coords[:, 1] - t_mean) / (t_std + 1e-8)).reshape(-1, 1)
    xt = np.hstack([X, Tn]).astype(np.float32)

    y_mean, y_std = targets.mean(), targets.std()
    y = ((targets - y_mean) / (y_std + 1e-8)).reshape(-1, 1).astype(np.float32)

    data_coords = torch.tensor(xt, dtype=torch.float32)
    data_targets = torch.tensor(y, dtype=torch.float32)

    pde = HeatPDE(alpha=0.01)
    model = PINNEngine(pde_module=pde, hidden=64, depth=3)

    model, history = train_pinn(model, data_coords, data_targets,
                                n_epochs=300, lr=1e-3, lambda_data=1.0,
                                lambda_pde=0.1, n_colloc=1000)

    ensure_dirs()
    ckpt_path = "results/checkpoints/heat_small.pth"
    torch.save(model.state_dict(), ckpt_path)

    # Predict and save figure
    model.eval()
    with torch.no_grad():
        pred = model(data_coords).numpy().reshape(len(t), len(x))
    # un-normalise
    pred_un = pred * y_std + y_mean

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(pred_un, origin='lower', aspect='auto',
                   extent=[x.min(), x.max(), t.min(), t.max()])
    ax.set_xlabel('x'); ax.set_ylabel('t'); ax.set_title('Predicted T(x,t)')
    fig.colorbar(im, ax=ax)
    fig_path = 'results/figures/heat_pred.png'
    fig.savefig(fig_path, dpi=150)
    print(f"Saved checkpoint: {ckpt_path}")
    print(f"Saved figure: {fig_path}")


if __name__ == '__main__':
    run()
