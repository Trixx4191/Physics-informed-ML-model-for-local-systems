import numpy as np
import torch
from core.pde_module import PINNEngine, sample_collocation, train_pinn
from domains.heat import HeatPDE, solve_heat_analytical
from domains.wave import WavePDE, solve_wave_analytical


def make_grid(xn=20, tn=20, L=1.0, T=1.0):
    x = np.linspace(0, L, xn)
    t = np.linspace(0, T, tn)
    xx, tt = np.meshgrid(x, t)
    coords = np.stack([xx.ravel(), tt.ravel()], axis=1).astype(np.float32)
    return x, t, coords


def norm_coords_targets(coords, targets):
    x_mean, x_std = coords[:, 0].mean(), coords[:, 0].std()
    t_mean, t_std = coords[:, 1].mean(), coords[:, 1].std()
    X = ((coords[:, 0] - x_mean) / (x_std + 1e-8)).reshape(-1, 1)
    T = ((coords[:, 1] - t_mean) / (t_std + 1e-8)).reshape(-1, 1)
    xt = np.hstack([X, T]).astype(np.float32)

    y_mean, y_std = targets.mean(), targets.std()
    y = ((targets - y_mean) / (y_std + 1e-8)).reshape(-1, 1).astype(np.float32)
    return xt, y


def test_heat_pinn_decreases_loss():
    x, t, coords = make_grid(16, 16, L=1.0, T=1.0)
    T_true = solve_heat_analytical(x, t, alpha=0.01, L=1.0, T0=1.0)
    targets = T_true.ravel().astype(np.float32)

    xt, y = norm_coords_targets(coords, targets)

    pde = HeatPDE(alpha=0.01)
    model = PINNEngine(pde_module=pde, hidden=32, depth=2)

    # compute initial total loss (data + PDE on small collocation)
    data_coords = torch.tensor(xt, dtype=torch.float32)
    data_targets = torch.tensor(y, dtype=torch.float32)
    coll = sample_collocation(200, input_dim=pde.input_dim, ranges=[(-1, 1), (-1, 1)])

    with torch.no_grad():
        pred = model(data_coords)
        loss_data = torch.mean((pred - data_targets) ** 2).item()
        residuals = model.pde_residual(coll)
        if isinstance(residuals, (tuple, list)):
            loss_pde = sum(torch.mean(r ** 2).item() for r in residuals)
        else:
            loss_pde = torch.mean(residuals ** 2).item()
    initial = loss_data + 0.1 * loss_pde

    # Train briefly
    _, history = train_pinn(model, data_coords, data_targets,
                            n_epochs=40, lr=1e-3, lambda_data=1.0, lambda_pde=0.1,
                            n_colloc=200)

    final = history["loss_total"][-1] if history["loss_total"] else initial
    assert final < initial


def test_wave_pinn_decreases_loss():
    x, t, coords = make_grid(16, 16, L=1.0, T=1.0)
    U_true = solve_wave_analytical(x, t, c=1.0, L=1.0, mode=1, amplitude=1.0)
    targets = U_true.ravel().astype(np.float32)

    xt, y = norm_coords_targets(coords, targets)

    pde = WavePDE(c=1.0)
    model = PINNEngine(pde_module=pde, hidden=32, depth=2)

    data_coords = torch.tensor(xt, dtype=torch.float32)
    data_targets = torch.tensor(y, dtype=torch.float32)

    coll = sample_collocation(200, input_dim=pde.input_dim, ranges=[(-1, 1), (-1, 1)])

    with torch.no_grad():
        pred = model(data_coords)
        loss_data = torch.mean((pred - data_targets) ** 2).item()
        residuals = model.pde_residual(coll)
        if isinstance(residuals, (tuple, list)):
            loss_pde = sum(torch.mean(r ** 2).item() for r in residuals)
        else:
            loss_pde = torch.mean(residuals ** 2).item()
    initial = loss_data + 0.1 * loss_pde

    _, history = train_pinn(model, data_coords, data_targets,
                            n_epochs=40, lr=1e-3, lambda_data=1.0, lambda_pde=0.1,
                            n_colloc=200)

    final = history["loss_total"][-1] if history["loss_total"] else initial
    assert final < initial
