"""
Uncertainty Quantification via Monte Carlo Dropout.

Wraps any trained ForwardPINN or InversePINN and produces
mean ± std confidence bands on predictions.

Key idea:
  - Add dropout layers to the MLP (kept ON at inference time)
  - Run N forward passes through the same input
  - Mean   = expected prediction
  - Std    = epistemic uncertainty (model doesn't know)

This is the Ryan Adams / Gal & Ghahramani (2016) approach —
publishable and interpretable.
"""

import torch
import torch.nn as nn
import numpy as np
from models.pinn_core import MLP


class MCDropoutPINN(nn.Module):
    """
    ForwardPINN with MC Dropout.
    dropout_p: dropout probability (0.05–0.15 typical for PINNs)
    """
    def __init__(self, hidden=64, depth=4, dropout_p=0.1,
                 g=9.81, S0=0.001):
        super().__init__()
        self.g = g
        self.S0 = S0
        self.dropout_p = dropout_p

        # Build MLP with dropout after each hidden layer
        layers = [nn.Linear(2, hidden), nn.Tanh(), nn.Dropout(dropout_p)]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh(), nn.Dropout(dropout_p)]
        layers.append(nn.Linear(hidden, 2))
        self.net = nn.Sequential(*layers)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xt):
        out = self.net(xt)
        h = torch.nn.functional.softplus(out[:, 0:1])
        u = out[:, 1:2]
        return h, u

    def pde_residual(self, xt, n_manning=0.03):
        xt.requires_grad_(True)
        h, u = self.forward(xt)

        dh = torch.autograd.grad(h.sum(), xt, create_graph=True)[0]
        dh_dt = dh[:, 1:2]
        q = u * h
        dq = torch.autograd.grad(q.sum(), xt, create_graph=True)[0]
        dq_dx = dq[:, 0:1]
        res_cont = dh_dt + dq_dx

        dq_dt = torch.autograd.grad(q.sum(), xt, create_graph=True)[0][:, 1:2]
        mf = u * q + 0.5 * self.g * h ** 2
        dmf_dx = torch.autograd.grad(mf.sum(), xt, create_graph=True)[0][:, 0:1]
        S_f = (n_manning ** 2) * u * torch.abs(u) / (h ** (4.0 / 3.0) + 1e-8)
        res_mom = dq_dt + dmf_dx + self.g * h * (S_f - self.S0)

        return res_cont, res_mom

    def predict_with_uncertainty(self, xt_tensor, n_samples=100, n_manning=0.03):
        """
        Run n_samples stochastic forward passes.
        Returns:
            h_mean, h_std  (N,) arrays — prediction + uncertainty
            u_mean, u_std
        """
        self.train()  # keep dropout active!
        h_samples = []
        u_samples = []

        with torch.no_grad():
            for _ in range(n_samples):
                h, u = self.forward(xt_tensor)
                h_samples.append(h.numpy())
                u_samples.append(u.numpy())

        h_stack = np.concatenate(h_samples, axis=1)   # (N, n_samples)
        u_stack = np.concatenate(u_samples, axis=1)

        return (h_stack.mean(axis=1), h_stack.std(axis=1),
                u_stack.mean(axis=1), u_stack.std(axis=1))


def train_mc_dropout(train_data, river_cfg,
                     n_epochs=3000, lr=1e-3,
                     lambda_pde=0.1, dropout_p=0.1,
                     n_colloc=3000, callback=None):
    """Train MCDropoutPINN — identical loss to ForwardPINN."""
    from experiments.trainer import to_tensor, sample_collocation
    import torch.optim as optim

    model = MCDropoutPINN(dropout_p=dropout_p,
                          S0=river_cfg.get("S0", 0.001))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, eta_min=1e-5)

    xt_d = to_tensor(train_data["xt"])
    h_d  = to_tensor(train_data["h"])
    u_d  = to_tensor(train_data["u"])
    n_man = river_cfg.get("n_manning", 0.03)

    history = {"epoch": [], "loss_total": [], "loss_data": [], "loss_pde": []}

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        h_pred, u_pred = model(xt_d)
        loss_data = (torch.mean((h_pred - h_d) ** 2) +
                     torch.mean((u_pred - u_d) ** 2))

        xt_col = sample_collocation(n_colloc)
        rc, rm = model.pde_residual(xt_col, n_manning=n_man)
        loss_pde = torch.mean(rc ** 2) + torch.mean(rm ** 2)

        loss = loss_data + lambda_pde * loss_pde
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
