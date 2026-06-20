"""
Dam domain — reservoir + downstream reach as a composite PINN.

ARCHITECTURAL NOTE (read before extending):
This does NOT fit the single-PDEModule pattern used by fluids/heat/wave/
gravity/elasticity. Those all map ONE set of inputs through ONE network
to outputs constrained by ONE governing equation. A dam is genuinely two
coupled physical systems:

  1. Reservoir: an ODE in time alone -- A*dZ/dt = Q_in(t) - Q_out(Z)
  2. Downstream reach: the full Saint-Venant PDE in (x,t), driven by the
     reservoir's gate outflow as its upstream boundary condition

Forcing this into one PDEModule.residual() would hide that coupling
structure. Instead this is a composite class that owns two MLPs (reusing
core.MLP, so it's still the same network building block as everywhere
else) and its own training loop, the same pattern InverseFluidsPDE uses
for its extra n_net.
"""

import torch
import torch.nn as nn
import numpy as np
from core.pde_module import MLP, sample_collocation


class DamSystem(nn.Module):
    """
    Two-network composite:
      reservoir_net: t -> Z(t)        (reservoir level)
      reach_net:     (x,t) -> h,u     (downstream flow field)
    Coupled via Q_out(Z) feeding into the reach's upstream boundary,
    and the reservoir mass-balance ODE residual.
    """
    def __init__(self, hidden=64, depth=4, g=9.81, S0=0.0005,
                reservoir_area=1e6, Cd=0.611, gate_width=10.0,
                n_manning=0.025):
        super().__init__()
        self.reservoir_net = MLP(in_dim=1, out_dim=1, hidden=32, depth=3)
        self.reach_net = MLP(in_dim=2, out_dim=2, hidden=hidden, depth=depth)
        self.g = g
        self.S0 = S0
        self.A = reservoir_area
        self.Cd = Cd
        self.W = gate_width
        self.n_manning = n_manning

    def reservoir_level(self, t_col):
        return torch.nn.functional.softplus(self.reservoir_net(t_col))

    def gate_outflow(self, Z):
        return self.Cd * self.W * (self.g ** 0.5) * (Z ** 1.5 + 1e-8)

    def reach_forward(self, xt):
        out = self.reach_net(xt)
        h = torch.nn.functional.softplus(out[:, 0:1])
        u = out[:, 1:2]
        return torch.cat([h, u], dim=1)

    def reservoir_residual(self, t_col, Q_in):
        """A * dZ/dt - Q_in + Q_out = 0"""
        t_col = t_col.clone().requires_grad_(True)
        Z = self.reservoir_level(t_col)
        dZ_dt = torch.autograd.grad(Z.sum(), t_col, create_graph=True)[0]
        Q_out = self.gate_outflow(Z)
        residual = self.A * dZ_dt - Q_in + Q_out
        return residual, Z, Q_out

    def reach_residual(self, xt):
        """Standard Saint-Venant residuals for the downstream reach."""
        xt = xt.clone().requires_grad_(True)
        out = self.reach_forward(xt)
        h, u = out[:, 0:1], out[:, 1:2]

        dh = torch.autograd.grad(h.sum(), xt, create_graph=True)[0]
        dh_dt = dh[:, 1:2]
        q = u * h
        dq = torch.autograd.grad(q.sum(), xt, create_graph=True)[0]
        dq_dx = dq[:, 0:1]
        res_cont = dh_dt + dq_dx

        dq_dt = torch.autograd.grad(q.sum(), xt, create_graph=True)[0][:, 1:2]
        mf = u * q + 0.5 * self.g * h ** 2
        dmf_dx = torch.autograd.grad(mf.sum(), xt, create_graph=True)[0][:, 0:1]
        S_f = (self.n_manning ** 2) * u * torch.abs(u) / (h ** (4.0 / 3.0) + 1e-8)
        res_mom = dq_dt + dmf_dx + self.g * h * (S_f - self.S0)

        return res_cont, res_mom


def train_dam_system(dam_data, n_epochs=2000, lr=1e-3,
                     lambda_res=1.0, lambda_reach=0.1, lambda_pde=0.05,
                     n_colloc=1500, callback=None):
    """
    Train DamSystem on synthetic dam_data (from data.generator.generate_dam_data,
    which itself now uses the FIXED Saint-Venant solver for the downstream reach).
    """
    import torch.optim as optim

    cfg = dam_data["cfg"]
    model = DamSystem(S0=cfg["S0"], reservoir_area=cfg["reservoir_area"],
                      Cd=cfg["Cd"], gate_width=cfg["gate_width"],
                      n_manning=cfg["n_manning"])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, eta_min=1e-5)

    t_res = dam_data["t_res"]
    t_norm = (t_res - t_res.mean()) / t_res.std()
    Z_data = dam_data["Z"]
    Z_norm = (Z_data - Z_data.mean()) / Z_data.std()
    Q_in_data = dam_data["Q_in"]

    t_col_res = torch.tensor(t_norm.reshape(-1, 1), dtype=torch.float32)
    Z_target = torch.tensor(Z_norm.reshape(-1, 1), dtype=torch.float32)
    Q_in_t = torch.tensor(Q_in_data.reshape(-1, 1), dtype=torch.float32)

    x_r, t_r = dam_data["x_reach"], dam_data["t_reach"]
    h_r, u_r = dam_data["h_reach"], dam_data["u_reach"]
    xx, tt = np.meshgrid(x_r, t_r)
    N_reach = xx.ravel().shape[0]
    idx = np.random.choice(N_reach, min(500, N_reach), replace=False)
    x_n = (xx.ravel()[idx] - x_r.mean()) / x_r.std()
    t_n = (tt.ravel()[idx] - t_r.mean()) / t_r.std()
    h_n = (h_r.ravel()[idx] - h_r.mean()) / h_r.std()
    u_n = (u_r.ravel()[idx] - u_r.mean()) / u_r.std()

    xt_reach = torch.tensor(np.stack([x_n, t_n], axis=1), dtype=torch.float32)
    h_reach_t = torch.tensor(h_n.reshape(-1, 1), dtype=torch.float32)
    u_reach_t = torch.tensor(u_n.reshape(-1, 1), dtype=torch.float32)

    history = {"epoch": [], "loss_total": [], "loss_reservoir": [],
              "loss_reach": [], "loss_pde": []}

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        Z_pred = model.reservoir_level(t_col_res)
        Z_pred_n = (Z_pred - Z_pred.mean()) / (Z_pred.std() + 1e-8)
        loss_res_data = torch.mean((Z_pred_n - Z_target) ** 2)

        res_reservoir, _, _ = model.reservoir_residual(t_col_res, Q_in_t)
        loss_res_pde = torch.mean(res_reservoir ** 2) * 1e-6

        out_reach = model.reach_forward(xt_reach)
        loss_reach = torch.mean((out_reach - torch.cat([h_reach_t, u_reach_t], dim=1)) ** 2)

        xt_col = sample_collocation(n_colloc, input_dim=2, seed=epoch)
        rc, rm = model.reach_residual(xt_col)
        loss_pde = torch.mean(rc ** 2) + torch.mean(rm ** 2)

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
