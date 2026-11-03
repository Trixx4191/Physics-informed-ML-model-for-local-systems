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
from core.pde_module import MLP, NormalizedPDE, sample_collocation
from domains.fluids import FluidsPDE


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
                n_manning=0.025, normalization=None, channel_width=30.0):
        super().__init__()
        self.reservoir_net = MLP(in_dim=1, out_dim=1, hidden=32, depth=3)
        self.reach_net = MLP(in_dim=2, out_dim=2, hidden=hidden, depth=depth)
        self.g = g
        self.S0 = S0
        self.A = reservoir_area
        self.Cd = Cd
        self.W = gate_width
        self.n_manning = n_manning
        self.normalization = normalization
        self.channel_width = channel_width
        self.reach_pde = None
        if normalization is not None:
            stats = normalization
            h_ref = max(abs(stats["h_m"]), stats["h_s"])
            u_ref = max(abs(stats["u_m"]), stats["u_s"])
            q_scale = h_ref * stats["u_s"] + u_ref * stats["h_s"]
            self.reach_pde = NormalizedPDE(
                FluidsPDE(g=g, S0=S0, n_manning=n_manning),
                [stats["x_m"], stats["t_m"]], [stats["x_s"], stats["t_s"]],
                [stats["h_m"], stats["u_m"]], [stats["h_s"], stats["u_s"]],
                positive_outputs=(0,),
                residual_scales=[max(stats["h_s"] / stats["t_s"] + q_scale / stats["x_s"], 1e-8),
                                 max(q_scale / stats["t_s"]
                                     + (2 * u_ref * q_scale + g * h_ref * stats["h_s"]) / stats["x_s"]
                                     + g * h_ref * (abs(S0) + n_manning ** 2 * u_ref ** 2 / h_ref ** (4 / 3)), 1e-8)])

    def reservoir_level(self, t_col):
        raw = self.reservoir_net(t_col)
        if self.normalization is None:
            return torch.nn.functional.softplus(raw)
        stats = self.normalization
        return stats["Z_s"] * torch.nn.functional.softplus(raw + stats["Z_m"] / stats["Z_s"])

    def gate_outflow(self, Z):
        return self.Cd * self.W * (self.g ** 0.5) * (Z ** 1.5 + 1e-8)

    def reach_forward(self, xt):
        out = self.reach_net(xt)
        if self.reach_pde is not None:
            return self.reach_pde.transform_output(out)
        h = torch.nn.functional.softplus(out[:, 0:1])
        u = out[:, 1:2]
        return torch.cat([h, u], dim=1)

    def reach_physical(self, xt):
        out = self.reach_forward(xt)
        return self.reach_pde.to_physical_output(out) if self.reach_pde is not None else out

    def reservoir_residual(self, t_col, Q_in):
        """A * dZ/dt - Q_in + Q_out = 0"""
        t_col = t_col.clone().requires_grad_(True)
        Z = self.reservoir_level(t_col)
        dZ_dt = torch.autograd.grad(Z.sum(), t_col, create_graph=True)[0]
        if self.normalization is not None:
            dZ_dt = dZ_dt / self.normalization["res_t_s"]
        Q_out = self.gate_outflow(Z)
        residual = self.A * dZ_dt - Q_in + Q_out
        return residual, Z, Q_out

    def reach_residual(self, xt):
        """Standard Saint-Venant residuals for the downstream reach."""
        xt = xt.clone().requires_grad_(True)
        if self.reach_pde is not None:
            return self.reach_pde.residual(xt, self.reach_forward)
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
                     n_colloc=1500, callback=None, fraction=None, seed=0):
    """
    Train DamSystem on synthetic dam_data (from data.generator.generate_dam_data,
    which itself now uses the FIXED Saint-Venant solver for the downstream reach).

    fraction selects observed reservoir levels and reach sensor locations;
    known inflow and physics collocation remain available throughout the domain.
    None preserves the original all-reservoir / 500-reach sampling for callers
    that do not request a data fraction.
    """
    import torch.optim as optim

    if fraction is not None and not 0 < fraction <= 1:
        raise ValueError("Data fraction must be greater than zero and at most one.")
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    cfg = dam_data["cfg"]
    t_res, Z_data = dam_data["t_res"], dam_data["Z"]
    x_r, t_r = dam_data["x_reach"], dam_data["t_reach"]
    h_r, u_r = dam_data["h_reach"], dam_data["u_reach"]
    normalization = {}
    for key, values in (("res_t", t_res), ("Z", Z_data), ("x", x_r),
                        ("t", t_r), ("h", h_r), ("u", u_r)):
        normalization[f"{key}_m"] = float(values.mean())
        normalization[f"{key}_s"] = max(float(values.std()), 1e-8)
    model = DamSystem(S0=cfg["S0"], reservoir_area=cfg["reservoir_area"],
                      Cd=cfg["Cd"], gate_width=cfg["gate_width"],
                      n_manning=cfg["n_manning"], normalization=normalization,
                      channel_width=cfg.get("channel_width", 30.0))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, eta_min=1e-5)

    t_norm = (t_res - normalization["res_t_m"]) / normalization["res_t_s"]
    Z_norm = (Z_data - normalization["Z_m"]) / normalization["Z_s"]
    Q_in_data = dam_data["Q_in"]

    t_col_res = torch.tensor(t_norm.reshape(-1, 1), dtype=torch.float32)
    Z_target = torch.tensor(Z_norm.reshape(-1, 1), dtype=torch.float32)
    Q_in_t = torch.tensor(Q_in_data.reshape(-1, 1), dtype=torch.float32)
    n_res = len(t_res) if fraction is None else min(len(t_res), max(2, int(len(t_res) * fraction)))
    res_idx = rng.choice(len(t_res), n_res, replace=False)

    xx, tt = np.meshgrid(x_r, t_r)
    N_reach = xx.ravel().shape[0]
    n_reach = min(500, N_reach) if fraction is None else min(N_reach, max(8, int(N_reach * fraction)))
    idx = rng.choice(N_reach, n_reach, replace=False)
    model.n_res_train = n_res
    model.n_reach_train = n_reach
    model.reservoir_train_idx = res_idx
    model.reach_train_idx = idx
    x_n = (xx.ravel()[idx] - x_r.mean()) / x_r.std()
    t_n = (tt.ravel()[idx] - t_r.mean()) / t_r.std()
    h_n = (h_r.ravel()[idx] - h_r.mean()) / h_r.std()
    u_n = (u_r.ravel()[idx] - u_r.mean()) / u_r.std()

    xt_reach = torch.tensor(np.stack([x_n, t_n], axis=1), dtype=torch.float32)
    h_reach_t = torch.tensor(h_n.reshape(-1, 1), dtype=torch.float32)
    u_reach_t = torch.tensor(u_n.reshape(-1, 1), dtype=torch.float32)
    gate_coords = torch.tensor(np.column_stack([
        np.full(len(t_res), -normalization["x_m"] / normalization["x_s"]),
        (t_res - normalization["t_m"]) / normalization["t_s"]]), dtype=torch.float32)
    flow_scale = max(float(np.abs(Q_in_data).mean()),
                     cfg["reservoir_area"] * normalization["Z_s"] / normalization["res_t_s"], 1.0)
    colloc_ranges = [((x_r.min() - x_r.mean()) / x_r.std(), (x_r.max() - x_r.mean()) / x_r.std()),
                     ((t_r.min() - t_r.mean()) / t_r.std(), (t_r.max() - t_r.mean()) / t_r.std())]

    history = {"epoch": [], "loss_total": [], "loss_reservoir": [],
              "loss_reach": [], "loss_pde": [], "loss_coupling": []}

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        Z_pred = model.reservoir_level(t_col_res)
        Z_pred_n = (Z_pred - normalization["Z_m"]) / normalization["Z_s"]
        loss_res_data = torch.mean((Z_pred_n[res_idx] - Z_target[res_idx]) ** 2)

        res_reservoir, _, _ = model.reservoir_residual(t_col_res, Q_in_t)
        loss_res_pde = torch.mean((res_reservoir / flow_scale) ** 2)

        out_reach = model.reach_forward(xt_reach)
        loss_reach = torch.mean((out_reach - torch.cat([h_reach_t, u_reach_t], dim=1)) ** 2)
        gate_reach = model.reach_physical(gate_coords)
        gate_discharge = model.channel_width * gate_reach[:, 0:1] * gate_reach[:, 1:2]
        loss_coupling = torch.mean(((gate_discharge - model.gate_outflow(Z_pred)) / flow_scale) ** 2)

        xt_col = sample_collocation(n_colloc, input_dim=2, ranges=colloc_ranges, seed=epoch)
        rc, rm = model.reach_residual(xt_col)
        loss_pde = torch.mean(rc ** 2) + torch.mean(rm ** 2)

        loss = (lambda_res * (loss_res_data + loss_res_pde) +
               lambda_reach * (loss_reach + loss_coupling) +
               lambda_pde * loss_pde)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 100 == 0 or epoch == n_epochs - 1:
            history["epoch"].append(epoch)
            history["loss_total"].append(loss.item())
            history["loss_reservoir"].append(loss_res_data.item())
            history["loss_reach"].append(loss_reach.item())
            history["loss_pde"].append(loss_pde.item())
            history["loss_coupling"].append(loss_coupling.item())
            if callback:
                callback(epoch, history)

    return model, history
