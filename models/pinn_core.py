"""
PINN Core — shared MLP backbone for all three modules.
Input:  (x, t) normalised to [-1, 1]
Output: [h, u]  water depth and velocity  (forward)
        [h, u, n_field]                   (inverse — n is also a network output)
"""

import torch
import torch.nn as nn
import numpy as np


class MLP(nn.Module):
    """Fully-connected network with tanh activations.
    tanh is essential — autograd needs smooth, differentiable activations
    to compute the PDE residuals via second-order derivatives.
    """
    def __init__(self, in_dim=2, out_dim=2, hidden=64, depth=4):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class ForwardPINN(nn.Module):
    """
    Forward problem: given known Manning's n, predict h(x,t) and u(x,t).
    Physics: 1D Saint-Venant equations (continuity + momentum)
      ∂h/∂t + ∂(uh)/∂x = 0                                      (continuity)
      ∂(uh)/∂t + ∂(u²h + g·h²/2)/∂x + g·h·(S_f - S_0) = 0     (momentum)
    where friction slope S_f = n²·u·|u| / (h^(4/3))
    """
    def __init__(self, hidden=64, depth=4, g=9.81, S0=0.001):
        super().__init__()
        self.net = MLP(in_dim=2, out_dim=2, hidden=hidden, depth=depth)
        self.g = g
        self.S0 = S0  # bed slope — set per river

    def forward(self, xt):
        """xt: (N, 2) tensor of [x, t] pairs, normalised"""
        out = self.net(xt)
        # softplus ensures h > 0 always (physically required)
        h = torch.nn.functional.softplus(out[:, 0:1])
        u = out[:, 1:2]
        return h, u

    def pde_residual(self, xt, n_manning=0.03):
        """
        Compute Saint-Venant residuals at collocation points.
        Uses autograd to get all required partial derivatives.
        Returns: (res_continuity, res_momentum) — should both be ~0
        """
        xt.requires_grad_(True)
        h, u = self.forward(xt)

        # ---- continuity: ∂h/∂t + ∂(uh)/∂x = 0 ----
        dh = torch.autograd.grad(h.sum(), xt, create_graph=True)[0]
        dh_dt = dh[:, 1:2]   # ∂h/∂t
        dh_dx = dh[:, 0:1]   # ∂h/∂x

        q = u * h  # discharge per unit width
        dq = torch.autograd.grad(q.sum(), xt, create_graph=True)[0]
        dq_dx = dq[:, 0:1]   # ∂(uh)/∂x

        res_cont = dh_dt + dq_dx

        # ---- momentum: ∂(uh)/∂t + ∂(u²h + g·h²/2)/∂x + g·h·(S_f - S_0) = 0 ----
        dq_dt = torch.autograd.grad(q.sum(), xt, create_graph=True)[0][:, 1:2]

        momentum_flux = u * q + 0.5 * self.g * h ** 2
        dmf = torch.autograd.grad(momentum_flux.sum(), xt, create_graph=True)[0]
        dmf_dx = dmf[:, 0:1]

        # friction slope: n²·u·|u| / h^(4/3)
        S_f = (n_manning ** 2) * u * torch.abs(u) / (h ** (4.0 / 3.0) + 1e-8)
        res_mom = dq_dt + dmf_dx + self.g * h * (S_f - self.S0)

        return res_cont, res_mom


class InversePINN(nn.Module):
    """
    Inverse problem: infer spatially-varying Manning's n(x) from observations.
    A second small network learns n(x) — treated as a trainable field.
    The main network still learns h(x,t) and u(x,t).
    """
    def __init__(self, hidden=64, depth=4, g=9.81, S0=0.001):
        super().__init__()
        self.flow_net = MLP(in_dim=2, out_dim=2, hidden=hidden, depth=depth)
        # n_net maps x → n(x); sigmoid scales output to (0.01, 0.15) — physical range
        self.n_net = MLP(in_dim=1, out_dim=1, hidden=32, depth=3)
        self.g = g
        self.S0 = S0

    def get_n(self, x_col):
        """Infer Manning's n at spatial locations x_col (N,1)"""
        raw = self.n_net(x_col)
        # constrain to physical range [0.01, 0.15]
        return 0.01 + 0.14 * torch.sigmoid(raw)

    def forward(self, xt):
        out = self.flow_net(xt)
        h = torch.nn.functional.softplus(out[:, 0:1])
        u = out[:, 1:2]
        return h, u

    def pde_residual(self, xt):
        """Same Saint-Venant residuals, but n is inferred from n_net."""
        xt.requires_grad_(True)
        h, u = self.forward(xt)

        x_col = xt[:, 0:1]
        n = self.get_n(x_col.detach())  # stop gradient through n here, separate path

        dh = torch.autograd.grad(h.sum(), xt, create_graph=True)[0]
        dh_dt = dh[:, 1:2]
        q = u * h
        dq = torch.autograd.grad(q.sum(), xt, create_graph=True)[0]
        dq_dx = dq[:, 0:1]
        res_cont = dh_dt + dq_dx

        dq_dt = torch.autograd.grad(q.sum(), xt, create_graph=True)[0][:, 1:2]
        momentum_flux = u * q + 0.5 * self.g * h ** 2
        dmf = torch.autograd.grad(momentum_flux.sum(), xt, create_graph=True)[0]
        dmf_dx = dmf[:, 0:1]

        S_f = (n ** 2) * u * torch.abs(u) / (h ** (4.0 / 3.0) + 1e-8)
        res_mom = dq_dt + dmf_dx + self.g * h * (S_f - self.S0)

        return res_cont, res_mom


class DamPINN(nn.Module):
    """
    Dam / reservoir module.
    Adds a storage term and gate-controlled outflow to Saint-Venant.
    Models:
      - Reservoir pool: level Z(t), storage S = A·Z
      - Gate outflow: Q_out = C_d · W · Z^(3/2)  (broad-crested weir)
      - Downstream reach: standard 1D Saint-Venant from gate to outlet
    The network learns Z(t) and h(x,t), u(x,t) in the downstream reach.
    """
    def __init__(self, hidden=64, depth=4, g=9.81, S0=0.0005,
                 reservoir_area=1e6, Cd=0.611, gate_width=10.0):
        super().__init__()
        # Z(t) network: time → reservoir level
        self.reservoir_net = MLP(in_dim=1, out_dim=1, hidden=32, depth=3)
        # downstream reach network: (x,t) → h, u
        self.reach_net = MLP(in_dim=2, out_dim=2, hidden=hidden, depth=depth)
        self.g = g
        self.S0 = S0
        self.A = reservoir_area   # reservoir surface area (m²)
        self.Cd = Cd
        self.W = gate_width       # gate width (m)

    def reservoir_level(self, t_col):
        """Predict reservoir water level Z(t). softplus ensures Z > 0."""
        return torch.nn.functional.softplus(self.reservoir_net(t_col))

    def gate_outflow(self, Z):
        """Broad-crested weir: Q = Cd · W · g^0.5 · Z^1.5"""
        return self.Cd * self.W * (self.g ** 0.5) * (Z ** 1.5 + 1e-8)

    def forward(self, xt):
        out = self.reach_net(xt)
        h = torch.nn.functional.softplus(out[:, 0:1])
        u = out[:, 1:2]
        return h, u

    def reservoir_residual(self, t_col, Q_in):
        """
        Mass balance ODE for reservoir:
          A · dZ/dt = Q_in(t) - Q_out(Z)
        Residual = A·dZ/dt - Q_in + Q_out
        """
        t_col.requires_grad_(True)
        Z = self.reservoir_level(t_col)
        dZ_dt = torch.autograd.grad(Z.sum(), t_col, create_graph=True)[0]
        Q_out = self.gate_outflow(Z)
        residual = self.A * dZ_dt - Q_in + Q_out
        return residual, Z, Q_out

    def reach_pde_residual(self, xt, n_manning=0.025):
        """Standard Saint-Venant for downstream reach."""
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
