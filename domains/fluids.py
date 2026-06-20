"""
Fluids domain — 1D Saint-Venant equations as a PDEModule.

This module implements the fluids PINN (Saint-Venant) within the general
multi-domain framework. Nothing about the physics changes — only how it's
wired into the engine.

Two variants:
  FluidsPDE         — forward problem, n_manning supplied as a constant
  InverseFluidsPDE   — inverse problem, n(x) learned as a trainable field
"""

import torch
import torch.nn as nn
from core.pde_module import PDEModule, MLP


class FluidsPDE(PDEModule):
    """
    1D shallow water / open-channel flow.
    Inputs: (x, t)
    Outputs: [h, u]  — water depth, velocity

    Governing equations (Saint-Venant):
        continuity: dh/dt + d(uh)/dx = 0
        momentum:   d(uh)/dt + d(u^2 h + 0.5 g h^2)/dx + g h (Sf - S0) = 0
        friction slope Sf = n^2 u |u| / h^(4/3)
    """
    def __init__(self, g=9.81, S0=0.001, n_manning=0.03):
        self.g = g
        self.S0 = S0
        self.n_manning = n_manning

    @property
    def name(self):
        return "Fluids — open channel flow (Saint-Venant)"

    @property
    def input_dim(self):
        return 2  # (x, t)

    @property
    def output_dim(self):
        return 2  # (h, u)

    @property
    def output_names(self):
        return ["h", "u"]

    def transform_output(self, raw_output):
        h = torch.nn.functional.softplus(raw_output[:, 0:1])  # h > 0 always
        u = raw_output[:, 1:2]
        return torch.cat([h, u], dim=1)

    def residual(self, coords, net_fn):
        out = net_fn(coords)
        h, u = out[:, 0:1], out[:, 1:2]

        dh = torch.autograd.grad(h.sum(), coords, create_graph=True)[0]
        dh_dt = dh[:, 1:2]

        q = u * h
        dq = torch.autograd.grad(q.sum(), coords, create_graph=True)[0]
        dq_dx = dq[:, 0:1]

        res_cont = dh_dt + dq_dx

        dq_dt = torch.autograd.grad(q.sum(), coords, create_graph=True)[0][:, 1:2]
        momentum_flux = u * q + 0.5 * self.g * h ** 2
        dmf = torch.autograd.grad(momentum_flux.sum(), coords, create_graph=True)[0]
        dmf_dx = dmf[:, 0:1]

        S_f = (self.n_manning ** 2) * u * torch.abs(u) / (h ** (4.0 / 3.0) + 1e-8)
        res_mom = dq_dt + dmf_dx + self.g * h * (S_f - self.S0)

        return res_cont, res_mom

    def describe_equation(self):
        return ("dh/dt + d(uh)/dx = 0   (continuity)\n"
                "d(uh)/dt + d(u^2h + 0.5gh^2)/dx + gh(Sf - S0) = 0   (momentum)")


class InverseFluidsPDE(PDEModule):
    """
    Inverse variant of Saint-Venant: Manning's n(x) is NOT supplied — it is
    a second small network learned jointly with the flow field. This is
    the "discover the physics" module: the most academically novel piece
    of the original project, now reframed as one FluidsPDE variant rather
    than a separate model class.

    The PDEModule interface wasn't built with a second learned sub-network
    in mind, so this class owns its own nn.Module for n(x) and exposes it
    via get_n() for the engine/UI to query and plot separately from the
    main h, u outputs.
    """
    def __init__(self, g=9.81, S0=0.001, n_min=0.01, n_max=0.15):
        self.g = g
        self.S0 = S0
        self.n_min = n_min
        self.n_max = n_max
        # Small separate network for n(x) — owned here, not by PINNEngine,
        # since PINNEngine.net only models the (h,u) outputs declared by
        # output_dim. The training loop must add this module's parameters
        # to the optimizer explicitly (see train_inverse_fluids below).
        self.n_net = MLP(in_dim=1, out_dim=1, hidden=32, depth=3)

    @property
    def name(self):
        return "Fluids — inverse roughness inference"

    @property
    def input_dim(self):
        return 2

    @property
    def output_dim(self):
        return 2

    @property
    def output_names(self):
        return ["h", "u"]

    def transform_output(self, raw_output):
        h = torch.nn.functional.softplus(raw_output[:, 0:1])
        u = raw_output[:, 1:2]
        return torch.cat([h, u], dim=1)

    def get_n(self, x_col):
        """Infer Manning's n at spatial locations x_col (N,1), constrained
        to a physical range via sigmoid scaling."""
        raw = self.n_net(x_col)
        return self.n_min + (self.n_max - self.n_min) * torch.sigmoid(raw)

    def residual(self, coords, net_fn):
        out = net_fn(coords)
        h, u = out[:, 0:1], out[:, 1:2]

        x_col = coords[:, 0:1]
        n = self.get_n(x_col)

        dh = torch.autograd.grad(h.sum(), coords, create_graph=True)[0]
        dh_dt = dh[:, 1:2]
        q = u * h
        dq = torch.autograd.grad(q.sum(), coords, create_graph=True)[0]
        dq_dx = dq[:, 0:1]
        res_cont = dh_dt + dq_dx

        dq_dt = torch.autograd.grad(q.sum(), coords, create_graph=True)[0][:, 1:2]
        mf = u * q + 0.5 * self.g * h ** 2
        dmf_dx = torch.autograd.grad(mf.sum(), coords, create_graph=True)[0][:, 0:1]
        S_f = (n ** 2) * u * torch.abs(u) / (h ** (4.0 / 3.0) + 1e-8)
        res_mom = dq_dt + dmf_dx + self.g * h * (S_f - self.S0)

        return res_cont, res_mom

    def smoothness_penalty(self, n_points=200):
        """Optional regulariser: penalise large spatial gradients in n(x)
        so the inferred field doesn't overfit to noise. Called explicitly
        by the training loop, not part of the standard residual()."""
        x_smooth = torch.linspace(-1, 1, n_points).unsqueeze(1).requires_grad_(True)
        n_smooth = self.get_n(x_smooth)
        dn_dx = torch.autograd.grad(n_smooth.sum(), x_smooth, create_graph=True)[0]
        return torch.mean(dn_dx ** 2)

    def describe_equation(self):
        return ("dh/dt + d(uh)/dx = 0\n"
                "d(uh)/dt + d(u^2h + 0.5gh^2)/dx + gh(Sf - S0) = 0\n"
                "n(x) learned, not supplied — inferred from data + physics jointly")
