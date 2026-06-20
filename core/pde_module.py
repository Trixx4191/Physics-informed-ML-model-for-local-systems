"""
PDE Module — the general physics-informed core.

Key architectural idea:
  A PINN is really two separable things:
    1. A neural network (the MLP) that maps inputs -> outputs
    2. A physics residual function that checks whether those outputs
       satisfy a governing equation, computed via autograd

  Almost every PINN paper (fluids, heat, waves, elasticity, EM) reuses
  the SAME network engine and training loop. Only the residual function
  changes. This file defines that separation cleanly so new physics
  domains can be added by writing ONE class, not rebuilding the trainer.

Usage pattern:
    module = HeatPDEModule(alpha=0.01)
    model  = PINNEngine(module, in_dim=2, out_dim=1)
    train(model, data, collocation_points)
"""

import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod


# ── The shared network engine ──────────────────────────────────────────────────

class MLP(nn.Module):
    """Generic fully-connected network. Tanh activations — required for
    smooth second derivatives under autograd."""
    def __init__(self, in_dim, out_dim, hidden=64, depth=4):
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


class PDEModule(ABC):
    """
    Abstract base class for a physics domain.
    Every domain (fluids, heat, waves, gravity, elasticity, EM)
    implements this interface. The training engine only ever calls
    these three methods — it never needs to know what equation is inside.
    """

    @property
    @abstractmethod
    def name(self):
        """Human-readable domain name, e.g. 'Heat diffusion'"""
        ...

    @property
    @abstractmethod
    def input_dim(self):
        """Dimensionality of the input (e.g. 2 for (x,t), 3 for (x,y,t))"""
        ...

    @property
    @abstractmethod
    def output_dim(self):
        """Dimensionality of the network output (e.g. 1 for temperature,
        2 for [h, u] in fluids)"""
        ...

    @property
    @abstractmethod
    def output_names(self):
        """List of output variable names, e.g. ['h', 'u'] or ['T']"""
        ...

    @abstractmethod
    def transform_output(self, raw_output):
        """Apply any physical constraints to the raw network output
        (e.g. softplus to keep depth > 0). Returns same shape."""
        ...

    @abstractmethod
    def residual(self, coords, raw_net_fn):
        """
        Compute the PDE residual(s) at the given coordinates.
        coords: (N, input_dim) tensor, requires_grad=True will be set internally
        raw_net_fn: callable, raw_net_fn(coords) -> transformed network output
        Returns: tensor or tuple of tensors, each residual should be ~0
                 when the PDE is satisfied
        """
        ...

    def describe_equation(self):
        """LaTeX-ish string describing the governing equation, for display."""
        return "Governing equation not specified"


class PINNEngine(nn.Module):
    """
    The general PINN wrapper. Takes any PDEModule and builds a trainable
    network around it. This replaces ForwardPINN/InversePINN/DamPINN with
    ONE class that works for every physics domain.
    """
    def __init__(self, pde_module: PDEModule, hidden=64, depth=4):
        super().__init__()
        self.pde_module = pde_module
        self.net = MLP(in_dim=pde_module.input_dim,
                       out_dim=pde_module.output_dim,
                       hidden=hidden, depth=depth)

    def forward(self, coords):
        raw = self.net(coords)
        return self.pde_module.transform_output(raw)

    def pde_residual(self, coords):
        coords = coords.clone().requires_grad_(True)
        return self.pde_module.residual(coords, self.forward)


# ── Generic collocation sampler (domain-agnostic) ───────────────────────────────

def sample_collocation(n, input_dim, ranges=None, seed=None):
    """
    Sample n random points in the input domain.
    ranges: list of (min, max) tuples, one per input dimension.
            Defaults to (-1, 1) for every dimension (normalised domain).
    """
    rng = np.random.RandomState(seed)
    if ranges is None:
        ranges = [(-1, 1)] * input_dim
    cols = [rng.uniform(lo, hi, n) for lo, hi in ranges]
    return torch.tensor(np.stack(cols, axis=1), dtype=torch.float32)


# ── Generic trainer — works for ANY PDEModule ───────────────────────────────────

def train_pinn(model: PINNEngine, data_coords, data_targets,
               n_epochs=3000, lr=1e-3, lambda_data=1.0, lambda_pde=0.1,
               n_colloc=3000, colloc_ranges=None, callback=None):
    """
    Universal training loop. Works identically whether model wraps
    FluidsPDE, HeatPDE, WavePDE, GravityPDE, etc.

    data_coords:  (N, input_dim) tensor — normalised input coordinates
    data_targets: (N, output_dim) tensor — normalised observed values
    """
    import torch.optim as optim

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, eta_min=1e-5)

    history = {"epoch": [], "loss_total": [], "loss_data": [], "loss_pde": []}

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        pred = model(data_coords)
        loss_data = torch.mean((pred - data_targets) ** 2)

        coll = sample_collocation(n_colloc, model.pde_module.input_dim,
                                  ranges=colloc_ranges, seed=epoch)
        residuals = model.pde_residual(coll)
        if isinstance(residuals, (tuple, list)):
            loss_pde = sum(torch.mean(r ** 2) for r in residuals)
        else:
            loss_pde = torch.mean(residuals ** 2)

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


# ── Trainer variant for modules that own extra trainable sub-networks ──────────

def train_pinn_with_extra_params(model: PINNEngine, data_coords, data_targets,
                                 extra_params, extra_penalty_fn=None,
                                 n_epochs=3000, lr=1e-3, lambda_data=1.0,
                                 lambda_pde=0.1, lambda_extra=0.01,
                                 n_colloc=3000, colloc_ranges=None, callback=None,
                                 history_extra_key=None, history_extra_fn=None):
    """
    Like train_pinn(), but for PDEModules that own extra learnable
    parameters outside the main PINNEngine.net (e.g. InverseFluidsPDE's
    n_net, which infers Manning's roughness alongside the flow field).

    extra_params: iterable of extra nn.Module parameters to add to the optimizer
    extra_penalty_fn: optional callable() -> scalar tensor, added to the loss
                      (e.g. a smoothness regulariser on the inferred field)
    history_extra_key / history_extra_fn: optionally track a custom scalar
                      in the history dict (e.g. mean inferred Manning's n)
    """
    import torch.optim as optim

    all_params = list(model.parameters()) + list(extra_params)
    optimizer = optim.Adam(all_params, lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, eta_min=1e-5)

    history = {"epoch": [], "loss_total": [], "loss_data": [], "loss_pde": []}
    if history_extra_key:
        history[history_extra_key] = []

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        pred = model(data_coords)
        loss_data = torch.mean((pred - data_targets) ** 2)

        coll = sample_collocation(n_colloc, model.pde_module.input_dim,
                                  ranges=colloc_ranges, seed=epoch)
        residuals = model.pde_residual(coll)
        if isinstance(residuals, (tuple, list)):
            loss_pde = sum(torch.mean(r ** 2) for r in residuals)
        else:
            loss_pde = torch.mean(residuals ** 2)

        loss = lambda_data * loss_data + lambda_pde * loss_pde

        if extra_penalty_fn is not None:
            loss = loss + lambda_extra * extra_penalty_fn()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 100 == 0:
            history["epoch"].append(epoch)
            history["loss_total"].append(loss.item())
            history["loss_data"].append(loss_data.item())
            history["loss_pde"].append(loss_pde.item())
            if history_extra_key and history_extra_fn:
                history[history_extra_key].append(history_extra_fn())
            if callback:
                callback(epoch, history)

    return model, history
