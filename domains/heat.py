"""
Heat domain — 1D heat / diffusion equation as a PDEModule.

This is the canonical first PINN example in almost every paper —
the cheapest way to prove the general framework works across domains.

Governing equation:
    dT/dt = alpha * d^2T/dx^2

Physical meaning: T(x,t) is temperature along a 1D rod. alpha is the
thermal diffusivity — how fast heat spreads. Higher alpha = faster spread.
"""

import torch
from core.pde_module import PDEModule


class HeatPDE(PDEModule):
    """
    1D heat equation.
    Inputs: (x, t)
    Outputs: [T]  — temperature

    dT/dt - alpha * d^2T/dx^2 = 0
    """
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    @property
    def name(self):
        return "Heat diffusion (1D)"

    @property
    def input_dim(self):
        return 2  # (x, t)

    @property
    def output_dim(self):
        return 1  # T

    @property
    def output_names(self):
        return ["T"]

    def transform_output(self, raw_output):
        # no constraint needed — temperature can be any real value
        # (normalised space; real temps recovered via rescaling outside)
        return raw_output

    def residual(self, coords, net_fn):
        T = net_fn(coords)

        dT = torch.autograd.grad(T.sum(), coords, create_graph=True)[0]
        dT_dt = dT[:, 1:2]
        dT_dx = dT[:, 0:1]

        d2T = torch.autograd.grad(dT_dx.sum(), coords, create_graph=True)[0]
        d2T_dx2 = d2T[:, 0:1]

        res = dT_dt - self.alpha * d2T_dx2
        return res

    def describe_equation(self):
        return "dT/dt = alpha * d^2T/dx^2   (1D heat / diffusion equation)"


def solve_heat_analytical(x, t, alpha=0.01, L=1.0, n_modes=20, T0=100.0):
    """
    Ground truth via Fourier series solution for a rod with fixed
    zero-temperature ends and an initial sinusoidal heat pulse.
    T(x,0) = T0 * sin(pi*x/L)
    T(x,t) = T0 * sin(pi*x/L) * exp(-alpha * (pi/L)^2 * t)   [exact for n=1 mode]
    Using just the first mode gives a clean, physically correct ground truth.
    """
    import numpy as np
    xx, tt = np.meshgrid(x, t)
    T = T0 * np.sin(np.pi * xx / L) * np.exp(-alpha * (np.pi / L) ** 2 * tt)
    return T  # shape (len(t), len(x))
