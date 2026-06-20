"""
Wave domain — 1D wave equation as a PDEModule.

Governing equation:
    d^2u/dt^2 = c^2 * d^2u/dx^2

Physical meaning: u(x,t) is displacement of a vibrating string (or
1D acoustic pressure wave). c is wave speed. This equation needs
SECOND time derivatives, which is a genuinely different residual
shape from heat (first time derivative) and fluids (coupled first
derivatives) — a real test of whether the abstraction generalises
or just happens to fit first-order systems.
"""

import torch
import numpy as np
from core.pde_module import PDEModule


class WavePDE(PDEModule):
    """
    1D wave equation.
    Inputs: (x, t)
    Outputs: [u]  — displacement

    d^2u/dt^2 - c^2 * d^2u/dx^2 = 0
    """
    def __init__(self, c=1.0):
        self.c = c

    @property
    def name(self):
        return "Wave propagation (1D)"

    @property
    def input_dim(self):
        return 2  # (x, t)

    @property
    def output_dim(self):
        return 1  # u (displacement)

    @property
    def output_names(self):
        return ["u"]

    def transform_output(self, raw_output):
        return raw_output  # displacement can be any sign

    def residual(self, coords, net_fn):
        u = net_fn(coords)

        du = torch.autograd.grad(u.sum(), coords, create_graph=True)[0]
        du_dt = du[:, 1:2]
        du_dx = du[:, 0:1]

        d2u_dt = torch.autograd.grad(du_dt.sum(), coords, create_graph=True)[0]
        d2u_dt2 = d2u_dt[:, 1:2]

        d2u_dx = torch.autograd.grad(du_dx.sum(), coords, create_graph=True)[0]
        d2u_dx2 = d2u_dx[:, 0:1]

        res = d2u_dt2 - (self.c ** 2) * d2u_dx2
        return res

    def describe_equation(self):
        return "d^2u/dt^2 = c^2 * d^2u/dx^2   (1D wave equation)"


def solve_wave_analytical(x, t, c=1.0, L=1.0, mode=1, amplitude=1.0):
    """
    Exact standing-wave solution for a string fixed at both ends:
    u(x,t) = A * sin(n*pi*x/L) * cos(n*pi*c*t/L)
    """
    xx, tt = np.meshgrid(x, t)
    n = mode
    u = amplitude * np.sin(n * np.pi * xx / L) * np.cos(n * np.pi * c * tt / L)
    return u  # shape (len(t), len(x))
