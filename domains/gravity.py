"""
Gravity domain — two-body orbital mechanics as a PDEModule.

IMPORTANT DISTINCTION (worth stating plainly, not glossing over):
Gravity here is NOT a partial differential equation over a continuous
field like fluids/heat/wave. It's an ORDINARY differential equation
system describing how a body's position evolves through time under
Newton's law of gravitation:

    d^2x/dt^2 = -G*M*x / |r|^3
    d^2y/dt^2 = -G*M*y / |r|^3

The network learns x(t), y(t) — the orbit trajectory — and the
"physics residual" checks whether the network's second time derivative
matches the gravitational acceleration at every point in time.

This still fits the PDEModule interface (single time input instead of
space+time), which is itself worth noting in a paper: the same PINN
machinery generalises from PDEs to ODEs because autograd doesn't care
which kind of equation it's differentiating.
"""

import torch
import numpy as np
from core.pde_module import PDEModule


class GravityPDE(PDEModule):
    """
    Two-body orbital mechanics (one body orbiting a fixed massive centre,
    e.g. a planet around the sun, simplified to the restricted two-body
    problem — the standard first case in orbital mechanics).

    Inputs: (t,)            — just time, normalised
    Outputs: [x, y]          — 2D position of the orbiting body

    Residual enforces:
        d^2x/dt^2 = -GM * x / r^3
        d^2y/dt^2 = -GM * y / r^3
        where r = sqrt(x^2 + y^2)
    """
    def __init__(self, GM=1.0):
        self.GM = GM  # combined gravitational parameter (G * M_central)

    @property
    def name(self):
        return "Gravity — two-body orbital mechanics"

    @property
    def input_dim(self):
        return 1  # t only

    @property
    def output_dim(self):
        return 2  # x, y

    @property
    def output_names(self):
        return ["x", "y"]

    def transform_output(self, raw_output):
        return raw_output  # position can be any sign

    def residual(self, coords, net_fn):
        pos = net_fn(coords)
        x, y = pos[:, 0:1], pos[:, 1:2]

        dx = torch.autograd.grad(x.sum(), coords, create_graph=True)[0]
        dx_dt = dx[:, 0:1]
        d2x = torch.autograd.grad(dx_dt.sum(), coords, create_graph=True)[0]
        d2x_dt2 = d2x[:, 0:1]

        dy = torch.autograd.grad(y.sum(), coords, create_graph=True)[0]
        dy_dt = dy[:, 0:1]
        d2y = torch.autograd.grad(dy_dt.sum(), coords, create_graph=True)[0]
        d2y_dt2 = d2y[:, 0:1]

        r = torch.sqrt(x ** 2 + y ** 2 + 1e-6)
        r3 = r ** 3

        res_x = d2x_dt2 + self.GM * x / r3
        res_y = d2y_dt2 + self.GM * y / r3

        return res_x, res_y

    def describe_equation(self):
        return ("d^2x/dt^2 = -GM*x/r^3\n"
                "d^2y/dt^2 = -GM*y/r^3   (Newtonian two-body gravity, r=sqrt(x^2+y^2))")


def solve_orbit_analytical(t, GM=1.0, a=1.0, e=0.0):
    """
    Exact Keplerian orbit solution for eccentricity e (0=circular, <1=ellipse).
    Solves Kepler's equation via Newton-Raphson, then converts to (x,y).
    a: semi-major axis. Period T = 2*pi*sqrt(a^3/GM).
    """
    T = 2 * np.pi * np.sqrt(a ** 3 / GM)
    M = 2 * np.pi * t / T  # mean anomaly

    # Solve Kepler's equation M = E - e*sin(E) via Newton-Raphson
    E = M.copy()
    for _ in range(50):
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))

    # True position in orbital plane
    x = a * (np.cos(E) - e)
    y = a * np.sqrt(1 - e ** 2) * np.sin(E)

    return x, y, T


def generate_orbit_data(GM=1.0, a=1.0, e=0.3, n_points=300, n_periods=2):
    """Generate ground-truth orbit trajectory for training/evaluation."""
    T = 2 * np.pi * np.sqrt(a ** 3 / GM)
    t = np.linspace(0, n_periods * T, n_points)
    x, y, T_period = solve_orbit_analytical(t, GM=GM, a=a, e=e)
    return t, x, y, T_period
