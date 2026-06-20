"""
Elasticity domain — Euler-Bernoulli beam bending as a PDEModule.

Governing equation (static beam bending under distributed load):
    EI * d^4v/dx^4 = q(x)

Physical meaning: v(x) is the vertical deflection of a beam at position x.
EI is flexural rigidity (E = Young's modulus, I = second moment of area —
together they describe how stiff the beam is). q(x) is the distributed
load (e.g. weight per unit length).

This is a FOURTH-order ODE in space, no time dependence — the third
distinct equation "shape" in the framework after first-order coupled
(fluids), second-order parabolic (heat), and second-order hyperbolic
(wave). A 4th-order spatial derivative requires autograd to differentiate
through itself three times — a genuine stress-test of the engine.

This ties directly back to your original brief: "structural loads on
Kumasi market roofs" or any beam/bridge/roof under load is this exact
equation.
"""

import torch
import numpy as np
from core.pde_module import PDEModule


class ElasticityPDE(PDEModule):
    """
    Euler-Bernoulli beam under distributed load.
    Inputs: (x,)       — position along the beam, normalised
    Outputs: [v]        — vertical deflection

    EI * d^4v/dx^4 - q(x) = 0
    """
    def __init__(self, EI=1.0, q0=1.0, load_type="uniform"):
        self.EI = EI
        self.q0 = q0
        self.load_type = load_type  # "uniform" or "point" (point at midspan)

    @property
    def name(self):
        return "Elasticity — beam bending (Euler-Bernoulli)"

    @property
    def input_dim(self):
        return 1  # x only (static problem, no time)

    @property
    def output_dim(self):
        return 1  # v (deflection)

    @property
    def output_names(self):
        return ["v"]

    def transform_output(self, raw_output):
        return raw_output  # deflection can be any sign

    def _load(self, x):
        if self.load_type == "uniform":
            return self.q0 * torch.ones_like(x)
        else:  # smooth approximation of a point load at midspan (Gaussian)
            return self.q0 * torch.exp(-((x - 0.0) ** 2) / 0.01) / np.sqrt(0.01 * np.pi)

    def residual(self, coords, net_fn):
        v = net_fn(coords)

        dv = torch.autograd.grad(v.sum(), coords, create_graph=True)[0]
        d2v = torch.autograd.grad(dv.sum(), coords, create_graph=True)[0]
        d3v = torch.autograd.grad(d2v.sum(), coords, create_graph=True)[0]
        d4v = torch.autograd.grad(d3v.sum(), coords, create_graph=True)[0]

        q = self._load(coords)
        res = self.EI * d4v - q
        return res

    def describe_equation(self):
        return "EI * d^4v/dx^4 = q(x)   (Euler-Bernoulli beam bending)"


def solve_beam_analytical(x, L=1.0, EI=1.0, q0=1.0, load_type="uniform"):
    """
    Exact solution for a simply-supported beam.
    Boundary conditions: v(0)=v(L)=0, v''(0)=v''(L)=0 (simple supports,
    zero moment at ends).

    For uniform load: v(x) = q0/(24*EI) * (x^4 - 2*L*x^3 + L^3*x)
    For central point load P=q0: piecewise classic simply-supported beam deflection.
    """
    if load_type == "uniform":
        v = q0 / (24 * EI) * (x ** 4 - 2 * L * x ** 3 + L ** 3 * x)
    else:
        # Point load at midspan
        mid = L / 2.0
        v = np.where(
            x <= mid,
            q0 * x * (3 * L ** 2 - 4 * x ** 2) / (48 * EI),
            q0 * (L - x) * (3 * L ** 2 - 4 * (L - x) ** 2) / (48 * EI)
        )
    return v
