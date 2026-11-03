"""Analytical checks for physical derivatives and reference-solver consistency."""

import numpy as np
import pytest
import torch

from core.pde_module import NormalizedPDE
from dash_app.domain_utils import generate_domain_data, prepare_domain_training_data
from data.generator import generate_dam_data, solve_saint_venant
from domains.dam import DamSystem
from domains.fluids import FluidsPDE


@pytest.mark.parametrize("domain,params", [
    ("heat", {"alpha": 0.12}),
    ("wave", {"c": 1.5, "mode": 2}),
    ("gravity", {"GM": 2.0, "eccentricity": 0.0}),
    ("elasticity", {"EI": 3.0, "L": 2.0, "q0": 0.7}),
])
def test_exact_physical_solutions_satisfy_normalized_residual(domain, params):
    data = generate_domain_data(domain, **params)
    pde, _, _, _, ranges = prepare_domain_training_data(domain, data)
    coords = torch.tensor([[0.2 * lo + 0.8 * hi for lo, hi in ranges],
                           [0.7 * lo + 0.3 * hi for lo, hi in ranges]],
                          dtype=torch.float64, requires_grad=True)

    def exact_normalized(inputs):
        physical = inputs * inputs.new_tensor(pde.input_std) + inputs.new_tensor(pde.input_mean)
        if domain == "heat":
            x, t = physical[:, :1], physical[:, 1:2]
            output = 100 * torch.sin(torch.pi * x) * torch.exp(-params["alpha"] * torch.pi ** 2 * t)
        elif domain == "wave":
            x, t = physical[:, :1], physical[:, 1:2]
            output = torch.sin(params["mode"] * torch.pi * x) * torch.cos(params["mode"] * torch.pi * params["c"] * t)
        elif domain == "gravity":
            phase = params["GM"] ** 0.5 * physical
            output = torch.cat([torch.cos(phase), torch.sin(phase)], dim=1)
        else:
            x, L = physical, params["L"]
            output = params["q0"] / (24 * params["EI"]) * (x ** 4 - 2 * L * x ** 3 + L ** 3 * x)
        return (output - inputs.new_tensor(pde.output_mean)) / inputs.new_tensor(pde.output_std)

    residuals = pde.residual(coords, exact_normalized)
    if not isinstance(residuals, tuple):
        residuals = (residuals,)
    # Gravity retains its existing 1e-6 singularity guard, giving O(1e-5)
    # dimensionless residual for an otherwise exact circular trajectory.
    assert max(float(value.detach().abs().max()) for value in residuals) < 1e-4


def test_equilibrium_fluid_residual_uses_physical_depth_velocity_and_coordinates():
    h, u, n = 4.0, 1.7, 0.035
    pde = NormalizedPDE(
        FluidsPDE(S0=n ** 2 * u ** 2 / h ** (4 / 3), n_manning=n),
        input_mean=[15000, 30000], input_std=[8000, 10000],
        output_mean=[3, 2], output_std=[0.5, 0.3], positive_outputs=(0,),
    )
    coords = torch.rand(5, 2, dtype=torch.float64, requires_grad=True)

    def exact(inputs):
        return torch.cat([inputs[:, :1] * 0 + (h - 3) / 0.5,
                          inputs[:, :1] * 0 + (u - 2) / 0.3], dim=1)

    continuity, momentum = pde.residual(coords, exact)
    assert continuity.abs().max() == 0
    assert momentum.abs().max() < 1e-8


def test_positive_physical_depth_allows_negative_standardized_depth():
    pde = NormalizedPDE(FluidsPDE(), [0, 0], [1, 1], [4, 1], [1, 0.2], positive_outputs=(0,))
    output = pde.transform_output(torch.tensor([[-1.0, -0.5]]))
    assert output[0, 0] < 0
    assert pde.to_physical_output(output)[0, 0] > 0


def test_saint_venant_reference_preserves_uniform_manning_equilibrium():
    h, u, n, width = 3.5, 1.2, 0.035, 10.0
    _, _, depths, velocities, _ = solve_saint_venant(
        "volta", Nx=12, Nt=10, T=1000,
        custom_params={"h0": h, "u0": u, "n_manning": n,
                       "S0": n ** 2 * u ** 2 / h ** (4 / 3), "flood_amp": 0},
        upstream_discharge=lambda _: width * h * u, channel_width=width,
    )
    np.testing.assert_allclose(depths, h, atol=1e-7)
    np.testing.assert_allclose(velocities, u, atol=1e-7)


def test_dam_reference_obeys_mass_balance_and_gate_boundary():
    data = generate_dam_data(T=3600, Nt_res=24, Nx=12, Nt_reach=24)
    cfg = data["cfg"]
    np.testing.assert_allclose(
        cfg["reservoir_area"] * np.diff(data["Z"]) / np.diff(data["t_res"]),
        data["Q_in"][:-1] - data["Q_out"][:-1], atol=1e-8,
    )
    np.testing.assert_allclose(
        cfg["channel_width"] * data["h_reach"][:, 0] * data["u_reach"][:, 0],
        np.interp(data["t_reach"], data["t_res"], data["Q_out"]), atol=1e-8,
    )
    changed = generate_dam_data(T=3600, Nt_res=24, Nx=12, Nt_reach=24, gate_width=15)
    assert not np.allclose(data["u_reach"][:, 0], changed["u_reach"][:, 0])


def test_dam_reservoir_residual_differentiates_physical_time_and_level():
    stats = dict(res_t_m=30.0, res_t_s=20.0, Z_m=4.0, Z_s=2.0,
                 x_m=100, x_s=50, t_m=30, t_s=20,
                 h_m=3.0, h_s=1.0, u_m=1.0, u_s=0.5)
    model = DamSystem(reservoir_area=1000, normalization=stats)

    class ExactLevel(torch.nn.Module):
        def forward(self, times):
            physical_time = times * stats["res_t_s"] + stats["res_t_m"]
            positive_value = (5 + 0.02 * physical_time) / stats["Z_s"]
            return torch.log(torch.expm1(positive_value)) - stats["Z_m"] / stats["Z_s"]

    model.reservoir_net = ExactLevel()
    times = torch.linspace(-1, 1, 5, dtype=torch.float64).reshape(-1, 1)
    exact_level = 5 + 0.02 * (times * stats["res_t_s"] + stats["res_t_m"])
    inflow = 1000 * 0.02 + model.gate_outflow(exact_level)
    residual, level, _ = model.reservoir_residual(times, inflow)
    torch.testing.assert_close(level, exact_level)
    assert residual.abs().max() < 1e-10
