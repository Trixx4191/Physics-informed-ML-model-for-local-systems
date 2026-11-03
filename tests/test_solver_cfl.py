"""The reference solver must satisfy CFL throughout the supported UI range."""

import numpy as np
import pytest

from data.generator import generate_dam_data, solve_saint_venant


def observed_cfl(x, t, h, u):
    assert np.isfinite(h).all() and np.isfinite(u).all()
    assert (h > 0).all()
    speed = np.max(np.abs(u) + np.sqrt(9.81 * h))
    return float(speed * np.max(np.diff(t)) / np.min(np.diff(x)))


@pytest.mark.parametrize("river", ["volta", "amazon", "rhine"])
def test_low_roughness_steep_slope_remains_within_cfl_limit(river):
    x, t, h, u, _ = solve_saint_venant(
        river, Nx=50, Nt=80,
        custom_params={"n_manning": 0.01, "S0": 0.002},
    )
    assert len(t) > 80
    assert observed_cfl(x, t, h, u) <= 0.9


@pytest.mark.parametrize("gate_width,reservoir_area", [(10.0, 1e6), (30.0, 1e5), (30.0, 5e6)])
def test_dam_default_and_extreme_controls_remain_within_cfl_limit(gate_width, reservoir_area):
    data = generate_dam_data(
        Nx=40, Nt_res=150, Nt_reach=150,
        gate_width=gate_width, reservoir_area=reservoir_area,
    )
    assert np.isfinite(data["Z"]).all()
    assert observed_cfl(data["x_reach"], data["t_reach"],
                        data["h_reach"], data["u_reach"]) <= 0.9
    cfg = data["cfg"]
    np.testing.assert_allclose(
        cfg["channel_width"] * data["h_reach"][:, 0] * data["u_reach"][:, 0],
        np.interp(data["t_reach"], data["t_res"], data["Q_out"]), atol=1e-8,
    )
