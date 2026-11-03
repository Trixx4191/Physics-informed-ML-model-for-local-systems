"""Operational regressions for the public Dash training dispatcher."""

import numpy as np
import pytest
import torch

from core.pde_module import PINNEngine, train_pinn
from dash_app import domain_utils
from domains.heat import HeatPDE
from experiments.sparsity_and_uncertainty import train_mc_dropout


@pytest.fixture(autouse=True)
def single_torch_thread():
    """Small derivative workloads are faster and repeatable on one CPU thread."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("domain", list(domain_utils.DOMAIN_INFO))
def test_every_domain_trains_and_reports_its_final_checkpoint(domain):
    data = domain_utils.generate_domain_data(domain)
    checkpoints = []
    _, history, _, extra = domain_utils.train_domain(
        domain, data, fraction=0.05, n_epochs=2, hidden=8, depth=1,
        n_colloc=8, callback=lambda epoch, _: checkpoints.append(epoch),
    )

    assert history["epoch"] == checkpoints == [0, 1]
    assert all(len(values) == 2 for values in history.values())
    assert all(np.isfinite(values).all() for values in history.values())
    assert extra["n_train"] > 0
    assert all(np.isfinite(value).all() for value in extra.values())
    if domain == "inverse_fluids":
        assert extra["n_final"] == history["n_inferred"][-1]
    if domain == "dam":
        assert extra["n_res_train"] == max(2, int(len(data["t_res"]) * 0.05))
        reach_points = data["h_reach"].size
        assert extra["n_reach_train"] == max(8, int(reach_points * 0.05))
        assert extra["n_train"] == extra["n_res_train"] + extra["n_reach_train"]


def test_hundred_epoch_run_has_a_visible_curve_and_final_progress():
    torch.manual_seed(7)
    model = PINNEngine(HeatPDE(), hidden=8, depth=1)
    coords = torch.rand(8, 2)
    checkpoints = []
    _, history = train_pinn(
        model, coords, torch.zeros(8, 1), n_epochs=100, n_colloc=8,
        callback=lambda epoch, _: checkpoints.append(epoch),
    )

    assert history["epoch"] == checkpoints == [0, 99]
    assert len(domain_utils.make_loss_curve_fig(history).data[0].x) == 2


def test_mc_dropout_reports_final_checkpoint_and_selected_dropout():
    torch.manual_seed(7)
    coords = torch.rand(8, 2)
    checkpoints = []
    model, history = train_mc_dropout(
        HeatPDE(alpha=0.12), coords, torch.zeros(8, 1),
        n_epochs=2, n_colloc=8, hidden=8, depth=1, dropout_p=0.2,
        callback=lambda epoch, _: checkpoints.append(epoch),
    )

    assert history["epoch"] == checkpoints == [0, 1]
    assert model.pde_module.alpha == 0.12
    assert model.dropout_p == 0.2
    means, stds = model.predict_with_uncertainty(coords, n_samples=8)
    assert means.shape == stds.shape == (8, 1)
    assert np.isfinite(means).all() and np.isfinite(stds).all()
    assert stds.mean() > 0


def test_elasticity_uses_selected_physics_weight(monkeypatch):
    captured = {}

    def capture_train(model, coords, targets, **kwargs):
        captured.update(kwargs)
        return model, {}

    monkeypatch.setattr(domain_utils, "train_pinn", capture_train)
    domain_utils.train_domain(
        "elasticity", domain_utils.generate_domain_data("elasticity"),
        lambda_pde=0.27, hidden=8, depth=1,
    )
    assert captured["lambda_pde"] == 0.27


def test_dispatcher_seed_repeats_initialization_and_training():
    data = domain_utils.generate_domain_data("heat", alpha=0.12)
    options = dict(n_epochs=2, hidden=8, depth=1, n_colloc=8, seed=42)
    _, history_a, _, extra_a = domain_utils.train_domain("heat", data, **options)
    _, history_b, _, extra_b = domain_utils.train_domain("heat", data, **options)
    assert history_a == history_b
    np.testing.assert_array_equal(extra_a["T_pred"], extra_b["T_pred"])


def test_dam_fraction_selects_observations_reproducibly():
    from domains.dam import train_dam_system

    data = domain_utils.generate_domain_data("dam")
    options = dict(n_epochs=1, n_colloc=8, seed=42)
    model_a, history_a = train_dam_system(data, fraction=0.1, **options)
    model_b, history_b = train_dam_system(data, fraction=0.1, **options)
    full_model, _ = train_dam_system(data, fraction=1.0, **options)

    assert len(model_a.reservoir_train_idx) == int(len(data["t_res"]) * 0.1)
    assert len(model_a.reach_train_idx) == int(data["h_reach"].size * 0.1)
    np.testing.assert_array_equal(model_a.reservoir_train_idx, model_b.reservoir_train_idx)
    np.testing.assert_array_equal(model_a.reach_train_idx, model_b.reach_train_idx)
    assert history_a == history_b
    assert full_model.n_res_train == len(data["t_res"])
    assert full_model.n_reach_train == data["h_reach"].size


def test_short_fluids_run_reaches_physics_phase():
    data = domain_utils.generate_domain_data("fluids")
    _, history, _, extra = domain_utils.train_domain(
        "fluids", data, fraction=0.005, n_epochs=100, n_colloc=8,
    )
    assert extra["warmup_epochs"] == 40
    assert history["lambda_adaptive"][-1] > 0
    assert extra["lambda_final"] == history["lambda_adaptive"][-1]


def test_fluids_weight_initializes_adaptive_trainer(monkeypatch):
    captured = {}

    def capture_train(model, coords, targets, **kwargs):
        captured.update(kwargs)
        return model, {"lambda_adaptive": [kwargs["lambda_init"]]}

    monkeypatch.setattr(domain_utils, "train_pinn_adaptive", capture_train)
    domain_utils.train_domain(
        "fluids", domain_utils.generate_domain_data("fluids"),
        lambda_pde=0.42, n_epochs=1,
    )
    assert captured["lambda_init"] == 0.42
    assert captured["warmup_epochs"] == 0


def test_sparsity_sweep_uses_five_observations_for_five_percent_of_hundred(monkeypatch):
    from experiments import sparsity_and_uncertainty as experiments

    pinn_counts, mlp_counts = [], []
    original_mlp = experiments.VanillaMLP

    def capture_pinn(model, coords, targets, **kwargs):
        pinn_counts.append(len(coords))
        assert kwargs["lambda_pde"] == 0.42
        return model, {}

    class CaptureMLP(original_mlp):
        def forward(self, coords):
            mlp_counts.append(len(coords))
            return super().forward(coords)

    monkeypatch.setattr(experiments, "train_pinn", capture_pinn)
    monkeypatch.setattr(experiments, "VanillaMLP", CaptureMLP)
    coords = torch.rand(100, 2)
    targets = torch.linspace(-1, 1, 100).reshape(-1, 1)
    experiments.run_sparsity_sweep(
        HeatPDE(), coords, targets, coords, targets[:, 0].numpy(),
        fractions=[0.05], n_seeds=1, n_epochs=1, lambda_pde=0.42,
    )
    assert pinn_counts == [5]
    assert mlp_counts == [5, 100]
