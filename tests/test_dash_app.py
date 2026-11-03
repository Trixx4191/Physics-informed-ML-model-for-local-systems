"""Keep scientific outputs and background jobs intact while changing the Dash UI.

Only numerical training is stubbed. Callbacks build their normal Dash components
and use the production Plotly helpers, including the domain-specific loss plots.
"""

from collections import Counter
from contextlib import contextmanager
from importlib import import_module
import runpy

import numpy as np
import pytest
from dash import dcc
from dash._callback_context import context_value
from dash._utils import AttributeDict


PARAMETERS = {
    "fluids": {"param-river": "volta", "param-n": 0.035, "param-s0": 0.0008},
    "heat": {"param-alpha": 0.05},
    "wave": {"param-c": 1.0, "param-mode": 1},
    "gravity": {"param-e": 0.3, "param-GM": 1.0},
    "elasticity": {"param-load": "uniform", "param-EI": 1.0, "param-L": 1.0, "param-q0": 1.0},
    "dam": {"param-gate-width": 10.0, "param-res-area": 1e6},
    "inverse_fluids": {"param-river": "volta", "param-true-n": 0.035},
}

HISTORY = {
    "epoch": [0, 4],
    "loss_total": [1.0, 0.2],
    "loss_data": [0.7, 0.1],
    "loss_pde": [0.3, 0.1],
    "lambda_adaptive": [0.1, 0.2],
    "loss_reservoir": [0.4, 0.1],
    "loss_reach": [0.3, 0.05],
}

TRAINING_INPUTS = {
    "n_clicks": 1,
    "fraction_pct": 35,
    "epochs": 400,
    "lambda_pde": 0.42,
    "river_key": "volta",
    "alpha_val": 0.08,
    "c_val": 1.8,
    "mode_val": 3,
    "e_val": 0.5,
    "load_val": "point",
    "n_val": 0.04,
    "s0_val": 0.001,
    "GM_val": 2.0,
    "EI_val": 2.5,
    "L_val": 1.5,
    "q0_val": 3.0,
    "gate_width_val": 12.0,
    "res_area_val": 2e6,
    "true_n_val": 0.045,
}


@pytest.fixture(scope="module")
def ui():
    return import_module("dash_app.app")


def walk(value):
    if isinstance(value, (tuple, list)):
        for child in value:
            yield from walk(child)
    else:
        yield value
        children = getattr(value, "children", None)
        if children is not None:
            yield from walk(children)


def text_content(value):
    return " ".join(str(part) for part in walk(value) if isinstance(part, (str, int, float)))


def by_id(value):
    return {part.id: part for part in walk(value) if getattr(part, "id", None)}


def graphs(value):
    return [part for part in walk(value) if isinstance(part, dcc.Graph)]


@pytest.fixture
def available_slot(ui, monkeypatch):
    events = []

    @contextmanager
    def slot(directory):
        events.append(("enter", directory))
        try:
            yield True
        finally:
            events.append(("exit", directory))

    monkeypatch.setattr(ui, "training_slot", slot)
    return events


def test_layout_retains_every_domain_training_control_and_experiment(ui):
    ids = by_id(ui.app.layout)
    domain_options = {option["value"] for option in ids["domain-select"].options}
    assert domain_options == set(PARAMETERS) == set(ui.DOMAIN_INFO)
    required = {
        "domain-params", "data-fraction", "epochs", "lambda-pde", "train-btn",
        "train-status", "train-progress", "equation-display", "results-area",
        "trained-results-store", "sparsity-btn", "sparsity-fractions", "sparsity-seeds",
        "mcdrop-btn", "dropout-p", "mc-samples", "experiment-results-area",
    }
    assert required <= ids.keys()
    component_ids = [part.id for part in walk(ui.app.layout) if getattr(part, "id", None)]
    assert len(component_ids) == len(set(component_ids)), "Duplicate IDs can break Dash callbacks"
    assert ids["sparsity-fractions"].value == [0.05, 0.10, 0.20, 0.50, 1.00]
    assert ids["sparsity-seeds"].value == 2
    assert ids["dropout-p"].value == 0.08
    assert ids["mc-samples"].value == 100


def test_tabs_preserve_mounted_results_and_keep_experiment_controls_out_of_toolbar(ui):
    controls = by_id(ui.control_bar)
    assert {"domain-select", "domain-params", "data-fraction", "epochs", "lambda-pde", "train-btn", "train-status", "train-progress"} <= controls.keys()
    assert not {"sparsity-btn", "sparsity-fractions", "sparsity-seeds", "mcdrop-btn", "dropout-p", "mc-samples"} & controls.keys()
    layout = by_id(ui.app.layout)
    assert "results-area" in by_id(layout["solution-panel"])
    assert {"sparsity-btn", "mcdrop-btn", "experiment-results-area"} <= by_id(layout["experiments-panel"]).keys()
    assert {tab.tab_id for tab in layout["workspace-tabs"].children} == {"solution", "experiments"}

    # Tab changes update visibility only; results and experiment settings stay
    # mounted as targets for in-flight background jobs and subsequent visits.
    callback_key = next(
        key for key, callback in ui.app.callback_map.items()
        if callback["inputs"] == [{"id": "workspace-tabs", "property": "active_tab"}]
    )
    client = ui.server.test_client()
    for active_tab in ("experiments", "solution", "experiments"):
        response = client.post("/_dash-update-component", json={
            "output": callback_key,
            "outputs": [{"id": "solution-panel", "property": "style"},
                        {"id": "experiments-panel", "property": "style"}],
            "inputs": [{"id": "workspace-tabs", "property": "active_tab", "value": active_tab}],
            "state": [],
            "changedPropIds": ["workspace-tabs.active_tab"],
        })
        assert response.status_code == 200
        panels = response.json["response"]
        assert panels == {
            "solution-panel": {"style": {"display": "block" if active_tab == "solution" else "none"}},
            "experiments-panel": {"style": {"display": "block" if active_tab == "experiments" else "none"}},
        }


@pytest.mark.parametrize("domain", PARAMETERS)
def test_dynamic_parameters_and_equation_remain_available(ui, domain):
    controls = by_id(ui.render_domain_params(domain))
    assert controls.keys() == PARAMETERS[domain].keys()
    layout_ids = by_id(ui.app.layout)
    for control_id, initial_value in PARAMETERS[domain].items():
        assert controls[control_id].value == initial_value
        assert layout_ids[f"{control_id}-store"].data == initial_value
        callbacks = [
            cb for cb in ui.app.callback_map.values()
            if cb["inputs"] == [{"id": control_id, "property": "value"}]
        ]
        assert len(callbacks) == 1
        assert str(callbacks[0]["output"]) == f"{control_id}-store.data"
    equation = text_content(ui.show_equation(domain))
    for field in ("label", "equation", "description"):
        assert ui.DOMAIN_INFO[domain][field] in equation


def test_training_and_experiments_are_registered_as_background_jobs(ui):
    training = next(cb for cb in ui.app.callback_map.values() if "train-btn" in str(cb["inputs"]))
    assert training["background"]["progress"][0].component_id == "train-progress"
    states = {state["id"] for state in training["state"]}
    assert {f"{control}-store" for params in PARAMETERS.values() for control in params} <= states
    experiments = next(cb for cb in ui.app.callback_map.values() if "sparsity-btn" in str(cb["inputs"]))
    assert experiments["background"] is not None
    specs = {cb["output"]: cb for cb in ui.app._callback_list}
    training_spec = next(cb for key, cb in specs.items() if "results-area.children" in key and "train-status" in key and any(i["id"] == "train-btn" for i in cb["inputs"]))
    assert training_spec["running"]["running"] == {"train-btn.disabled": True}
    assert training_spec["running"]["runningOff"] == {"train-btn.disabled": False}
    experiment_spec = next(cb for key, cb in specs.items() if key == "experiment-results-area.children")
    assert experiment_spec["running"]["running"] == {
        "sparsity-btn.disabled": True, "mcdrop-btn.disabled": True,
    }
    assert training_spec["prevent_initial_call"] and experiment_spec["prevent_initial_call"]


DOMAIN_EXPECTATIONS = {
    "fluids": (5, {"make_3d_surface_pair": 2, "make_adaptive_loss_fig": 1},
               ["R² depth h", "R² velocity u", "λ (self-calibrated)", "Training pts", "Dual-head PINN"]),
    "heat": (3, {"make_3d_surface_pair": 1, "make_loss_curve_fig": 1},
             ["R²", "Training pts", "α (diffusivity)"]),
    "wave": (3, {"make_3d_surface_pair": 1, "make_loss_curve_fig": 1},
             ["R²", "Training pts", "Wave speed c"]),
    "gravity": (2, {"make_orbit_3d": 1, "make_loss_curve_fig": 1},
                ["R² (x)", "R² (y)", "Training pts", "Eccentricity", "GM"]),
    "elasticity": (2, {"make_beam_3d": 1, "make_loss_curve_fig": 1},
                   ["R²", "Training pts", "Load type", "EI", "Beam L", "q0"]),
    "dam": (4, {"make_dam_reservoir_fig": 1, "make_dam_reach_surface": 1, "make_dam_loss_curve_fig": 1},
            ["Reservoir level R²", "Training pts", "Reservoir area", "Downstream reach"]),
    "inverse_fluids": (2, {"make_n_field_fig": 1, "make_loss_curve_fig": 1},
                       ["True n", "Inferred n", "Relative error", "Training pts", "roughness inference depends"]),
}


@pytest.mark.parametrize("domain", PARAMETERS)
def test_training_keeps_scientific_figures_metrics_progress_and_parameters(ui, monkeypatch, available_slot, domain):
    x = np.linspace(0, 2, 5)
    t = np.linspace(0, 4, 5)
    field = np.arange(25, dtype=float).reshape(5, 5) + 1
    data = dict(
        x=x, t=t, y=x**2, h=field, u=field / 3, T=field, U=field,
        v=x / 10, c=1.8, e=0.5, GM=2.0, load_type="point", EI=2.5, L=1.5, q0=3.0,
        t_res=t, Z=x + 3, Q_in=x + 10, Q_out=x + 9,
        x_reach=x, t_reach=t, h_reach=field,
        cfg=dict(name="Volta test", length=2000, reservoir_area=2e6),
    )
    extra = dict(
        h_pred=field, u_pred=field / 3, T_pred=field, U_pred=field,
        x_pred=x, y_pred=x**2, v_pred=x / 10, Z_pred=x + 3,
        n_field=np.full(200, 0.04), true_n=0.045, n_final=0.04, err_pct=11.1,
        r2=0.91, r2_u=0.88, r2x=0.90, r2y=0.89, r2_z=0.92, n_train=3, lambda_final=0.2,
    )
    generated, trained, called_helpers = [], [], []

    def generate(key, **kwargs):
        generated.append((key, kwargs))
        return data

    def train(key, training_data, **kwargs):
        trained.append((key, training_data, kwargs))
        kwargs["callback"](4, HISTORY)
        return object(), HISTORY, {}, extra

    monkeypatch.setattr(ui, "generate_domain_data", generate)
    monkeypatch.setattr(ui, "train_domain", train)
    for helper in {name for _, helpers, _ in DOMAIN_EXPECTATIONS.values() for name in helpers}:
        original = getattr(ui, helper)

        def record(*args, _name=helper, _original=original, **kwargs):
            called_helpers.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(ui, helper, record)

    progress = []
    content, status = ui.run_training(progress.append, domain_key=domain, **TRAINING_INPUTS)
    assert status != "Failed", text_content(content)
    assert progress == [f"Training {domain}: epoch 5 / 400"]
    assert len(trained) == 1 and trained[0][:2] == (domain, data)
    training_options = trained[0][2]
    assert training_options["fraction"] == 0.35
    assert training_options["n_epochs"] == 400
    assert training_options["lambda_pde"] == 0.42
    assert training_options["n_colloc"] == ui.DASH_COLLOCATION_POINTS
    assert generated == [(domain, {
        "fluids": dict(river_key="volta", n_manning=0.04, S0=0.001),
        "heat": dict(alpha=0.08),
        "wave": dict(c=1.8, mode=3),
        "gravity": dict(eccentricity=0.5, GM=2.0),
        "elasticity": dict(load_type="point", EI=2.5, q0=3.0, L=1.5),
        "dam": dict(gate_width=12.0, reservoir_area=2e6),
        "inverse_fluids": dict(river_key="volta", true_n=0.045),
    }[domain])]
    n_graphs, expected_helpers, labels = DOMAIN_EXPECTATIONS[domain]
    rendered_graphs = graphs(content)
    assert len(rendered_graphs) == n_graphs
    assert Counter(called_helpers) == expected_helpers
    for label in labels:
        assert label in text_content(content)
    assert all(graph.figure.data for graph in rendered_graphs)
    if domain == "gravity":
        assert {trace.name for trace in rendered_graphs[0].figure.data} == {
            "True orbit", "PINN orbit", "Training points (sparse)", "Central body",
        }
    elif domain == "elasticity":
        assert {trace.name for trace in rendered_graphs[0].figure.data} == {
            "True beam shape", "PINN prediction", "Sensor points",
        }
    elif domain == "fluids":
        assert [graph.figure.data[0].type for graph in rendered_graphs[:4]] == ["surface"] * 4
        assert "λ (adaptive)" in {trace.name for trace in rendered_graphs[-1].figure.data}
    elif domain == "dam":
        assert {trace.name for trace in rendered_graphs[1].figure.data} == {"Inflow Q_in", "Gate outflow Q_out"}
        assert {trace.name for trace in rendered_graphs[-1].figure.data} == {"Total", "Reservoir", "Reach data", "Reach physics"}
    assert available_slot == [("enter", ui.background_cache.directory), ("exit", ui.background_cache.directory)]


def test_training_content_is_retained_when_another_job_holds_the_slot(ui, monkeypatch):
    @contextmanager
    def unavailable_slot(_directory):
        yield False

    def should_not_train(*_args, **_kwargs):
        pytest.fail("A busy training slot must reject work before data generation")

    monkeypatch.setattr(ui, "training_slot", unavailable_slot)
    monkeypatch.setattr(ui, "generate_domain_data", should_not_train)
    content, status = ui.run_training(should_not_train, domain_key="heat", **TRAINING_INPUTS)
    assert content is ui.dash.no_update
    assert "Another training or experiment job is running" in status
    warning = ui.run_experiment(1, 0, "heat", [0.1], 1, 100, 0.08, 100, "volta", 0.035, 0.0008)
    assert "Another training or experiment job is running" in text_content(warning)


def test_training_error_is_visible_and_releases_the_slot(ui, monkeypatch, available_slot):
    def broken_data(*_args, **_kwargs):
        raise ValueError("test data error")

    monkeypatch.setattr(ui, "generate_domain_data", broken_data)
    content, status = ui.run_training(lambda _message: None, domain_key="heat", **TRAINING_INPUTS)
    assert status == "Failed"
    assert "Training error: test data error" in text_content(content)
    assert [event for event, _directory in available_slot] == ["enter", "exit"]


@pytest.mark.parametrize("domain", ["fluids", "heat", "wave", "gravity", "elasticity"])
def test_sparsity_sweep_retains_comparison_error_bars_and_results_table(ui, monkeypatch, available_slot, domain):
    if domain == "fluids":
        field = np.arange(25, dtype=float).reshape(5, 5)
        monkeypatch.setattr(ui, "generate_domain_data", lambda *_args, **_kwargs: {
            "X": field.ravel(), "T": field.T.ravel(), "H": field.ravel() + 2,
            "U": field.ravel() + 3, "cfg": {"name": "Volta", "S0": 0.0008, "n_manning": 0.035},
        })
    recorded = []

    def sweep(*_args, **kwargs):
        recorded.append(kwargs)
        return {
            "fractions": kwargs["fractions"], "pinn_r2": [0.5, 0.9], "mlp_r2": [0.6, 0.85],
            "pinn_r2_std": [0.02, 0.01], "mlp_r2_std": [0.03, 0.02],
            "pinn_rmse": [0.2, 0.1], "mlp_rmse": [0.4, 0.15],
        }

    monkeypatch.setattr(ui, "run_sparsity_sweep", sweep)
    token = context_value.set(AttributeDict(triggered_inputs=[{"prop_id": "sparsity-btn.n_clicks", "value": 1}]))
    try:
        content = ui.run_experiment(1, 0, domain, [0.5, 0.1], 3, 400, 0.08, 100, "volta", 0.04, 0.001)
    finally:
        context_value.reset(token)
    assert recorded[0]["fractions"] == [0.1, 0.5]
    assert recorded[0]["n_seeds"] == 3
    assert recorded[0]["n_epochs"] == 400
    r2_graph, rmse_graph = graphs(content)
    pinn, mlp = r2_graph.figure.data[:2]
    assert pinn.name == "PINN (physics-informed)"
    assert mlp.name == "Vanilla MLP (same capacity, no physics)"
    assert list(pinn.error_y.array) == [0.02, 0.01] and pinn.error_y.visible
    assert list(mlp.error_y.array) == [0.03, 0.02] and mlp.error_y.visible
    assert len(rmse_graph.figure.data) == 2
    assert "Results table" in text_content(content)
    assert "PINN R²" in text_content(content) and "MLP R²" in text_content(content)
    assert "Higher mean PINN R² at tested fractions: 50%" in text_content(content)
    assert "and below" not in text_content(content)
    assert "not held-out" in text_content(content)


@pytest.mark.parametrize("domain", ["heat", "gravity"])
def test_mc_dropout_retains_field_or_band_uncertainty_and_loss(ui, monkeypatch, available_slot, domain):
    training_options, prediction_samples = [], []

    class MCModel:
        def predict_with_uncertainty(self, coords, n_samples):
            prediction_samples.append(n_samples)
            return np.zeros((len(coords), 2)), np.full((len(coords), 2), 0.1)

    def train(_pde, coords, targets, **kwargs):
        training_options.append({**kwargs, "n_train": len(coords)})
        assert len(coords) == len(targets)
        return MCModel(), HISTORY

    monkeypatch.setattr(ui, "train_mc_dropout", train)
    token = context_value.set(AttributeDict(triggered_inputs=[{"prop_id": "mcdrop-btn.n_clicks", "value": 1}]))
    try:
        content = ui.run_experiment(0, 1, domain, [0.1], 2, 400, 0.12, 150, "volta", 0.035, 0.0008,
                                    fraction_pct=30, lambda_pde=0.42)
    finally:
        context_value.reset(token)
    assert training_options[0]["dropout_p"] == 0.12
    assert training_options[0]["n_epochs"] == 400
    assert training_options[0]["lambda_pde"] == 0.42
    assert training_options[0]["n_train"] == (750 if domain == "heat" else 90)
    assert prediction_samples == [150]
    uncertainty, distribution, loss = graphs(content)
    if domain == "heat":
        assert uncertainty.figure.data[0].type == "surface"
    else:
        assert {trace.name for trace in uncertainty.figure.data} == {"True", "MC mean", "±2σ dropout spread", "±1σ dropout spread"}
    assert distribution.figure.data[0].type == "histogram"
    assert np.allclose(distribution.figure.data[0].x, 0.4)
    assert {trace.name for trace in loss.figure.data} == {"Total", "Data", "Physics"}
    for label in ("MC mean R²", "Mean ±2σ width", "MC samples", "Dropout p", "150", "0.12"):
        assert label in text_content(content)
    assert "0.4000" in text_content(content)


def test_home_and_workspace_routes_preserve_the_mounted_dashboard(ui):
    ids = by_id(ui.app.layout)
    assert {"landing-page", "workspace-page", "page-location", "workspace-id", "run-recovery"} <= ids.keys()
    assert "results-area" in by_id(ids["workspace-page"])
    assert "experiment-results-area" in by_id(ids["workspace-page"])
    links = [part for part in walk(ids["landing-page"]) if isinstance(part, dcc.Link)]
    assert links and all(link.href == "/workspace" and link.refresh is False for link in links)
    assert ui.show_page("/") == ({"display": "block"}, {"display": "none"}, True)
    assert ui.show_page("/workspace") == ({"display": "none"}, {"display": "block"}, False)


def test_workspace_controls_restore_values_including_zero_eccentricity(ui):
    ids = by_id(ui.app.layout)
    assert ids["workspace-id"].storage_type == "session"
    assert ids["workspace-tabs"].persistence_type == "session"
    for param in ui.PARAM_DEFAULTS:
        assert ids[f"{param}-store"].storage_type == "session"
    for control in ("domain-select", "data-fraction", "epochs", "lambda-pde",
                    "sparsity-fractions", "sparsity-seeds", "dropout-p", "mc-samples"):
        assert ids[control].persistence and ids[control].persistence_type == "session"
    values = {**ui.PARAM_DEFAULTS, "param-e": 0.0, "param-GM": 3.0}
    controls = by_id(ui.render_domain_params("gravity", *values.values()))
    assert controls["param-e"].value == 0.0
    assert controls["param-GM"].value == 3.0


def test_recovery_restores_outputs_and_avoids_republishing_unchanged_figures(ui, monkeypatch):
    saved_result = {"type": "Div", "namespace": "dash_html_components", "props": {"children": "saved heat solution"}}
    record = dict(id="run-one", updated=1.5, state="completed", domain_key="heat",
                  inputs={"epochs": 200, "fraction_pct": 25}, result=[saved_result, "Heat trained"])
    monkeypatch.setattr(ui, "get_run", lambda _cache, kind, _session: record if kind == "training" else None)
    restored = ui.recover_runs(1, "session", None)
    assert restored[:2] == (saved_result, "Heat trained")
    assert restored[5:8] == (False, False, False)
    assert "Heat diffusion" in restored[9] and "200 epochs" in restored[9]
    repeated = ui.recover_runs(2, "session", restored[8])
    assert all(value is ui.dash.no_update for value in (*repeated[:5], repeated[8], repeated[9]))


def test_recovery_keeps_an_active_run_guarded_and_reports_interruption(ui, monkeypatch):
    record = dict(id="run-two", updated=2.5, state="running", domain_key="wave",
                  inputs={}, result=["old result", "old status"], progress="Training wave: epoch 10 / 200")
    monkeypatch.setattr(ui, "get_run", lambda _cache, kind, _session: record if kind == "training" else None)
    restored = ui.recover_runs(1, "session", None)
    assert restored[0] is ui.dash.no_update
    assert restored[2] == record["progress"]
    assert restored[5:8] == (True, True, True)
    record.update(state="interrupted", result=None)
    interrupted = ui.recover_runs(2, "session", restored[8])
    assert "interrupted" in interrupted[1].lower()
    assert interrupted[5:8] == (False, False, False)
    record.update(updated=3.5, result=["old heat result", "Heat trained"],
                  result_domain_key="heat", result_inputs={"epochs": 100, "fraction_pct": 50})
    retained = ui.recover_runs(3, "session", interrupted[8])
    assert retained[0] == "old heat result"
    assert "Heat diffusion" in retained[9] and "100 epochs" in retained[9]
    assert "Wave propagation" not in retained[9]


@pytest.mark.parametrize("domain, settings, expected", [
    ("heat", {"alpha_val": 0.08}, {"alpha": 0.08}),
    ("wave", {"c_val": 1.8, "mode_val": 3}, {"c": 1.8, "mode": 3}),
    ("gravity", {"e_val": 0.0, "GM_val": 2.0}, {"eccentricity": 0.0, "GM": 2.0}),
    ("elasticity", {"load_val": "point", "EI_val": 2.5, "L_val": 2.0, "q0_val": 3.0},
     {"load_type": "point", "EI": 2.5, "L": 2.0, "q0": 3.0}),
])
def test_experiments_use_the_selected_domain_parameters(ui, monkeypatch, available_slot, domain, settings, expected):
    received = []

    def capture(key, **params):
        received.append((key, params))
        raise ValueError("stop after checking generation parameters")

    monkeypatch.setattr(ui, "generate_domain_data", capture)
    token = context_value.set(AttributeDict(triggered_inputs=[{"prop_id": "sparsity-btn.n_clicks", "value": 1}]))
    try:
        ui.run_experiment(1, 0, domain, [0.1], 1, 100, 0.08, 100, "volta", 0.035, 0.0008, **settings)
    finally:
        context_value.reset(token)
    assert received == [(domain, expected)]


def test_dash_layout_dependencies_and_health_are_served(ui):
    client = ui.server.test_client()
    assert client.get("/healthz").data == b"ok"
    assert client.get("/_dash-layout").status_code == 200
    dependencies = client.get("/_dash-dependencies")
    assert dependencies.status_code == 200
    assert any("train-status.children" in callback["output"] for callback in dependencies.json)
    assert any(callback["output"] == "experiment-results-area.children" for callback in dependencies.json)


def test_documented_launch_registers_experiments_and_parameter_sync_before_starting(ui, monkeypatch):
    launched_callbacks = []

    def capture_run(app, **_kwargs):
        launched_callbacks.extend(app.callback_map)

    monkeypatch.setattr(ui.dash.Dash, "run", capture_run)
    runpy.run_path(ui.__file__, run_name="__main__")
    assert "experiment-results-area.children" in launched_callbacks
    assert {
        f"{control}-store.data" for params in PARAMETERS.values() for control in params
    } <= set(launched_callbacks)
