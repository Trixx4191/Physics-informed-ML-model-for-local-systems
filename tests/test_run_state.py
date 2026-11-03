"""Recover scientific Dash outputs without weakening the existing job guard."""

import inspect
import json
import multiprocessing
import os
from uuid import uuid4

from dash import dcc, html, no_update
import dash_bootstrap_components as dbc
import diskcache
import numpy as np
import plotly.graph_objects as go
import psutil
import pytest

from dash_app.run_state import RUN_EXPIRY_SECONDS, get_run, remember_run


@pytest.fixture
def cache(tmp_path):
    with diskcache.Cache(str(tmp_path / "runs")) as run_cache:
        yield run_cache


def test_without_workspace_keeps_original_call_and_return_identity(cache):
    expected = (html.Div("unchanged"), "Ready")

    @remember_run(cache, "training")
    def train(domain_key):
        """A callback docstring."""
        assert domain_key == "heat"
        return expected

    assert train("heat") is expected
    assert train("heat", workspace_id="malformed") is expected
    assert list(cache) == []
    assert train.__name__ == "train"
    assert train.__doc__ == "A callback docstring."
    assert inspect.signature(train).parameters["workspace_id"].default is None
    assert get_run(cache, "training", None) is None


def test_completed_output_restores_dash_components_figures_arrays_and_settings(cache):
    workspace_id = str(uuid4())
    figure = go.Figure(go.Scatter(x=np.arange(3), y=np.array([1.0, 2.0, 3.0])))
    expected = (html.Div([html.H3("Scientific result"), dcc.Graph(figure=figure)]), "Trained heat")

    @remember_run(cache, "training")
    def train(domain_key, epochs, fractions, workspace_id=None):
        return expected

    assert train("heat", 100, [0.1, 0.5], workspace_id) is expected
    record = get_run(cache, "training", workspace_id)
    assert record["state"] == "completed"
    assert record["domain_key"] == "heat"
    assert record["inputs"] == {"domain_key": "heat", "epochs": 100, "fractions": [0.1, 0.5]}
    assert record["result_domain_key"] == "heat"
    assert record["result_inputs"] == record["inputs"]
    assert record["progress"] == "Trained heat"
    assert record["result"][0]["type"] == "Div"
    assert record["result"][0]["props"]["children"][1]["type"] == "Graph"
    assert record["result"][1] == "Trained heat"
    assert json.loads(json.dumps(record)) == record
    assert record["id"] == record["nonce"]
    assert record["updated"] == record["updated_at"]
    _, expires = cache.get(next(iter(cache)), expire_time=True)
    assert abs(expires - record["updated"] - RUN_EXPIRY_SECONDS) < 2


def test_progress_forwards_every_update_and_flushes_last_throttled_update(cache):
    workspace_id = str(uuid4())
    received = []

    @remember_run(cache, "training", progress=True)
    def train(set_progress, domain_key, workspace_id=None):
        set_progress("epoch 1")
        record = get_run(cache, "training", workspace_id)
        assert record["state"] == "running"
        assert record["progress"] == "epoch 1"
        set_progress("epoch 2")
        set_progress("epoch 3")
        return html.Div("done")

    train(received.append, "wave", workspace_id)
    assert received == ["epoch 1", "epoch 2", "epoch 3"]
    assert get_run(cache, "training", workspace_id)["progress"] == "done"
    assert "set_progress" not in get_run(cache, "training", workspace_id)["inputs"]


def test_latest_progress_survives_without_a_string_status(cache):
    workspace_id = str(uuid4())

    @remember_run(cache, "training", progress=True)
    def train(set_progress, workspace_id=None):
        set_progress("epoch 1")
        set_progress("epoch 8")
        return [html.Div("result")]

    train(lambda _: None, workspace_id)
    assert get_run(cache, "training", workspace_id)["progress"] == "epoch 8"


@pytest.mark.parametrize("result", [
    (dbc.Alert("Training error", color="danger"), "Failed"),
    dbc.Alert("Sweep error", color="danger"),
])
def test_existing_caught_failure_alerts_are_marked_failed(cache, result):
    workspace_id = str(uuid4())

    @remember_run(cache, "training")
    def train():
        return result

    assert train(workspace_id=workspace_id) is result
    assert get_run(cache, "training", workspace_id)["state"] == "failed"


def test_worker_exception_is_recorded_and_propagated(cache):
    workspace_id = str(uuid4())

    @remember_run(cache, "training")
    def train():
        raise RuntimeError("numerical failure")

    with pytest.raises(RuntimeError, match="numerical failure"):
        train(workspace_id=workspace_id)
    record = get_run(cache, "training", workspace_id)
    assert record["state"] == "failed"
    assert record["progress"] == "Failed: numerical failure"


def test_live_same_workspace_record_cannot_be_clobbered_by_competing_invocation(cache):
    workspace_id = str(uuid4())

    @remember_run(cache, "training")
    def contender():
        return no_update, "Another training or experiment job is running."

    @remember_run(cache, "training")
    def admitted():
        initial = get_run(cache, "training", workspace_id)
        assert contender(workspace_id=workspace_id)[0] is no_update
        assert get_run(cache, "training", workspace_id) == initial
        return html.Div("admitted result"), "Trained"

    admitted(workspace_id=workspace_id)
    assert get_run(cache, "training", workspace_id)["result"][1] == "Trained"


@pytest.mark.parametrize("busy_result", [
    (no_update, "Another training or experiment job is running."),
    dbc.Alert("Another training or experiment job is running. Please try again when it finishes.", color="info"),
])
def test_busy_outputs_preserve_prior_result_and_its_domain_settings(cache, busy_result):
    workspace_id = str(uuid4())

    @remember_run(cache, "training")
    def train(domain_key, epochs):
        return html.Div("completed heat result"), "Trained heat"

    @remember_run(cache, "training")
    def busy(domain_key, epochs):
        return busy_result

    train("heat", 200, workspace_id)
    previous = get_run(cache, "training", workspace_id)
    assert busy("wave", 500, workspace_id) is busy_result
    retained = get_run(cache, "training", workspace_id)
    assert retained["state"] == "completed"
    assert retained["result"] == previous["result"]
    assert retained["domain_key"] == "heat"
    assert retained["inputs"] == {"domain_key": "heat", "epochs": 200}
    assert retained["result_domain_key"] == "heat"
    assert retained["result_inputs"] == {"domain_key": "heat", "epochs": 200}


def test_interrupted_new_domain_keeps_the_previous_result_domain_and_settings(cache):
    workspace_id = str(uuid4())

    @remember_run(cache, "training")
    def train(domain_key, epochs):
        if domain_key == "wave":
            running = get_run(cache, "training", workspace_id)
            assert running["domain_key"] == "wave"
            assert running["inputs"]["epochs"] == 500
            assert running["result_domain_key"] == "heat"
            assert running["result_inputs"] == {"domain_key": "heat", "epochs": 200}
            raise RuntimeError("new worker interrupted")
        return html.Div("heat result"), "Trained heat"

    train("heat", 200, workspace_id)
    with pytest.raises(RuntimeError, match="new worker interrupted"):
        train("wave", 500, workspace_id)
    key = next(iter(cache))
    record = cache[key]
    record.update(state="running", create_time=psutil.Process().create_time() - 5)
    cache[key] = record
    interrupted = get_run(cache, "training", workspace_id)
    assert interrupted["state"] == "interrupted"
    assert interrupted["domain_key"] == "wave"
    assert interrupted["result_domain_key"] == "heat"
    assert interrupted["result_inputs"] == {"domain_key": "heat", "epochs": 200}
    assert interrupted["result"][0]["props"]["children"] == "heat result"


def test_records_are_isolated_by_session_and_job_kind(cache):
    first, second = str(uuid4()), str(uuid4())

    @remember_run(cache, "training")
    def train(domain_key):
        return html.Div(domain_key), f"Trained {domain_key}"

    @remember_run(cache, "experiment")
    def experiment():
        return html.Div("Sweep")

    train("heat", first)
    train("wave", second)
    experiment(workspace_id=first)
    assert get_run(cache, "training", first)["domain_key"] == "heat"
    assert get_run(cache, "training", second)["domain_key"] == "wave"
    assert get_run(cache, "experiment", first)["result"]["props"]["children"] == "Sweep"
    assert get_run(cache, "experiment", second) is None
    assert len(cache) == 3


def _hold_run(directory, workspace_id, ready, release):
    with diskcache.Cache(directory) as cache:
        @remember_run(cache, "training")
        def train():
            ready.set()
            release.wait(timeout=20)
            return html.Div("finished")

        train(workspace_id=workspace_id)


def test_actual_worker_liveness_and_process_death_recovery(cache):
    workspace_id = str(uuid4())
    context = multiprocessing.get_context("spawn")
    ready, release = context.Event(), context.Event()
    worker = context.Process(target=_hold_run, args=(cache.directory, workspace_id, ready, release))
    worker.start()
    try:
        assert ready.wait(timeout=15)
        record = get_run(cache, "training", workspace_id)
        assert record["state"] == "running"
        assert record["pid"] == worker.pid
        worker.kill()
        worker.join(timeout=10)
        record = get_run(cache, "training", workspace_id)
        assert record["state"] == "interrupted"
        assert "interrupted" in record["progress"]
    finally:
        if worker.is_alive():
            worker.terminate()
        worker.join(timeout=10)


def test_reused_pid_or_stale_record_does_not_look_live(cache):
    workspace_id = str(uuid4())

    @remember_run(cache, "training")
    def train():
        return html.Div("recovered")

    train(workspace_id=workspace_id)
    key = next(iter(cache))
    record = cache[key]
    record.update(state="running", pid=os.getpid(), create_time=psutil.Process().create_time() - 5)
    cache[key] = record
    assert get_run(cache, "training", workspace_id)["state"] == "interrupted"
    train(workspace_id=workspace_id)
    assert get_run(cache, "training", workspace_id)["state"] == "completed"


def test_successive_runs_reuse_single_expiring_session_record(cache):
    workspace_id = str(uuid4())

    @remember_run(cache, "training")
    def train(epochs):
        return html.Div(str(epochs)), "Trained"

    train(100, workspace_id)
    previous = get_run(cache, "training", workspace_id)
    train(200, workspace_id)
    latest = get_run(cache, "training", workspace_id)
    assert len(cache) == 1
    assert latest["id"] != previous["id"]
    assert latest["result"][0]["props"]["children"] == "200"
    assert latest["inputs"] == {"epochs": 200}
