"""
PINN Multi-Physics Explorer — Dash app.

Replaces the Streamlit app. Each control change only re-runs its own
callback (Dash's partial-update model), which matters once every domain
renders a 3D Plotly scene — Streamlit's full-script rerun would re-draw
all of them on every slider move.

Launch: python dash_app/app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import dash
import diskcache
from dash import dcc, html, Input, Output, State, DiskcacheManager
import dash_bootstrap_components as dbc
import numpy as np

from dash_app.domain_utils import (
    DOMAIN_INFO, generate_domain_data, train_domain, prepare_domain_training_data,
    make_3d_surface_pair, make_orbit_3d, make_beam_3d, make_loss_curve_fig,
    make_dam_reservoir_fig, make_dam_reach_surface, make_n_field_fig,
    make_dam_loss_curve_fig, make_adaptive_loss_fig
)
from experiments.sparsity_and_uncertainty import (
    MCDropoutEngine, train_mc_dropout, run_sparsity_sweep
)
from data.generator import RIVERS, generate_sparse_observations
from dash_app.job_guard import training_slot
from dash_app.run_state import get_run, remember_run
from dash_app.landing import build_landing_page

background_cache = diskcache.Cache(
    os.environ.get("DASH_CACHE_DIR", "/tmp/pinn-dash-cache")
)
DASH_COLLOCATION_POINTS = 512
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
    background_callback_manager=DiskcacheManager(background_cache),
)
app.title = "PINN Multi-Physics Explorer"
server = app.server  # exposed for gunicorn / Hugging Face Spaces

@server.route("/healthz")
def healthz():
    return "ok"

# ── Layout ────────────────────────────────────────────────────────────────────

PARAM_DEFAULTS = {
    "param-river": "volta", "param-alpha": 0.05, "param-c": 1.0,
    "param-mode": 1, "param-e": 0.3, "param-load": "uniform",
    "param-n": 0.035, "param-s0": 0.0008, "param-GM": 1.0,
    "param-EI": 1.0, "param-L": 1.0, "param-q0": 1.0,
    "param-gate-width": 10.0, "param-res-area": 1000000.0,
    "param-true-n": 0.035,
}


def control_field(label, component, *, wide=False):
    """A compact labelled control, shared by the toolbar and experiments."""
    if isinstance(component, dcc.Slider):
        component.tooltip = {"placement": "bottom", "always_visible": False}
    component.persistence = True
    component.persistence_type = "session"
    return html.Div([
        html.Label(label, htmlFor=component.id, className="control-label"),
        component,
    ], className="control-field" + (" control-field-wide" if wide else ""))


control_bar = html.Div([
    html.Div([
        control_field("Domain", dcc.Dropdown(
            id="domain-select",
            options=[{"label": v["label"], "value": k} for k, v in DOMAIN_INFO.items()],
            value="fluids", clearable=False,
        ), wide=True),
        html.Div(id="domain-params", className="domain-params"),
        html.Div([
            control_field("Data fraction (%)", dcc.Slider(
                id="data-fraction", min=5, max=100, step=5, value=15,
                marks={5: "5%", 50: "50%", 100: "100%"})),
            control_field("Epochs", dcc.Slider(
                id="epochs", min=100, max=1000, step=100, value=200,
                marks={100: "100", 500: "500", 1000: "1k"})),
            control_field("Physics weight λ₂", dcc.Slider(
                id="lambda-pde", min=0.01, max=1.0, step=0.01, value=0.1,
                marks={0.01: "0.01", 0.5: "0.5", 1.0: "1.0"})),
        ], className="training-params"),
    ], className="control-inputs", tabIndex=0,
       **{"aria-label": "Domain and training controls; scroll horizontally on smaller screens"}),
    html.Div([
        dbc.Button("Train PINN", id="train-btn", color="primary", n_clicks=0),
        html.Div([
            html.Div(id="train-status"),
            html.Div(id="train-progress"),
        ], className="training-status", role="status", **{"aria-live": "polite"}),
    ], className="training-actions"),
], className="control-bar", role="region", **{"aria-label": "Training controls"})

experiment_controls = html.Div([
    html.Section([
        html.H2("Sparsity sweep", className="section-title"),
        html.P("Compare PINN and vanilla MLP accuracy across observation fractions.",
               className="section-description"),
        html.Div([
            control_field("Fractions to test", dcc.Checklist(
                id="sparsity-fractions",
                options=[{"label": f" {p}%", "value": p/100}
                         for p in [5, 10, 20, 50, 100]],
                value=[0.05, 0.10, 0.20, 0.50, 1.00], inline=True,
                className="fraction-checklist")),
            control_field("Seeds per fraction", dcc.Slider(
                id="sparsity-seeds", min=1, max=5, step=1, value=2,
                marks={i: str(i) for i in range(1, 6)})),
            dbc.Button("Sparsity sweep", id="sparsity-btn", color="secondary",
                       outline=True, n_clicks=0),
        ], id="sparsity-controls", className="experiment-fields"),
    ], className="experiment-section"),
    html.Section([
        html.H2("MC Dropout uncertainty", className="section-title"),
        html.P("Estimate epistemic uncertainty with repeated stochastic predictions.",
               className="section-description"),
        html.Div([
            control_field("Dropout p", dcc.Slider(
                id="dropout-p", min=0.02, max=0.20, step=0.02, value=0.08,
                marks={0.02: "0.02", 0.10: "0.10", 0.20: "0.20"})),
            control_field("MC samples", dcc.Slider(
                id="mc-samples", min=50, max=300, step=50, value=100,
                marks={50: "50", 150: "150", 300: "300"})),
            dbc.Button("MC Dropout uncertainty", id="mcdrop-btn", color="secondary",
                       outline=True, n_clicks=0),
        ], id="mcdrop-controls", className="experiment-fields"),
    ], className="experiment-section"),
], className="experiment-controls")

workspace_layout = html.Div([
    html.Header([
        html.H1("PINN Multi-Physics Explorer"),
        html.Span("Scientific workspace · 7 domains"),
        dcc.Link("← Home", href="/", refresh=False, className="workspace-home"),
    ], className="workspace-header"),
    control_bar,
    html.Div(id="equation-display", className="equation-strip"),
    html.Main([
        dbc.Tabs([
            dbc.Tab(label="Solution", tab_id="solution"),
            dbc.Tab(label="Experiments", tab_id="experiments"),
        ], id="workspace-tabs", active_tab="solution", className="workspace-tabs",
           persistence=True, persistence_type="session"),
        # Keep both panels mounted: switching tabs must preserve controls,
        # figures, and the targets of in-flight background callbacks.
        html.Section([
            html.Div(id="run-context", className="run-context"),
            # Progress and disabled run buttons show job activity. A Loading
            # wrapper here would hide saved figures on every recovery poll.
            html.Div(id="loading-train", children=html.Div(
                id="results-area", children=html.Div([
                    html.H2("Solution workspace", className="section-title"),
                    html.P("Choose a domain and training parameters, then train a PINN "
                           "to compare its solution with the reference data."),
                ], className="empty-state"))),
        ], id="solution-panel"),
        html.Section([
            experiment_controls,
            html.Div(id="experiment-status", className="experiment-status", role="status",
                     **{"aria-live": "polite"}),
            html.Div(id="loading-experiments",
                        children=html.Div(id="experiment-results-area")),
        ], id="experiments-panel", style={"display": "none"}),
    ], className="workspace-main"),
    # Dynamic controls write into permanent stores so background callbacks
    # can safely read their values even when another domain is selected.
    *[dcc.Store(id=f"{param}-store", data=value, storage_type="session")
      for param, value in PARAM_DEFAULTS.items()],
    dcc.Store(id="trained-results-store"),
], id="workspace-page", className="laboratory-workspace", style={"display": "none"})

app.layout = html.Div([
    dcc.Location(id="page-location", refresh=False),
    build_landing_page(),
    workspace_layout,
    dcc.Store(id="workspace-id", storage_type="session"),
    dcc.Interval(id="run-recovery", interval=1000, disabled=True),
])


app.clientside_callback(
    """function(_, current) {
        if (current) return window.dash_clientside.no_update;
        if (window.crypto.randomUUID) return window.crypto.randomUUID();
        return '10000000-1000-4000-8000-100000000000'.replace(/[018]/g, c =>
            (c ^ window.crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> c / 4).toString(16));
    }""",
    Output("workspace-id", "data"), Input("page-location", "pathname"),
    State("workspace-id", "data"),
)


@app.callback(
    Output("landing-page", "style"), Output("workspace-page", "style"),
    Output("run-recovery", "disabled"), Input("page-location", "pathname"),
)
def show_page(pathname):
    if (pathname or "/").rstrip("/") == "/workspace":
        return {"display": "none"}, {"display": "block"}, False
    return {"display": "block"}, {"display": "none"}, True


@app.callback(
    Output("results-area", "children", allow_duplicate=True),
    Output("train-status", "children", allow_duplicate=True),
    Output("train-progress", "children", allow_duplicate=True),
    Output("experiment-results-area", "children", allow_duplicate=True),
    Output("experiment-status", "children"),
    Output("train-btn", "disabled", allow_duplicate=True),
    Output("sparsity-btn", "disabled", allow_duplicate=True),
    Output("mcdrop-btn", "disabled", allow_duplicate=True),
    Output("trained-results-store", "data"),
    Output("run-context", "children"),
    Input("run-recovery", "n_intervals"), Input("workspace-id", "data"),
    State("trained-results-store", "data"),
    prevent_initial_call=True,
)
def recover_runs(_ticks, workspace_id, revisions):
    """Reattach to this browser session's job records after navigation/reload."""
    records = {kind: get_run(background_cache, kind, workspace_id)
               for kind in ("training", "experiment")}
    revisions = dict(revisions or {})
    results = status = progress = experiments = experiment_status = context = dash.no_update
    changed = False
    for kind, record in records.items():
        if not record:
            continue
        revision = [record["id"], record["updated"], record["state"]]
        if revisions.get(kind) == revision:
            continue
        changed = True
        revisions[kind] = revision
        state = record["state"]
        label = DOMAIN_INFO.get(record.get("domain_key"), {}).get("label", "Run")
        if kind == "training":
            if state == "running":
                status = f"Training · {label}"
                progress = record.get("progress", "Starting run…")
            elif record.get("result") is not None:
                results, status = record["result"]
                progress = "" if state == "completed" else record.get("progress", "")
                result_label = DOMAIN_INFO.get(record.get("result_domain_key", record.get("domain_key")), {}).get("label", label)
                settings = record.get("result_inputs", record.get("inputs", {}))
                context = (f"Last solution · {result_label} · {settings.get('epochs', '—')} epochs · "
                           f"{settings.get('fraction_pct', '—')}% observations")
            else:
                status = record.get("progress", "Run interrupted. Start a new run.")
                progress = ""
            if state == "interrupted":
                status = "Run interrupted. Start a new run to continue."
        else:
            if state != "running" and record.get("result") is not None:
                experiments = record["result"]
            verb = {"running": "Running experiment", "completed": "Experiment complete",
                    "failed": "Experiment failed", "interrupted": "Experiment interrupted"}[state]
            experiment_status = f"{verb} · {label}"
    busy = any(record and record["state"] == "running" for record in records.values())
    return (results, status, progress, experiments, experiment_status,
            busy, busy, busy, revisions if changed else dash.no_update, context)


@app.callback(
    Output("solution-panel", "style"), Output("experiments-panel", "style"),
    Input("workspace-tabs", "active_tab"),
)
def switch_workspace(active_tab):
    if active_tab == "experiments":
        return {"display": "none"}, {"display": "block"}
    return {"display": "block"}, {"display": "none"}


# ── Dynamic per-domain parameter controls ───────────────────────────────────────

@app.callback(
    Output("domain-params", "children"), Input("domain-select", "value"),
    *[State(f"{param}-store", "data") for param in PARAM_DEFAULTS],
)
def render_domain_params(domain_key, *saved_values):
    controls = domain_controls(domain_key)
    values = dict(zip(PARAM_DEFAULTS, saved_values))
    for field in controls.children or []:
        component = field.children[1]
        value = values.get(component.id)
        if value is not None:
            component.value = value
    return controls


def domain_controls(domain_key):
    if domain_key == "fluids":
        river_options = [{"label": v["name"], "value": k}
                         for k, v in RIVERS.items() if k != "dam"]
        return html.Div([
            control_field("River", dcc.Dropdown(id="param-river", options=river_options,
                          value="volta", clearable=False), wide=True),
            control_field("Manning n", dcc.Slider(
                id="param-n", min=0.01, max=0.06, step=0.005, value=0.035,
                marks={0.01: "0.01", 0.03: "0.03", 0.06: "0.06"})),
            control_field("Bed slope S0", dcc.Slider(
                id="param-s0", min=0.0001, max=0.002, step=0.0001, value=0.0008,
                marks={0.0001: "1e-4", 0.0008: "8e-4", 0.002: "2e-3"})),
        ], className="domain-fields")
    if domain_key == "heat":
        return html.Div([
            control_field("Thermal diffusivity α", dcc.Slider(
                id="param-alpha", min=0.005, max=0.15, step=0.005, value=0.05,
                marks={0.005: "0.005", 0.05: "0.05", 0.15: "0.15"})),
        ], className="domain-fields")
    if domain_key == "wave":
        return html.Div([
            control_field("Wave speed c", dcc.Slider(
                id="param-c", min=0.5, max=3.0, step=0.1, value=1.0,
                marks={0.5: "0.5", 1.5: "1.5", 3.0: "3.0"})),
            control_field("Vibration mode", dcc.Slider(
                id="param-mode", min=1, max=4, step=1, value=1,
                marks={i: str(i) for i in range(1, 5)})),
        ], className="domain-fields")
    if domain_key == "gravity":
        return html.Div([
            control_field("Orbital eccentricity", dcc.Slider(
                id="param-e", min=0.0, max=0.8, step=0.05, value=0.3,
                marks={0: "circular", 0.4: "0.4", 0.8: "0.8"})),
            control_field("Central mass GM", dcc.Slider(
                id="param-GM", min=0.1, max=5.0, step=0.1, value=1.0,
                marks={0.1: "0.1", 1.0: "1.0", 5.0: "5.0"})),
        ], className="domain-fields")
    if domain_key == "elasticity":
        return html.Div([
            control_field("Load type", dcc.Dropdown(id="param-load", options=[
                {"label": "Uniform load", "value": "uniform"},
                {"label": "Point load (midspan)", "value": "point"},
            ], value="uniform", clearable=False), wide=True),
            control_field("Flexural rigidity EI", dcc.Slider(
                id="param-EI", min=0.1, max=5.0, step=0.1, value=1.0,
                marks={0.1: "0.1", 1.0: "1.0", 5.0: "5.0"})),
            control_field("Beam length L", dcc.Slider(
                id="param-L", min=0.5, max=3.0, step=0.1, value=1.0,
                marks={0.5: "0.5", 1.0: "1.0", 3.0: "3.0"})),
            control_field("Load amplitude q0", dcc.Slider(
                id="param-q0", min=0.2, max=5.0, step=0.2, value=1.0,
                marks={0.2: "0.2", 1.0: "1.0", 5.0: "5.0"})),
        ], className="domain-fields")
    if domain_key == "dam":
        return html.Div([
            control_field("Gate width", dcc.Slider(
                id="param-gate-width", min=2.0, max=30.0, step=1.0, value=10.0,
                marks={2: "2", 10: "10", 30: "30"})),
            control_field("Reservoir area", dcc.Slider(
                id="param-res-area", min=1e5, max=5e6, step=1e5, value=1e6,
                marks={1e5: "1e5", 1e6: "1e6", 5e6: "5e6"})),
        ], className="domain-fields")
    if domain_key == "inverse_fluids":
        river_options = [{"label": v["name"], "value": k}
                         for k, v in RIVERS.items() if k != "dam"]
        return html.Div([
            control_field("River", dcc.Dropdown(id="param-river", options=river_options,
                          value="volta", clearable=False), wide=True),
            control_field("True Manning n", dcc.Slider(
                id="param-true-n", min=0.01, max=0.06, step=0.005, value=0.035,
                marks={0.01: "0.01", 0.03: "0.03", 0.06: "0.06"})),
        ], className="domain-fields")
    return html.Div()


@app.callback(Output("equation-display", "children"), Input("domain-select", "value"))
def show_equation(domain_key):
    info = DOMAIN_INFO[domain_key]
    notes = {
        "dam": "Adjust gate geometry and reservoir storage for better operational realism.",
        "inverse_fluids": "River (sets true n, hidden from the model). "
                          "The PINN infers n(x) from sparse observations and the physics residual.",
        "elasticity": "Point-load physics uses a smooth approximation of the load at midspan.",
    }
    return html.Div([
        html.Div([
            html.Strong(info["label"]),
            html.Code(info["equation"]),
        ], className="equation-line"),
        html.P(info.get("description", ""), className="equation-description"),
        *([html.P(notes[domain_key], className="equation-note")] if domain_key in notes else []),
    ])


# ── Main training callback ──────────────────────────────────────────────────────

@app.callback(
    Output("results-area", "children"),
    Output("train-status", "children"),
    Input("train-btn", "n_clicks"),
    State("domain-select", "value"),
    State("data-fraction", "value"),
    State("epochs", "value"),
    State("lambda-pde", "value"),
    State("param-river-store", "data"),
    State("param-alpha-store", "data"),
    State("param-c-store", "data"),
    State("param-mode-store", "data"),
    State("param-e-store", "data"),
    State("param-load-store", "data"),
    State("param-n-store", "data"),
    State("param-s0-store", "data"),
    State("param-GM-store", "data"),
    State("param-EI-store", "data"),
    State("param-L-store", "data"),
    State("param-q0-store", "data"),
    State("param-gate-width-store", "data"),
    State("param-res-area-store", "data"),
    State("param-true-n-store", "data"),
    State("workspace-id", "data"),
    prevent_initial_call=True,
    background=True,
    progress=Output("train-progress", "children"),
    progress_default="",
    running=[(Output("train-btn", "disabled"), True, False)],
)
@remember_run(background_cache, "training", progress=True)
def run_training(set_progress, n_clicks, domain_key, fraction_pct, epochs, lambda_pde,
                 river_key, alpha_val, c_val, mode_val, e_val, load_val,
                 n_val, s0_val, GM_val, EI_val, L_val, q0_val,
                 gate_width_val, res_area_val, true_n_val, workspace_id=None):
    with training_slot(background_cache.directory) as acquired:
        if not acquired:
            return dash.no_update, "Another training or experiment job is running. Please try again when it finishes."
        fraction = fraction_pct / 100.0

        def report_progress(epoch, _history):
            set_progress(f"Training {domain_key}: epoch {epoch + 1} / {epochs}")

        try:
            if domain_key == "fluids":
                data = generate_domain_data(
                    "fluids", river_key=river_key or "volta",
                    n_manning=n_val, S0=s0_val)
                model, hist, stats, extra = train_domain(
                    "fluids", data, fraction=fraction, n_epochs=epochs, lambda_pde=lambda_pde,
                    n_colloc=DASH_COLLOCATION_POINTS, callback=report_progress)

                fig_true, fig_pred = make_3d_surface_pair(
                    data["x"]/1000, data["t"]/3600, data["h"], extra["h_pred"],
                    x_label="Distance (km)", t_label="Time (hr)", z_label="Depth h (m)",
                    title_true="Ground truth h(x,t)", title_pred="PINN prediction")

                # Also show u prediction surface
                fig_u_true, fig_u_pred = make_3d_surface_pair(
                    data["x"]/1000, data["t"]/3600, data["u"], extra["u_pred"],
                    x_label="Distance (km)", t_label="Time (hr)", z_label="Velocity u (m/s)",
                    title_true="Ground truth u(x,t)", title_pred="PINN prediction")

                lam_final = extra.get("lambda_final", 0.0)
                r2_u = extra.get("r2_u", 0.0)

                content = html.Div([
                    metric_row([
                        ("R² depth h", f"{extra['r2']:.4f}"),
                        ("R² velocity u", f"{r2_u:.4f}"),
                        ("λ (self-calibrated)", f"{lam_final:.4f}"),
                        ("Training pts", extra["n_train"]),
                    ]),
                    html.Div([
                        html.H6("Water depth h(x,t)", className="figure-heading"),
                        html.Div([
                            dcc.Graph(figure=fig_true),
                            dcc.Graph(figure=fig_pred),
                        ], className="figure-grid"),
                        html.H6("Velocity u(x,t)", className="figure-heading"),
                        html.Div([
                            dcc.Graph(figure=fig_u_true),
                            dcc.Graph(figure=fig_u_pred),
                        ], className="figure-grid"),
                    ], className="primary-visualisations"),
                    html.Section([
                        html.H6("Training loss (adaptive λ)", className="figure-heading"),
                        dcc.Graph(figure=make_adaptive_loss_fig(hist)),
                    ], className="loss-panel"),
                    dbc.Alert([
                        html.Strong("Architecture: "),
                        "Dual-head PINN (separate h and u decoders) + Wang et al. 2021 "
                        "adaptive λ weighting. Physics residuals are evaluated in physical "
                        "coordinates while the network trains on standardized observations. "
                        "λ₂ initializes the adaptive weight; the final weight is learned. "
                        "R² measures fit to the synthetic reference, including observed points; "
                        "it is not a held-out accuracy or a guarantee of physical consistency."
                    ], color="info", className="domain-note"),
                ], className="solution-results")
                return content, f"Trained on {data['cfg']['name']} — {epochs} epochs"

            if domain_key == "heat":
                alpha = alpha_val or 0.05
                data = generate_domain_data("heat", alpha=alpha)
                model, hist, stats, extra = train_domain(
                    "heat", data, fraction=fraction, n_epochs=epochs, lambda_pde=lambda_pde,
                    n_colloc=DASH_COLLOCATION_POINTS, callback=report_progress)

                fig_true, fig_pred = make_3d_surface_pair(
                    data["x"], data["t"], data["T"], extra["T_pred"],
                    x_label="Position x", t_label="Time t", z_label="Temperature T",
                    title_true="Ground truth T(x,t)", title_pred="PINN prediction")

                content = html.Div([
                    metric_row([("R²", f"{extra['r2']:.4f}"),
                               ("Training pts", extra["n_train"]),
                               ("α (diffusivity)", alpha)]),
                    html.Div([
                        dcc.Graph(figure=fig_true),
                        dcc.Graph(figure=fig_pred),
                    ], className="figure-grid primary-visualisations"),
                    html.Section([
                        html.H6("Training loss", className="figure-heading"),
                        dcc.Graph(figure=make_loss_curve_fig(hist)),
                    ], className="loss-panel"),
                ], className="solution-results")
                return content, f"Heat diffusion trained — {epochs} epochs"

            if domain_key == "wave":
                data = generate_domain_data("wave", c=c_val or 1.0, mode=mode_val or 1)
                model, hist, stats, extra = train_domain(
                    "wave", data, fraction=fraction, n_epochs=epochs, lambda_pde=lambda_pde,
                    n_colloc=DASH_COLLOCATION_POINTS, callback=report_progress)

                fig_true, fig_pred = make_3d_surface_pair(
                    data["x"], data["t"], data["U"], extra["U_pred"],
                    x_label="Position x", t_label="Time t", z_label="Displacement u",
                    title_true="Ground truth u(x,t)", title_pred="PINN prediction")

                content = html.Div([
                    metric_row([("R²", f"{extra['r2']:.4f}"),
                               ("Training pts", extra["n_train"]),
                               ("Wave speed c", data["c"])]),
                    html.Div([
                        dcc.Graph(figure=fig_true),
                        dcc.Graph(figure=fig_pred),
                    ], className="figure-grid primary-visualisations"),
                    html.Section([
                        html.H6("Training loss", className="figure-heading"),
                        dcc.Graph(figure=make_loss_curve_fig(hist)),
                    ], className="loss-panel"),
                ], className="solution-results")
                return content, f"Wave equation trained — {epochs} epochs"

            if domain_key == "gravity":
                data = generate_domain_data("gravity", eccentricity=0.3 if e_val is None else e_val,
                                            GM=GM_val or 1.0)
                model, hist, stats, extra = train_domain(
                    "gravity", data, fraction=fraction, n_epochs=epochs, lambda_pde=lambda_pde,
                    n_colloc=DASH_COLLOCATION_POINTS, callback=report_progress)

                n_show = extra["n_train"]
                train_idx = np.random.RandomState(0).choice(len(data["t"]), n_show, replace=False)
                fig_orbit = make_orbit_3d(data["x"], data["y"],
                                          extra["x_pred"], extra["y_pred"], train_idx)

                content = html.Div([
                    metric_row([("R² (x)", f"{extra['r2x']:.4f}"),
                               ("R² (y)", f"{extra['r2y']:.4f}"),
                               ("Training pts", extra["n_train"]),
                               ("Eccentricity", data["e"]),
                               ("GM", f"{data['GM']:.2f}")]),
                    html.Div(dcc.Graph(figure=fig_orbit), className="primary-visualisations"),
                    html.Section([
                        html.H6("Training loss", className="figure-heading"),
                        dcc.Graph(figure=make_loss_curve_fig(hist)),
                    ], className="loss-panel"),
                ], className="solution-results")
                return content, f"Orbit reconstructed from {fraction_pct}% of trajectory"

            if domain_key == "elasticity":
                data = generate_domain_data(
                    "elasticity", load_type=load_val or "uniform",
                    EI=EI_val or 1.0, q0=q0_val or 1.0, L=L_val or 1.0)
                model, hist, stats, extra = train_domain(
                    "elasticity", data, fraction=fraction, n_epochs=epochs, lambda_pde=lambda_pde,
                    n_colloc=DASH_COLLOCATION_POINTS, callback=report_progress)

                n_show = extra["n_train"]
                train_idx = np.random.RandomState(0).choice(len(data["x"]), n_show, replace=False)
                fig_beam = make_beam_3d(data["x"], data["v"], extra["v_pred"], train_idx)

                content = html.Div([
                    metric_row([("R²", f"{extra['r2']:.4f}"),
                               ("Training pts", extra["n_train"]),
                               ("Load type", data["load_type"]),
                               ("EI", f"{data['EI']:.2f}"),
                               ("Beam L", f"{data['L']:.1f}"),
                               ("q0", f"{data['q0']:.2f}")]),
                    html.Div(dcc.Graph(figure=fig_beam), className="primary-visualisations"),
                    html.Section([
                        html.H6("Training loss", className="figure-heading"),
                        dcc.Graph(figure=make_loss_curve_fig(hist)),
                    ], className="loss-panel"),
                ], className="solution-results")
                return content, f"Beam deflection trained — {epochs} epochs"

            if domain_key == "dam":
                data = generate_domain_data(
                    "dam", gate_width=gate_width_val or 10.0,
                    reservoir_area=res_area_val or 1e6)
                model, hist, stats, extra = train_domain(
                    "dam", data, fraction=fraction, n_epochs=epochs, lambda_pde=lambda_pde,
                    n_colloc=DASH_COLLOCATION_POINTS, callback=report_progress)

                fig_z, fig_q = make_dam_reservoir_fig(
                    data["t_res"], data["Z"], extra["Z_pred"],
                    data["Q_in"], data["Q_out"])
                fig_reach = make_dam_reach_surface(
                    data["x_reach"], data["t_reach"], data["h_reach"])

                content = html.Div([
                    metric_row([("Reservoir level R²", f"{extra['r2_z']:.4f}"),
                               ("Training pts", extra["n_train"]),
                               ("Reservoir / reach pts",
                                f"{extra.get('n_res_train', extra['n_train'])} / "
                                f"{extra.get('n_reach_train', '—')}"),
                               ("Channel width", f"{data['cfg'].get('channel_width', 30.0):g} m"),
                               ("Reservoir area", f"{data['cfg']['reservoir_area']:.0e} m²")]),
                    html.Div([
                        html.Div([
                            dcc.Graph(figure=fig_z),
                            dcc.Graph(figure=fig_q),
                        ], className="figure-grid"),
                        html.H6("Downstream reach (ground truth, fixed solver)", className="figure-heading"),
                        dcc.Graph(figure=fig_reach),
                    ], className="primary-visualisations"),
                    html.Section([
                        html.H6("Training loss", className="figure-heading"),
                        dcc.Graph(figure=make_dam_loss_curve_fig(hist)),
                    ], className="loss-panel"),
                ], className="solution-results")
                return content, f"Dam system trained — {epochs} epochs"

            if domain_key == "inverse_fluids":
                data = generate_domain_data("inverse_fluids", river_key=river_key or "volta",
                                            true_n=true_n_val or 0.035)
                model, hist, stats, extra = train_domain(
                    "inverse_fluids", data, fraction=fraction, n_epochs=epochs, lambda_pde=lambda_pde,
                    n_colloc=DASH_COLLOCATION_POINTS, callback=report_progress)

                x_km = np.linspace(0, data["cfg"]["length"] / 1000, 200)
                fig_n = make_n_field_fig(x_km, extra["n_field"], extra["true_n"])

                content = html.Div([
                    metric_row([("True n", f"{extra['true_n']:.4f}"),
                               ("Inferred n", f"{extra['n_final']:.4f}"),
                               ("Relative error", f"{extra['err_pct']:.1f}%"),
                               ("Training pts", extra["n_train"])]),
                    html.Div(dcc.Graph(figure=fig_n), className="primary-visualisations"),
                    html.Section([
                        html.H6("Training loss", className="figure-heading"),
                        dcc.Graph(figure=make_loss_curve_fig(hist)),
                    ], className="loss-panel"),
                    dbc.Alert(
                        "Note: roughness inference depends on the forward flow fit and on "
                        "how well sparse observations identify n(x). Treat the error % as a "
                        "synthetic-reference diagnostic; it is not a validated estimate of "
                        "real river roughness or a calibrated uncertainty bound.",
                        color="warning", className="domain-note"
                    ),
                ], className="solution-results")
                return content, f"Inverse roughness trained — {epochs} epochs"

        except Exception as e:
            return dbc.Alert(f"Training error: {e}", color="danger"), "Failed"

        return html.Div(), ""


def metric_row(pairs):
    return html.Dl([
        html.Div([
            html.Dt(str(label), className="metric-label"),
            html.Dd(str(value), className="metric-value"),
        ], className="metric")
        for label, value in pairs
    ], className="metric-row")


# ── Sync controls locally before a training click reads their stores ────────────
# Avoid a server round trip: changing a value and immediately pressing Train
# must submit that value, including zero eccentricity.
for param in PARAM_DEFAULTS:
    app.clientside_callback(
        """function(value) {
            return value == null ? window.dash_clientside.no_update : value;
        }""",
        Output(f"{param}-store", "data"), Input(param, "value"),
        prevent_initial_call=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SPARSITY SWEEP CALLBACK
# ═══════════════════════════════════════════════════════════════════════════════

@app.callback(
    Output("experiment-results-area", "children"),
    Input("sparsity-btn", "n_clicks"),
    Input("mcdrop-btn", "n_clicks"),
    State("domain-select", "value"),
    State("sparsity-fractions", "value"),
    State("sparsity-seeds", "value"),
    State("epochs", "value"),
    State("dropout-p", "value"),
    State("mc-samples", "value"),
    State("param-river-store", "data"),
    State("param-n-store", "data"),
    State("param-s0-store", "data"),
    *[State(f"{param}-store", "data") for param in (
        "param-alpha", "param-c", "param-mode", "param-e", "param-GM",
        "param-load", "param-EI", "param-L", "param-q0")],
    State("data-fraction", "value"), State("lambda-pde", "value"),
    State("workspace-id", "data"),
    prevent_initial_call=True,
    background=True,
    running=[
        (Output("sparsity-btn", "disabled"), True, False),
        (Output("mcdrop-btn", "disabled"), True, False),
    ],
)
@remember_run(background_cache, "experiment")
def run_experiment(sparsity_clicks, mcdrop_clicks,
                   domain_key, fractions, n_seeds, n_epochs,
                   dropout_p, mc_samples, river_key, n_val, s0_val,
                   alpha_val=0.05, c_val=1.0, mode_val=1, e_val=0.3, GM_val=1.0,
                   load_val="uniform", EI_val=1.0, L_val=1.0, q0_val=1.0,
                   fraction_pct=15, lambda_pde=0.1, workspace_id=None):
    """Single callback handles both experiment buttons via callback_context."""
    with training_slot(background_cache.directory) as acquired:
        if not acquired:
            return dbc.Alert("Another training or experiment job is running. Please try again when it finishes.", color="info")
        from dash import callback_context
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import numpy as np, torch

        if not callback_context.triggered:
            return html.Div()

        triggered_id = callback_context.triggered[0]["prop_id"].split(".")[0]

        # Use the same selected physical parameters and normalization as the
        # solution run, then evaluate both experiments on the full reference.
        if domain_key not in {"fluids", "heat", "wave", "gravity", "elasticity"}:
            return dbc.Alert(
                f"Sparsity sweep and MC Dropout not yet configured for '{domain_key}'. "
                "Select fluids, heat, wave, gravity, or elasticity.", color="warning")
        parameters = {
            "fluids": dict(river_key=river_key or "volta", n_manning=n_val, S0=s0_val),
            "heat": dict(alpha=alpha_val),
            "wave": dict(c=c_val, mode=mode_val),
            "gravity": dict(eccentricity=e_val, GM=GM_val),
            "elasticity": dict(load_type=load_val, EI=EI_val, L=L_val, q0=q0_val),
        }
        try:
            data = generate_domain_data(domain_key, **parameters[domain_key])
            pde, all_coords, all_targets, _stats, colloc_ranges = prepare_domain_training_data(
                domain_key, data)
            output_idx = 0
            full_y_true = all_targets[:, output_idx].numpy()
            domain_label = data.get("cfg", {}).get("name", DOMAIN_INFO[domain_key]["label"])
        except Exception as e:
            return dbc.Alert(f"Data generation error: {e}", color="danger")

        # ─────────────────────────────────────────────────────────────────────────
        # SPARSITY SWEEP
        # ─────────────────────────────────────────────────────────────────────────
        if triggered_id == "sparsity-btn":
            if not fractions:
                return dbc.Alert("Choose at least one data fraction to test.", color="warning")
            fracs = sorted(fractions)
            try:
                results = run_sparsity_sweep(
                    pde, all_coords, all_targets,
                    all_coords, full_y_true,
                    fractions=fracs,
                    n_epochs=min(n_epochs, 1000),
                    n_seeds=n_seeds or 2,
                    output_idx=output_idx,
                    colloc_ranges=colloc_ranges,
                    lambda_pde=lambda_pde,
                )
            except Exception as e:
                return dbc.Alert(f"Sweep error: {e}", color="danger")

            frac_pct = [f*100 for f in results["fractions"]]
            PINN_C, MLP_C = "#534AB7", "#D85A30"

            # Main figure: R2 vs data fraction — the publishable result
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=frac_pct, y=results["pinn_r2"], mode="lines+markers",
                name="PINN (physics-informed)",
                line=dict(color=PINN_C, width=2.5),
                marker=dict(size=8),
                error_y=dict(type="data", array=results["pinn_r2_std"],
                            visible=True, color=PINN_C, thickness=1.5, width=4)
            ))
            fig.add_trace(go.Scatter(
                x=frac_pct, y=results["mlp_r2"], mode="lines+markers",
                name="Vanilla MLP (same capacity, no physics)",
                line=dict(color=MLP_C, width=2.5, dash="dash"),
                marker=dict(size=8, symbol="square"),
                error_y=dict(type="data", array=results["mlp_r2_std"],
                            visible=True, color=MLP_C, thickness=1.5, width=4)
            ))

            # Shade the PINN advantage region
            pinn_advantage = [p > m for p,m in zip(results["pinn_r2"], results["mlp_r2"])]
            if any(pinn_advantage):
                adv_fracs = [f for f,a in zip(frac_pct, pinn_advantage) if a]
                adv_pinn  = [p for p,a in zip(results["pinn_r2"], pinn_advantage) if a]
                adv_mlp   = [m for m,a in zip(results["mlp_r2"], pinn_advantage) if a]
                fig.add_trace(go.Scatter(
                    x=adv_fracs + adv_fracs[::-1],
                    y=adv_pinn + adv_mlp[::-1],
                    fill="toself", fillcolor="rgba(83,74,183,0.10)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="PINN advantage zone", showlegend=True
                ))

            fig.update_layout(
                title=f"Sparsity benchmark — {domain_label} | {n_seeds} seed(s) per fraction",
                xaxis_title="Training data fraction (%)",
                yaxis_title="R² score",
                height=420,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", y=1.12),
            )

            # RMSE figure
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=frac_pct, y=results["pinn_rmse"], mode="lines+markers",
                name="PINN", line=dict(color=PINN_C, width=2.5), marker=dict(size=7)))
            fig2.add_trace(go.Scatter(
                x=frac_pct, y=results["mlp_rmse"], mode="lines+markers",
                name="Vanilla MLP", line=dict(color=MLP_C, width=2.5, dash="dash"),
                marker=dict(size=7, symbol="square")))
            fig2.update_layout(
                title="RMSE vs data fraction",
                xaxis_title="Training data fraction (%)", yaxis_title="RMSE (normalised)",
                height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", y=1.12),
            )

            winning_fractions = [f"{fraction:.0f}%" for fraction, pinn, mlp
                                 in zip(frac_pct, results["pinn_r2"], results["mlp_r2"])
                                 if pinn > mlp]
            crossover_msg = ("Higher mean PINN R² at tested fractions: " + ", ".join(winning_fractions)
                             if winning_fractions else "No PINN advantage detected at tested fractions.")

            import pandas as pd
            df = pd.DataFrame({
                "Fraction (%)": [f"{f:.0f}" for f in frac_pct],
                "PINN R²": [f"{v:.4f} ± {s:.4f}"
                            for v,s in zip(results["pinn_r2"], results["pinn_r2_std"])],
                "MLP R²":  [f"{v:.4f} ± {s:.4f}"
                            for v,s in zip(results["mlp_r2"], results["mlp_r2_std"])],
                "PINN RMSE": [f"{v:.4f}" for v in results["pinn_rmse"]],
                "MLP RMSE":  [f"{v:.4f}" for v in results["mlp_rmse"]],
            })

            return html.Div([
                html.H5("Sparsity sweep — PINN vs vanilla MLP", className="mt-3"),
                dbc.Alert(crossover_msg + " Scores compare against the full synthetic reference, "
                          "including training points; they are not held-out generalization scores.",
                          color="success" if winning_fractions else "warning",
                          className="small"),
                dcc.Graph(figure=fig),
                dcc.Graph(figure=fig2),
                html.H6("Results table", className="mt-3"),
                dbc.Table.from_dataframe(df, striped=True, bordered=False,
                                         hover=True, size="sm"),
            ])

        # ─────────────────────────────────────────────────────────────────────────
        # MC DROPOUT UNCERTAINTY
        # ─────────────────────────────────────────────────────────────────────────
        elif triggered_id == "mcdrop-btn":
            try:
                n_train = max(1, int(len(all_coords) * fraction_pct / 100))
                train_idx = np.random.RandomState(0).choice(len(all_coords), n_train, replace=False)
                mc_model, hist_mc = train_mc_dropout(
                    pde, all_coords[train_idx], all_targets[train_idx],
                    n_epochs=min(n_epochs, 1500),
                    lr=1e-3, lambda_pde=lambda_pde,
                    n_colloc=1500, colloc_ranges=colloc_ranges,
                    dropout_p=dropout_p or 0.08,
                )
            except Exception as e:
                return dbc.Alert(f"MC Dropout training error: {e}", color="danger")

            means, stds = mc_model.predict_with_uncertainty(
                all_coords, n_samples=int(mc_samples or 100))

            y_mean = means[:, output_idx]
            y_std  = stds[:, output_idx]
            y_true = full_y_true

            # For 1D domains (elasticity, gravity) — line plot with bands
            is_1d = (all_coords.shape[1] == 1)

            if is_1d:
                x_vals = all_coords[:, 0].numpy()
                sort_idx = np.argsort(x_vals)
                x_s_arr = x_vals[sort_idx]
                y_m_s = y_mean[sort_idx]
                y_std_s = y_std[sort_idx]
                y_t_s = y_true[sort_idx]

                fig_mc = go.Figure()
                fig_mc.add_trace(go.Scatter(
                    x=x_s_arr, y=y_t_s, mode="lines",
                    name="True", line=dict(color="#111111", width=2.5)))
                fig_mc.add_trace(go.Scatter(
                    x=x_s_arr, y=y_m_s, mode="lines",
                    name="MC mean", line=dict(color="#534AB7", width=2, dash="dash")))
                fig_mc.add_trace(go.Scatter(
                    x=np.concatenate([x_s_arr, x_s_arr[::-1]]),
                    y=np.concatenate([y_m_s+2*y_std_s, (y_m_s-2*y_std_s)[::-1]]),
                    fill="toself", fillcolor="rgba(83,74,183,0.18)",
                    line=dict(color="rgba(255,255,255,0)"), name="±2σ dropout spread"))
                fig_mc.add_trace(go.Scatter(
                    x=np.concatenate([x_s_arr, x_s_arr[::-1]]),
                    y=np.concatenate([y_m_s+y_std_s, (y_m_s-y_std_s)[::-1]]),
                    fill="toself", fillcolor="rgba(83,74,183,0.30)",
                    line=dict(color="rgba(255,255,255,0)"), name="±1σ dropout spread"))
                fig_mc.update_layout(
                    title=f"MC Dropout uncertainty — {domain_label} "
                          f"({int(mc_samples)} samples, p={dropout_p})",
                    xaxis_title="Input coordinate (normalised)",
                    yaxis_title="Output (normalised)",
                    height=400,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    legend=dict(orientation="h", y=1.12),
                )

            else:
                # 2D field — show uncertainty as a separate surface
                n_t = int(np.sqrt(len(y_std))) if int(np.sqrt(len(y_std)))**2 == len(y_std) else 50
                n_x = len(y_std) // n_t
                if n_t * n_x != len(y_std):
                    n_t, n_x = 50, len(y_std) // 50

                y_std_grid = y_std[:n_t*n_x].reshape(n_t, n_x)
                fig_mc = go.Figure(data=[go.Surface(
                    z=y_std_grid * 4,   # Full width of the mean ±2σ band
                    colorscale="Oranges", showscale=True,
                    contours=dict(z=dict(show=True, usecolormap=True, project_z=True))
                )])
                fig_mc.update_layout(
                    scene=dict(
                        xaxis_title="x", yaxis_title="t",
                        zaxis_title="±2σ spread width",
                        camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
                    ),
                    title=f"Epistemic uncertainty — {domain_label} "
                          f"({int(mc_samples)} samples, p={dropout_p})",
                    height=450,
                    paper_bgcolor="rgba(0,0,0,0)",
                )

            # Uncertainty width histogram
            fig_hist = go.Figure(data=[go.Histogram(
                x=y_std * 4, nbinsx=40,
                marker_color="#EF9F27", opacity=0.8,
                name="±2σ spread width distribution"
            )])
            fig_hist.update_layout(
                title="Distribution of dropout spread widths (mean ±2σ)",
                xaxis_title="Full ±2σ width (normalised units)",
                yaxis_title="Count", height=280,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )

            # Loss curve
            fig_loss = make_loss_curve_fig(hist_mc)

            mean_ci = float(y_std.mean() * 4)
            mean_r2 = float(1 - np.mean((y_true - y_mean)**2) /
                            (np.var(y_true) + 1e-8))

            return html.Div([
                html.H5("MC Dropout uncertainty quantification", className="mt-3"),
                metric_row([
                    ("MC mean R²", f"{mean_r2:.4f}"),
                    ("Mean ±2σ width", f"{mean_ci:.4f}"),
                    ("MC samples", int(mc_samples)),
                    ("Dropout p", dropout_p),
                    ("Training pts", n_train),
                ]),
                dbc.Alert(
                    "Dropout spread describes variability across stochastic model predictions. "
                    "The ±1σ and ±2σ bands are descriptive, not calibrated confidence intervals; "
                    "they do not establish coverage or guarantee the best sensor locations. "
                    "R² compares with the full synthetic reference, including training points. "
                    "Method: Gal & Ghahramani, ICML 2016.",
                    color="info", className="small"
                ),
                dcc.Graph(figure=fig_mc),
                dcc.Graph(figure=fig_hist),
                html.H6("Training loss", className="mt-3"),
                dcc.Graph(figure=fig_loss),
            ])

        return html.Div()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=os.environ.get("DASH_DEBUG") == "1", host="0.0.0.0", port=port)
