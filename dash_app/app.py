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
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import numpy as np

from dash_app.domain_utils import (
    DOMAIN_INFO, generate_domain_data, train_domain,
    make_3d_surface_pair, make_orbit_3d, make_beam_3d, make_loss_curve_fig,
    make_dam_reservoir_fig, make_dam_reach_surface, make_n_field_fig,
    make_dam_loss_curve_fig, make_adaptive_loss_fig
)
from experiments.sparsity_and_uncertainty import (
    MCDropoutEngine, train_mc_dropout, run_sparsity_sweep
)
from data.generator import RIVERS, generate_sparse_observations

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY],
               suppress_callback_exceptions=True)
app.title = "PINN Multi-Physics Explorer"
server = app.server  # exposed for gunicorn / Hugging Face Spaces

@server.route("/healthz")
def healthz():
    return "ok"

# ── Layout ────────────────────────────────────────────────────────────────────

sidebar = dbc.Card([
    dbc.CardBody([
        html.H4("PINN Explorer", className="mb-1"),
        html.P("Physics-informed ML across 5 domains", className="text-muted small mb-3"),
        html.Hr(),

        html.Label("Physics domain", className="fw-bold small"),
        dcc.Dropdown(
            id="domain-select",
            options=[{"label": v["label"], "value": k} for k, v in DOMAIN_INFO.items()],
            value="fluids", clearable=False, className="mb-3"
        ),

        html.Div(id="domain-params"),  # populated dynamically per domain
        # Hidden stores hold the CURRENT domain-specific param value, kept in
        # sync by sync_param_stores(). The training callback reads these
        # instead of binding directly to dynamically-created components,
        # which Dash cannot do safely (component may not exist in the DOM yet).
        dcc.Store(id="param-river-store", data="volta"),
        dcc.Store(id="param-alpha-store", data=0.05),
        dcc.Store(id="param-c-store", data=1.0),
        dcc.Store(id="param-mode-store", data=1),
        dcc.Store(id="param-e-store", data=0.3),
        dcc.Store(id="param-load-store", data="uniform"),
        dcc.Store(id="param-n-store", data=0.035),
        dcc.Store(id="param-s0-store", data=0.0008),
        dcc.Store(id="param-GM-store", data=1.0),
        dcc.Store(id="param-EI-store", data=1.0),
        dcc.Store(id="param-L-store", data=1.0),
        dcc.Store(id="param-q0-store", data=1.0),
        dcc.Store(id="param-gate-width-store", data=10.0),
        dcc.Store(id="param-res-area-store", data=1000000.0),
        dcc.Store(id="param-true-n-store", data=0.035),

        html.Hr(),
        html.Label("Training data fraction (%)", className="fw-bold small"),
        dcc.Slider(id="data-fraction", min=5, max=100, step=5, value=15,
                  marks={5: "5%", 25: "25%", 50: "50%", 100: "100%"}),

        html.Label("Epochs", className="fw-bold small mt-3"),
        dcc.Slider(id="epochs", min=500, max=4000, step=500, value=1500,
                  marks={500: "500", 2000: "2k", 4000: "4k"}),

        html.Label("Physics weight (λ₂)", className="fw-bold small mt-3"),
        dcc.Slider(id="lambda-pde", min=0.01, max=1.0, step=0.01, value=0.1,
                  marks={0.01: "0.01", 0.5: "0.5", 1.0: "1.0"}),

        dbc.Button("Train PINN", id="train-btn", color="primary",
                  className="w-100 mt-4", n_clicks=0),

        html.Div(id="train-status", className="mt-3 small text-muted"),

        html.Hr(),
        html.P("Research experiments", className="fw-bold small mb-2"),

        dbc.Button("Sparsity sweep", id="sparsity-btn", color="secondary",
                  outline=True, className="w-100 mb-2", n_clicks=0),

        html.Div([
            html.Label("Fractions to test", className="small text-muted"),
            dcc.Checklist(
                id="sparsity-fractions",
                options=[{"label": f" {p}%", "value": p/100}
                         for p in [5, 10, 20, 50, 100]],
                value=[0.05, 0.10, 0.20, 0.50, 1.00],
                inline=True, className="small"
            ),
            html.Label("Seeds per fraction", className="small text-muted mt-1"),
            dcc.Slider(id="sparsity-seeds", min=1, max=5, step=1, value=2,
                      marks={i: str(i) for i in range(1,6)},
                      className="mb-2"),
        ], id="sparsity-controls"),

        dbc.Button("MC Dropout uncertainty", id="mcdrop-btn", color="secondary",
                  outline=True, className="w-100 mb-2", n_clicks=0),

        html.Div([
            html.Label("Dropout p", className="small text-muted"),
            dcc.Slider(id="dropout-p", min=0.02, max=0.20, step=0.02, value=0.08,
                      marks={0.02:"0.02", 0.10:"0.10", 0.20:"0.20"}),
            html.Label("MC samples", className="small text-muted mt-1"),
            dcc.Slider(id="mc-samples", min=50, max=300, step=50, value=100,
                      marks={50:"50", 150:"150", 300:"300"}),
        ], id="mcdrop-controls"),
    ])
], className="h-100")

main_panel = html.Div([
    html.Div(id="equation-display", className="mb-3"),
    dcc.Loading(
        id="loading-train",
        type="circle",
        children=html.Div(id="results-area")
    ),
    html.Hr(className="mt-4"),
    dcc.Loading(
        id="loading-experiments",
        type="dot",
        children=html.Div(id="experiment-results-area")
    ),
])

app.layout = dbc.Container([
    dbc.Row(
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.H2("PINN Multi-Physics Explorer", className="mb-1"),
                    html.P(
                        "Interactive Dash dashboard for training and visualising PINN models across multiple physics domains.",
                        className="text-muted mb-0"
                    ),
                ])
            ], className="shadow-sm mb-3"),
            width=12
        )
    ),
    dbc.Row([
        dbc.Col(sidebar, width=3, className="py-3"),
        dbc.Col(main_panel, width=9, className="py-3"),
    ], className="g-0"),
    dcc.Store(id="trained-results-store"),
], fluid=True)


# ── Dynamic per-domain parameter controls ───────────────────────────────────────

@app.callback(Output("domain-params", "children"), Input("domain-select", "value"))
def render_domain_params(domain_key):
    if domain_key == "fluids":
        river_options = [{"label": v["name"], "value": k}
                         for k, v in RIVERS.items() if k != "dam"]
        return html.Div([
            html.Label("River", className="fw-bold small mt-2"),
            dcc.Dropdown(id="param-river", options=river_options,
                        value="volta", clearable=False),
            html.Label("Manning n", className="fw-bold small mt-3"),
            dcc.Slider(id="param-n", min=0.01, max=0.06, step=0.005, value=0.035,
                       marks={0.01: "0.01", 0.03: "0.03", 0.06: "0.06"}),
            html.Label("Bed slope S0", className="fw-bold small mt-3"),
            dcc.Slider(id="param-s0", min=0.0001, max=0.002, step=0.0001, value=0.0008,
                       marks={0.0001: "1e-4", 0.0008: "8e-4", 0.002: "2e-3"}),
        ])
    if domain_key == "heat":
        return html.Div([
            html.Label("Thermal diffusivity α", className="fw-bold small mt-2"),
            dcc.Slider(id="param-alpha", min=0.005, max=0.15, step=0.005, value=0.05,
                      marks={0.005: "0.005", 0.05: "0.05", 0.15: "0.15"}),
        ])
    if domain_key == "wave":
        return html.Div([
            html.Label("Wave speed c", className="fw-bold small mt-2"),
            dcc.Slider(id="param-c", min=0.5, max=3.0, step=0.1, value=1.0,
                      marks={0.5: "0.5", 1.5: "1.5", 3.0: "3.0"}),
            html.Label("Vibration mode", className="fw-bold small mt-2"),
            dcc.Slider(id="param-mode", min=1, max=4, step=1, value=1,
                      marks={i: str(i) for i in range(1, 5)}),
        ])
    if domain_key == "gravity":
        return html.Div([
            html.Label("Orbital eccentricity", className="fw-bold small mt-2"),
            dcc.Slider(id="param-e", min=0.0, max=0.8, step=0.05, value=0.3,
                      marks={0: "circular", 0.4: "0.4", 0.8: "0.8"}),
            html.Label("Central mass parameter GM", className="fw-bold small mt-3"),
            dcc.Slider(id="param-GM", min=0.1, max=5.0, step=0.1, value=1.0,
                      marks={0.1: "0.1", 1.0: "1.0", 5.0: "5.0"}),
        ])
    if domain_key == "elasticity":
        return html.Div([
            html.Label("Load type", className="fw-bold small mt-2"),
            dcc.Dropdown(id="param-load", options=[
                {"label": "Uniform load", "value": "uniform"},
                {"label": "Point load (midspan)", "value": "point"},
            ], value="uniform", clearable=False),
            html.Label("Flexural rigidity EI", className="fw-bold small mt-3"),
            dcc.Slider(id="param-EI", min=0.1, max=5.0, step=0.1, value=1.0,
                       marks={0.1: "0.1", 1.0: "1.0", 5.0: "5.0"}),
            html.Label("Beam length L", className="fw-bold small mt-3"),
            dcc.Slider(id="param-L", min=0.5, max=3.0, step=0.1, value=1.0,
                       marks={0.5: "0.5", 1.0: "1.0", 3.0: "3.0"}),
            html.Label("Load amplitude q0", className="fw-bold small mt-3"),
            dcc.Slider(id="param-q0", min=0.2, max=5.0, step=0.2, value=1.0,
                       marks={0.2: "0.2", 1.0: "1.0", 5.0: "5.0"}),
        ])
    if domain_key == "dam":
        return html.Div([
            html.Label("Gate width", className="fw-bold small mt-2"),
            dcc.Slider(id="param-gate-width", min=2.0, max=30.0, step=1.0, value=10.0,
                       marks={2: "2", 10: "10", 30: "30"}),
            html.Label("Reservoir area", className="fw-bold small mt-3"),
            dcc.Slider(id="param-res-area", min=1e5, max=5e6, step=1e5, value=1e6,
                       marks={1e5: "1e5", 1e6: "1e6", 5e6: "5e6"}),
            html.P("Adjust gate geometry and reservoir storage for better operational realism.",
                   className="text-muted small mt-2"),
        ])
    if domain_key == "inverse_fluids":
        river_options = [{"label": v["name"], "value": k}
                         for k, v in RIVERS.items() if k != "dam"]
        return html.Div([
            html.Label("River (sets true n, hidden from the model)",
                      className="fw-bold small mt-2"),
            dcc.Dropdown(id="param-river", options=river_options,
                        value="volta", clearable=False),
            html.Label("True Manning n", className="fw-bold small mt-3"),
            dcc.Slider(id="param-true-n", min=0.01, max=0.06, step=0.005, value=0.035,
                       marks={0.01: "0.01", 0.03: "0.03", 0.06: "0.06"}),
            html.P("The PINN infers n(x) from sparse observations and the physics residual.",
                   className="text-muted small mt-2"),
        ])
    return html.Div()


@app.callback(Output("equation-display", "children"), Input("domain-select", "value"))
def show_equation(domain_key):
    info = DOMAIN_INFO[domain_key]
    return dbc.Alert([
        html.Div([html.Strong(info["label"])], className="mb-2"),
        html.Code(info["equation"], className="small mb-2 d-block"),
        html.P(info.get("description", ""), className="small mb-0 text-muted"),
    ], color="light", className="border")


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
    prevent_initial_call=True,
)
def run_training(n_clicks, domain_key, fraction_pct, epochs, lambda_pde,
                 river_key, alpha_val, c_val, mode_val, e_val, load_val,
                 n_val, s0_val, GM_val, EI_val, L_val, q0_val,
                 gate_width_val, res_area_val, true_n_val):
    fraction = fraction_pct / 100.0

    try:
        if domain_key == "fluids":
            data = generate_domain_data(
                "fluids", river_key=river_key or "volta",
                n_manning=n_val, S0=s0_val)
            model, hist, stats, extra = train_domain(
                "fluids", data, fraction=fraction, n_epochs=epochs, lambda_pde=lambda_pde)

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
                dbc.Alert([
                    html.Strong("Architecture: "),
                    "Dual-head PINN (separate h and u decoders) + Wang et al. 2021 "
                    "adaptive λ weighting. Physics constraint improves physical "
                    "consistency and extrapolation accuracy. For interpolation, the "
                    f"data-only ceiling for h is R²≈0.47 — physics regularisation "
                    "trades some interpolation accuracy for guaranteed physical consistency."
                ], color="info", className="small mt-2 mb-3"),
                html.H6("Water depth h(x,t)", className="mt-2"),
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig_true), width=6),
                    dbc.Col(dcc.Graph(figure=fig_pred), width=6),
                ]),
                html.H6("Velocity u(x,t)", className="mt-3"),
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig_u_true), width=6),
                    dbc.Col(dcc.Graph(figure=fig_u_pred), width=6),
                ]),
                html.H6("Training loss (adaptive λ)", className="mt-3"),
                dcc.Graph(figure=make_adaptive_loss_fig(hist)),
            ])
            return content, f"Trained on {data['cfg']['name']} — {epochs} epochs"

        if domain_key == "heat":
            alpha = alpha_val or 0.05
            data = generate_domain_data("heat", alpha=alpha)
            model, hist, stats, extra = train_domain(
                "heat", data, fraction=fraction, n_epochs=epochs, lambda_pde=lambda_pde)

            fig_true, fig_pred = make_3d_surface_pair(
                data["x"], data["t"], data["T"], extra["T_pred"],
                x_label="Position x", t_label="Time t", z_label="Temperature T",
                title_true="Ground truth T(x,t)", title_pred="PINN prediction")

            content = html.Div([
                metric_row([("R²", f"{extra['r2']:.4f}"),
                           ("Training pts", extra["n_train"]),
                           ("α (diffusivity)", alpha)]),
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig_true), width=6),
                    dbc.Col(dcc.Graph(figure=fig_pred), width=6),
                ]),
                html.H6("Training loss", className="mt-3"),
                dcc.Graph(figure=make_loss_curve_fig(hist)),
            ])
            return content, f"Heat diffusion trained — {epochs} epochs"

        if domain_key == "wave":
            data = generate_domain_data("wave", c=c_val or 1.0, mode=mode_val or 1)
            model, hist, stats, extra = train_domain(
                "wave", data, fraction=fraction, n_epochs=epochs, lambda_pde=lambda_pde)

            fig_true, fig_pred = make_3d_surface_pair(
                data["x"], data["t"], data["U"], extra["U_pred"],
                x_label="Position x", t_label="Time t", z_label="Displacement u",
                title_true="Ground truth u(x,t)", title_pred="PINN prediction")

            content = html.Div([
                metric_row([("R²", f"{extra['r2']:.4f}"),
                           ("Training pts", extra["n_train"]),
                           ("Wave speed c", data["c"])]),
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig_true), width=6),
                    dbc.Col(dcc.Graph(figure=fig_pred), width=6),
                ]),
                html.H6("Training loss", className="mt-3"),
                dcc.Graph(figure=make_loss_curve_fig(hist)),
            ])
            return content, f"Wave equation trained — {epochs} epochs"

        if domain_key == "gravity":
            data = generate_domain_data("gravity", eccentricity=e_val or 0.3, GM=GM_val or 1.0)
            model, hist, stats, extra = train_domain(
                "gravity", data, fraction=fraction, n_epochs=epochs, lambda_pde=lambda_pde)

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
                dcc.Graph(figure=fig_orbit),
                html.H6("Training loss", className="mt-3"),
                dcc.Graph(figure=make_loss_curve_fig(hist)),
            ])
            return content, f"Orbit reconstructed from {fraction_pct}% of trajectory"

        if domain_key == "elasticity":
            data = generate_domain_data(
                "elasticity", load_type=load_val or "uniform",
                EI=EI_val or 1.0, q0=q0_val or 1.0, L=L_val or 1.0)
            model, hist, stats, extra = train_domain(
                "elasticity", data, fraction=fraction, n_epochs=epochs, lambda_pde=lambda_pde)

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
                dcc.Graph(figure=fig_beam),
                html.H6("Training loss", className="mt-3"),
                dcc.Graph(figure=make_loss_curve_fig(hist)),
            ])
            return content, f"Beam deflection trained — {epochs} epochs"

        if domain_key == "dam":
            data = generate_domain_data(
                "dam", gate_width=gate_width_val or 10.0,
                reservoir_area=res_area_val or 1e6)
            model, hist, stats, extra = train_domain(
                "dam", data, fraction=fraction, n_epochs=epochs, lambda_pde=lambda_pde)

            fig_z, fig_q = make_dam_reservoir_fig(
                data["t_res"], data["Z"], extra["Z_pred"],
                data["Q_in"], data["Q_out"])
            fig_reach = make_dam_reach_surface(
                data["x_reach"], data["t_reach"], data["h_reach"])

            content = html.Div([
                metric_row([("Reservoir level R²", f"{extra['r2_z']:.4f}"),
                           ("Time steps", extra["n_train"]),
                           ("Reservoir area", f"{data['cfg']['reservoir_area']:.0e} m²")]),
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig_z), width=6),
                    dbc.Col(dcc.Graph(figure=fig_q), width=6),
                ]),
                html.H6("Downstream reach (ground truth, fixed solver)", className="mt-3"),
                dcc.Graph(figure=fig_reach),
                html.H6("Training loss", className="mt-3"),
                dcc.Graph(figure=make_dam_loss_curve_fig(hist)),
            ])
            return content, f"Dam system trained — {epochs} epochs"

        if domain_key == "inverse_fluids":
            data = generate_domain_data("inverse_fluids", river_key=river_key or "volta",
                                        true_n=true_n_val or 0.035)
            model, hist, stats, extra = train_domain(
                "inverse_fluids", data, fraction=fraction, n_epochs=epochs, lambda_pde=lambda_pde)

            x_km = np.linspace(0, data["cfg"]["length"] / 1000, 200)
            fig_n = make_n_field_fig(x_km, extra["n_field"], extra["true_n"])

            content = html.Div([
                metric_row([("True n", f"{extra['true_n']:.4f}"),
                           ("Inferred n", f"{extra['n_final']:.4f}"),
                           ("Relative error", f"{extra['err_pct']:.1f}%"),
                           ("Training pts", extra["n_train"])]),
                dcc.Graph(figure=fig_n),
                html.H6("Training loss", className="mt-3"),
                dcc.Graph(figure=make_loss_curve_fig(hist)),
                dbc.Alert(
                    "Note: roughness inference accuracy is currently capped "
                    "by the same forward-fit ceiling affecting the base "
                    "fluids domain on this flood-wave signal — see project "
                    "notes. Treat the error % as a diagnostic, not a final "
                    "result, until the forward capacity issue is resolved.",
                    color="warning", className="small mt-3"
                ),
            ])
            return content, f"Inverse roughness trained — {epochs} epochs"

    except Exception as e:
        return dbc.Alert(f"Training error: {e}", color="danger"), "Failed"

    return html.Div(), ""


def metric_row(pairs):
    cols = []
    for label, value in pairs:
        cols.append(dbc.Col(
            html.Div([
                html.Div(str(label), className="text-muted small"),
                html.Div(str(value), className="fs-5 fw-bold"),
            ], className="p-2 bg-light rounded text-center"),
            width=12 // len(pairs)
        ))
    return dbc.Row(cols, className="mb-3 g-2")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=os.environ.get("DASH_DEBUG") == "1", host="0.0.0.0", port=port)


# ── Sync dynamic per-domain controls into hidden stores ─────────────────────────
# Dash cannot bind a State/Input directly to a component that is created
# dynamically inside another callback's output -- it may not exist in the
# DOM yet, which raises a "nonexistent object" error unless callback
# exceptions are suppressed (set below). The fix is the standard Dash
# pattern: each dynamic control writes into a permanent, always-present
# dcc.Store, and the training callback reads from the stores instead of
# binding to the dynamic controls directly.

@app.callback(Output("param-river-store", "data"),
              Input("param-river", "value"), prevent_initial_call=True)
def sync_river(v):
    return v


@app.callback(Output("param-alpha-store", "data"),
              Input("param-alpha", "value"), prevent_initial_call=True)
def sync_alpha(v):
    return v


@app.callback(Output("param-c-store", "data"),
              Input("param-c", "value"), prevent_initial_call=True)
def sync_c(v):
    return v


@app.callback(Output("param-mode-store", "data"),
              Input("param-mode", "value"), prevent_initial_call=True)
def sync_mode(v):
    return v


@app.callback(Output("param-e-store", "data"),
              Input("param-e", "value"), prevent_initial_call=True)
def sync_e(v):
    return v


@app.callback(Output("param-load-store", "data"),
              Input("param-load", "value"), prevent_initial_call=True)
def sync_load(v):
    return v


@app.callback(Output("param-n-store", "data"),
              Input("param-n", "value"), prevent_initial_call=True)
def sync_n(v):
    return v


@app.callback(Output("param-s0-store", "data"),
              Input("param-s0", "value"), prevent_initial_call=True)
def sync_s0(v):
    return v


@app.callback(Output("param-GM-store", "data"),
              Input("param-GM", "value"), prevent_initial_call=True)
def sync_GM(v):
    return v


@app.callback(Output("param-EI-store", "data"),
              Input("param-EI", "value"), prevent_initial_call=True)
def sync_EI(v):
    return v


@app.callback(Output("param-L-store", "data"),
              Input("param-L", "value"), prevent_initial_call=True)
def sync_L(v):
    return v


@app.callback(Output("param-q0-store", "data"),
              Input("param-q0", "value"), prevent_initial_call=True)
def sync_q0(v):
    return v


@app.callback(Output("param-gate-width-store", "data"),
              Input("param-gate-width", "value"), prevent_initial_call=True)
def sync_gate_width(v):
    return v


@app.callback(Output("param-res-area-store", "data"),
              Input("param-res-area", "value"), prevent_initial_call=True)
def sync_res_area(v):
    return v


@app.callback(Output("param-true-n-store", "data"),
              Input("param-true-n", "value"), prevent_initial_call=True)
def sync_true_n(v):
    return v


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
    prevent_initial_call=True,
)
def run_experiment(sparsity_clicks, mcdrop_clicks,
                   domain_key, fractions, n_seeds, n_epochs,
                   dropout_p, mc_samples, river_key, n_val, s0_val):
    """Single callback handles both experiment buttons via callback_context."""
    from dash import callback_context
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np, torch

    if not callback_context.triggered:
        return html.Div()

    triggered_id = callback_context.triggered[0]["prop_id"].split(".")[0]

    # ── shared: generate domain data ─────────────────────────────────────────
    try:
        if domain_key == "fluids":
            data = generate_domain_data("fluids", river_key=river_key or "volta",
                                        n_manning=n_val, S0=s0_val)
            from domains.fluids import FluidsPDE
            pde = FluidsPDE(S0=data["cfg"]["S0"], n_manning=data["cfg"]["n_manning"])
            x_m, x_s = data["X"].mean(), data["X"].std()
            t_m, t_s = data["T"].mean(), data["T"].std()
            h_m, h_s = data["H"].mean(), data["H"].std()
            u_m, u_s = data["U"].mean(), data["U"].std()
            all_coords = torch.tensor(
                np.stack([(data["X"]-x_m)/x_s, (data["T"]-t_m)/t_s], axis=1),
                dtype=torch.float32)
            all_targets = torch.tensor(
                np.stack([(data["H"]-h_m)/h_s, (data["U"]-u_m)/u_s], axis=1),
                dtype=torch.float32)
            full_y_true = (all_targets[:, 0].numpy())  # normalised h for R2
            domain_label = data["cfg"]["name"]
            colloc_ranges = None
            output_idx = 0

        elif domain_key == "heat":
            from domains.heat import HeatPDE, solve_heat_analytical
            alpha = 0.05
            x = np.linspace(0, 1, 50); t = np.linspace(0, 2, 50)
            T_field = solve_heat_analytical(x, t, alpha=alpha, T0=100.0)
            xx, tt = np.meshgrid(x, t)
            X, Tt, Y = xx.ravel(), tt.ravel(), T_field.ravel()
            x_m,x_s = X.mean(),X.std(); t_m,t_s = Tt.mean(),Tt.std()
            y_m,y_s = Y.mean(),Y.std()+1e-8
            all_coords = torch.tensor(
                np.stack([(X-x_m)/x_s,(Tt-t_m)/t_s],axis=1),dtype=torch.float32)
            all_targets = torch.tensor(
                ((Y-y_m)/y_s).reshape(-1,1),dtype=torch.float32)
            pde = HeatPDE(alpha=alpha)
            full_y_true = all_targets[:,0].numpy()
            domain_label = "Heat diffusion"; colloc_ranges = None; output_idx = 0

        elif domain_key == "wave":
            from domains.wave import WavePDE, solve_wave_analytical
            x = np.linspace(0,1,50); t = np.linspace(0,2,50)
            U_field = solve_wave_analytical(x,t,c=1.0,mode=1,amplitude=1.0)
            xx,tt = np.meshgrid(x,t)
            X,Tt,Y = xx.ravel(),tt.ravel(),U_field.ravel()
            x_m,x_s=X.mean(),X.std(); t_m,t_s=Tt.mean(),Tt.std()
            y_m,y_s=Y.mean(),Y.std()+1e-8
            all_coords = torch.tensor(
                np.stack([(X-x_m)/x_s,(Tt-t_m)/t_s],axis=1),dtype=torch.float32)
            all_targets = torch.tensor(
                ((Y-y_m)/y_s).reshape(-1,1),dtype=torch.float32)
            pde = WavePDE(c=1.0)
            full_y_true = all_targets[:,0].numpy()
            domain_label = "Wave propagation"; colloc_ranges = None; output_idx = 0

        elif domain_key == "gravity":
            from domains.gravity import GravityPDE, generate_orbit_data
            t,x,y,period = generate_orbit_data(GM=1.0,a=1.0,e=0.3,n_points=200,n_periods=1)
            t_m,t_s = t.mean(),t.std()
            x_m,x_s = x.mean(),x.std(); y_m,y_s = y.mean(),y.std()
            all_coords = torch.tensor(((t-t_m)/t_s).reshape(-1,1),dtype=torch.float32)
            all_targets = torch.tensor(
                np.stack([(x-x_m)/x_s,(y-y_m)/y_s],axis=1),dtype=torch.float32)
            pde = GravityPDE(GM=1.0)
            full_y_true = all_targets[:,0].numpy()
            domain_label = "Orbital mechanics"; colloc_ranges=[(-2,2)]; output_idx=0

        elif domain_key == "elasticity":
            from domains.elasticity import ElasticityPDE, solve_beam_analytical
            x = np.linspace(0,1,100)
            v = solve_beam_analytical(x,L=1.0,EI=1.0,q0=1.0)
            x_m,x_s = x.mean(),x.std(); v_m,v_s = v.mean(),v.std()+1e-8
            all_coords = torch.tensor(((x-x_m)/x_s).reshape(-1,1),dtype=torch.float32)
            all_targets = torch.tensor(((v-v_m)/v_s).reshape(-1,1),dtype=torch.float32)
            pde = ElasticityPDE(EI=1.0,q0=1.0)
            full_y_true = all_targets[:,0].numpy()
            domain_label = "Beam elasticity"; colloc_ranges=[(-2,2)]; output_idx=0

        else:
            return dbc.Alert(
                f"Sparsity sweep and MC Dropout not yet configured for '{domain_key}'. "
                "Select fluids, heat, wave, gravity, or elasticity.",
                color="warning")

    except Exception as e:
        return dbc.Alert(f"Data generation error: {e}", color="danger")

    # ─────────────────────────────────────────────────────────────────────────
    # SPARSITY SWEEP
    # ─────────────────────────────────────────────────────────────────────────
    if triggered_id == "sparsity-btn":
        fracs = sorted(fractions or [0.05, 0.10, 0.20, 0.50, 1.00])
        try:
            results = run_sparsity_sweep(
                pde, all_coords, all_targets,
                all_coords, full_y_true,
                fractions=fracs,
                n_epochs=min(n_epochs, 1000),
                n_seeds=n_seeds or 2,
                output_idx=output_idx,
                colloc_ranges=colloc_ranges,
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

        # Find crossover
        crossover_msg = ""
        for i, (p, m) in enumerate(zip(results["pinn_r2"], results["mlp_r2"])):
            if p > m:
                crossover_msg = (f"PINN outperforms vanilla MLP at "
                                 f"{frac_pct[i]:.0f}% data and below.")
                break

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
            dbc.Alert(crossover_msg or "No PINN advantage detected at tested fractions.",
                      color="success" if crossover_msg else "warning",
                      className="small"),
            dcc.Graph(figure=fig),
            dcc.Graph(figure=fig2),
            html.H6("Results table", className="mt-3"),
            dbc.Table.from_dataframe(df, striped=True, bordered=True,
                                     hover=True, size="sm"),
        ])

    # ─────────────────────────────────────────────────────────────────────────
    # MC DROPOUT UNCERTAINTY
    # ─────────────────────────────────────────────────────────────────────────
    elif triggered_id == "mcdrop-btn":
        try:
            mc_model, hist_mc = train_mc_dropout(
                pde, all_coords, all_targets,
                n_epochs=min(n_epochs, 1500),
                lr=1e-3, lambda_pde=0.05,
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
                line=dict(color="rgba(255,255,255,0)"), name="95% CI"))
            fig_mc.add_trace(go.Scatter(
                x=np.concatenate([x_s_arr, x_s_arr[::-1]]),
                y=np.concatenate([y_m_s+y_std_s, (y_m_s-y_std_s)[::-1]]),
                fill="toself", fillcolor="rgba(83,74,183,0.30)",
                line=dict(color="rgba(255,255,255,0)"), name="68% CI"))
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
                z=y_std_grid * 2,   # 95% CI width
                colorscale="Oranges", showscale=True,
                contours=dict(z=dict(show=True, usecolormap=True, project_z=True))
            )])
            fig_mc.update_layout(
                scene=dict(
                    xaxis_title="x", yaxis_title="t",
                    zaxis_title="95% CI width",
                    camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
                ),
                title=f"Epistemic uncertainty — {domain_label} "
                      f"({int(mc_samples)} samples, p={dropout_p})",
                height=450,
                paper_bgcolor="rgba(0,0,0,0)",
            )

        # Uncertainty width histogram
        fig_hist = go.Figure(data=[go.Histogram(
            x=y_std * 2, nbinsx=40,
            marker_color="#EF9F27", opacity=0.8,
            name="95% CI width distribution"
        )])
        fig_hist.update_layout(
            title="Distribution of 95% confidence interval widths",
            xaxis_title="95% CI width (normalised units)",
            yaxis_title="Count", height=280,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )

        # Loss curve
        fig_loss = make_loss_curve_fig(hist_mc)

        mean_ci = float(y_std.mean() * 2)
        mean_r2 = float(1 - np.mean((y_true - y_mean)**2) /
                        (np.var(y_true) + 1e-8))

        return html.Div([
            html.H5("MC Dropout uncertainty quantification", className="mt-3"),
            metric_row([
                ("MC mean R²", f"{mean_r2:.4f}"),
                ("Mean 95% CI width", f"{mean_ci:.4f}"),
                ("MC samples", int(mc_samples)),
                ("Dropout p", dropout_p),
            ]),
            dbc.Alert(
                "Wider bands = higher epistemic uncertainty = where additional "
                "sensors or training data would most improve predictions. "
                "This is directly actionable for sensor network planning. "
                "Citation: Gal & Ghahramani, ICML 2016.",
                color="info", className="small"
            ),
            dcc.Graph(figure=fig_mc),
            dcc.Graph(figure=fig_hist),
            html.H6("Training loss", className="mt-3"),
            dcc.Graph(figure=fig_loss),
        ])

    return html.Div()
