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
    make_dam_loss_curve_fig
)
from data.generator import RIVERS

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY],
               suppress_callback_exceptions=True)
app.title = "PINN Multi-Physics Explorer"


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
    ])
], className="h-100")

main_panel = html.Div([
    html.Div(id="equation-display", className="mb-3"),
    dcc.Loading(
        id="loading-train",
        type="circle",
        children=html.Div(id="results-area")
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
        ])
    if domain_key == "elasticity":
        return html.Div([
            html.Label("Load type", className="fw-bold small mt-2"),
            dcc.Dropdown(id="param-load", options=[
                {"label": "Uniform load", "value": "uniform"},
                {"label": "Point load (midspan)", "value": "point"},
            ], value="uniform", clearable=False),
        ])
    if domain_key == "dam":
        return html.Div([
            html.P("Coupled reservoir mass-balance ODE + downstream "
                  "Saint-Venant reach. No extra params — uses the "
                  "generic dam preset.", className="text-muted small mt-2"),
        ])
    if domain_key == "inverse_fluids":
        river_options = [{"label": v["name"], "value": k}
                         for k, v in RIVERS.items() if k != "dam"]
        return html.Div([
            html.Label("River (sets true n, hidden from the model)",
                      className="fw-bold small mt-2"),
            dcc.Dropdown(id="param-river", options=river_options,
                        value="volta", clearable=False),
            html.P("The PINN never sees this value — it infers n(x) "
                  "purely from sparse h,u observations and the physics "
                  "residual.", className="text-muted small mt-2"),
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
    prevent_initial_call=True,
)
def run_training(n_clicks, domain_key, fraction_pct, epochs, lambda_pde,
                 river_key, alpha_val, c_val, mode_val, e_val, load_val):
    fraction = fraction_pct / 100.0

    try:
        if domain_key == "fluids":
            data = generate_domain_data("fluids", river_key=river_key or "volta")
            model, hist, stats, extra = train_domain(
                "fluids", data, fraction=fraction, n_epochs=epochs, lambda_pde=lambda_pde)

            fig_true, fig_pred = make_3d_surface_pair(
                data["x"]/1000, data["t"]/3600, data["h"], extra["h_pred"],
                x_label="Distance (km)", t_label="Time (hr)", z_label="Depth h (m)",
                title_true="Ground truth h(x,t)", title_pred="PINN prediction")

            content = html.Div([
                metric_row([("R²", f"{extra['r2']:.4f}"),
                           ("Training pts", extra["n_train"]),
                           ("River", data["cfg"]["name"])]),
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig_true), width=6),
                    dbc.Col(dcc.Graph(figure=fig_pred), width=6),
                ]),
                html.H6("Training loss", className="mt-3"),
                dcc.Graph(figure=make_loss_curve_fig(hist)),
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
            data = generate_domain_data("gravity", eccentricity=e_val or 0.3, GM=1.0)
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
                           ("Eccentricity", data["e"])]),
                dcc.Graph(figure=fig_orbit),
                html.H6("Training loss", className="mt-3"),
                dcc.Graph(figure=make_loss_curve_fig(hist)),
            ])
            return content, f"Orbit reconstructed from {fraction_pct}% of trajectory"

        if domain_key == "elasticity":
            data = generate_domain_data("elasticity", load_type=load_val or "uniform", EI=1.0, q0=1.0)
            model, hist, stats, extra = train_domain(
                "elasticity", data, fraction=fraction, n_epochs=epochs, lambda_pde=lambda_pde)

            n_show = extra["n_train"]
            train_idx = np.random.RandomState(0).choice(len(data["x"]), n_show, replace=False)
            fig_beam = make_beam_3d(data["x"], data["v"], extra["v_pred"], train_idx)

            content = html.Div([
                metric_row([("R²", f"{extra['r2']:.4f}"),
                           ("Training pts", extra["n_train"]),
                           ("Load type", data["load_type"])]),
                dcc.Graph(figure=fig_beam),
                html.H6("Training loss", className="mt-3"),
                dcc.Graph(figure=make_loss_curve_fig(hist)),
            ])
            return content, f"Beam deflection trained — {epochs} epochs"

        if domain_key == "dam":
            data = generate_domain_data("dam")
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
            data = generate_domain_data("inverse_fluids", river_key=river_key or "volta")
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
    app.run(debug=True, host="0.0.0.0", port=8050)


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
