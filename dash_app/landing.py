"""The image-led physics poster at the entrance to the workspace."""

from dash import dcc, html


def _sub(base, index):
    return html.Span([base, html.Sub(index)], className="chalk-term")


def _fraction(numerator, denominator):
    return html.Span([
        html.Span(numerator, className="chalk-numerator"),
        html.Span(denominator, className="chalk-denominator"),
    ], className="chalk-fraction")


def _equation(children, label):
    return html.Div(children, className="chalk-equation", role="math",
                    **{"aria-label": label})


def _note(title, equation, label):
    return html.Section([
        html.H2(title),
        _equation(equation, label),
    ], className="chalk-note")


def build_landing_page():
    """Keep this page mounted alongside the dashboard during a running job."""
    return html.Div([
        html.Header([
            html.Span("PINN", className="poster-brand"),
            html.Span("MULTI-PHYSICS EXPLORER", className="poster-name"),
        ], className="poster-header"),
        html.Main([
            html.Figure([
                html.Img(
                    src="/assets/physics-poster.jpeg",
                    alt="The supplied physics poster: white orbital trails around a blue spiral and a bright centre.",
                    className="poster-image",
                    draggable="false",
                ),
            ], className="poster-art"),
            html.Div([
                html.H1("Physics-informed neural networks", className="poster-title"),
                html.Section([
                    html.H2("Saint-Venant"),
                    _equation([
                        _fraction("∂h", "∂t"), " + ", _fraction("∂(hu)", "∂x"), " = 0",
                    ], "Continuity: partial h over partial t plus partial h u over partial x equals zero."),
                    _equation([
                        _sub("(hu)", "t"), " + ",
                        _sub(html.Span(["(hu", html.Sup("2"), " + ½gh", html.Sup("2"), ")"]), "x"),
                        " = gh(", _sub("S", "0"), " − ", _sub("S", "f"), ")",
                    ], "Momentum: derivative of h u in time plus derivative of h u squared plus one half g h squared in space equals g h times bed slope minus friction slope."),
                ], className="chalk-note chalk-flow"),
                html.Div([
                    _note("Heat", [
                        _fraction("∂T", "∂t"), " = α ", _fraction(["∂", html.Sup("2"), "T"], ["∂x", html.Sup("2")]),
                    ], "Heat equation: partial T over partial t equals alpha times the second spatial derivative of T."),
                    _note("Waves", [
                        _fraction(["∂", html.Sup("2"), "u"], ["∂t", html.Sup("2")]),
                        " = c", html.Sup("2"), " ",
                        _fraction(["∂", html.Sup("2"), "u"], ["∂x", html.Sup("2")]),
                    ], "Wave equation: the second time derivative of u equals c squared times the second spatial derivative of u."),
                    _note("Gravity", [
                        "r̈ = −", _fraction("GM r", ["|r|", html.Sup("3")]),
                    ], "Orbital acceleration equals minus G M r divided by the magnitude of r cubed."),
                    _note("Elasticity", [
                        "EI ", _fraction(["d", html.Sup("4"), "v"], ["dx", html.Sup("4")]), " = q(x)",
                    ], "Euler Bernoulli beam: E I times the fourth spatial derivative of v equals q of x."),
                    _note("Dam & reservoir", [
                        "A ", _fraction("dZ", "dt"), " = ", _sub("Q", "in"), " − ", _sub("Q", "out"), "(Z)",
                    ], "Reservoir mass balance: area times change in level equals inflow minus outflow."),
                    _note("Inverse fluids", [
                        _sub("S", "f"), " = ",
                        _fraction(["n(x)", html.Sup("2"), "u|u|"], ["h", html.Sup("4/3")]),
                    ], "Friction slope equals learned Manning roughness n of x squared times u absolute u divided by h to the four thirds."),
                ], className="chalk-equations"),
                _equation([
                    "ℒ = ", _sub("ℒ", "data"), " + ", _sub("λ", "2"), " ", _sub("ℒ", "physics"),
                ], "Training objective: data loss plus lambda two times physics loss."),
                dcc.Link([
                    "Open workspace", html.Span("→", **{"aria-hidden": "true"}),
                ], href="/workspace", refresh=False, className="poster-workspace-link"),
            ], className="poster-notes"),
        ], className="physics-poster"),
    ], id="landing-page", className="physics-landing")
