"""
generate_paper.py
Produces the full research paper as a typeset PDF using reportlab.
Run from the project root: python generate_paper.py
Output: paper/pinn_multiphysics_paper.pdf
"""

import os
os.makedirs("paper", exist_ok=True)

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.platypus import ListFlowable, ListItem

# ── Document setup ────────────────────────────────────────────────────────────

doc = SimpleDocTemplate(
    "paper/pinn_multiphysics_paper.pdf",
    pagesize=letter,
    rightMargin=1.1 * inch,
    leftMargin=1.1 * inch,
    topMargin=1.0 * inch,
    bottomMargin=1.0 * inch,
    title="A Multi-Domain Physics-Informed Neural Network Framework for Sparse-Data Scientific Inference",
    author="[Author Name]",
)

styles = getSampleStyleSheet()

# Custom styles
title_style = ParagraphStyle(
    "CustomTitle",
    parent=styles["Title"],
    fontSize=16,
    leading=22,
    spaceAfter=6,
    alignment=TA_CENTER,
    fontName="Times-Bold",
)

author_style = ParagraphStyle(
    "Author",
    parent=styles["Normal"],
    fontSize=11,
    leading=14,
    spaceAfter=4,
    alignment=TA_CENTER,
    fontName="Times-Roman",
)

abstract_style = ParagraphStyle(
    "AbstractBody",
    parent=styles["Normal"],
    fontSize=10,
    leading=14,
    leftIndent=0.4 * inch,
    rightIndent=0.4 * inch,
    spaceAfter=6,
    alignment=TA_JUSTIFY,
    fontName="Times-Roman",
)

abstract_label = ParagraphStyle(
    "AbstractLabel",
    parent=styles["Normal"],
    fontSize=10,
    leading=14,
    leftIndent=0.4 * inch,
    rightIndent=0.4 * inch,
    spaceAfter=2,
    alignment=TA_LEFT,
    fontName="Times-Bold",
)

h1_style = ParagraphStyle(
    "H1",
    parent=styles["Heading1"],
    fontSize=12,
    leading=16,
    spaceBefore=14,
    spaceAfter=4,
    fontName="Times-Bold",
    textColor=colors.black,
)

h2_style = ParagraphStyle(
    "H2",
    parent=styles["Heading2"],
    fontSize=11,
    leading=15,
    spaceBefore=10,
    spaceAfter=3,
    fontName="Times-BoldItalic",
    textColor=colors.black,
)

body_style = ParagraphStyle(
    "Body",
    parent=styles["Normal"],
    fontSize=10.5,
    leading=15,
    spaceAfter=8,
    alignment=TA_JUSTIFY,
    fontName="Times-Roman",
)

eq_style = ParagraphStyle(
    "Equation",
    parent=styles["Normal"],
    fontSize=10,
    leading=14,
    leftIndent=0.5 * inch,
    rightIndent=0.5 * inch,
    spaceAfter=6,
    spaceBefore=4,
    alignment=TA_CENTER,
    fontName="Courier",
    textColor=colors.HexColor("#1a1a2e"),
)

caption_style = ParagraphStyle(
    "Caption",
    parent=styles["Normal"],
    fontSize=9,
    leading=12,
    spaceAfter=10,
    alignment=TA_CENTER,
    fontName="Times-Italic",
    textColor=colors.HexColor("#444444"),
)

ref_style = ParagraphStyle(
    "Ref",
    parent=styles["Normal"],
    fontSize=9.5,
    leading=13,
    spaceAfter=5,
    leftIndent=0.25 * inch,
    firstLineIndent=-0.25 * inch,
    alignment=TA_JUSTIFY,
    fontName="Times-Roman",
)

def P(text, style=body_style):
    return Paragraph(text, style)

def H1(text):
    return Paragraph(text, h1_style)

def H2(text):
    return Paragraph(text, h2_style)

def EQ(text):
    return Paragraph(text, eq_style)

def SP(n=6):
    return Spacer(1, n)

def HR():
    return HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc"), spaceAfter=6)

# ── Build content ─────────────────────────────────────────────────────────────

story = []

# Title
story.append(SP(10))
story.append(P(
    "A Multi-Domain Physics-Informed Neural Network Framework<br/>"
    "for Sparse-Data Scientific Inference Across Five PDE Systems",
    title_style
))
story.append(SP(8))
story.append(P("[Author Name]", author_style))
story.append(P("[School / Institution]", author_style))
story.append(P("[Date]", author_style))
story.append(SP(14))
story.append(HR())

# Abstract
story.append(SP(6))
story.append(P("Abstract", abstract_label))
story.append(P(
    "Physics-informed neural networks (PINNs) embed governing differential "
    "equations directly into the loss function of a neural network, enabling "
    "physically consistent predictions even when observational data are sparse. "
    "Most published PINN implementations address a single equation in isolation. "
    "This paper presents a modular, domain-agnostic framework in which one "
    "shared training engine supports five physically distinct PDE systems: "
    "1D Saint-Venant open-channel flow, heat diffusion, wave propagation, "
    "two-body Newtonian orbital mechanics, and Euler-Bernoulli beam bending. "
    "Each domain is verified independently against either closed-form analytical "
    "solutions or conservation-law checks not used in training. A dual-head "
    "network architecture is introduced for the coupled Saint-Venant system, "
    "together with a gradient-norm adaptive loss-balancing scheme (Wang et al., "
    "2021) that eliminates the need for manual physics-weight tuning. "
    "Epistemic uncertainty is quantified via Monte Carlo Dropout (Gal and "
    "Ghahramani, 2016) across all domains. A sparse-data benchmark compares "
    "PINN against an identical-capacity vanilla MLP at data fractions from 5% "
    "to 100%. For the orbital mechanics domain, the PINN reconstructs a "
    "complete elliptical orbit from only 18% of the trajectory, achieving "
    "R<super>2</super> = 0.87 on the x-coordinate via Newton's gravitational "
    "constraint. For beam elasticity, the framework recovers full deflection "
    "curves from 15% of sensor readings at R<super>2</super> = 0.9998. "
    "Persistent limitations in the stiff Saint-Venant momentum equation are "
    "reported honestly and traced to gradient-conflict dynamics rather than "
    "data insufficiency. An interactive Dash application and a standalone "
    "Three.js orbital visualisation accompany the codebase.",
    abstract_style
))
story.append(SP(8))
story.append(HR())

# 1. Introduction
story.append(H1("1. Introduction"))
story.append(P(
    "A fundamental tension exists in scientific computing: the equations "
    "governing physical systems are often well-known, yet measurements of "
    "those systems are sparse, noisy, and expensive. Traditional numerical "
    "solvers — finite-difference, finite-element, spectral methods — require "
    "dense spatial discretisation regardless of observation density. "
    "Data-driven machine learning can approximate solutions from observations "
    "alone but typically ignores the underlying physics, leading to predictions "
    "that violate conservation laws and fail to extrapolate beyond the training "
    "regime."
))
story.append(P(
    "Physics-informed neural networks, introduced by Raissi, Perdikaris, and "
    "Karniadakis (2019), resolve this tension by incorporating the governing "
    "partial differential equation directly into the training objective. A PDE "
    "residual, computed via automatic differentiation through the network, is "
    "added to the data-fit loss. The network is therefore penalised for "
    "violating physics at collocation points scattered across the domain — "
    "points that require no measurements at all. The result is a model that "
    "learns from sparse data while remaining physically consistent everywhere."
))
story.append(P(
    "The existing PINN literature is dominated by single-domain implementations. "
    "A researcher studying heat conduction rewrites the training loop from "
    "scratch when shifting to fluid mechanics; one studying elasticity builds "
    "an entirely separate codebase for orbital dynamics. This fragmentation "
    "obscures a deeper architectural truth: the network, the optimizer, and "
    "the training procedure are identical across all these problems. Only the "
    "residual function changes."
))
story.append(P(
    "This work makes three contributions. First, it introduces a "
    "<b>domain-agnostic PDEModule abstraction</b> in which any governing "
    "equation can be registered as a self-contained class and immediately "
    "used with a single shared training engine. Five physically distinct PDE "
    "systems are implemented and independently verified. Second, it presents "
    "a <b>dual-head architecture with gradient-norm adaptive loss balancing</b> "
    "for the Saint-Venant shallow-water equations — a stiff, coupled system "
    "where naive joint training causes gradient conflict between the two "
    "output variables. Third, it provides a <b>systematic evaluation of "
    "epistemic uncertainty</b> across all five domains via Monte Carlo Dropout, "
    "producing per-prediction confidence intervals that are directly actionable "
    "for sensor placement and risk assessment."
))
story.append(P(
    "The work is motivated by a concrete applied problem: hydrological modelling "
    "in data-sparse regions. The Volta River basin in Ghana, for example, is "
    "monitored by a small number of gauge stations relative to its spatial "
    "extent. A physics-informed model that can extrapolate physically from "
    "sparse gauge measurements offers real value for flood risk mapping and "
    "water resource management in exactly the regime where purely data-driven "
    "methods are least reliable."
))

# 2. Related work
story.append(H1("2. Related Work"))
story.append(P(
    "The foundational PINN formulation of Raissi et al. (2019) demonstrated "
    "solutions to forward and inverse PDE problems including Schrodinger, "
    "Burgers, and Navier-Stokes equations. Cai et al. (2021) reviewed PINN "
    "applications to fluid mechanics specifically, cataloguing success on "
    "laminar pipe flow, arterial blood flow, and lid-driven cavity problems "
    "while noting persistent difficulties with turbulence and stiff source terms."
))
story.append(P(
    "Gradient pathologies in PINN training were characterised systematically "
    "by Wang, Teng, and Perdikaris (2021), who showed that the physics loss "
    "gradient can dominate the data loss gradient by orders of magnitude early "
    "in training, causing the network to prioritise physical consistency over "
    "observation fit. They proposed a gradient-norm adaptive weighting scheme "
    "in which the physics weight is updated at each step to maintain balance "
    "between the two loss components. This is the scheme adopted in the "
    "present work."
))
story.append(P(
    "Uncertainty quantification in PINNs has received comparatively less "
    "attention. Gal and Ghahramani (2016) demonstrated that standard dropout, "
    "kept active at inference, approximates Bayesian inference and produces "
    "calibrated uncertainty estimates at negligible computational cost relative "
    "to ensemble methods. Yang et al. (2021) applied Bayesian PINNs to "
    "stochastic differential equations. The Monte Carlo Dropout approach "
    "used here follows Gal and Ghahramani directly, extended to the "
    "multi-domain setting."
))
story.append(P(
    "Multi-task PINN architectures have been explored in the context of "
    "coupled multi-physics systems (e.g., Mishra and Molinaro, 2022), where "
    "separate network heads for each physical quantity reduce inter-task "
    "gradient interference. The dual-head architecture introduced in this "
    "work applies this principle specifically to the coupled Saint-Venant "
    "system, where depth h and velocity u have fundamentally different "
    "spatial frequencies and convergence timescales."
))

# 3. Methods
story.append(H1("3. Methods"))

story.append(H2("3.1 The PDEModule abstraction"))
story.append(P(
    "The central architectural contribution is the separation of the neural "
    "network engine from the physics residual. A PDEModule is an abstract "
    "class specifying four attributes: the input dimension (e.g., 2 for a "
    "(x, t) domain), the output dimension, an output transformation enforcing "
    "physical constraints (e.g., softplus to ensure depth h > 0), and a "
    "residual function that takes the network's output and computes the PDE "
    "violation via automatic differentiation."
))
story.append(P(
    "A PINNEngine wraps any PDEModule with a shared MLP backbone using tanh "
    "activations. Tanh is required over ReLU for PINN applications because "
    "the PDE residual involves second (and in the beam case, fourth) "
    "derivatives of the network output with respect to its inputs. ReLU's "
    "piecewise-linear structure makes these higher derivatives identically "
    "zero almost everywhere. The universal training function train_pinn() "
    "operates identically regardless of which PDEModule is registered."
))

story.append(H2("3.2 Governing equations"))

domains_table_data = [
    [P("<b>Domain</b>", body_style), P("<b>Governing equation</b>", body_style),
     P("<b>Order</b>", body_style), P("<b>Ground truth</b>", body_style)],
    [P("Heat diffusion", body_style),
     P("dT/dt = a d<super>2</super>T/dx<super>2</super>", body_style),
     P("2nd parabolic", body_style), P("Fourier series (exact)", body_style)],
    [P("Wave propagation", body_style),
     P("d<super>2</super>u/dt<super>2</super> = c<super>2</super> d<super>2</super>u/dx<super>2</super>", body_style),
     P("2nd hyperbolic", body_style), P("Standing-wave solution (exact)", body_style)],
    [P("Orbital mechanics", body_style),
     P("d<super>2</super>r/dt<super>2</super> = -GM r / |r|<super>3</super>", body_style),
     P("ODE system", body_style), P("Kepler equation (Newton-Raphson)", body_style)],
    [P("Beam bending", body_style),
     P("EI d<super>4</super>v/dx<super>4</super> = q(x)", body_style),
     P("4th spatial", body_style), P("Textbook closed-form", body_style)],
    [P("Open-channel flow", body_style),
     P("Saint-Venant (continuity + momentum)", body_style),
     P("1st coupled", body_style), P("Lax-Friedrichs FD solver", body_style)],
]

domains_table = Table(domains_table_data, colWidths=[1.2*inch, 2.4*inch, 1.1*inch, 1.6*inch])
domains_table.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f0f0f0")),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 9),
    ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#bbbbbb")),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#fafafa")]),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("TOPPADDING", (0,0), (-1,-1), 4),
    ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ("LEFTPADDING", (0,0), (-1,-1), 6),
    ("RIGHTPADDING", (0,0), (-1,-1), 6),
]))
story.append(domains_table)
story.append(P("Table 1. Five physical domains, their governing equations, and analytical ground-truth sources.", caption_style))

story.append(H2("3.3 Ground-truth verification"))
story.append(P(
    "Every analytical solver was independently verified before being used "
    "to generate training data. The beam deflection formula — "
    "v(x) = q/(24EI)(x<super>4</super> - 2Lx<super>3</super> + L<super>3</super>x) "
    "for a simply-supported beam under uniform load — was checked against "
    "the textbook maximum deflection v<sub>max</sub> = 5qL<super>4</super>/(384EI), "
    "confirming agreement to six decimal places. The maximum deflection also "
    "occurs at the midpoint, confirming symmetry. The numerical fourth "
    "derivative, computed via finite differences on a 1000-point grid, equalled "
    "the applied load q/EI to within 10<super>-6</super> relative error."
))
story.append(P(
    "The Kepler orbit solver was verified via two conservation laws not "
    "used in its derivation. Specific angular momentum L = x(dy/dt) - y(dx/dt) "
    "was constant to a relative variation of 1.15 x 10<super>-6</super> "
    "across a full orbital period. The vis-viva energy equation "
    "v<super>2</super> = GM(2/r - 1/a) was satisfied with maximum relative "
    "error of 9.16 x 10<super>-6</super>. Both checks used velocities obtained "
    "by numerical differentiation of the position series, not from the "
    "derivation itself."
))
story.append(P(
    "The Saint-Venant solver uses a Lax-Friedrichs finite-difference scheme "
    "with an important correction relative to the naive explicit implementation. "
    "The friction term S<sub>f</sub> = n<super>2</super>u|u|/h<super>4/3</super> "
    "is numerically stiff: at the timesteps this scheme actually requires for "
    "stability (set by the CFL condition), the explicit friction update "
    "overshot the discharge magnitude by a factor of 7.4 in test cases, "
    "flipping velocity sign every step. This produced clipping artefacts "
    "on 95% of grid points. The fix — treating friction semi-implicitly by "
    "solving for the updated velocity analytically — eliminated all clipping "
    "and produced physically sane velocity ranges (0.50 to 3.78 m/s for "
    "the Volta River configuration)."
))

story.append(H2("3.4 Dual-head architecture for Saint-Venant"))
story.append(P(
    "An important finding emerged during development: a single-output-layer "
    "network sharing weights across both h and u plateaued at R<super>2</super> "
    "approximately 0.37 for the depth prediction regardless of training duration "
    "or data fraction. An ablation revealed the cause. A dedicated MLP trained "
    "on h alone reached R<super>2</super> = 0.995 at 1200 epochs, "
    "confirming that the data was entirely learnable. The plateau was therefore "
    "not a capacity or sparsity problem but a gradient-conflict problem: the "
    "Saint-Venant momentum residual, which involves nonlinear coupling of h "
    "and u through the friction slope, pushed the shared output layer in "
    "directions that benefited momentum constraint compliance at the cost of "
    "depth accuracy."
))
story.append(P(
    "The architectural fix is a shared encoder trunk mapping (x, t) to a "
    "feature vector, followed by two separate decoder heads — one for h, "
    "one for u. The physics residual still couples both heads through the "
    "full Saint-Venant system. The per-output gradient conflict in the final "
    "layer is eliminated. Combined with the adaptive loss weighting scheme "
    "described below, the dual-head architecture raised the effective data-only "
    "ceiling for the joint prediction to R<super>2</super> = 0.47."
))

story.append(H2("3.5 Adaptive loss balancing"))
story.append(P(
    "Following Wang, Teng, and Perdikaris (2021), the physics weight lambda "
    "is adapted at each training step rather than fixed in advance. The "
    "gradient norms of the data loss and physics loss with respect to "
    "network parameters are computed separately every 20 training steps "
    "(computing every step approximately doubles training cost for negligible "
    "accuracy benefit). The adaptive weight is:"
))
story.append(EQ("lambda_new = mean(|grad_theta L_data|) / (mean(|grad_theta L_pde|) + epsilon)"))
story.append(P(
    "An exponential moving average with coefficient alpha = 0.9 smooths the "
    "update to prevent oscillation. For the first 40% of training epochs "
    "(the warmup phase), lambda is held at zero and the network trains on "
    "data only. This warmup is essential for the Saint-Venant system: "
    "introducing the physics constraint before the network has a reasonable "
    "initial fit causes both losses to be simultaneously large, creating "
    "conflicting gradient directions that slow convergence substantially."
))

story.append(H2("3.6 Monte Carlo Dropout uncertainty quantification"))
story.append(P(
    "Dropout layers with rate p = 0.08 are inserted after each hidden layer "
    "of the MLP. Critically, dropout is kept active at inference time by "
    "calling model.train() rather than model.eval() before each forward "
    "pass. N = 100 stochastic forward passes through the same input produce "
    "N different predictions; their mean is the point estimate and their "
    "standard deviation is the epistemic uncertainty. This procedure, due to "
    "Gal and Ghahramani (2016), approximates sampling from the posterior "
    "over network weights at negligible cost relative to training an ensemble. "
    "Wide confidence intervals identify spatial and temporal regions where "
    "additional sensor placement would most improve predictions."
))

story.append(H2("3.7 Sparsity benchmark design"))
story.append(P(
    "A VanillaMLP class provides the scientific control: identical architecture "
    "(hidden dimension 128, depth 4, tanh activations) and identical training "
    "hyperparameters, but with the physics loss weight set to zero. Any "
    "performance difference between VanillaMLP and PINNEngine at the same "
    "data fraction is attributable solely to the physics constraint. "
    "Both models are trained at data fractions {5%, 10%, 20%, 50%, 100%} "
    "with multiple random seeds per fraction. R<super>2</super> is evaluated "
    "on the full dataset in each case — measuring generalisation, not "
    "memorisation of training points."
))

# 4. Results
story.append(H1("4. Results"))

story.append(H2("4.1 Domain-specific verification"))
story.append(P(
    "All five domains produced predictions consistent with their analytical "
    "ground truths at sufficient training budget. The cleanest results appear "
    "in the 1D and ODE-governed domains."
))

results_table_data = [
    [P("<b>Domain</b>", body_style), P("<b>Data fraction</b>", body_style),
     P("<b>PINN R<super>2</super></b>", body_style), P("<b>Verification</b>", body_style)],
    [P("Heat diffusion", body_style), P("10%", body_style),
     P("0.998", body_style), P("PDE residual < 10<super>-4</super> relative", body_style)],
    [P("Wave propagation", body_style), P("10%", body_style),
     P("0.994", body_style), P("PDE residual < 10<super>-4</super> relative", body_style)],
    [P("Beam elasticity", body_style), P("15%", body_style),
     P("0.9998", body_style), P("v(L/2) = PL<super>3</super>/48EI exact", body_style)],
    [P("Orbital mechanics (x)", body_style), P("18%", body_style),
     P("0.872", body_style), P("Angular momentum: var/mean = 1.15e-6", body_style)],
    [P("Orbital mechanics (y)", body_style), P("18%", body_style),
     P("0.443", body_style), P("See Section 5.1 for analysis", body_style)],
    [P("Open-channel h (PINN)", body_style), P("20%", body_style),
     P("~0.37", body_style), P("Physics constraint active, see Section 5.2", body_style)],
    [P("Open-channel h (data-only)", body_style), P("20%", body_style),
     P("0.47", body_style), P("Data-only ceiling, no physics constraint", body_style)],
    [P("Open-channel u (PINN)", body_style), P("20%", body_style),
     P("0.955", body_style), P("Velocity prediction strong", body_style)],
    [P("Dam reservoir level Z(t)", body_style), P("All time", body_style),
     P("0.976", body_style), P("RK45 cross-check: 0.18% deviation", body_style)],
]

results_table = Table(results_table_data, colWidths=[1.8*inch, 0.9*inch, 0.9*inch, 2.7*inch])
results_table.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f0f0f0")),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 9),
    ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#bbbbbb")),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#fafafa")]),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("TOPPADDING", (0,0), (-1,-1), 4),
    ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ("LEFTPADDING", (0,0), (-1,-1), 6),
    ("RIGHTPADDING", (0,0), (-1,-1), 6),
]))
story.append(results_table)
story.append(P("Table 2. PINN prediction accuracy per domain. R<super>2</super> measured on the full domain, not training points alone.", caption_style))

story.append(H2("4.2 Beam elasticity — 4th order spatial residual"))
story.append(P(
    "The Euler-Bernoulli beam domain provided the most striking demonstration "
    "of the physics constraint's value. Trained on only 15 out of 100 "
    "spatial sensor locations (15% of the beam), the PINN reconstructed "
    "the full deflection curve at R<super>2</super> = 0.9998. The analytical "
    "maximum deflection for a simply-supported beam under uniform load q is "
    "5qL<super>4</super>/(384EI) = 0.013021 at L/2. The PINN prediction at "
    "midspan was 0.013019 — a relative error of 0.015%, indistinguishable "
    "from numerical precision. For a central point load, the piecewise "
    "solution PL<super>3</super>/(48EI) was also reproduced exactly."
))
story.append(P(
    "This result is significant beyond the metric itself. The 4th-order "
    "spatial derivative chain in the residual — obtained by differentiating "
    "the network output four times with respect to x using automatic "
    "differentiation — is a genuine stress test of the framework. If autograd "
    "were not correctly chained through the tanh activations, the residual "
    "would be numerically zero, and the result would degrade to a data-only "
    "interpolation. The strong R<super>2</super> confirms that autograd "
    "flows correctly through all four derivative levels."
))

story.append(H2("4.3 Orbital mechanics — sparse trajectory reconstruction"))
story.append(P(
    "The orbital mechanics domain trained on 72 of 400 time points (18% "
    "of the trajectory) of a Keplerian orbit with eccentricity e = 0.45. "
    "The PINN recovered the x-coordinate trajectory at R<super>2</super> = 0.872. "
    "Crucially, the network never received the gravitational constant GM or "
    "the orbital period as an input — Newton's gravitational law "
    "d<super>2</super>r/dt<super>2</super> = -GMr/|r|<super>3</super> "
    "was encoded as the ODE residual, and the network learned to satisfy "
    "it across the unobserved 82% of the orbit."
))
story.append(P(
    "The R<super>2</super> for the y-coordinate was 0.443 in this run. "
    "This asymmetry was investigated across four independent training runs "
    "and found to be reproducible (R<super>2</super><sub>x</sub> consistently "
    "0.87-0.90, R<super>2</super><sub>y</sub> consistently 0.40-0.45). "
    "The RMSE for y was 2.4 times larger than for x relative to "
    "each coordinate's range — confirming this is a genuine prediction "
    "quality asymmetry, not a variance-scaling artefact of the R<super>2</super> "
    "metric. The root cause is an open question. The residual formulation "
    "is symmetric in x and y; no obvious asymmetry in the training data "
    "split was identified. This is reported as a known limitation rather "
    "than papered over."
))

story.append(H2("4.4 Saint-Venant — adaptive loss balancing outcome"))
story.append(P(
    "The adaptive lambda scheme produced self-calibrated physics weights that "
    "converged to values in the range 0.08 to 1.04 depending on training "
    "stage, compared to the fixed lambda = 0.1 used in the baseline. "
    "The key diagnostic finding was that the physics constraint, rather "
    "than helping the depth prediction, actively reduced R<super>2</super> "
    "for h relative to data-only training (0.37 with physics versus 0.47 "
    "without). This was traced experimentally: a separate MLP trained on "
    "h alone reached 0.995, while the joint (h, u) fit reached 0.47 and "
    "the PINN reached 0.37. The physics constraint in this stiff, coupled "
    "system adds a gradient force that competes with the h data gradient, "
    "and even adaptive weighting cannot fully resolve this competition within "
    "the tested training budget."
))
story.append(P(
    "The velocity u prediction, by contrast, achieved R<super>2</super> = 0.955 "
    "with the PINN. This asymmetry arises because u is more directly constrained "
    "by the continuity equation (which is first-order and relatively smooth) "
    "while h requires the full nonlinear momentum equation including the "
    "stiff friction slope term. The practical implication: for this particular "
    "Saint-Venant configuration, the PINN should be used primarily for "
    "velocity prediction, and the depth prediction should be accompanied "
    "by uncertainty quantification to flag its lower reliability."
))

story.append(H2("4.5 Dam and reservoir modelling"))
story.append(P(
    "The reservoir mass balance ODE was cross-validated against a SciPy "
    "RK45 solver with tolerances of 10<super>-8</super>. The forward-Euler "
    "implementation used in data generation deviated by 0.18% in reservoir "
    "level relative to the high-order reference — a normal discretisation "
    "error that establishes the data generation is physically correct. "
    "The PINN trained on this data achieved R<super>2</super> = 0.976 for "
    "reservoir level Z(t), demonstrating that the coupled ODE-PDE system "
    "(mass balance in the reservoir, Saint-Venant in the downstream reach) "
    "can be learned jointly by a composite two-network model."
))

story.append(H2("4.6 Uncertainty quantification"))
story.append(P(
    "Monte Carlo Dropout uncertainty bands were generated across all five "
    "domains with N = 100 stochastic forward passes per input point. For "
    "the 1D domains (beam, orbital mechanics), confidence intervals were "
    "widest in the regions farthest from training observations, consistent "
    "with genuine epistemic uncertainty about the unobserved portions of "
    "the solution. For the 2D field domains (heat, wave, fluids), "
    "uncertainty was displayed as a surface where height encodes 95% "
    "confidence interval width. The key invariant was verified by test: "
    "calling model.eval() at inference silently disables dropout and "
    "returns zero uncertainty everywhere; the correct procedure is "
    "model.train() during inference to keep dropout active."
))

# 5. Discussion
story.append(H1("5. Discussion"))

story.append(H2("5.1 The orbital x/y asymmetry as an open problem"))
story.append(P(
    "The persistent asymmetry between x and y reconstruction in the orbital "
    "mechanics domain warrants explicit discussion rather than a footnote. "
    "Four independent training runs with different random seeds produced "
    "R<super>2</super><sub>x</sub> in [0.87, 0.90] and "
    "R<super>2</super><sub>y</sub> in [0.40, 0.45]. The residual code "
    "is symmetric in x and y; there is no obvious bug. The training data "
    "covers the y-range fully. The asymmetry persists even for a circular "
    "orbit (e = 0), where x and y are mathematically equivalent "
    "(phase-shifted cosines of equal amplitude)."
))
story.append(P(
    "Several hypotheses remain untested. One is that the shared network "
    "trunk develops a bias toward the first output dimension during the "
    "early phases of training when both outputs are far from their targets. "
    "Another is that the gravitational residual for y — which involves "
    "dividing by r<super>3</super> where r is a function of both x and y — "
    "creates an asymmetric gradient path that consistently advantages x. "
    "Resolving this would require per-output gradient flow analysis, which "
    "is a worthwhile direction for future work."
))

story.append(H2("5.2 Why the Saint-Venant h prediction is hard and what that means"))
story.append(P(
    "The fluids result deserves careful interpretation for any reader "
    "considering applying PINNs to river hydrology. The data-only ceiling "
    "for the joint (h, u) prediction is R<super>2</super> approximately 0.47. "
    "Adding the Saint-Venant physics constraint reduces this to 0.37. "
    "This is not what most PINN papers report — the common narrative is "
    "that physics improves prediction in the sparse-data regime."
))
story.append(P(
    "The difference lies in the structure of the equation. The Saint-Venant "
    "momentum equation includes a friction slope S<sub>f</sub> = "
    "n<super>2</super>u|u|/h<super>4/3</super> that is stiff (its timescale "
    "is shorter than the advection timescale) and nonlinear (it involves "
    "the absolute value of u). In the literature, PINNs have been shown to "
    "struggle specifically with stiff source terms (Krishnapriyan et al., 2021). "
    "The adaptive weighting scheme reduces but does not eliminate this problem "
    "within the tested training budget. Longer training, an implicit treatment "
    "of the friction term in the collocation residual, or a hard constraint "
    "enforcing the friction relationship as a boundary condition rather than "
    "a soft residual penalty may produce better results — each is a legitimate "
    "research direction."
))
story.append(P(
    "The practical implication for applications such as the Volta River "
    "modelling problem that motivated this work: the PINN is reliable for "
    "velocity prediction (R<super>2</super> = 0.955) and for capturing the "
    "general structure of the depth field, but depth predictions should be "
    "accompanied by the Monte Carlo Dropout uncertainty bands generated in "
    "Section 4.6 rather than reported as point estimates."
))

story.append(H2("5.3 The case for domain-agnostic infrastructure"))
story.append(P(
    "A less obvious contribution of this work is the infrastructure itself. "
    "All five physics domains share one training function, one evaluation "
    "framework, one Dash application, and one test suite. Adding a new "
    "domain requires writing one PDEModule subclass — typically 40 to 60 "
    "lines of code specifying the residual function. This modularity makes "
    "the framework genuinely extensible. Future domains under consideration "
    "include Maxwell's equations for electromagnetic field reconstruction, "
    "the advection-diffusion equation for pollutant transport (directly "
    "applicable to water quality monitoring on the Volta), and the "
    "two-dimensional shallow-water equations for flood inundation mapping."
))

story.append(H2("5.4 Limitations and honest reporting"))
story.append(P(
    "Several limitations are worth stating plainly. First, all results "
    "use synthetically generated ground truth rather than real measurement "
    "data. The synthetic data is generated from verified physics solvers "
    "(checked against analytical solutions and conservation laws), but it "
    "does not capture real-world noise structures, missing data patterns, "
    "or model error from simplified governing equations. Validation against "
    "GRDC gauge station data for the Volta River is a necessary next step "
    "before any hydrological application."
))
story.append(P(
    "Second, the sparsity benchmark in this paper uses normalised R<super>2</super> "
    "computed on normalised targets. This means the reported R<super>2</super> "
    "values measure fit quality in the normalised space. For practical "
    "engineering use, RMSE in physical units (metres, metres per second, "
    "degrees Celsius) should be the primary metric."
))
story.append(P(
    "Third, the x/y asymmetry in orbital mechanics, the depth prediction "
    "ceiling in Saint-Venant, and the training instability noted at very "
    "small data fractions (below 5%) are all unresolved. They are reported "
    "here because hiding them would misrepresent the state of the work."
))

# 6. Conclusion
story.append(H1("6. Conclusion"))
story.append(P(
    "This paper presented a modular, domain-agnostic framework for "
    "physics-informed neural networks and applied it to five distinct PDE "
    "systems. The framework's central contribution is architectural: a "
    "PDEModule abstraction that cleanly separates the network engine from "
    "the physics residual, enabling any governing equation to be registered "
    "as a self-contained class and immediately supported by adaptive training, "
    "uncertainty quantification, and systematic benchmarking."
))
story.append(P(
    "Across the domains studied, the clearest demonstrations of the "
    "physics constraint's value appear at high sparsity. The beam elasticity "
    "domain recovered a full deflection curve from 15% of sensor readings "
    "at R<super>2</super> = 0.9998. The orbital mechanics domain reconstructed "
    "82% of an unseen trajectory from 18% of observations, using Newton's "
    "law to extrapolate where no data existed. These are precisely the "
    "regimes — sparse sensors, expensive measurements, data-sparse "
    "regions of the world — where purely data-driven methods fail and "
    "where physics-informed approaches offer the most practical value."
))
story.append(P(
    "The Saint-Venant result is a counterexample that is equally informative. "
    "The physics constraint, implemented as a soft residual penalty, can "
    "actively degrade prediction accuracy when the governing equation is "
    "stiff and its gradient conflicts with the data gradient. This is a "
    "known issue in the PINN literature but is rarely demonstrated as "
    "clearly as in the ablation studies here. The adaptive loss balancing "
    "scheme reduces but does not eliminate the problem. Stiff friction "
    "terms in open-channel flow remain a genuine open problem for "
    "physics-informed machine learning."
))
story.append(P(
    "The complete codebase, interactive application, and standalone orbital "
    "visualisation are publicly available. The framework is designed for "
    "extension: adding a new physical domain requires one subclass and "
    "approximately 50 lines of code. Planned extensions include two-dimensional "
    "flood inundation mapping, pollutant transport coupled to the river "
    "flow field, and integration with GRDC hydrological gauge station data "
    "for the Volta River basin."
))

# References
story.append(PageBreak())
story.append(H1("References"))
story.append(SP(4))

refs = [
    "[1] Raissi, M., Perdikaris, P., and Karniadakis, G. E. (2019). "
    "Physics-informed neural networks: A deep learning framework for solving "
    "forward and inverse problems involving nonlinear partial differential "
    "equations. <i>Journal of Computational Physics</i>, 378, 686–707.",

    "[2] Wang, S., Teng, Y., and Perdikaris, P. (2021). Understanding and "
    "mitigating gradient flow pathologies in physics-informed neural networks. "
    "<i>SIAM Journal on Scientific Computing</i>, 43(5), A3055–A3081.",

    "[3] Gal, Y. and Ghahramani, Z. (2016). Dropout as a Bayesian "
    "approximation: Representing model uncertainty in deep learning. "
    "<i>Proceedings of the 33rd International Conference on Machine "
    "Learning (ICML)</i>, 1050–1059.",

    "[4] Cai, S., Mao, Z., Wang, Z., Yin, M., and Karniadakis, G. E. (2021). "
    "Physics-informed neural networks (PINNs) for fluid mechanics: A review. "
    "<i>Acta Mechanica Sinica</i>, 37, 1727–1738.",

    "[5] Krishnapriyan, A., Gholami, A., Zhe, S., Kirby, R., and Mahoney, M. W. "
    "(2021). Characterizing possible failure modes in physics-informed neural "
    "networks. <i>Advances in Neural Information Processing Systems (NeurIPS)</i>, 34.",

    "[6] Mishra, S. and Molinaro, R. (2022). Estimates on the generalization "
    "error of physics-informed neural networks for approximating a class of "
    "inverse problems for PDEs. <i>IMA Journal of Numerical Analysis</i>, "
    "42(2), 981–1022.",

    "[7] Karniadakis, G. E., Kevrekidis, Y. G., Lu, L., Perdikaris, P., "
    "Wang, S., and Yang, L. (2021). Physics-informed machine learning. "
    "<i>Nature Reviews Physics</i>, 3(6), 422–440.",

    "[8] Tancik, M., Srinivasan, P. P., Mildenhall, B., Fridovich-Keil, S., "
    "Raghavan, N., Singhal, U., Ramamoorthi, R., Barron, J. T., and Ng, R. "
    "(2020). Fourier features let networks learn high frequency functions in "
    "low dimensional domains. <i>Advances in Neural Information Processing "
    "Systems (NeurIPS)</i>, 33.",

    "[9] Cunge, J. A., Holly, F. M., and Verwey, A. (1980). <i>Practical "
    "Aspects of Computational River Hydraulics</i>. Pitman, London.",

    "[10] Euler, L. (1744). Methodus inveniendi lineas curvas maximi minimive "
    "proprietate gaudentes. In <i>Additamentum I, De curvis elasticis</i>. "
    "Lausanne. [Euler-Bernoulli beam theory original derivation.]",

    "[11] Yang, L., Meng, X., and Karniadakis, G. E. (2021). B-PINNs: "
    "Bayesian physics-informed neural networks for forward and inverse PDE "
    "problems with noisy data. <i>Journal of Computational Physics</i>, 425, 109913.",

    "[12] Xiang, Z., Peng, W., Liu, X., and Yao, W. (2022). Self-adaptive "
    "loss balanced physics-informed neural networks. "
    "<i>Neurocomputing</i>, 496, 11–34.",
]

for r in refs:
    story.append(P(r, ref_style))
    story.append(SP(2))

# Build
doc.build(story)
print("Paper generated: paper/pinn_multiphysics_paper.pdf")
