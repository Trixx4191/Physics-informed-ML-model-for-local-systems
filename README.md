# PINN River Modeller

Physics-Informed Neural Networks for 1D river hydrodynamics.
Penn-level research project — three-tier contribution.

## Three modules

| Module | What it does | Physics |
|--------|-------------|---------|
| **Forward prediction** | Predict h(x,t) and u(x,t) from sparse sensor data | 1D Saint-Venant PDEs |
| **Dam & reservoir** | Model reservoir level Z(t) + downstream reach | Mass-balance ODE + Saint-Venant |
| **Inverse: infer roughness** | Back-calculate Manning's n(x) from flow observations | Saint-Venant + spatial regularisation |

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the web app
python dash_app/app.py
```

## River presets
Volta (Ghana) · Amazon (Brazil) · Rhine (Germany) · Generic dam · Custom

## Project structure

```
pinn_river/
├── models/
│   └── pinn_core.py          # ForwardPINN, InversePINN, DamPINN
├── data/
│   └── generator.py          # Lax-Friedrichs Saint-Venant solver + sparse sampler
├── experiments/
│   └── trainer.py            # Training loops for all three modules
├── dash_app/
│   └── app.py                # Web UI (Dash)
└── requirements.txt
```

## Governing equations

**1D Saint-Venant (continuity + momentum):**
```
∂h/∂t + ∂(uh)/∂x = 0
∂(uh)/∂t + ∂(u²h + g·h²/2)/∂x + g·h·(Sf − S0) = 0
```
where friction slope `Sf = n²·u·|u| / h^(4/3)`

**Reservoir mass balance (dam module):**
```
A · dZ/dt = Q_in(t) - Q_out(Z)
Q_out = Cd · W · √g · Z^1.5   (broad-crested weir)
```

**Physics loss (PINN training):**
```
L = λ₁·MSE(data) + λ₂·MSE(PDE residuals at collocation points)
```

## Key result
The sparsity benchmark module produces the central figure:
PINN vs vanilla MLP R² as a function of training data fraction (5%→100%).
The crossover point — where PINN outperforms — is the main scientific contribution.

## References
- Raissi, M., Perdikaris, P., Lagaris, G.E. (2019). Physics-informed neural networks. *Journal of Computational Physics*, 378, 686–707.
- Shi et al. (2020). Saint-Venant equations for river flow.
- Cunge, J.A., Holly, F.M., Verwey, A. (1980). *Practical Aspects of Computational River Hydraulics*. Pitman.
