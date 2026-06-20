# PINN Multi-Physics Explorer

Physics-Informed Neural Networks across multiple 1D domains (fluids, heat,
waves, elasticity, gravity, dam/reservoir). Penn-level research project —
multi-concept framework for PINNs.

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

## Domain presets
Example presets (fluids/dam): Volta (Ghana) · Amazon (Brazil) · Rhine (Germany) · Generic dam · Custom

## Project structure

```
pinn_multiphysics/
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

## Examples & tests

There are quick reproducible examples and tests included to validate the
multi-domain capability:

- `examples/reproduce_heat.py` — small heat PINN training that saves a
	checkpoint and a figure in `results/`.
- `tests/` — pytest smoke tests for the data generator and the PINN engine
	(heat and wave small training runs).

### Standalone experiment runner

The CLI runner can execute fluids/dam experiments and the newer physics
modules directly from the command line:

```bash
python run_experiment.py --domain fluids --river volta --module all --epochs 3000
python run_experiment.py --domain dam --epochs 3000
python run_experiment.py --domain heat --epochs 2000
python run_experiment.py --domain wave --epochs 2500
python run_experiment.py --domain gravity --epochs 3000
python run_experiment.py --domain elasticity --epochs 2000
```

For the `fluids` domain, use `--module` to choose between `all`,
`forward`, `sparsity`, `inverse`, `dam`, and `uncertainty`.

Run them locally as described in `RUNNING.md`.

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
