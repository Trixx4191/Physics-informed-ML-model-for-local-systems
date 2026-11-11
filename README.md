# PINN Multi-Physics Explorer

An interactive Dash workspace for physics-informed neural networks. The application includes river flow, dam and reservoir dynamics, heat diffusion, wave propagation, orbital motion and beam bending, using synthetic reference data.

[Live application](https://d203piwre8zpf0.cloudfront.net) · [Research paper](docs/Alhassan_Salifu_Babamu_PINN_Multiphysics_Paper.pdf)

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python dash_app/app.py
```

Open `http://localhost:8050`. The **Open workspace** control opens the dashboard at `/workspace`.

## Command-line experiments

```bash
python run_experiment.py --domain heat --epochs 2000
```

The `--domain` options are `fluids`, `dam`, `heat`, `wave`, `gravity` and `elasticity`. For fluids, `--module` selects forward prediction, a sparsity sweep, inverse roughness estimation or Monte Carlo Dropout. Experiment outputs are saved under `results/`.

These workflows use synthetic references. Their results do not establish field performance, and dropout spread is not a calibrated confidence interval. See the research paper for the scope and results of its separate comparative benchmark.

## Tests

```bash
python -m pip install -r requirements-dev.txt
pytest -q
```

## Code and documentation

- `core/` contains the shared PINN training engine.
- `domains/`, `models/` and `data/` define physical systems, networks and synthetic data.
- `dash_app/` contains the interactive application.
- `experiments/` and `run_experiment.py` provide experiment workflows.
- `threejs_orbit/` contains the standalone orbit visualisation.

[RUNNING.md](RUNNING.md) describes workflow settings and cached runs. [DEPLOYMENT.md](DEPLOYMENT.md) covers deployment. The project is released under the [MIT licence](LICENSE).
