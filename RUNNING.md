Quick reproducible run and notes

1) Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

2) Run the Dash app locally

```bash
python dash_app/app.py
# Open http://localhost:8050
```

3) Run command-line experiments

```bash
python run_experiment.py --domain fluids --river volta --module all --epochs 3000
python run_experiment.py --domain heat --epochs 2000
python run_experiment.py --domain wave --epochs 2500
python run_experiment.py --domain gravity --epochs 3000
python run_experiment.py --domain elasticity --epochs 2000
```

For `fluids`, `--module` selects the workflow; other domains currently
run a single domain-specific experiment.

4) Run tests

```bash
pytest -q
```

Reproducibility notes:
- Use fixed random seeds in experiments (see `data/generator.py` default seeds).
- For paper figures, run `run_experiment.py` which saves results into `results/`.
