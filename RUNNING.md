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

3) Run tests

```bash
pytest -q
```

Reproducibility notes:
- Use fixed random seeds in experiments (see `data/generator.py` default seeds).
- For paper figures, run `run_experiment.py` which saves results into `results/`.
