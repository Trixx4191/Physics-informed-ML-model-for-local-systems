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

The home page shows the supplied physics artwork beside the governing equations. **Open workspace** takes you to `/workspace`; this URL
also opens the scientific dashboard directly.

Workspace controls, the selected domain, and the selected tab survive refreshes
in the same browser session. Completed training and experiment results are
recovered from the dashboard's disk cache, and an active job continues while
you refresh or visit the home page. A missing worker is reported as interrupted;
refreshing never silently starts a replacement run. The workspace identifier is
scoped to the browser tab's session storage. Cached run records expire after seven days;
retain `DASH_CACHE_DIR` if results must survive a server/container restart.

The experiments use the selected domain parameters and physics weight.
MC Dropout trains on the selected observation fraction; the sparsity sweep uses
its own fractions checklist. R² compares predictions with the full synthetic
reference, including observed points. Dropout bands describe stochastic model
spread and are not calibrated confidence intervals.

The numerical path evaluates PDE derivatives and nonlinear terms in physical
units, using a normalization adapter around standardized network inputs and
outputs. Fluids retains its dual-head model and adaptive physics weighting;
λ₂ supplies the initial adaptive weight, with the first 40% of epochs used for
data warmup. Loss histories include the final epoch. Dam observations use the
selected fraction, and gate outflow drives the downstream reference boundary
and the model's coupling constraint.

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
- `run_experiment.py` saves workflow outputs into `results/`. The research paper describes its separate comparative benchmark.
