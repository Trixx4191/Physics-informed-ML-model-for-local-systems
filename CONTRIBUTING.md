Thank you for contributing to PINN River Modeller.

Key guidelines to make this project admissions-grade:

- Reproducibility: include a short `RUNNING.md` or expand `README.md` with exact commands to reproduce main figures and experiments. Provide seed values and data generation scripts.
- Tests: add unit tests for core numerical solvers (small grids), the data generator, and training harness. Aim for a small CI test suite that runs in <10 minutes on GitHub Actions.
- Lightweight examples: provide one or two small example notebooks or scripts that run end-to-end on tiny datasets and produce the paper figure(s).
- Documentation: add a `docs/` folder with a short project overview, key equations, and figure-generation steps (use Sphinx or simple Markdown).
- Code quality: enforce style with `black` and `flake8`, and include a pre-commit config.
- Packaging: add `setup.cfg`/`pyproject.toml` and a `requirements-dev.txt` for developer tools.

Running locally (quick):
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python dash_app/app.py
```

Please open small, focused PRs: one per feature/fix, with a descriptive title and tests where applicable.
