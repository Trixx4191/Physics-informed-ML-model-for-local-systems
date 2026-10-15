# Deployment guide

This project is already structured for a live Dash deployment.

## Recommended first target

Use the Render Blueprint defined in `render.yaml`. It configures a native Python
web service, installs `requirements.txt`, starts the Dash server with Gunicorn,
and checks `/healthz`.

## Fastest path

1. Push this repo to GitHub.
2. In Render, create a new Blueprint and connect the repo.
3. Confirm the service uses the settings from `render.yaml`; no separate build,
   start, or health-check command is needed.

The repository also includes a Dockerfile for Docker-based deployments. Do not
select Docker when deploying the Blueprint above.

## Local production-style check

```bash
export PORT=7860
gunicorn dash_app.app:server --bind 0.0.0.0:$PORT --workers 1 --timeout 300
```

## Notes

- The app uses the Dash server object, not the experiment script.
- The live service should default to a single small training run rather than the full benchmark suite.
- Heavy sweeps remain in `run_experiment.py` for offline research work.
