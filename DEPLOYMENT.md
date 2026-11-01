# Deployment guide

## AWS host

A dedicated Amazon Lightsail VM, `pinn-explorer-01`, was created on 1 November
2026 in `eu-north-1a` (Stockholm), using Ubuntu 24.04 LTS. It has 2 vCPUs,
4 GB RAM and an 80 GB disk (`medium_3_0`, $24/month base price).

The host has a dedicated static IP, restricted key-based SSH access, an
unprivileged `pinn` user, and a Python virtual environment at
`/opt/pinn-explorer/venv`. Writable cache directories are prepared separately
from the application code.

**Status:** VM provisioning is complete; the full Dash application has not yet
been published on this VM. GitHub Pages is disabled. No existing VM or app was
reconfigured for this host.

The app source is staged under `/opt/pinn-explorer/releases`. Runtime
dependencies, the app service and the public proxy route remain to be installed.
The public service should use CPU-only PyTorch, one native numerical thread and
a global limit of one training job at a time.

## Render alternative

Use the Render Blueprint defined in `render.yaml`. It configures a native Python
web service, installs `requirements.txt`, starts the Dash server with Gunicorn,
and checks `/healthz`.

## Render steps

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
