# Deployment guide

## Live AWS deployment

The full Dash app is available at **https://d203piwre8zpf0.cloudfront.net**.
GitHub Pages is disabled. The live app includes interactive training and research
tools.

CloudFront provides viewer HTTPS and redirects HTTP visitors to HTTPS. It
connects to nginx over HTTP. nginx validates a private custom origin header
before proxying to the app. Dash responses are not cached; callback POST
requests, query strings and cookies are forwarded.

### Runtime

The versioned configuration templates are in `deploy/aws/`:

- `pinn-explorer.service` runs Gunicorn as an unprivileged user on loopback,
  with one worker, two threads, automatic startup and restart after failure.
- `gunicorn.conf.py` configures the application server.
- `nginx.conf.template` configures the reverse proxy. Render its
  `__PINN_ORIGIN_TOKEN__` placeholder with the private value configured on the
  CloudFront origin. Keep the rendered configuration and token outside Git.

PyTorch is CPU-only, and numerical libraries use one native thread. Training and
research tools run in the background and share one process lock. A competing
submission receives a busy message while the interface remains available.
The service configuration limits memory and CPU use and keeps writable caches
separate from application code.

### Operations

For a fresh Python environment, install CPU-only PyTorch before the other
requirements:

```bash
/opt/pinn-explorer/venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
/opt/pinn-explorer/venv/bin/python -m pip install -r /opt/pinn-explorer/current/requirements.txt
```

Stage a release under `/opt/pinn-explorer/releases`, update the `current`
symlink, then restart and check the app:

```bash
sudo systemctl restart pinn-explorer
sudo systemctl status pinn-explorer nginx
curl --fail http://127.0.0.1:8050/healthz
sudo journalctl -u pinn-explorer -n 50 --no-pager
```

Check public routing independently:

```bash
curl --fail https://d203piwre8zpf0.cloudfront.net/healthz
```

To roll back, select a previously verified release with the `current` symlink
and restart the service. Keep host identifiers, origin credentials and account
billing details in private operations notes.

## Render alternative

Use the Render Blueprint defined in `render.yaml`. It configures a native Python
web service, installs `requirements.txt`, starts the Dash server with Gunicorn,
and checks `/healthz`.

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

The app uses the Dash server object. The full benchmark suite can also be run
offline with `run_experiment.py`.
