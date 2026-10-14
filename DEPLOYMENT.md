# Deployment guide

This project is already structured for a live Dash deployment.

## Recommended first target

Use Render or Railway with the included Dockerfile.

## Fastest path

1. Push this repo to GitHub.
2. Create a new Render web service.
3. Connect the repo.
4. Use the existing Dockerfile.
5. Set the start command to:

   gunicorn dash_app.app:server --bind 0.0.0.0:$PORT --workers 1 --timeout 300

6. Set health check path to `/healthz`.

## Local production-style check

```bash
export PORT=7860
gunicorn dash_app.app:server --bind 0.0.0.0:$PORT --workers 1 --timeout 300
```

## Notes

- The app uses the Dash server object, not the experiment script.
- The live service should default to a single small training run rather than the full benchmark suite.
- Heavy sweeps remain in `run_experiment.py` for offline research work.
