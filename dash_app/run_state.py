"""Recover background results after a browser refresh without owning the job lock.

The dashboard's ``training_slot`` remains responsible for admission. These small,
session-scoped records only remember progress and renderable callback outputs;
models, optimizers, and training tensors are never persisted here.
"""

from functools import wraps
import inspect
import json
import logging
import os
import time
from uuid import UUID, uuid4

from dash import no_update
from plotly.utils import PlotlyJSONEncoder
import psutil


RUN_EXPIRY_SECONDS = 7 * 24 * 60 * 60
PROGRESS_SAVE_INTERVAL = 0.25
_KEY_PREFIX = "pinn-workspace-run-v1"
_logger = logging.getLogger(__name__)


def _key(kind, workspace_id):
    """Reject malformed identifiers instead of sharing an anonymous run."""
    if not isinstance(workspace_id, str) or not isinstance(kind, str) or not kind:
        return None
    try:
        workspace_uuid = UUID(workspace_id)
    except (ValueError, AttributeError):
        return None
    return (_KEY_PREFIX, kind, str(workspace_uuid))


def _json_safe(value):
    # This is the same encoder Dash uses for components, Plotly figures and
    # NumPy arrays. Round-tripping also avoids storing live component objects.
    return json.loads(json.dumps(value, cls=PlotlyJSONEncoder))


def _is_alive(record):
    """A PID is only live when it still belongs to the original worker."""
    try:
        process = psutil.Process(record["pid"])
        return (
            process.is_running()
            and process.status() != psutil.STATUS_ZOMBIE
            and abs(process.create_time() - record["create_time"]) < 0.01
        )
    except (psutil.Error, KeyError, TypeError, ValueError, OverflowError):
        return False


def _stamp(record):
    record["updated"] = record["updated_at"] = time.time()
    return record


def get_run(cache, kind, workspace_id):
    """Return one JSON-compatible run, marking a vanished worker interrupted.

    Unknown or invalid workspace identifiers return ``None``. A worker's PID
    birth time prevents a reused PID from making an abandoned run look active.
    """
    key = _key(kind, workspace_id)
    if key is None:
        return None
    with cache.transact():
        record = cache.get(key)
        if not isinstance(record, dict):
            return None
        if record.get("state") == "running" and not _is_alive(record):
            record = dict(record)
            record["state"] = "interrupted"
            record["progress"] = "Run interrupted. Start a new run to continue."
            cache.set(key, _stamp(record), expire=RUN_EXPIRY_SECONDS)
    return record


def _primitive(value):
    """Allow scalar settings and short settings lists, never numerical payloads."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)) and len(value) <= 100:
        if all(item is None or isinstance(item, (str, bool, int, float)) for item in value):
            return list(value)
    raise TypeError("Not a scalar dashboard setting")


def _settings(arguments):
    settings = {}
    for name, value in arguments.items():
        if name in {"workspace_id", "set_progress"} or callable(value):
            continue
        try:
            settings[name] = _primitive(value)
        except TypeError:
            continue
    return _json_safe(settings)


def _output_text(result):
    if isinstance(result, (tuple, list)) and result and isinstance(result[-1], str):
        return result[-1]
    payload = result.to_plotly_json() if hasattr(result, "to_plotly_json") else result
    if isinstance(payload, dict):
        children = payload.get("props", {}).get("children", "")
        if isinstance(children, str):
            return children
    return ""


def _busy(result):
    if result is no_update:
        return True
    if isinstance(result, (tuple, list)) and any(part is no_update for part in result):
        return True
    return "another training or experiment job is running" in _output_text(result).lower()


def _failed(result):
    if _output_text(result).strip().lower() == "failed":
        return True
    parts = result if isinstance(result, (tuple, list)) else [result]
    for part in parts:
        if hasattr(part, "to_plotly_json"):
            payload = part.to_plotly_json()
            if payload.get("type") == "Alert" and payload.get("props", {}).get("color") == "danger":
                return True
    return False


def remember_run(cache, kind, progress=False):
    """Decorate a background callback with optional session result persistence.

    ``workspace_id=None`` may be appended to the callback's original signature;
    callbacks without that parameter are supported too. Calls without a valid
    UUID behave exactly as before. ``progress=True`` wraps the first positional
    callback argument, forwarding every Dash progress update unchanged.

    A live record for the same session is never replaced by a competing worker.
    The wrapped function still runs and lets its existing job guard decide what
    to return. Every update checks a nonce so an old worker cannot overwrite a
    newer run after its process has been replaced.
    """
    def decorate(function):
        original_signature = inspect.signature(function)
        has_workspace_id = "workspace_id" in original_signature.parameters
        public_signature = original_signature
        if not has_workspace_id:
            parameters = list(original_signature.parameters.values())
            needs_keyword_only = any(
                item.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.KEYWORD_ONLY}
                for item in parameters
            )
            parameter = inspect.Parameter(
                "workspace_id",
                kind=inspect.Parameter.KEYWORD_ONLY if needs_keyword_only else inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=None,
            )
            insertion = next(
                (index for index, item in enumerate(parameters) if item.kind == inspect.Parameter.VAR_KEYWORD),
                len(parameters),
            )
            parameters.insert(insertion, parameter)
            public_signature = original_signature.replace(parameters=parameters)

        @wraps(function)
        def wrapped(*args, **kwargs):
            bound = public_signature.bind(*args, **kwargs)
            bound.apply_defaults()
            workspace_id = bound.arguments.get("workspace_id")
            key = _key(kind, workspace_id)
            if not has_workspace_id:
                bound.arguments.pop("workspace_id", None)

            def call_original():
                return function(*bound.args, **bound.kwargs)

            if key is None:
                return call_original()

            previous = None
            nonce = str(uuid4())
            settings = _settings(bound.arguments)
            process = psutil.Process(os.getpid())
            record = _stamp({
                "id": nonce,
                "nonce": nonce,
                "pid": process.pid,
                "create_time": process.create_time(),
                "kind": kind,
                "workspace_id": key[-1],
                "domain_key": settings.get("domain_key"),
                "inputs": settings,
                "state": "running",
                "progress": "Starting run…",
                "result": None,
                "result_domain_key": None,
                "result_inputs": None,
            })
            try:
                with cache.transact():
                    previous = cache.get(key)
                    if isinstance(previous, dict) and previous.get("state") == "running" and _is_alive(previous):
                        # This worker may immediately lose training_slot. Its
                        # return must not replace the admitted worker's record.
                        record = None
                    else:
                        if isinstance(previous, dict):
                            record["result"] = previous.get("result")
                            record["result_domain_key"] = previous.get("result_domain_key", previous.get("domain_key"))
                            record["result_inputs"] = previous.get("result_inputs", previous.get("inputs"))
                        cache.set(key, record, expire=RUN_EXPIRY_SECONDS)
            except Exception:
                _logger.exception("Could not remember background run")
                record = None
            if record is None:
                return call_original()

            def save(**changes):
                try:
                    with cache.transact():
                        current = cache.get(key)
                        if not isinstance(current, dict) or current.get("nonce") != nonce:
                            return
                        record.update(changes)
                        cache.set(key, _stamp(record), expire=RUN_EXPIRY_SECONDS)
                except Exception:
                    # Persistence must never interrupt numerical training.
                    _logger.exception("Could not save background run progress")

            last_save = float("-inf")
            latest_progress = record["progress"]
            if progress:
                progress_name = next(iter(original_signature.parameters))
                forward_progress = bound.arguments[progress_name]

                def track_progress(*values, **progress_kwargs):
                    nonlocal last_save, latest_progress
                    forward_progress(*values, **progress_kwargs)
                    value = values[0] if values else progress_kwargs
                    latest_progress = value if isinstance(value, str) else str(value)
                    now = time.monotonic()
                    if now - last_save >= PROGRESS_SAVE_INTERVAL:
                        save(progress=latest_progress)
                        last_save = now

                bound.arguments[progress_name] = track_progress
            try:
                result = call_original()
            except BaseException as error:
                save(state="failed", progress=f"Failed: {error}")
                raise

            if _busy(result):
                # Busy is not a new scientific result, including on sessions
                # that already have completed plots worth recovering.
                previous_state = previous.get("state", "completed") if isinstance(previous, dict) else "completed"
                save(
                    state="interrupted" if previous_state == "running" else previous_state,
                    progress=_output_text(result),
                    result=previous.get("result") if isinstance(previous, dict) else None,
                    domain_key=previous.get("domain_key") if isinstance(previous, dict) else record["domain_key"],
                    inputs=previous.get("inputs", {}) if isinstance(previous, dict) else record["inputs"],
                    result_domain_key=(
                        previous.get("result_domain_key", previous.get("domain_key"))
                        if isinstance(previous, dict) else None
                    ),
                    result_inputs=(
                        previous.get("result_inputs", previous.get("inputs"))
                        if isinstance(previous, dict) else None
                    ),
                )
            else:
                try:
                    persisted_result = _json_safe(result)
                except (TypeError, ValueError, OverflowError):
                    _logger.exception("Background output could not be serialized for recovery")
                    save(state="failed", progress="Could not save this result for recovery.")
                else:
                    save(
                        state="failed" if _failed(result) else "completed",
                        progress=_output_text(result) or latest_progress,
                        result=persisted_result,
                        result_domain_key=settings.get("domain_key"),
                        result_inputs=settings,
                    )
            return result

        wrapped.__signature__ = public_signature
        return wrapped
    return decorate
