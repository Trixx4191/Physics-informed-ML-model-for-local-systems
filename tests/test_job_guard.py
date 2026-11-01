"""Exercise the production training slot using independent operating processes."""

import multiprocessing
import time

import pytest

from dash_app.job_guard import training_slot


def _hold_slot(directory, ready, release):
    with training_slot(directory) as acquired:
        assert acquired
        ready.set()
        assert release.wait(timeout=15)


def _try_slot(directory, results):
    started = time.monotonic()
    with training_slot(directory) as acquired:
        results.put((acquired, time.monotonic() - started))


@pytest.fixture
def process_context():
    # Spawn avoids inheriting a parent's lock descriptor into the contender.
    return multiprocessing.get_context("spawn")


def _start_holder(context, directory):
    ready, release = context.Event(), context.Event()
    process = context.Process(target=_hold_slot, args=(str(directory), ready, release))
    process.start()
    assert ready.wait(timeout=10)
    return process, release


def test_competing_process_is_rejected_without_waiting(tmp_path, process_context):
    holder, release = _start_holder(process_context, tmp_path)
    try:
        results = process_context.Queue()
        contender = process_context.Process(target=_try_slot, args=(str(tmp_path), results))
        contender.start()
        try:
            acquired, duration = results.get(timeout=10)
            contender.join(timeout=10)
            assert contender.exitcode == 0
            assert acquired is False
            assert duration < 1.0
            assert holder.is_alive()
        finally:
            if contender.is_alive():
                contender.terminate()
                contender.join(timeout=5)
            results.close()
            results.join_thread()
    finally:
        release.set()
        holder.join(timeout=10)
        if holder.is_alive():
            holder.terminate()
            holder.join(timeout=5)
    assert holder.exitcode == 0
    with training_slot(tmp_path) as acquired:
        assert acquired


def test_slot_released_after_exception(tmp_path):
    with pytest.raises(RuntimeError, match="job failed"):
        with training_slot(tmp_path) as acquired:
            assert acquired
            raise RuntimeError("job failed")
    with training_slot(tmp_path) as acquired:
        assert acquired


def test_slot_released_after_process_death(tmp_path, process_context):
    holder, release = _start_holder(process_context, tmp_path)
    try:
        holder.kill()
        holder.join(timeout=10)
        assert not holder.is_alive()
        with training_slot(tmp_path) as acquired:
            assert acquired
    finally:
        # A killed process may die while holding an Event's internal semaphore.
        # Do not touch its synchronization primitives after termination.
        if holder.is_alive():
            holder.terminate()
            holder.join(timeout=5)


def test_independent_open_in_same_process_cannot_reenter(tmp_path):
    with training_slot(tmp_path) as first:
        assert first
        with training_slot(tmp_path) as second:
            assert second is False
    with training_slot(tmp_path) as acquired:
        assert acquired


def test_different_cache_directories_have_independent_slots(tmp_path):
    with training_slot(tmp_path / "first") as first:
        with training_slot(tmp_path / "second") as second:
            assert first and second
