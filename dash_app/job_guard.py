"""Admit one training or research job across the dashboard's worker processes."""

from contextlib import contextmanager
import fcntl
from pathlib import Path


@contextmanager
def training_slot(cache_directory):
    """Yield immediately with whether this process acquired the shared job slot.

    The kernel releases this lock even if a job process is terminated. Keep the
    lock file in place: unlinking it could let processes lock different inodes.
    """
    directory = Path(cache_directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "training.lock").open("a") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
