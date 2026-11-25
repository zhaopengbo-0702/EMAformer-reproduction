from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@contextmanager
def exclusive_lock(path):
    """Kernel released lock, never silently take over an active runner."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+b")
    locked = False
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except (OSError, IOError) as error:
            raise RuntimeError(f"Another queue owns {path}; no second training process was started") from error
        yield
    finally:
        if locked:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def valid_metrics(data):
    return all(isinstance(data.get(key), (int, float)) and math.isfinite(data[key]) and data[key] >= 0
               for key in ("mse", "mae", "rmse"))


def verified_completion(folder, signature):
    """Resume requires successful status, matching identity and intact artifacts."""
    folder = Path(folder)
    try:
        state = json.loads((folder / "status.json").read_text(encoding="utf-8"))
        if state.get("state") != "complete" or state.get("signature") != signature:
            return False
        metrics = json.loads((folder / state["metrics"]["path"]).read_text(encoding="utf-8"))
        if not valid_metrics(metrics) or not state.get("checkpoints"):
            return False
        all_artifacts = [state["metrics"]] + state["checkpoints"]
        if "secondary_metrics" in state:
            secondary = json.loads((folder / state["secondary_metrics"]["path"]).read_text(encoding="utf-8"))
            if not valid_metrics(secondary):
                return False
            all_artifacts.append(state["secondary_metrics"])
        for artifact in all_artifacts:
            path = folder / artifact["path"]
            if not path.is_file() or path.stat().st_size == 0 or sha256(path) != artifact["sha256"]:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False
