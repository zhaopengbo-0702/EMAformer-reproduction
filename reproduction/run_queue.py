"""One kernel-locked queue, one child, one CUDA device; no cloud API actions."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile

from common import atomic_json, exclusive_lock, sha256, utcnow, valid_metrics, verified_completion
from manifest import digest, options


def source_inventory(repo):
    """Content hash includes all relevant code, not just a possibly dirty git HEAD."""
    source_dirs = {"data_provider", "experiments", "layers", "model", "scripts", "utils"}
    return {p.relative_to(repo).as_posix(): sha256(p) for p in sorted(repo.rglob("*"))
            if p.is_file() and p.suffix in (".py", ".sh", ".txt")
            and (p.parent == repo or p.relative_to(repo).parts[0] in source_dirs)}


def artifact(path, folder):
    return {"path": path.relative_to(folder).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=["release", "best-checkpoint", "dual"])
    parser.add_argument("--evaluation", choices=["original", "streaming"], default="streaming")
    parser.add_argument("--groups", nargs="+", default=["multivariate_forecasting"])
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--max-runs", type=int, default=0, help="0 = all selected runs")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--training-lock", type=Path, default=Path(tempfile.gettempdir()) / "emaformer-single-training.lock",
                        help="Global child-held GPU training lock; use one shared path for this machine")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    args.output = args.output.resolve()
    args.data_root = args.data_root.resolve()
    args.training_lock = args.training_lock.resolve()
    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected = [run for run in data["runs"] if run["group"] in args.groups
                and (args.ids is None or run["id"] in args.ids)
                and (args.seeds is None or run["seed"] in args.seeds)]
    if not selected:
        parser.error("No manifest runs selected")
    if len({r["id"] for r in selected}) != len(selected):
        parser.error("Duplicate manifest run identifiers")
    for run in selected:
        if not run["id"] or any(c in run["id"] for c in ("/", "\\", ":")):
            parser.error("Unsafe manifest run id")
        opts = options(run["argv"])
        if "use_multi_gpu" in opts:
            parser.error("Multi-GPU manifest is prohibited")
    args.output.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(args.output / "queue.lock"):
        with exclusive_lock(args.training_lock):
            pass  # Refuse even a different-output queue while an orphan trains.
        dataset_files = {}
        for run in selected:
            opts = options(run["argv"])
            relative = Path(opts["root_path"].replace("\\", "/"))
            if relative.parts and relative.parts[0] in ("data", "dataset"):
                relative = Path(*relative.parts[1:])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Manifest dataset path escapes data root")
            data_file = (args.data_root / relative / opts["data_path"]).resolve()
            data_file.relative_to(args.data_root)
            if data_file not in dataset_files:
                if not data_file.is_file():
                    raise FileNotFoundError(f"Required dataset is missing: {data_file}")
                dataset_files[data_file] = {"path": str(data_file), "sha256": sha256(data_file),
                                             "bytes": data_file.stat().st_size, "mtime_ns": data_file.stat().st_mtime_ns}
        atomic_json(args.output / "dataset_inventory.json", {"created_utc": utcnow(),
                    "files": list(dataset_files.values())})
        inventory = source_inventory(args.repo)
        instrument_files = list(Path(__file__).parent.glob("*.py"))
        extension_dir = Path(__file__).resolve().parent.parent / "reproduction_extensions"
        if extension_dir.exists():
            instrument_files += list(extension_dir.rglob("*.py"))
        instrument_hash = digest({"reproduction/" + p.name if p.parent == Path(__file__).parent else "reproduction_extensions/" + p.relative_to(extension_dir).as_posix(): sha256(p)
                                  for p in instrument_files})
        code_hash = digest(inventory)
        atomic_json(args.output / "source_inventory.json", {"upstream_commit": data.get("upstream_commit"),
                    "files": inventory, "sha256": code_hash, "instrumentation_sha256": instrument_hash})
        snapshot = args.output / "runtime_snapshots" / digest({"source": code_hash, "instrumentation": instrument_hash})
        if not (snapshot / "READY.json").exists():
            for relative in inventory:
                destination = snapshot / "upstream" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(args.repo / relative, destination)
            for source in instrument_files:
                relative = (Path("reproduction") / source.name if source.parent == Path(__file__).parent
                            else Path("reproduction_extensions") / source.relative_to(extension_dir))
                destination = snapshot / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            atomic_json(snapshot / "READY.json", {"created_utc": utcnow(), "source_sha256": code_hash,
                        "instrumentation_sha256": instrument_hash})
        total = Counter()
        stop_requested = False
        active_child = None

        def stop(signum, frame):
            nonlocal stop_requested
            stop_requested = True
            if active_child is not None and active_child.poll() is None:
                active_child.terminate()
        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        for run in selected:
            if stop_requested:
                break
            opts = options(run["argv"])
            relative = Path(opts["root_path"].replace("\\", "/"))
            if relative.parts and relative.parts[0] in ("data", "dataset"):
                relative = Path(*relative.parts[1:])
            dataset = dataset_files[(args.data_root / relative / opts["data_path"]).resolve()]
            current_stat = Path(dataset["path"]).stat()
            if current_stat.st_size != dataset["bytes"] or current_stat.st_mtime_ns != dataset["mtime_ns"]:
                raise RuntimeError("Dataset changed after queue preflight; stop and re-hash in a new queue")
            signature = digest({"run": run, "mode": args.mode, "evaluation": args.evaluation, "code": code_hash,
                                "dataset_sha256": dataset["sha256"],
                                "instrumentation": instrument_hash, "num_workers": args.num_workers})
            run_root = args.output / args.mode / run["id"]
            index = run_root / "index.json"
            if index.exists():
                prior = json.loads(index.read_text(encoding="utf-8"))
                attempt = run_root / prior["attempt"]
                if verified_completion(attempt, signature):
                    total["resumed_complete"] += 1
                    continue
            if args.max_runs and total["attempted"] >= args.max_runs:
                break
            attempts = [int(p.name.split("-")[-1]) for p in run_root.glob("attempt-*") if p.is_dir()]
            attempt = run_root / f"attempt-{max(attempts, default=0) + 1:03d}"
            attempt.mkdir(parents=True)
            atomic_json(attempt / "run.json", run)
            atomic_json(attempt / "dataset.json", dataset)
            atomic_json(index, {"attempt": attempt.name, "signature": signature})
            state = {"state": "running", "run_id": run["id"], "signature": signature,
                     "started_utc": utcnow(), "mode": args.mode, "pid": os.getpid(),
                     "upstream_commit": data.get("upstream_commit"), "source_sha256": code_hash,
                     "dataset": dataset,
                     "instrumentation_sha256": instrument_hash}
            atomic_json(attempt / "status.json", state)
            command = [sys.executable, "-u", str(snapshot / "reproduction" / "child.py"),
                       "--repo", str(snapshot / "upstream"), "--run-json", str(attempt / "run.json"),
                       "--mode", args.mode, "--evaluation", args.evaluation, "--data-root", str(args.data_root),
                       "--num-workers", str(args.num_workers), "--training-lock", str(args.training_lock)]
            env = dict(os.environ)
            env.update({"PYTHONUNBUFFERED": "1", "PYTHONHASHSEED": str(run["seed"]),
                        "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
            atomic_json(attempt / "invocation.json", {"command": command, "cwd": str(attempt)})
            print(f"START {run['id']} ({attempt.name})", flush=True)
            total["attempted"] += 1
            with open(attempt / "train.log", "w", encoding="utf-8") as log:
                active_child = subprocess.Popen(command, cwd=attempt, env=env, stdout=log, stderr=subprocess.STDOUT)
                state["child_pid"] = active_child.pid
                atomic_json(attempt / "status.json", state)
                code = active_child.wait()
                active_child = None
            state.update({"returncode": code, "finished_utc": utcnow(), "state": "interrupted" if stop_requested else "failed"})
            metrics_path = attempt / "machine_metrics.json"
            checkpoints = sorted((attempt / "checkpoints").rglob("checkpoint*.pth"))
            if code == 0 and metrics_path.exists() and checkpoints:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                if valid_metrics(metrics):
                    state["metrics"] = artifact(metrics_path, attempt)
                    state["checkpoints"] = [artifact(p, attempt) for p in checkpoints]
                    state["state"] = "complete"
                    if args.mode == "dual":
                        best_path = attempt / "machine_metrics_best.json"
                        if best_path.exists() and valid_metrics(json.loads(best_path.read_text(encoding="utf-8"))):
                            state["secondary_metrics"] = artifact(best_path, attempt)
                        else:
                            state["state"] = "failed"
            atomic_json(attempt / "status.json", state)
            total[state["state"]] += 1
            atomic_json(args.output / "queue_progress.json", {"updated_utc": utcnow(), "selected": len(selected),
                        "counts": dict(total), "last_run": run["id"], "last_state": state["state"]})
            print(f"{state['state'].upper()} {run['id']}", flush=True)
            if state["state"] != "complete" and not args.continue_on_error:
                break
        atomic_json(args.output / "queue_progress.json", {"updated_utc": utcnow(), "selected": len(selected),
                    "counts": dict(total), "queue_state": "interrupted" if stop_requested else "exited"})
        print(json.dumps(dict(total)), flush=True)
        if total["failed"] or total["interrupted"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
