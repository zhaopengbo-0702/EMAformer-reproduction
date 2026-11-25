"""Summarize verified measured runs; never copy paper values into measured cells."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics

from common import atomic_json, utcnow, verified_completion
from manifest import options


def summarize(manifest, output_root, mode, reference=None):
    rows, progress = [], []
    for run in manifest["runs"]:
        root = output_root / mode / run["id"]
        state = "pending"
        attempt = None
        try:
            index = json.loads((root / "index.json").read_text(encoding="utf-8"))
            attempt = root / index["attempt"]
            status = json.loads((attempt / "status.json").read_text(encoding="utf-8"))
            state = status["state"]
            if state == "complete" and not verified_completion(attempt, status["signature"]):
                state = "invalid_artifacts"
        except (OSError, ValueError, KeyError):
            pass
        progress.append({"id": run["id"], "group": run["group"], "seed": run["seed"], "state": state})
        if state != "complete":
            continue
        config = options(run["argv"])
        dataset = config["model_id"].split("_")[0]
        for role, file in (("release" if mode in ("dual", "release") else "best-validation", "machine_metrics.json"),
                           *(([("best-validation", "machine_metrics_best.json")]) if mode == "dual" else [])):
            metrics = json.loads((attempt / file).read_text(encoding="utf-8"))
            rows.append({"run_id": run["id"], "group": run["group"], "dataset": dataset,
                         "model": config["model"], "seq_len": int(config["seq_len"]),
                         "pred_len": int(config["pred_len"]), "seed": run["seed"], "checkpoint_role": role,
                         "mse": metrics["mse"], "mae": metrics["mae"], "rmse": metrics["rmse"],
                         "evaluation": metrics.get("evaluation", "original"), "attempt": str(attempt)})
    from collections import Counter, defaultdict
    grouped = defaultdict(list)
    for row in rows:
        if row["group"] == "multivariate_forecasting":
            grouped[(row["dataset"], row["pred_len"], row["checkpoint_role"], row["evaluation"])].append(row)
    main = []
    for key, values in sorted(grouped.items()):
        dataset, horizon, role, evaluation = key
        item = {"dataset": dataset, "pred_len": horizon, "checkpoint_role": role, "evaluation": evaluation,
                "n_seeds": len(values), "seeds": sorted(v["seed"] for v in values),
                "six_seed_complete": len(values) == 6}
        for metric in ("mse", "mae"):
            measurements = [v[metric] for v in values]
            item[metric + "_mean"] = statistics.mean(measurements)
            item[metric + "_sd_population"] = statistics.pstdev(measurements)
            item[metric + "_sd_sample"] = statistics.stdev(measurements) if len(measurements) > 1 else None
        # Reference schema is explicit; unknown key remains absent.
        ref_dataset = {"solar": "Solar", "traffic": "Traffic", "weather": "Weather"}.get(dataset, dataset)
        published = (reference or {}).get("table5_six_seed_results", {}).get(ref_dataset, {}).get(str(horizon))
        if published:
            item["published_reference"] = published
            item["published_reference_provenance"] = "paper Table5, not measured"
            for metric in ("mse", "mae"):
                item[metric + "_delta_vs_published"] = item[metric + "_mean"] - published[metric]
        main.append(item)
    return {"generated_utc": utcnow(), "mode": mode, "planned_training_runs": len(manifest["runs"]),
            "states": dict(Counter(p["state"] for p in progress)), "progress": progress,
            "measured_rows": rows, "main_by_horizon": main,
            "scope_note": "Training completeness only. Full-paper status also requires analyses, independent comparison baselines and verified local backup. SD ddof undisclosed in paper: both conventions shown."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=["release", "best-checkpoint", "dual"])
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    reference = json.loads(args.reference.read_text(encoding="utf-8")) if args.reference else None
    result = summarize(manifest, args.output, args.mode, reference)
    atomic_json(args.report, result)
    csv_path = args.report.with_suffix(".csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as handle:
        fields = ["run_id", "group", "dataset", "model", "seq_len", "pred_len", "seed", "checkpoint_role",
                  "mse", "mae", "rmse", "evaluation", "attempt"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["measured_rows"])
    print(json.dumps({"planned": len(manifest['runs']), "states": result['states'], "measured_rows": len(result['measured_rows'])}))


if __name__ == "__main__":
    main()
