"""Strict main-only acceptance and comparisons. Never trains or imports torch."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis_tools"))
sys.path.insert(0, str(ROOT / "reproduction"))
from aggregate import aggregate
from verified import collect, frozen_snapshot, read, sha256


def selected_manifest(manifest):
    runs = [row for row in manifest["runs"] if row["group"] == "multivariate_forecasting"]
    if len(runs) != 288 or len({r["id"] for r in runs}) != 288:
        raise ValueError("Expected exactly288 unique main experiment runs")
    return {**manifest, "runs": runs, "scope": "user-approved main288 only; branches are checklist only"}


def metric_columns(measured, reference):
    result = {}
    for metric in ("mse", "mae"):
        stats = measured[metric]
        value = reference.get(metric) if reference else None
        reference_sd = reference.get(metric + "_sd") if reference else None
        result.update({metric + "_measured_mean": stats["mean"],
                       metric + "_sd_population": stats["sd_population"], metric + "_sd_sample": stats["sd_sample"],
                       metric + "_published_reference": value,
                       metric + "_published_sd": reference_sd,
                       metric + "_population_sd_difference": stats["sd_population"] - reference_sd if stats["sd_population"] is not None and reference_sd is not None else None,
                       metric + "_sample_sd_difference": stats["sd_sample"] - reference_sd if stats["sd_sample"] is not None and reference_sd is not None else None,
                       metric + "_difference": stats["mean"] - value if value is not None else None,
                       metric + "_difference_percent": 100 * (stats["mean"] - value) / value if value else None})
    return result


def tables(result, reference):
    per_horizon, averaged = [], []
    for row in result["table5_by_horizon"]:
        published = reference.get("table5_six_seed_results", {}).get(row["dataset"], {}).get(str(row["horizon"]), {})
        per_horizon.append({"dataset": row["dataset"], "horizon": row["horizon"], "checkpoint_role": row["checkpoint_role"],
            "n_seeds": row["n_seeds"], "seeds": ",".join(map(str, row["seeds"])), "six_seed_complete": row["six_seed_complete"],
            "protocol": row["protocol"], **metric_columns(row, published)})
    for row in result["table1_horizon_averages"]:
        published = reference.get("table1_horizon_averaged_results", {}).get(row["dataset"], {}).get("EMAformer", {})
        averaged.append({"dataset": row["dataset"], "checkpoint_role": row["checkpoint_role"],
            "n_complete_seeds": row["n_complete_seeds"], "seeds": ",".join(map(str, row["seeds"])),
            "six_seed_complete": row["six_seed_complete"], "protocol": row["protocol"], **metric_columns(row, published)})
    roles = {}
    for row in per_horizon:
        pair = roles.setdefault((row["dataset"], row["horizon"]), {})
        if row["checkpoint_role"] in pair:
            raise ValueError("Multiple protocols/configurations for one dataset/horizon/role; report separately before closeout")
        pair[row["checkpoint_role"]] = row
    paired = []
    for (dataset, horizon), pair in sorted(roles.items()):
        if set(pair) != {"release", "best-validation"}:
            continue
        a, b = pair["release"], pair["best-validation"]
        if a["seeds"] != b["seeds"]:
            raise ValueError("Checkpoint role comparison requires same seeds")
        paired.append({"dataset": dataset, "horizon": horizon, "seeds": a["seeds"], "n_seeds": a["n_seeds"],
            "six_seed_complete": a["six_seed_complete"] and b["six_seed_complete"],
            **{metric + "_best_minus_release": b[metric + "_measured_mean"] - a[metric + "_measured_mean"] for metric in ("mse", "mae")}})
    return per_horizon, averaged, paired


def backup_union(manifest, roots):
    identities, conflicts, evidence = {}, [], []
    for root in roots:
        items, progress = collect(manifest, root, "dual")
        for item in items:
            frozen_snapshot(item)
            run_id, state = item["run"]["id"], item["state"]
            identity = {"signature": state["signature"],
                        "artifact_hashes": sorted((a["path"], a["sha256"]) for a in [state["metrics"], state["secondary_metrics"]] + state["checkpoints"])}
            if run_id in identities and identities[run_id] != identity:
                conflicts.append(run_id)
            identities[run_id] = identity
            evidence.append({"id": run_id, "signature": state["signature"], "result_root": str(Path(root).resolve()),
                             "attempt": item["attempt"]})
    if conflicts:
        raise ValueError("Conflicting downloaded copies for IDs: " + ",".join(sorted(set(conflicts))))
    return {"declared_local_result_roots": [str(Path(root).resolve()) for root in roots],
            "verified_unique_runs": len(identities), "complete": len(identities) == 288,
            "verified_ids": sorted(identities), "evidence": evidence,
            "note": "Caller declares these directories are downloaded local copies; same-NFS cloud copies must not be supplied as local evidence."}


def export_csv(path, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--local-results", type=Path, nargs="*", default=[])
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    manifest = selected_manifest(read(args.manifest))
    reference = read(args.reference)
    result = aggregate(manifest, args.output, "dual", reference)
    per_horizon, averaged, paired = tables(result, reference)
    backup = backup_union(manifest, args.local_results)
    selected_items, selected_progress = collect(manifest, args.output, "dual")
    selected_identities = {item["run"]["id"]: item["state"]["signature"] for item in selected_items}
    downloaded_identities = {item["id"]: item["signature"] for item in backup["evidence"]}
    backup["matching_selected_output_runs"] = sum(downloaded_identities.get(run_id) == signature
                                                  for run_id, signature in selected_identities.items())
    backup["selected_output_fully_backed_up"] = len(selected_identities) == 288 and backup["matching_selected_output_runs"] == 288
    complete_table5 = len(per_horizon) == 96 and all(row["six_seed_complete"] for row in per_horizon)
    complete_table1 = len(averaged) == 24 and all(row["six_seed_complete"] for row in averaged)
    acceptance = {"kind": "main288_only_closeout", "training_complete": result["verified_training_runs"] == 288,
        "verified_main_runs": result["verified_training_runs"], "planned_main_runs": 288,
        "table5_EMAformer_roles_complete": complete_table5, "table1_EMAformer_roles_complete": complete_table1,
        "paired_role_comparisons": len(paired), "local_backup": backup,
        "main_closeout_complete": result["verified_training_runs"] == 288 and complete_table5 and complete_table1 and backup["selected_output_fully_backed_up"],
        "full_paper_reproduction_claim": False,
        "approved_scope": "Main288 only. Branches and comparison baselines remain checklist; no additional training requested.",
        "reference_sha256": sha256(args.reference), "manifest_sha256": sha256(args.manifest),
        "limitations": ["Published values are rounded, SD convention and original seed identities undisclosed.",
                        "Release retains upstream final-weight behavior; best-validation is separately reported.",
                        "No independent baseline rerun or full-paper ranking claim is made."]}
    args.destination.mkdir(parents=True, exist_ok=False)
    for name, value in (("strict_aggregate.json", result), ("acceptance.json", acceptance)):
        (args.destination / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    for name, rows in (("Table5_EMAformer_per_horizon.csv", per_horizon), ("Table1_EMAformer_horizon_average.csv", averaged), ("release_vs_best.csv", paired)):
        export_csv(args.destination / name, rows)
    report = ["# EMAformer 主实验验收", "", f"已严格验收 {result['verified_training_runs']}/288 次训练。",
              f"Table5两角色完整：{complete_table5}；Table1两角色完整：{complete_table1}。",
              f"已声明本地结果目录去重验签 {backup['verified_unique_runs']}/288 次。",
              f"主实验收尾完成：{acceptance['main_closeout_complete']}。", "",
              "本报告仅涵盖EMAformer主表六seed；release与best分列，未齐六次不填SD。",
              "534次作者变体与独立比较基线仅保留待办清单，不执行。不能称全文复现。",
              "论文数值只列为published_reference，并未填入实测单元格。"]
    (args.destination / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"verified_main_runs": result["verified_training_runs"], "local_backup_runs": backup["verified_unique_runs"],
                      "main_closeout_complete": acceptance["main_closeout_complete"], "full_paper_claim": False}))


if __name__ == "__main__":
    main()
