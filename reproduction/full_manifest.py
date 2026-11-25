"""Construct the paper-owned study matrix; no paper reference values are run data."""
from __future__ import annotations

import argparse
from copy import deepcopy
import itertools
import json
from pathlib import Path

from manifest import digest, make_manifest, options, replace_option


def paper_manifest(repo):
    released = make_manifest(repo)
    main = [r for r in released["runs"] if r["group"] == "multivariate_forecasting"]
    one = [r for r in main if r["seed"] == 2023]
    runs = deepcopy(main)
    def add(base, group, extensions=None, changes=None, provenance="reconstructed_from_paper_and_official_script"):
        row = deepcopy(base)
        row["group"] = group
        row["seed"] = 2023
        row["extensions"] = extensions or {}
        for key, value in (changes or {}).items():
            row["argv"] = replace_option(row["argv"], key, value)
        row["provenance"] = provenance
        opts = options(row["argv"])
        row["signature"] = digest({"argv": row["argv"], "seed": row["seed"], "extensions": row["extensions"]})
        row["id"] = f"{group}-{opts['model_id']}-{opts['model']}-s2023-{row['signature'][:10]}"
        runs.append(row)
    for base in one:
        opts = options(base["argv"])
        for c, p, j in itertools.product((0, 1), repeat=3):
            if (c, p, j) != (1, 1, 1):
                add(base, "armor_ablation", {"armor_channel": c, "armor_phase": p, "armor_joint": j})
        dataset = opts["model_id"].split("_")[0].lower()
        if dataset not in ("etth1", "etth2", "ettm1", "ettm2"):
            add(base, "mlp_backbone", {"backbone": "mlp"}, {"model": "EMAformerMLP"})
        if dataset in ("etth2", "ettm2", "solar", "weather"):
            add(base, "mean_input", {"mean_input": 1}, {"model": "EMAformerMeanInput"})
        if not dataset.startswith("pems"):
            day = 24 if dataset in ("etth1", "etth2", "ecl", "traffic") else (96 if dataset.startswith("ettm") else 144)
            current = int(opts.get("cycle", 24))
            alternative = day if current == 7 * day else 7 * day
            add(base, "alternative_period", changes={"cycle": alternative})
    for base in released["runs"]:
        opts = options(base["argv"])
        if base["group"] == "boost_performance" and opts["model"] in ("iInformer", "iReformer", "iFlowformer"):
            for stage in ("original", "inverted", "armor"):
                changes = {"model": "Original" + opts["model"][1:]} if stage == "original" else {}
                ext = {"armor_channel": 0, "armor_phase": 0, "armor_joint": 0} if stage == "inverted" else {}
                add(base, "transfer_" + stage, ext, changes,
                    "upstream_boost_script_with_explicit_model_or_armor_stage")
    for dataset in ("ecl", "traffic"):
        # Base only L96/H96 configs. Other L values are explicit reconstructions;
        # released Traffic loop covers only 528/720, and ECL may differ from main.
        base = next(r for r in one if options(r["argv"])["model_id"].lower().startswith(dataset + "_")
                    and options(r["argv"])["pred_len"] == "96")
        for length in (48, 192, 336, 528, 720):
            add(base, "lookback_emaformer", changes={"seq_len": length,
                "model_id": f"{dataset}_{length}_96"})
    result = deepcopy(released)
    result["runs"] = runs
    result["scope"] = "Paper-owned experiments: main six seeds, each other configuration one seed. Analysis reuses checkpoints. Independent comparison baseline reruns not included."
    result["protocol_choices"] = {
        "seeds": "2023..2028 chosen by reproduction; original six identities undisclosed",
        "non_main_repeats": "one seed 2023; paper does not disclose repeats",
        "model_variants": "See reproduction_extensions/README.md for MeanInput/MLP/phase ambiguities",
        "lookback": "L96 main hyperparameters held fixed for reconstructed other L",
        "transfer": "Official boost configs; temporal original, inverted with armor disabled, inverted with armor enabled",
        "checkpoint": "Queue requires explicit release or best-checkpoint mode; never pool modes",
        "baseline_status": "Paper reference values are not independently rerun baseline results"}
    result["expected_counts"] = {g: sum(r["group"] == g for r in runs) for g in sorted({r["group"] for r in runs})}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = paper_manifest(args.repo)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"runs": len(data['runs']), "groups": data['expected_counts']}))


if __name__ == "__main__":
    main()
