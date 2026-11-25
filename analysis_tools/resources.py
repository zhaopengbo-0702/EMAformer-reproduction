"""Aggregate recorded training/test resource measurements, never retrain."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import statistics
from verified import collect,dataset,digest,options,protocol,read,sha256

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument("--manifest",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--mode",required=True,choices=["release","best-checkpoint","dual"]);p.add_argument("--report",type=Path,required=True)
    a=p.parse_args();items,progress=collect(read(a.manifest),a.output,a.mode);rows=[];excluded=[]
    for item in items:
        run=item["run"];cfg=options(run["argv"]);ext=run.get("extensions",{})
        flags=tuple(int(ext.get("armor_"+k,1)) for k in ("channel","phase","joint"))
        variant="full" if run["group"]=="multivariate_forecasting" else "token_only" if run["group"]=="armor_ablation" and flags==(0,0,0) else None
        if variant is None:continue
        path=Path(item["attempt"])
        try:
            resources=read(path/"epoch_resources.json");env=read(path/"environment.json")
            log=(path/"train.log").read_text(encoding="utf-8")
            logged=[(int(n),float(s)) for n,s in re.findall(r"Epoch: (\d+) cost time: ([0-9.]+)",log)]
            observed=[(e["epoch"],e["official_epoch_seconds"]) for e in resources["epochs"]]
            if logged!=observed or not observed:raise ValueError("epoch resource values disagree with training log")
            if env["run"]!=run or env["gpu_count"]!=1:raise ValueError("environment identity/GPU count inconsistent")
            for role,file in ([("release","machine_metrics.json"),("best-validation","machine_metrics_best.json")] if a.mode=="dual" else [("release" if a.mode=="release" else "best-validation","machine_metrics.json")]):
                metric=read(path/file)
                if metric.get("test_seconds") is None:raise ValueError("full test timing unavailable")
                effective={k:v for k,v in item["effective_args"].items() if not k.startswith("armor_") and k not in ("checkpoints","root_path","model_id","des","gpu","seed","random_seed","fix_seed")}
                rows.append({"run_id":run["id"],"protocol":protocol(item,role),"comparability_signature":digest(effective),"checkpoint_role":role,"evaluation":item["evaluation"],"dataset":dataset(item),"seed":run["seed"],"horizon":int(cfg["pred_len"]),"variant":variant,"gpu":env["gpu"],"torch":env["torch"],"cuda":env["cuda"],"batch_size":int(cfg.get("batch_size",32)),
                    "epoch_seconds_mean":statistics.mean(x[1] for x in observed),"epoch_count":len(observed),"epoch_seconds":[x[1] for x in observed],
                    "training_peak_allocated_bytes":max(x["max_cuda_memory_allocated"] for x in resources["epochs"]),
                    "training_peak_reserved_bytes":max(x["max_cuda_memory_reserved"] for x in resources["epochs"]),
                    "test_seconds":metric["test_seconds"],"test_peak_allocated_bytes":metric.get("max_cuda_memory_allocated"),
                    "auxiliary_evidence_hashes":{name:sha256(path/name) for name in ("environment.json","epoch_resources.json","train.log")},
                    "evidence_note":"metrics/checkpoint completion hashes verified; epoch/environment/log auxiliary files were cross-checked, but original queue status did not seal their hashes"})
        except (OSError,KeyError,ValueError) as error:excluded.append({"run_id":run["id"],"reason":str(error)})
    paired=defaultdict(list)
    for r in rows:paired[(r["protocol"],r["dataset"],r["seed"],r["horizon"],r["gpu"],r["torch"],r["cuda"],r["batch_size"],r["comparability_signature"])].append(r)
    pairs=[]
    for key,value in sorted(paired.items()):
        if len(value)==2 and {r["variant"] for r in value}=={"full","token_only"}:pairs.append({"key":key,"measurements":value})
    result={"kind":"recorded_resources_not_new_training","verified_complete_runs":len(items),"measurements":rows,"matched_comparisons":pairs,"excluded":excluded,
        "notes":["Only pair same seed/horizon/hardware/batch/evaluation/checkpoint role.","Epoch means include all completed epochs, including first epoch, as explicitly recorded; original paper warmup policy unknown.","Training peaks and test peaks are distinct; allocated and reserved not conflated.","V100 measurements cannot be claimed as a numeric reproduction of the original RTX4090 resource figure.","Full Fig8 requires complete four horizons for all12 datasets and then ETT/PEMS group averaging; partial pairs remain partial."]}
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"measurements":len(rows),"matched_pairs":len(pairs),"excluded":len(excluded),"training_runs_created":0}))
if __name__=="__main__":main()
