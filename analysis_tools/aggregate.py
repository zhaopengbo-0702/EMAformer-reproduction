"""CPU-only aggregation of verified runs. No model loading and no training."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import statistics
from verified import collect,dataset,digest,options,protocol,read

VARIANTS={(0,0,0):"token_only",(1,0,0):"channel",(0,1,0):"phase",(0,0,1):"joint",(1,1,0):"channel_phase",(1,0,1):"channel_joint",(0,1,1):"phase_joint",(1,1,1):"full"}
DATASETS=["ETTh1","ETTh2","ETTm1","ETTm2","ECL","Solar","Traffic","Weather","PEMS03","PEMS04","PEMS07","PEMS08"]
def horizons(ds):return [12,24,48,96] if ds.startswith("PEMS") else [96,192,336,720]
def stats(values,complete=False):
    return {"mean":statistics.mean(values),"sd_population":statistics.pstdev(values) if complete else None,
        "sd_sample":statistics.stdev(values) if complete and len(values)>1 else None,
        "sd_status":"six declared seeds complete" if complete else "withheld until all six declared seeds complete"}

def aggregate(manifest,output,mode,reference=None):
    verified,progress=collect(manifest,output,mode)
    measured=[]
    for item in verified:
        cfg=options(item["run"]["argv"])
        effective={k:v for k,v in item["effective_args"].items() if k not in ("checkpoints","root_path","model_id","des","gpu","num_workers","seed","random_seed","fix_seed")}
        compatibility=digest(effective)
        ablation_compatibility=digest({k:v for k,v in effective.items() if not k.startswith("armor_")})
        roles=[("best-validation","machine_metrics_best.json"),("release","machine_metrics.json")] if mode=="dual" else [("release" if mode=="release" else "best-validation","machine_metrics.json")]
        for role,file in roles:
            metrics=read(Path(item["attempt"])/file)
            measured.append({"run_id":item["run"]["id"],"group":item["run"]["group"],"dataset":dataset(item),
                "seed":item["run"]["seed"],"pred_len":int(cfg["pred_len"]),"seq_len":int(cfg["seq_len"]),
                "model":cfg["model"],"extensions":item["run"].get("extensions",{}),"checkpoint_role":role,
                "config_signature":compatibility,"ablation_comparability_signature":ablation_compatibility,
                "evaluation":item["evaluation"],"protocol":protocol(item,role),"dataset_sha256":item["state"]["dataset"]["sha256"],
                "mse":metrics["mse"],"mae":metrics["mae"],"rmse":metrics["rmse"],"attempt":item["attempt"]})
    main_groups=defaultdict(list)
    for row in measured:
        if row["group"]=="multivariate_forecasting":
            main_groups[(row["protocol"],row["dataset"],row["pred_len"],row["dataset_sha256"],row["config_signature"])].append(row)
    by_horizon=[]
    seed_averages=[]
    for key,rows in sorted(main_groups.items()):
        p,ds,h,sha,config=key
        seeds=[r["seed"] for r in rows]
        if len(seeds)!=len(set(seeds)):raise ValueError("duplicate main dataset/horizon/seed")
        expected={r["seed"] for r in manifest["runs"] if r["group"]=="multivariate_forecasting" and options(r["argv"])["model_id"].split("_")[0].lower()==ds.lower() and int(options(r["argv"])["pred_len"])==h}
        entry={"protocol":p,"dataset":ds,"horizon":h,"checkpoint_role":rows[0]["checkpoint_role"],"evaluation":rows[0]["evaluation"],
            "seeds":sorted(seeds),"n_seeds":len(seeds),"expected_seeds":sorted(expected),"all_manifest_seeds_complete":set(seeds)==expected,
            "six_seed_complete":len(set(seeds))==6 and set(seeds)==expected,"dataset_sha256":sha,"config_signature":config}
        entry.update({m:stats([r[m] for r in rows],entry["six_seed_complete"]) for m in ("mse","mae")})
        published=(reference or {}).get("table5_six_seed_results",{}).get(ds,{}).get(str(h))
        if published:entry["published_reference_not_measured"]=published
        by_horizon.append(entry)
    across=defaultdict(list)
    for row in measured:
        if row["group"]=="multivariate_forecasting":across[(row["protocol"],row["dataset"],row["seed"],row["dataset_sha256"])].append(row)
    for (p,ds,seed,sha),rows in sorted(across.items()):
        hs=[r["pred_len"] for r in rows]
        if sorted(hs)!=horizons(ds):continue
        seed_averages.append({"protocol":p,"dataset":ds,"seed":seed,"checkpoint_role":rows[0]["checkpoint_role"],"evaluation":rows[0]["evaluation"],"dataset_sha256":sha,
            "mse":statistics.mean(r["mse"] for r in rows),"mae":statistics.mean(r["mae"] for r in rows),"horizons":hs,
            "horizon_config_signatures":{str(r["pred_len"]):r["config_signature"] for r in rows}})
    average_groups=defaultdict(list)
    for row in seed_averages:average_groups[(row["protocol"],row["dataset"],row["dataset_sha256"],digest(row["horizon_config_signatures"]))].append(row)
    table1=[]
    for (p,ds,sha,configs),rows in sorted(average_groups.items()):
        expected={r["seed"] for r in manifest["runs"] if r["group"]=="multivariate_forecasting" and options(r["argv"])["model_id"].split("_")[0].lower()==ds.lower()}
        complete=len({r["seed"] for r in rows})==6 and {r["seed"] for r in rows}==expected
        table1.append({"protocol":p,"dataset":ds,"checkpoint_role":rows[0]["checkpoint_role"],"evaluation":rows[0]["evaluation"],"n_complete_seeds":len(rows),"seeds":sorted(r["seed"] for r in rows),
            "expected_seeds":sorted(expected),"six_seed_complete":complete,"horizon_config_set_signature":configs,
            "mse":stats([r["mse"] for r in rows],complete),"mae":stats([r["mae"] for r in rows],complete),
            "interpretation":"First average four horizons within each seed; SD then describes seed-to-seed horizon-averaged scores."})
    # Radar uses paired seed(s), never full's six-seed mean against single-seed ablations.
    radar_groups=defaultdict(list)
    for row in measured:
        if row["group"] not in ("multivariate_forecasting","armor_ablation"):continue
        flags=tuple(int(row["extensions"].get("armor_"+k,1)) for k in ("channel","phase","joint"))
        variant=VARIANTS[flags]
        radar_groups[(row["protocol"],row["seed"],row["dataset"],row["pred_len"],row["dataset_sha256"],row["ablation_comparability_signature"])].append((variant,row))
    paired=defaultdict(list)
    for (p,seed,ds,h,sha,config),pairs in radar_groups.items():
        names=[v for v,r in pairs]
        if len(names)!=len(set(names)):raise ValueError("duplicate ablation combination")
        if set(names)!=set(VARIANTS.values()):continue
        for variant,row in pairs:paired[(p,seed,ds,variant)].append(row)
    per_ds=[]
    for (p,seed,ds,v),rows in sorted(paired.items()):
        if sorted(r["pred_len"] for r in rows)!=horizons(ds):continue
        per_ds.append({"protocol":p,"seed":seed,"dataset":ds,"variant":v,"mae":statistics.mean(r["mae"] for r in rows)})
    axes={"ETT":["ETTh1","ETTh2","ETTm1","ETTm2"],"PEMS":["PEMS03","PEMS04","PEMS07","PEMS08"],**{x:[x] for x in ("ECL","Solar","Traffic","Weather")}}
    radar=[]
    for p,seed in sorted({(r["protocol"],r["seed"]) for r in per_ds}):
        for axis,members in axes.items():
            values={}
            for v in VARIANTS.values():
                rows=[r for r in per_ds if r["protocol"]==p and r["seed"]==seed and r["variant"]==v and r["dataset"] in members]
                if sorted(r["dataset"] for r in rows)!=sorted(members):break
                values[v]=statistics.mean(r["mae"] for r in rows)
            if len(values)!=8:continue
            lo,hi=min(values.values()),max(values.values())
            radar.append({"protocol":p,"seed":seed,"axis":axis,"datasets":members,"raw_mae":values,
                "normalized_improvement":{v:(hi-x)/(hi-lo) if hi>lo else 0.0 for v,x in values.items()},
                "min_mae":lo,"max_mae":hi,"token_only_is_maximum":values["token_only"]==hi,
                "normalization":"(max_over_8_variants - MAE) / (max-min); lower errors farther outward. Equal range gives 0. No assumption token_only is worst."})
    return {"kind":"verified_measured_aggregation","planned_training_runs":len(manifest["runs"]),"verified_training_runs":len(verified),"states":dict(Counter(r["state"] for r in progress)),"progress":progress,
        "measured_rows":measured,"table5_by_horizon":by_horizon,"main_horizon_averages_per_seed":seed_averages,"table1_horizon_averages":table1,"ablation_dataset_horizon_averages":per_ds,"figure3_radar":radar,
        "complete_radar_groups":[{"protocol":p,"seed":s} for p,s in sorted({(r["protocol"],r["seed"]) for r in radar}) if len([r for r in radar if r["protocol"]==p and r["seed"]==s])==6],
        "scope_note":"Only verified completed runs enter measured outputs; no paper values are used as measurements. Zero verified runs means no forecasting result reproduced. Full-paper completion also needs baselines and checkpoint analyses.",
        "sd_note":"Both population and sample SD reported because paper ddof is unspecified; horizon-averaged SD is computed per seed, not across horizons."}

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--manifest",required=True,type=Path);p.add_argument("--output",required=True,type=Path)
    p.add_argument("--mode",required=True,choices=["release","best-checkpoint","dual"])
    p.add_argument("--reference",type=Path);p.add_argument("--report",required=True,type=Path)
    a=p.parse_args();result=aggregate(read(a.manifest),a.output,a.mode,read(a.reference) if a.reference else None)
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"verified_training_runs":result["verified_training_runs"],"states":result["states"],"radar_complete_groups":len(result["complete_radar_groups"])}))
if __name__=="__main__":main()
