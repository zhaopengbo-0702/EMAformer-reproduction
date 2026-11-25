"""Verify phase0 entropy analysis receipts and form Table2 only with full horizons."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
from verified import checkpoint,dataset,options,protocol,read,sha256,validate

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument("--manifest",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--mode",required=True,choices=["release","best-checkpoint","dual"]);p.add_argument("--analyses",type=Path,nargs="+",required=True);p.add_argument("--report",type=Path,required=True)
    a=p.parse_args();manifest=read(a.manifest);byid={r["id"]:r for r in manifest["runs"]};rows=[];rejected=[]
    for path in a.analyses:
        try:
            meta=read(path/"analysis.json");item=validate(a.output,a.mode,byid[meta["run_id"]]);role=meta["checkpoint_role"]
            if meta["checkpoint_sha256"]!=sha256(checkpoint(item,role)) or meta["training_signature"]!=item["state"]["signature"]:raise ValueError("checkpoint/analysis identity mismatch")
            entropy=meta["attention_entropy"]
            if entropy["base"]!=2 or entropy["phase"]!=0 or entropy["selected_samples"]<=0:raise ValueError("not a valid phase0 base2 entropy observation")
            ext=item["run"].get("extensions",{});flags=tuple(int(ext.get("armor_"+k,1)) for k in ("channel","phase","joint"))
            variant={(0,0,0):"token_only",(1,0,0):"token_channel",(1,1,1):"full"}.get(flags)
            if variant is None:raise ValueError("variant is outside Table2")
            cfg=options(item["run"]["argv"])
            rows.append({"run_id":meta["run_id"],"dataset":dataset(item),"horizon":int(cfg["pred_len"]),"seed":item["run"]["seed"],"variant":variant,"checkpoint_role":role,"protocol":protocol(item,role),
                "entropy":entropy["entropy"],"base":2,"selected_samples":entropy["selected_samples"],"eligible_samples":meta["sample_selection"]["eligible_windows"],"all_phase0_windows":entropy["selected_samples"]==meta["sample_selection"]["eligible_windows"],"phase_offset":entropy["phase_offset"],"analysis_sha256":sha256(path/"analysis.json")})
        except (OSError,KeyError,ValueError) as error:rejected.append({"analysis":str(path),"reason":str(error)})
    grouped=defaultdict(list)
    for r in rows:grouped[(r["protocol"],r["dataset"],r["seed"],r["variant"],r["phase_offset"])].append(r)
    table=[]
    for (pro,ds,seed,v,offset),measurements in sorted(grouped.items()):
        hs=[r["horizon"] for r in measurements]
        if len(hs)!=len(set(hs)):raise ValueError("duplicate entropy horizon; refuse double counting")
        if sorted(hs)!=[96,192,336,720]:continue
        table.append({"protocol":pro,"dataset":ds,"seed":seed,"variant":v,"phase_offset":offset,"four_horizons_complete":True,"all_phase0_windows":all(r["all_phase0_windows"] for r in measurements),
            "entropy_horizon_mean":statistics.mean(r["entropy"] for r in measurements),"selected_samples":sum(r["selected_samples"] for r in measurements),"aggregation_choice":"equal average of four horizon entropies; paper horizon aggregation unspecified"})
    result={"kind":"verified_entropy_analysis_not_training","observations":rows,"table2_candidate_rows":table,"rejected":rejected,"notes":["Only verified checkpoint analyses included. Analysis receipt hash is computed now; original queue did not seal subsequent analysis receipts.","Bounded sample analyses remain partial; original paper aggregation details not disclosed.","Do not compare differing phase_offset/horizon/seed/checkpoint roles as identical protocol."]}
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"observations":len(rows),"four_horizon_candidates":len(table),"rejected":len(rejected),"training_runs_created":0}))
if __name__=="__main__":main()
