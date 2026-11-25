"""Fail-closed validation shared by analysis entry points."""
from __future__ import annotations
import json
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/"reproduction"))
from common import sha256, valid_metrics
from manifest import digest, options

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def inside(root,relative):
    path=(root/relative).resolve()
    path.relative_to(root.resolve())
    return path

def validate(output,mode,run):
    """Validate identity, queue signature, metrics and every checkpoint hash.

    Returns metadata only. Never imports torch or deserializes a checkpoint.
    """
    root=Path(output).resolve()/mode/run["id"]
    if any(c in run["id"] for c in ("/","\\",":")):
        raise ValueError("unsafe run id")
    index=read(root/"index.json")
    attempt=inside(root,index["attempt"])
    state=read(attempt/"status.json")
    if state["state"]!="complete":
        raise ValueError("latest attempt is "+state["state"])
    if state.get("returncode")!=0 or state["run_id"]!=run["id"] or state["mode"]!=mode:
        raise ValueError("inconsistent completion identity")
    if read(attempt/"run.json")!=run:
        raise ValueError("stored run differs from requested manifest")
    effective=read(attempt/"effective_args.json")
    metrics=read(attempt/"machine_metrics.json")
    evaluation=metrics.get("evaluation","original")
    expected=digest({"run":run,"mode":mode,"evaluation":evaluation,"code":state["source_sha256"],
        "dataset_sha256":state["dataset"]["sha256"],"instrumentation":state["instrumentation_sha256"],
        "num_workers":effective["num_workers"]})
    if expected!=state["signature"] or expected!=index["signature"]:
        raise ValueError("queue signature does not match manifest/evaluation/provenance")
    if not valid_metrics(metrics):
        raise ValueError("invalid metrics")
    artifacts=[state["metrics"]]+state.get("checkpoints",[])
    if not state.get("checkpoints"):
        raise ValueError("checkpoint missing")
    if mode=="dual":
        artifacts.append(state["secondary_metrics"])
        if not valid_metrics(read(inside(attempt,state["secondary_metrics"]["path"]))):
            raise ValueError("invalid secondary metrics")
    for artifact in artifacts:
        path=inside(attempt,artifact["path"])
        if not path.is_file() or path.stat().st_size<=0 or sha256(path)!=artifact["sha256"]:
            raise ValueError("artifact hash failed: "+artifact["path"])
    if state["metrics"]["path"]!="machine_metrics.json":
        raise ValueError("unexpected primary metric file")
    return {"run":run,"state":state,"attempt":str(attempt),"effective_args":effective,"evaluation":evaluation}

def collect(manifest,output,mode):
    valid=[]
    progress=[]
    for run in manifest["runs"]:
        root=Path(output)/mode/run["id"]
        if not (root/"index.json").exists():
            progress.append({"id":run["id"],"state":"pending"})
            continue
        try:
            row=validate(output,mode,run)
            valid.append(row)
            progress.append({"id":run["id"],"state":"verified_complete"})
        except (OSError,ValueError,KeyError,TypeError) as error:
            progress.append({"id":run["id"],"state":"unverified_or_incomplete","reason":str(error)})
    return valid,progress

def protocol(item,role):
    s=item["state"]
    return digest({"source":s["source_sha256"],"instrumentation":s["instrumentation_sha256"],
        "mode":s["mode"],"evaluation":item["evaluation"],"checkpoint_role":role})

def dataset(item):
    stem=options(item["run"]["argv"])["model_id"].split("_")[0]
    return {"solar":"Solar","traffic":"Traffic","weather":"Weather","ecl":"ECL"}.get(stem.lower(),stem)

def checkpoint(item,role):
    name="checkpoint_best.pth" if item["state"]["mode"]=="dual" and role=="best-validation" else "checkpoint.pth"
    matches=[a for a in item["state"]["checkpoints"] if Path(a["path"]).name==name]
    if len(matches)!=1:
        raise ValueError("expected exactly one checkpoint for role "+role)
    return inside(Path(item["attempt"]),matches[0]["path"])

def frozen_snapshot(item):
    """Find and verify the exact source snapshot before importing its code."""
    invocation=read(Path(item["attempt"])/"invocation.json")["command"]
    source=Path(invocation[invocation.index("--repo")+1])
    if not source.exists():
        # Allow a complete result tree downloaded from Linux onto Windows.
        output=Path(item["attempt"]).parents[2]
        key=digest({"source":item["state"]["source_sha256"],"instrumentation":item["state"]["instrumentation_sha256"]})
        source=output/"runtime_snapshots"/key/"upstream"
    snapshot=source.parent
    ready=read(snapshot/"READY.json")
    if ready["source_sha256"]!=item["state"]["source_sha256"] or ready["instrumentation_sha256"]!=item["state"]["instrumentation_sha256"]:
        raise ValueError("snapshot identity differs from completed run")
    allowed={"data_provider","experiments","layers","model","scripts","utils"}
    files={p.relative_to(source).as_posix():sha256(p) for p in sorted(source.rglob("*"))
        if p.is_file() and p.suffix in (".py",".sh",".txt") and (p.parent==source or p.relative_to(source).parts[0] in allowed)}
    if digest(files)!=ready["source_sha256"]:
        raise ValueError("frozen upstream source hash failed")
    instrument={"reproduction/"+p.name:sha256(p) for p in (snapshot/"reproduction").glob("*.py")}
    instrument.update({"reproduction_extensions/"+p.relative_to(snapshot/"reproduction_extensions").as_posix():sha256(p)
        for p in (snapshot/"reproduction_extensions").rglob("*.py")})
    if digest(instrument)!=ready["instrumentation_sha256"]:
        raise ValueError("frozen instrumentation hash failed")
    return source,snapshot
