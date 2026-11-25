"""Bounded read-only checkpoint analyses; never constructs optimizer or trains."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from verified import checkpoint,dataset,frozen_snapshot,read,sha256,validate

def safe_checkpoint_load(torch_module,path,max_mib=1536):
    """Use only explicitly supported safe torch.load arguments.

    Torch2.0 has weights_only but not mmap. Never retry without weights_only.
    """
    import inspect
    path=Path(path)
    if path.stat().st_size>max_mib*1024**2:
        raise ValueError("checkpoint exceeds explicit CPU budget")
    parameters=inspect.signature(torch_module.load).parameters
    if "weights_only" not in parameters:
        raise RuntimeError("torch.load weights_only support required; no unsafe pickle fallback")
    kwargs={"map_location":"cpu","weights_only":True}
    use_mmap="mmap" in parameters
    if use_mmap:kwargs["mmap"]=True
    state=torch_module.load(path,**kwargs)
    return state,{"weights_only":True,"map_location":"cpu","mmap":use_mmap,
        "strategy":"weights_only_cpu_mmap" if use_mmap else "weights_only_cpu_regular",
        "checkpoint_bytes":path.stat().st_size,"maximum_checkpoint_mib":max_mib}

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--manifest",required=True,type=Path);p.add_argument("--output",required=True,type=Path)
    p.add_argument("--mode",required=True,choices=["release","best-checkpoint","dual"])
    p.add_argument("--run-id",required=True);p.add_argument("--checkpoint-role",required=True,choices=["release","best-validation"])
    p.add_argument("--analysis-dir",required=True,type=Path)
    p.add_argument("--actions",nargs="+",choices=["embeddings","entropy","forward-latency"],default=["embeddings"])
    p.add_argument("--device",choices=["cpu","cuda:0"],default="cpu")
    p.add_argument("--data-root",type=Path);p.add_argument("--max-samples",type=int,default=64)
    p.add_argument("--batch-size",type=int,default=1);p.add_argument("--max-checkpoint-mib",type=int,default=1536)
    p.add_argument("--training-lock",type=Path,default=Path(tempfile.gettempdir())/"emaformer-single-training.lock")
    a=p.parse_args()
    if not 1<=a.max_samples<=512 or not 1<=a.batch_size<=4: p.error("max-samples must be1..512 and batch-size1..4")
    if a.mode=="release" and a.checkpoint_role!="release":p.error("release has no best-validation artifact")
    if a.mode=="best-checkpoint" and a.checkpoint_role!="best-validation":p.error("best-checkpoint has no release artifact")
    matches=[r for r in read(a.manifest)["runs"] if r["id"]==a.run_id]
    if len(matches)!=1:p.error("run-id must identify one manifest entry")
    item=validate(a.output,a.mode,matches[0]);cp=checkpoint(item,a.checkpoint_role)
    if cp.stat().st_size>a.max_checkpoint_mib*1024**2:raise ValueError("checkpoint exceeds explicit CPU budget; select larger budget deliberately")
    source,snapshot=frozen_snapshot(item)
    sys.path.insert(0,str(source));sys.path.insert(0,str(snapshot))
    os.environ.setdefault("OMP_NUM_THREADS","2");os.environ.setdefault("MKL_NUM_THREADS","2")
    import numpy as np
    import torch
    torch.set_num_threads(2)
    if a.device=="cuda:0" and (not torch.cuda.is_available() or torch.cuda.device_count()!=1):
        raise RuntimeError("GPU analysis requires exactly one platform-visible GPU; preserve allocation mask")
    from common import exclusive_lock
    # Same lock as actual training child: analysis cannot overlap GPU training.
    # CPU analyses also use it to avoid heavy simultaneous checkpoint loading.
    with exclusive_lock(a.training_lock):
        state,load_strategy=safe_checkpoint_load(torch,cp,a.max_checkpoint_mib)
        cfg=SimpleNamespace(**item["effective_args"])
        result={"kind":"verified_checkpoint_analysis_not_training","run_id":a.run_id,"dataset":dataset(item),
            "checkpoint_role":a.checkpoint_role,"checkpoint_path":str(cp),"checkpoint_sha256":sha256(cp),
            "training_signature":item["state"]["signature"],"source_sha256":item["state"]["source_sha256"],
            "device":a.device,"torch":torch.__version__,"threads":2,"actions":a.actions,"checkpoint_load":load_strategy,
            "limitations":["Analysis creates no training results.","Bounded forward latency is not full-test time or training-epoch time.","Bounded entropy samples do not prove exact Table2 replication; original horizon/seed/sample aggregation was not disclosed."]}
        arrays={}
        if "embeddings" in a.actions:
            if "channel_embedding" in state:arrays["channel"]=state["channel_embedding"].detach().cpu().numpy().copy()
            for key in ("phase_embedding.weight","temporal_embedding.weight"):
                if key in state:arrays["phase"]=state[key].detach().cpu().numpy().copy();break
            for key in ("joint_embedding.weight","cc_embedding.weight"):
                if key in state:
                    arrays["joint_channel_0"]=state[key][:,:cfg.d_model].detach().cpu().numpy().copy();break
            result["embedding_shapes"]={k:list(v.shape) for k,v in arrays.items()}
        if set(a.actions)&{"entropy","forward-latency"}:
            if a.data_root is None:p.error("--data-root required for inference analysis")
            from experiments.exp_basic import Exp_Basic
            from reproduction_extensions import register_models
            from reproduction_extensions.analysis import AttentionEntropy,measure_inference,parameter_overhead
            # Registry construction is read-only; do not instantiate Exp_Basic,
            # whose device initializer rewrites CUDA_VISIBLE_DEVICES.
            import importlib
            names=["Transformer","Informer","Reformer","Flowformer","Flashformer","iTransformer","iInformer","iReformer","iFlowformer","iFlashformer","EMAformer"]
            registry={n:importlib.import_module("model."+n) for n in names}
            if item["run"].get("extensions") or cfg.model in ("EMAformerMLP","EMAformerMeanInput") or cfg.model.startswith("Original"):
                register_models(registry)
            entropy_requested="entropy" in a.actions
            if entropy_requested and (cfg.model!="EMAformer" or getattr(cfg,"backbone","transformer")!="transformer"):
                raise ValueError("Table2 entropy is supported only for EMAformer Transformer variants")
            cfg.output_attention=entropy_requested
            model=registry[cfg.model].Model(cfg).float()
            model.load_state_dict(state,strict=True)
            del state
            model.to(a.device).eval()
            result["parameters"]=parameter_overhead(model)
            from data_provider.data_loader import Dataset_ETT_hour,Dataset_ETT_minute,Dataset_Custom,Dataset_Solar,Dataset_PEMS
            providers={"ETTh1":Dataset_ETT_hour,"ETTh2":Dataset_ETT_hour,"ETTm1":Dataset_ETT_minute,"ETTm2":Dataset_ETT_minute,"custom":Dataset_Custom,"Solar":Dataset_Solar,"PEMS":Dataset_PEMS}
            from manifest import options
            opts=options(item["run"]["argv"])
            relative=Path(opts["root_path"].replace("\\","/"))
            if relative.parts and relative.parts[0] in ("dataset","data"):relative=Path(*relative.parts[1:])
            datafile=(a.data_root.resolve()/relative/cfg.data_path).resolve();datafile.relative_to(a.data_root.resolve())
            if sha256(datafile)!=item["state"]["dataset"]["sha256"]:raise ValueError("data hash differs from completed training")
            data=providers[cfg.data](root_path=str(datafile.parent),data_path=datafile.name,flag="test",
                size=[cfg.seq_len,cfg.label_len,cfg.pred_len],features=cfg.features,target=cfg.target,timeenc=0 if cfg.embed!="timeF" else 1,freq=cfg.freq,cycle=cfg.cycle)
            phase_offset=int(getattr(cfg,"phase_offset",0))
            candidates=[i for i in range(len(data)) if int((data.cycle_index[i+cfg.seq_len]+phase_offset)%cfg.cycle)==0] if entropy_requested else list(range(min(len(data),a.max_samples)))
            selected=candidates[:a.max_samples]
            if not selected:raise ValueError("no phase0 test windows under declared indexing")
            from torch.utils.data import DataLoader,Subset
            loader=DataLoader(Subset(data,selected),batch_size=a.batch_size,num_workers=0,shuffle=False)
            result["sample_selection"]={"total_test_windows":len(data),"eligible_windows":len(candidates),"selected_indices":selected,"phase":"model effective phase0" if entropy_requested else "first test windows","max_samples":a.max_samples,"batch_size":a.batch_size,"phase_offset":phase_offset,"data_sha256":sha256(datafile)}
            entropy=AttentionEntropy(phase=0,base=2,phase_offset=phase_offset,cycle=cfg.cycle) if entropy_requested else None
            with torch.no_grad():
                for bx,by,bxm,bym,phase in loader:
                    bx=bx.float().to(a.device);by=by.float().to(a.device);phase=phase.int().to(a.device)
                    bxm,bym=(None,None) if cfg.data in ("PEMS","Solar") else (bxm.float().to(a.device),bym.float().to(a.device))
                    dec=torch.cat([by[:,:cfg.label_len,:],torch.zeros_like(by[:,-cfg.pred_len:,:])],dim=1)
                    forward=(bx,bxm,dec,bym,phase)
                    if entropy is not None:
                        pred,attns=model(*forward);entropy.update(attns,phase)
                    if "forward-latency" in a.actions:
                        result["forward_latency"]=measure_inference(model,forward,warmup=2,repeats=5)
                        result["forward_latency"]["attention_enabled"]=cfg.output_attention
                        # One batch only; repeats do not traverse all test data.
                        if entropy is None:break
            if entropy is not None:result["attention_entropy"]=entropy.result()
            result["hardware"]={"gpu":torch.cuda.get_device_name(0) if a.device=="cuda:0" else None,"gpu_visible_count":torch.cuda.device_count() if a.device=="cuda:0" else 0}
        a.analysis_dir.mkdir(parents=True,exist_ok=False)
        if arrays:np.savez_compressed(a.analysis_dir/"learned_embeddings.npz",**arrays)
        result["generated_artifacts"]={p.name:sha256(p) for p in a.analysis_dir.iterdir() if p.is_file()}
        (a.analysis_dir/"analysis.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        print(json.dumps({"analysis_dir":str(a.analysis_dir),"actions":a.actions,"training_runs_created":0}))
if __name__=="__main__":main()
