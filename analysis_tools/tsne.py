"""Fixed-seed CPU t-SNE of an embedding archive tied to a verified checkpoint."""
import argparse
import inspect
import json
import os
from pathlib import Path
from verified import checkpoint,read,sha256,validate

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument("--manifest",required=True,type=Path);p.add_argument("--output",required=True,type=Path)
    p.add_argument("--mode",required=True,choices=["release","best-checkpoint","dual"]);p.add_argument("--analysis-dir",required=True,type=Path)
    p.add_argument("--destination",required=True,type=Path);p.add_argument("--seed",type=int,default=0)
    p.add_argument("--max-points",type=int,default=1024);p.add_argument("--perplexity",type=float,default=30)
    a=p.parse_args();meta=read(a.analysis_dir/"analysis.json")
    run=next(r for r in read(a.manifest)["runs"] if r["id"]==meta["run_id"])
    item=validate(a.output,a.mode,run);cp=checkpoint(item,meta["checkpoint_role"])
    archive=a.analysis_dir/"learned_embeddings.npz"
    if meta["training_signature"]!=item["state"]["signature"] or meta["checkpoint_sha256"]!=sha256(cp) or sha256(archive)!=meta["generated_artifacts"][archive.name]:raise ValueError("embedding/checkpoint provenance mismatch")
    if not 3<=a.max_points<=2048 or a.perplexity<=0:p.error("bounded max-points3..2048 and positive perplexity required")
    os.environ.setdefault("OMP_NUM_THREADS","2");os.environ.setdefault("MKL_NUM_THREADS","2")
    import numpy as np
    import sklearn
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from threadpoolctl import threadpool_limits
    a.destination.mkdir(parents=True,exist_ok=False);results={}
    with np.load(archive,allow_pickle=False) as embeddings,threadpool_limits(limits=2):
        for name,array in embeddings.items():
            if array.ndim!=2 or array.shape[0]<3:continue
            indices=np.arange(min(len(array),a.max_points));values=array[indices].astype(np.float64)
            if not np.isfinite(values).all():raise ValueError("nonfinite embedding")
            components=min(50,values.shape[0]-1,values.shape[1]);values=PCA(n_components=components,svd_solver="full").fit_transform(values)
            perplexity=min(a.perplexity,float(len(values)-1))
            kwargs=dict(n_components=2,perplexity=perplexity,init="pca",learning_rate="auto",random_state=a.seed,method="barnes_hut",n_jobs=2)
            kwargs["max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"]=1000
            coordinates=TSNE(**kwargs).fit_transform(values)
            np.savez_compressed(a.destination/(name+"_tsne.npz"),coordinates=coordinates,source_indices=indices)
            results[name]={"total_source_points":len(array),"plotted_points":len(indices),"selected":"first max-points, deterministic; no post-hoc selection","PCA_components":components,"perplexity":perplexity,"seed":a.seed,"iterations":1000,"init":"pca","learning_rate":"auto","n_jobs":2}
    report={"kind":"checkpoint_embedding_projection_not_training","checkpoint_analysis":str(a.analysis_dir),"source_archive_sha256":sha256(archive),"training_signature":meta["training_signature"],"checkpoint_sha256":meta["checkpoint_sha256"],"sklearn":sklearn.__version__,"arrays":results,"limitations":["Paper t-SNE hyperparameters and selected horizon/checkpoint were not disclosed; these fixed settings are reproduction choices.","Coordinates are qualitative projections; spatial clusters alone do not establish forecast causation."]}
    (a.destination/"tsne.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"arrays":list(results),"training_runs_created":0}))
if __name__=="__main__":main()
