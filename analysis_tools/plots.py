"""Plot only complete verified aggregation groups; never draw reference values as data."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument("--aggregate",type=Path,required=True);p.add_argument("--destination",type=Path,required=True)
    p.add_argument("--style-kernel",type=Path,required=True,help="figure-style/kernel.py")
    a=p.parse_args();data=json.loads(a.aggregate.read_text(encoding="utf-8"))
    if data["kind"]!="verified_measured_aggregation":raise ValueError("requires verified aggregation")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    spec=importlib.util.spec_from_file_location("figure_style",a.style_kernel);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.apply_figure_style(frame="open",sizes=(10,9,8))
    a.destination.mkdir(parents=True,exist_ok=True);artifacts=[]
    order=["ETT","PEMS","ECL","Solar","Traffic","Weather"]
    for group in data["complete_radar_groups"]:
        rows=[r for r in data["figure3_radar"] if r["protocol"]==group["protocol"] and r["seed"]==group["seed"]]
        values={r["axis"]:r for r in rows}
        if set(values)!=set(order):raise ValueError("radar axis completeness mismatch")
        theta=np.linspace(0,2*np.pi,len(order),endpoint=False);theta=np.r_[theta,theta[0]]
        figure,ax=plt.subplots(figsize=(7.5,6),subplot_kw={"projection":"polar"})
        variants=list(values[order[0]]["raw_mae"])
        for i,v in enumerate(variants):
            numbers=[values[k]["normalized_improvement"][v] for k in order];numbers.append(numbers[0])
            ax.plot(theta,numbers,label=v.replace("_"," "),linewidth=2 if v=="full" else 1,marker="o",markersize=3)
        ax.set_xticks(theta[:-1],order);ax.set_ylim(0,1.03);ax.set_yticks([0,.5,1]);ax.set_title(f"Embedding ablation, paired seed {group['seed']}",pad=20)
        ax.legend(loc="upper left",bbox_to_anchor=(1.1,1.05));figure.text(.02,.01,"Each axis: mean MAE over all four horizons; ETT/PEMS averaged over four datasets.\nRadius = (max MAE - MAE)/(max-min); larger means lower error. All eight variants complete.",fontsize=9)
        figure.tight_layout(rect=(0,.07,1,1));stem=f"figure3_radar_{group['protocol'][:12]}_s{group['seed']}"
        for ext in ("png","svg","pdf"):figure.savefig(a.destination/(stem+"."+ext))
        plt.close(figure);artifacts.append(stem)
    report={"verified_training_runs":data["verified_training_runs"],"radar_figures":artifacts,"notes":["No placeholder or paper-reference data plotted.","If empty, verified results do not yet cover a complete radar group.","Render images and inspect layout before delivery."]}
    (a.destination/"plot_status.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report))
if __name__=="__main__":main()
