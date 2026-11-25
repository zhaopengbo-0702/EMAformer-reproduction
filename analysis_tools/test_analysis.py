"""Synthetic integrity checks; fixtures are never scientific measured results."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from aggregate import aggregate
from checkpoint import safe_checkpoint_load
from verified import digest,sha256

class AnalysisChecks(unittest.TestCase):
    def fixture(self,root,seeds=(2023,),horizons=(96,192,336,720),variants=((1,1,1),)):
        runs=[]
        for seed in seeds:
            for h in horizons:
                for flags in variants:
                    ext={} if flags==(1,1,1) else dict(zip(("armor_channel","armor_phase","armor_joint"),flags))
                    run={"id":f"test-h{h}-s{seed}-v{''.join(map(str,flags))}","group":"multivariate_forecasting" if not ext else "armor_ablation","seed":seed,"extensions":ext,
                        "argv":["--model_id",f"ETTh1_96_{h}","--model","EMAformer","--seq_len","96","--pred_len",str(h)]}
                    runs.append(run);folder=root/"dual"/run["id"]/"attempt-001";folder.mkdir(parents=True)
                    effective={"num_workers":2,"model":"EMAformer","seq_len":96,"pred_len":h,"d_model":256,**ext}
                    write=lambda name,value:(folder/name).write_text(json.dumps(value),encoding="utf-8")
                    write("run.json",run);write("effective_args.json",effective)
                    for suffix in ("","_best"):
                        write("machine_metrics"+suffix+".json",{"mse":.3+(seed-2023)*.001,"mae":.4-.01*sum(flags),"rmse":.55,"evaluation":"streaming"})
                    cp=folder/"checkpoints";cp.mkdir();(cp/"checkpoint.pth").write_bytes(b"synthetic only");(cp/"checkpoint_best.pth").write_bytes(b"synthetic best only")
                    signature=digest({"run":run,"mode":"dual","evaluation":"streaming","code":"codehash","dataset_sha256":"datahash","instrumentation":"instrumenthash","num_workers":2})
                    artifact=lambda f:{"path":f,"sha256":sha256(folder/f)}
                    status={"state":"complete","returncode":0,"run_id":run["id"],"mode":"dual","signature":signature,"source_sha256":"codehash","instrumentation_sha256":"instrumenthash","dataset":{"sha256":"datahash"},
                        "metrics":artifact("machine_metrics.json"),"secondary_metrics":artifact("machine_metrics_best.json"),"checkpoints":[artifact("checkpoints/checkpoint.pth"),artifact("checkpoints/checkpoint_best.pth")]}
                    write("status.json",status);(folder.parent/"index.json").write_text(json.dumps({"attempt":"attempt-001","signature":signature}),encoding="utf-8")
        return {"runs":runs}

    def test_zero_results_stay_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            m={"runs":[{"id":"pending"}]};r=aggregate(m,Path(tmp),"dual")
            self.assertEqual(r["verified_training_runs"],0);self.assertEqual(r["table5_by_horizon"],[])

    def test_missing_seed_sd_withheld_and_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);m=self.fixture(root)
            r=aggregate(m,root,"dual");self.assertEqual(len(r["table5_by_horizon"]),8)
            self.assertTrue(all(x["mse"]["sd_population"] is None for x in r["table5_by_horizon"]))
            cp=root/"dual"/m["runs"][0]["id"]/"attempt-001/checkpoints/checkpoint.pth";cp.write_bytes(b"tampered")
            self.assertEqual(aggregate(m,root,"dual")["verified_training_runs"],3)

    def test_six_seeds_and_roles_keep_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);m=self.fixture(root,seeds=range(2023,2029))
            r=aggregate(m,root,"dual")
            self.assertEqual(r["verified_training_runs"],24)
            self.assertTrue(all(x["six_seed_complete"] for x in r["table5_by_horizon"]))
            self.assertEqual(len(r["table1_horizon_averages"]),2)
            self.assertTrue(all(x["n_complete_seeds"]==6 for x in r["table1_horizon_averages"]))

    def test_duplicate_seed_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);m=self.fixture(root)
            m["runs"].append(copy.deepcopy(m["runs"][0]))
            with self.assertRaisesRegex(ValueError,"duplicate"):aggregate(m,root,"dual")

    def test_config_drift_cannot_form_six_seed_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);m=self.fixture(root,seeds=range(2023,2029),horizons=(96,))
            folder=root/"dual"/m["runs"][0]["id"]/"attempt-001"
            effective=json.loads((folder/"effective_args.json").read_text());effective["d_model"]=512
            (folder/"effective_args.json").write_text(json.dumps(effective))
            r=aggregate(m,root,"dual")
            self.assertFalse(any(x["six_seed_complete"] for x in r["table5_by_horizon"]))
            self.assertTrue(all(x["mse"]["sd_population"] is None for x in r["table5_by_horizon"]))

    def test_radar_requires_all8_and_same_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);m=self.fixture(root,variants=((1,1,1),(0,0,0)))
            r=aggregate(m,root,"dual");self.assertEqual(r["figure3_radar"],[])

    def test_safe_loader_torch20_and_mmap_signatures(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"checkpoint.pth";path.write_bytes(b"fixture")
            calls=[]
            def load20(f,map_location=None,*,weights_only=False,**pickle_load_args):
                calls.append(dict(map_location=map_location,weights_only=weights_only,**pickle_load_args));return {"state":"fixture"}
            _,strategy=safe_checkpoint_load(SimpleNamespace(load=load20),path)
            self.assertFalse(strategy["mmap"]);self.assertEqual(calls[-1],{"map_location":"cpu","weights_only":True})
            def load21(f,map_location=None,*,weights_only=False,mmap=None,**pickle_load_args):
                calls.append(dict(map_location=map_location,weights_only=weights_only,mmap=mmap));return {}
            _,strategy=safe_checkpoint_load(SimpleNamespace(load=load21),path)
            self.assertTrue(strategy["mmap"]);self.assertTrue(calls[-1]["weights_only"])
            def unsafe(f,map_location=None,**kwargs):return {}
            with self.assertRaisesRegex(RuntimeError,"weights_only"):safe_checkpoint_load(SimpleNamespace(load=unsafe),path)
            def fail(f,map_location=None,*,weights_only=False,**kwargs):raise TypeError("safe load failure")
            with self.assertRaisesRegex(TypeError,"safe load failure"):safe_checkpoint_load(SimpleNamespace(load=fail),path)

if __name__=="__main__":unittest.main()
