"""Execute one upstream training run with explicit instrumentation and mode."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from common import atomic_json, exclusive_lock, utcnow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--run-json", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=["release", "best-checkpoint", "dual"])
    parser.add_argument("--evaluation", choices=["original", "streaming"], default="streaming")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--training-lock", required=True, type=Path)
    args = parser.parse_args()
    # Held by the actual training child, not only its supervising queue. A
    # killed supervisor therefore cannot let a restart overlap an orphan.
    import atexit
    child_lock = exclusive_lock(args.training_lock)
    child_lock.__enter__()
    atexit.register(child_lock.__exit__, None, None, None)
    repo = args.repo.resolve()
    run = json.loads(args.run_json.read_text(encoding="utf-8"))
    # Do not clear the platform-assigned mask or select host physical GPU ids.
    visible_mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_mask is not None and len([s for s in visible_mask.split(",") if s.strip()]) != 1:
        raise RuntimeError(f"Expected exactly one platform-visible GPU, got mask {visible_mask!r}")
    import torch
    import numpy as np
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"Single-GPU policy requires exactly 1 visible CUDA GPU; found {torch.cuda.device_count()}")
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from experiments.exp_basic import Exp_Basic
    from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
    from utils.tools import EarlyStopping
    import experiments.exp_long_term_forecasting as exp_module
    from manifest import replace_option, options

    def acquire(self):
        if self.args.use_multi_gpu:
            raise RuntimeError("Multi-GPU mode prohibited")
        return torch.device("cuda:0")
    Exp_Basic._acquire_device = acquire
    if args.mode in ("best-checkpoint", "dual"):
        original_save = EarlyStopping.save_checkpoint
        def cloned_save(self, val_loss, model, path):
            original_save(self, val_loss, model, path)
            clone = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if args.mode == "best-checkpoint":
                self.checkpoint_in_gpu = clone
            else:
                # Separate secondary artifact; original shallow-reference
                # checkpoint remains unchanged for release evaluation.
                torch.save(clone, Path(path) / "checkpoint_best.pth")
        EarlyStopping.save_checkpoint = cloned_save

    original_visual = exp_module.visual
    def close_visual(*visual_args, **visual_kwargs):
        original_visual(*visual_args, **visual_kwargs)
        import matplotlib.pyplot as plt
        plt.close("all")
    exp_module.visual = close_visual

    original_build = Exp_Long_Term_Forecast._build_model
    def build(self):
        for key, value in run.get("extensions", {}).items():
            setattr(self.args, key, value)
        if run.get("extensions") or self.args.model in ("EMAformerMLP", "EMAformerMeanInput") or self.args.model.startswith("Original"):
            import reproduction_extensions
            reproduction_extensions.register_models(self.model_dict)
        self.args.gpu = 0
        self.args.use_gpu = True
        self.args.use_multi_gpu = False
        atomic_json("effective_args.json", vars(self.args))
        return original_build(self)
    Exp_Long_Term_Forecast._build_model = build

    original_metric = exp_module.metric
    start = time.monotonic()
    def metric(pred, true):
        values = original_metric(pred, true)
        names = ["mae", "mse", "rmse", "mape", "mspe"]
        record = {name: (float(value) if np.isfinite(value) else None) for name, value in zip(names, values)}
        record.update({"prediction_shape": list(pred.shape), "target_shape": list(true.shape),
                       "seed": run["seed"], "mode": args.mode, "extensions": run.get("extensions", {}),
                       "finished_utc": utcnow(), "elapsed_seconds": time.monotonic() - start,
                       "max_cuda_memory_allocated": torch.cuda.max_memory_allocated(),
                       "max_cuda_memory_reserved": torch.cuda.max_memory_reserved()})
        suffix = "_best" if getattr(exp_module, "_reproduction_secondary", False) else ""
        atomic_json("machine_metrics" + suffix + ".json", record)
        # Compact evidence of exact samplewise scoring; avoid huge full Traffic arrays.
        np.savez_compressed("prediction_sample" + suffix + ".npz", pred=pred[:16], true=true[:16])
        return values
    exp_module.metric = metric
    if args.evaluation == "streaming":
        from streaming import install_streaming_test
        install_streaming_test(Exp_Long_Term_Forecast, args.mode)
    original_test = Exp_Long_Term_Forecast.test
    def test_with_secondary(self, setting, test=0):
        result = original_test(self, setting, test=test)
        if args.evaluation == "streaming":
            primary_path = Path("machine_metrics.json")
            primary_record = json.loads(primary_path.read_text(encoding="utf-8"))
            primary_record.update({"seed": run["seed"], "extensions": run.get("extensions", {}),
                                   "checkpoint_role": "release" if args.mode in ("release", "dual") else "best-validation"})
            atomic_json(primary_path, primary_record)
        if args.mode == "dual":
            self.model.load_state_dict(torch.load(Path(self.args.checkpoints) / setting / "checkpoint_best.pth", map_location=self.device))
            self._reproduction_secondary = True
            exp_module._reproduction_secondary = True
            try:
                original_test(self, setting, test=0)
                secondary_path = Path("machine_metrics_best.json")
                secondary_record = json.loads(secondary_path.read_text(encoding="utf-8"))
                secondary_record.update({"seed": run["seed"], "extensions": run.get("extensions", {}),
                                         "checkpoint_role": "best-validation"})
                atomic_json(secondary_path, secondary_record)
            finally:
                self._reproduction_secondary = False
                exp_module._reproduction_secondary = False
        return result
    Exp_Long_Term_Forecast.test = test_with_secondary

    # Official progress already emits one epoch timing line. Capture the actual
    # synchronized CUDA peak per epoch without changing training math.
    import builtins
    original_print = builtins.print
    epochs = []
    original_train = Exp_Long_Term_Forecast.train
    def resource_train(self, setting):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        return original_train(self, setting)
    Exp_Long_Term_Forecast.train = resource_train
    original_vali = Exp_Long_Term_Forecast.vali
    def resource_vali(self, *vali_args, **vali_kwargs):
        result = original_vali(self, *vali_args, **vali_kwargs)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        return result
    Exp_Long_Term_Forecast.vali = resource_vali
    def resource_print(*values, **kwargs):
        message = " ".join(str(value) for value in values)
        if message.startswith("Epoch: ") and " cost time: " in message:
            torch.cuda.synchronize()
            left, seconds = message.split(" cost time: ", 1)
            epochs.append({"epoch": int(left.split(": ", 1)[1]), "official_epoch_seconds": float(seconds),
                "max_cuda_memory_allocated": torch.cuda.max_memory_allocated(),
                "max_cuda_memory_reserved": torch.cuda.max_memory_reserved()})
            atomic_json("epoch_resources.json", {"epochs": epochs, "timing": "upstream train loop wall time; CUDA sync before peak capture"})
            torch.cuda.reset_peak_memory_stats()
        original_print(*values, **kwargs)
    builtins.print = resource_print
    argv = list(run["argv"])
    opts = options(argv)
    if "use_multi_gpu" in opts or opts.get("itr", "1") != "1":
        raise RuntimeError("Manifest requests parallel/repeated upstream invocation")
    relative_data = Path(opts["root_path"].replace("\\", "/"))
    parts = relative_data.parts
    if parts and parts[0] in ("dataset", "data"):
        relative_data = Path(*parts[1:])
    elif relative_data.is_absolute() or ".." in parts:
        raise RuntimeError("Manifest data path must be under the dataset root")
    argv = replace_option(argv, "root_path", args.data_root.resolve() / relative_data)
    argv = replace_option(argv, "checkpoints", Path("checkpoints").resolve())
    argv = replace_option(argv, "num_workers", args.num_workers)
    argv = replace_option(argv, "gpu", 0)
    atomic_json("environment.json", {"time_utc": utcnow(), "python": sys.version,
                "platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__,
                "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                "gpu": torch.cuda.get_device_name(0), "gpu_count": torch.cuda.device_count(),
                "platform_cuda_mask": visible_mask, "checkpoint_mode": args.mode, "evaluation": args.evaluation,
                "dataset": json.loads(Path("dataset.json").read_text(encoding="utf-8")),
                "instrumentation": ["logical CUDA index, preserve allocation mask", "seed override", "machine metrics", "compact prediction sample"],
                "argv": argv, "run": run})
    try:
        freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        Path("pip-freeze.txt").write_text(freeze, encoding="utf-8")
    except subprocess.SubprocessError:
        pass
    source = (repo / "run.py").read_text(encoding="utf-8")
    if source.count("fix_seed = 2023") != 1:
        raise RuntimeError("Upstream seed anchor changed; refusing silent patch")
    source = source.replace("fix_seed = 2023", f"fix_seed = {int(run['seed'])}")
    sys.argv = [str(repo / "run.py")] + argv
    exec(compile(source, str(repo / "run.py"), "exec"), {"__name__": "__main__", "__file__": str(repo / "run.py")})


if __name__ == "__main__":
    main()
