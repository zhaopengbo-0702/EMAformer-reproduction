"""Bounded-memory metric reductions; explicit difference from NumPy float32 sum."""
from __future__ import annotations


class StreamingMetrics:
    def __init__(self):
        self.count = 0
        self.absolute_sum = 0.0
        self.squared_sum = 0.0
        self.percentage_sum = 0.0
        self.squared_percentage_sum = 0.0
        self.percentage_finite = True
        self.squared_percentage_finite = True

    def update(self, pred, true):
        import numpy as np
        # Match original float32 subtraction/square before the more stable
        # float64 global reduction. Do not average unequal batch means.
        error = pred - true
        self.count += error.size
        self.absolute_sum += float(np.abs(error).sum(dtype=np.float64))
        self.squared_sum += float(np.square(error).sum(dtype=np.float64))
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            relative = error / true
            absolute = np.abs(relative).sum(dtype=np.float64)
            squared = np.square(relative).sum(dtype=np.float64)
        if np.isfinite(absolute):
            self.percentage_sum += float(absolute)
        else:
            self.percentage_finite = False
        if np.isfinite(squared):
            self.squared_percentage_sum += float(squared)
        else:
            self.squared_percentage_finite = False

    def result(self):
        import math
        if not self.count:
            raise ValueError("No test elements accumulated")
        mse = self.squared_sum / self.count
        return {"mae": self.absolute_sum / self.count, "mse": mse, "rmse": math.sqrt(mse),
                "mape": self.percentage_sum / self.count if self.percentage_finite else None,
                "mspe": self.squared_percentage_sum / self.count if self.squared_percentage_finite else None,
                "metric_elements": self.count,
                "metric_reduction": "float32 elementwise error and square; float64 global sums divided by element count"}


def install_streaming_test(experiment_class, mode, secondary=False):
    """Keep official data/forward/slicing; replace only array accumulation."""
    import json
    from pathlib import Path
    import time
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from common import atomic_json, utcnow

    def evaluate(self, setting, test=0):
        test_data, test_loader = self._get_data(flag="test")
        if test:
            path = Path(self.args.checkpoints) / setting / "checkpoint.pth"
            self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        metrics = StreamingMetrics()
        samples_pred, samples_true, sample_count = [], [], 0
        windows = 0
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle in test_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)
                if "PEMS" in self.args.data or "Solar" in self.args.data:
                    batch_x_mark, batch_y_mark = None, None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                decoder = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                decoder = torch.cat([batch_y[:, :self.args.label_len, :], decoder], dim=1).float().to(self.device)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, decoder, batch_y_mark, batch_cycle)
                else:
                    outputs = self.model(batch_x, batch_x_mark, decoder, batch_y_mark, batch_cycle)
                if self.args.output_attention:
                    outputs = outputs[0]
                first = -1 if self.args.features == "MS" else 0
                pred = outputs[:, -self.args.pred_len:, first:].detach().cpu().numpy()
                true = batch_y[:, -self.args.pred_len:, first:].detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = pred.shape
                    pred = test_data.inverse_transform(pred.squeeze(0)).reshape(shape)
                    true = test_data.inverse_transform(true.squeeze(0)).reshape(shape)
                metrics.update(pred, true)
                windows += pred.shape[0]
                take = min(16 - sample_count, pred.shape[0])
                if take:
                    samples_pred.append(pred[:take].copy())
                    samples_true.append(true[:take].copy())
                    sample_count += take
        torch.cuda.synchronize()
        record = metrics.result()
        record.update({"prediction_shape": [windows, self.args.pred_len, pred.shape[-1]],
                       "mode": mode, "evaluation": "streaming", "finished_utc": utcnow(),
                       "test_seconds": time.monotonic() - start,
                       "max_cuda_memory_allocated": torch.cuda.max_memory_allocated(),
                       "max_cuda_memory_reserved": torch.cuda.max_memory_reserved()})
        suffix = "_best" if getattr(self, "_reproduction_secondary", False) else ""
        atomic_json("machine_metrics" + suffix + ".json", record)
        sample_pred, sample_true = np.concatenate(samples_pred), np.concatenate(samples_true)
        np.savez_compressed("prediction_sample" + suffix + ".npz", pred=sample_pred, true=sample_true)
        figure, axis = plt.subplots()
        axis.plot(sample_true[0, :, -1], label="GroundTruth")
        axis.plot(sample_pred[0, :, -1], label="Prediction")
        axis.legend()
        figure.savefig("prediction_sample" + suffix + ".png", dpi=120, bbox_inches="tight")
        plt.close(figure)
        print("mse:{}, mae:{} (streaming, {})".format(record["mse"], record["mae"], mode), flush=True)
        return record

    experiment_class.test = evaluate
