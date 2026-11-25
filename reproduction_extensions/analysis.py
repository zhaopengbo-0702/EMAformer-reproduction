"""Bounded exports and measurement helpers; these do not run training."""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch


def parameter_overhead(model):
    """Actual parameter counts after disabled components have been removed."""
    groups = {"armor_channel": 0, "armor_phase": 0, "armor_joint": 0,
              "encoder": 0, "projector": 0, "other": 0}
    total = trainable = 0
    for name, param in model.named_parameters():
        count = param.numel()
        total += count
        trainable += count if param.requires_grad else 0
        if name == "channel_embedding":
            group = "armor_channel"
        elif name.startswith(("phase_embedding.", "temporal_embedding.")):
            group = "armor_phase"
        elif name.startswith(("joint_embedding.", "cc_embedding.")):
            group = "armor_joint"
        elif name.startswith("encoder."):
            group = "encoder"
        elif name.startswith("projector."):
            group = "projector"
        else:
            group = "other"
        groups[group] += count
    armor = sum(groups[k] for k in ("armor_channel", "armor_phase", "armor_joint"))
    return {"total_parameters": total, "trainable_parameters": trainable,
            "armor_parameters": armor,
            "armor_fraction_of_total": armor / total if total else 0,
            "groups": groups}


def export_embeddings(model, directory, channel=0):
    """Export learned weights without loading the full joint embedding to RAM.

    Only one joint channel is exported by default, matching Fig. 6b's channel 0.
    No PCA/t-SNE choice is silently imposed; the saved tensors remain reusable.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    output = {}
    channel_param = getattr(model, "channel_embedding", None)
    if channel_param is not None:
        output["channel"] = channel_param.detach().cpu().numpy()
    phase_module = getattr(model, "phase_embedding", None)
    if phase_module is None:
        phase_module = getattr(model, "temporal_embedding", None)
    if phase_module is not None:
        output["phase"] = phase_module.weight.detach().cpu().numpy()
    joint_module = getattr(model, "joint_embedding", None)
    if joint_module is None:
        joint_module = getattr(model, "cc_embedding", None)
    if joint_module is not None:
        if not 0 <= channel < model.enc_in:
            raise ValueError("joint export channel is out of range")
        start = channel * model.d_model
        output["joint_channel_{}".format(channel)] = (
            joint_module.weight[:, start:start + model.d_model].detach().cpu().numpy()
        )
    target = directory / "learned_embeddings.npz"
    np.savez_compressed(str(target), **output)
    meta = {"joint_channel": channel,
            "arrays": {name: list(value.shape) for name, value in output.items()},
            "parameters": parameter_overhead(model),
            "extension_spec": model.extension_spec() if hasattr(model, "extension_spec") else None}
    (directory / "learned_embeddings.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


class AttentionEntropy:
    """Last-layer head-averaged entropy at a chosen phase; constant memory.

    Phase is the dataset phase unless phase_offset is supplied explicitly.
    Evaluation attention must contain probabilities before head averaging.
    Base 2 follows the paper's numerical statement log(7) approx 2.81.
    """

    def __init__(self, phase=0, base=2.0, phase_offset=0, cycle=None):
        if base <= 0 or base == 1:
            raise ValueError("entropy base must be positive and different from 1")
        self.phase = phase
        self.base = base
        self.phase_offset = phase_offset
        self.cycle = cycle
        self.entropy_sum = 0.0
        self.rows = 0
        self.samples = 0
        self.skipped_samples = 0
        self.channels = None

    @torch.no_grad()
    def update(self, attention_layers, phase):
        if not attention_layers or attention_layers[-1] is None:
            raise ValueError("last-layer attention unavailable; set output_attention=True on a supported model")
        attention = attention_layers[-1]
        if attention.ndim != 4 or attention.shape[-1] != attention.shape[-2]:
            raise ValueError("expected attention [batch,heads,channels,channels]")
        phase = phase.reshape(-1).to(attention.device)
        if self.phase_offset:
            if not self.cycle:
                raise ValueError("cycle is required with phase_offset")
            phase = (phase + self.phase_offset) % self.cycle
        if len(phase) != len(attention):
            raise ValueError("phase and attention batch sizes differ")
        selected = phase == self.phase
        self.skipped_samples += int((~selected).sum().item())
        if not selected.any():
            return
        # Head average BEFORE entropy, exactly as Appendix B specifies.
        probs = attention[selected].detach().double().mean(dim=1)
        if not torch.isfinite(probs).all() or torch.any(probs < 0):
            raise ValueError("attention probabilities must be finite and nonnegative")
        row_sum = probs.sum(dim=-1)
        if not torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-5, rtol=1e-5):
            raise ValueError("attention rows are not probabilities; run model.eval() to disable attention dropout")
        logs = torch.where(probs > 0, probs.clamp_min(torch.finfo(probs.dtype).tiny).log(), torch.zeros_like(probs))
        entropy = -(probs * logs).sum(dim=-1) / math.log(self.base)
        self.entropy_sum += entropy.sum().item()
        self.rows += entropy.numel()
        self.samples += int(selected.sum().item())
        self.channels = probs.shape[-1]

    def result(self):
        return {"entropy": self.entropy_sum / self.rows if self.rows else None,
                "base": self.base, "phase": self.phase,
                "phase_offset": self.phase_offset, "cycle": self.cycle,
                "selected_samples": self.samples, "averaged_rows": self.rows,
                "skipped_samples": self.skipped_samples, "channels": self.channels,
                "maximum_entropy": math.log(self.channels, self.base) if self.channels else None,
                "protocol": "last layer; average heads, then row entropy, then average rows and selected samples"}


class AttentionCapture:
    """Optional hook for bounded attention inspection without changing outputs.

    Keep this off during throughput measurements. The hook captures at most
    max_samples and moves only head-averaged selected samples to CPU.
    """

    def __init__(self, model, max_samples=4):
        if max_samples < 1:
            raise ValueError("max_samples must be at least one")
        self.model = model
        self.max_samples = max_samples
        self.samples = []
        self.count = 0
        self.handle = None

    def __enter__(self):
        layers = getattr(getattr(self.model, "encoder", None), "attn_layers", None)
        if not layers:
            raise ValueError("model has no Transformer attention layers")
        # EncoderLayer returns (tokens, attention). FullAttention must be
        # configured with output_attention=True before model construction.
        def capture(module, args, output):
            attention = output[1]
            if attention is None:
                raise ValueError("attention unavailable; construct model with output_attention=True")
            remaining = self.max_samples - self.count
            if remaining > 0:
                value = attention[:remaining].detach().mean(dim=1).cpu()
                self.samples.append(value)
                self.count += len(value)
        self.handle = layers[-1].register_forward_hook(capture)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.handle is not None:
            self.handle.remove()

    def export(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        values = torch.cat(self.samples, dim=0).numpy() if self.samples else np.empty((0,))
        np.save(str(path), values)
        return {"path": str(path), "shape": list(values.shape), "head_averaged": True}


def measure_inference(model, forward_args, warmup=2, repeats=5):
    """Single-model forward latency with bounded repeats, CUDA synchronization.

    Reports model forwards only: no data I/O, no training, no hidden batching.
    Call on exactly the same hardware, shapes, dtype and output_attention setting
    for each comparator. Does not claim GPU results when run on CPU.
    """
    if repeats < 1 or warmup < 0:
        raise ValueError("invalid measurement repeat count")
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    elapsed = []
    try:
        with torch.no_grad():
            for _ in range(warmup):
                model(*forward_args)
            sync()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            for _ in range(repeats):
                sync()
                start = time.perf_counter()
                model(*forward_args)
                sync()
                elapsed.append(time.perf_counter() - start)
        return {"device": str(device), "warmup": warmup, "repeats": repeats,
                "forward_seconds_median": float(np.median(elapsed)),
                "forward_seconds_min": min(elapsed), "forward_seconds": elapsed,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
                "parameters": parameter_overhead(model),
                "protocol": "eval/no_grad; model forward only; synchronized CUDA when available"}
    finally:
        model.train(was_training)
