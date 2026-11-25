"""Meaningful CPU checks for added model semantics; not a training experiment."""

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "EMAformer"))
local_deps = ROOT / "tools" / "model_check_deps"
if local_deps.exists():
    sys.path.insert(0, str(local_deps))

import torch

from model.EMAformer import Model as UpEMAformer
from model.iInformer import Model as UpInformer
from model.iReformer import Model as UpReformer
from model.iFlowformer import Model as UpFlowformer
from reproduction_extensions.models import (
    ABLATIONS, EMAformer, IInformer, IReformer, IFlowformer,
    OriginalTransformer, OriginalInformer, OriginalReformer, OriginalFlowformer,
)
from reproduction_extensions.analysis import AttentionEntropy, AttentionCapture, parameter_overhead


def config(**kwargs):
    values = dict(seq_len=8, pred_len=4, output_attention=True, use_norm=1,
                  d_model=16, cycle=6, enc_in=8, embed="timeF", freq="h",
                  dropout=0.0, class_strategy="projection", factor=2,
                  n_heads=4, d_ff=24, e_layers=2, activation="gelu",
                  output_proj_dropout=0.0, dec_in=8, c_out=8,
                  channel_independence=False, d_layers=1, distil=False,
                  label_len=4)
    values.update(kwargs)
    return SimpleNamespace(**values)


def main():
    torch.set_num_threads(1)
    report = {"device": "cpu", "torch": torch.__version__, "checks": []}
    x = torch.randn(2, 8, 8)
    marks = torch.randn(2, 8, 4)
    phase = torch.tensor([0, 3], dtype=torch.int64)
    args = (x, marks, None, None, phase)
    for label, official, extended in (
        ("EMAformer", UpEMAformer, EMAformer),
        ("iInformer", UpInformer, IInformer),
        ("iReformer", UpReformer, IReformer),
        ("iFlowformer", UpFlowformer, IFlowformer),
    ):
        torch.manual_seed(2023)
        original = official(config()).eval()
        torch.manual_seed(2023)
        candidate = extended(config()).eval()
        assert original.state_dict().keys() == candidate.state_dict().keys()
        for name, tensor in original.state_dict().items():
            assert torch.equal(tensor, candidate.state_dict()[name]), name
        torch.manual_seed(30)
        expected = original(*args)[0]
        torch.manual_seed(30)
        actual = candidate(*args)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # Nontrivial gradient check: all enabled embeddings must influence the
        # forecast and receive finite nonzero gradients.
        candidate.train()
        candidate(*args)[0].square().mean().backward()
        names = ["channel_embedding", candidate._phase_name + ".weight", candidate._joint_name + ".weight"]
        params = dict(candidate.named_parameters())
        for name in names:
            assert params[name].grad is not None and torch.isfinite(params[name].grad).all()
            assert params[name].grad.abs().sum() > 0
        report["checks"].append({"name": label + " default exact parity and embedding gradient", "passed": True})

    for variant, flags in ABLATIONS.items():
        model = EMAformer(config(armor_variant=variant)).eval()
        output = model(*args)[0]
        assert output.shape == (2, 4, 8) and torch.isfinite(output).all()
        counts = parameter_overhead(model)
        expected_count = flags[0] * 8 * 16 + flags[1] * 6 * 16 + flags[2] * 6 * 8 * 16
        assert counts["armor_parameters"] == expected_count
    report["checks"].append({"name": "all eight ablation shapes and actual parameter removal", "passed": True})

    model = EMAformer(config(backbone="mlp", use_norm=0)).eval()
    x_grad = x.clone().requires_grad_(True)
    output = model(x_grad, None, None, None, phase)[0]
    output[:, :, 0].sum().backward()
    assert x_grad.grad[:, :, 0].abs().sum() > 0
    assert torch.count_nonzero(x_grad.grad[:, :, 1:]) == 0, "MLP must not mix channels"
    assert parameter_overhead(model)["groups"]["encoder"] == 0
    report["checks"].append({"name": "MLP removes encoder and has no cross-channel mixing", "passed": True})

    model = EMAformer(config(mean_input=1, mean_input_strategy="zero", use_norm=0)).eval()
    a = model(x, None, None, None, phase)[0]
    b = model(x + torch.randn_like(x) * 5, None, None, None, phase)[0]
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    report["checks"].append({"name": "zero standardized history independent of history when use_norm=0", "passed": True})

    model = EMAformer(config(phase_offset=-1)).eval()
    source = EMAformer(config()).eval()
    source.load_state_dict(model.state_dict())
    torch.testing.assert_close(model(*args)[0], source(x, marks, None, None, (phase - 1) % 6)[0], atol=0, rtol=0)
    report["checks"].append({"name": "phase offset matches explicit shifted lookup", "passed": True})

    uniform = torch.full((2, 3, 7, 7), 1.0 / 7)
    entropy = AttentionEntropy()
    entropy.update([uniform], torch.tensor([0, 1]))
    assert abs(entropy.result()["entropy"] - math.log2(7)) < 1e-6
    assert entropy.result()["selected_samples"] == 1
    deterministic = torch.eye(7).expand(2, 3, 7, 7)
    entropy2 = AttentionEntropy()
    entropy2.update([deterministic], torch.tensor([0, 0]))
    assert entropy2.result()["entropy"] == 0
    report["checks"].append({"name": "entropy uniform/deterministic bounds and phase filtering", "passed": True})
    model = EMAformer(config()).eval()
    with AttentionCapture(model, max_samples=1) as capture:
        model(*args)
        model(*args)
    assert capture.count == 1 and capture.samples[0].shape == (1, 8, 8)
    report["checks"].append({"name": "attention capture honors sample bound", "passed": True})
    for adapter in (OriginalTransformer, OriginalInformer, OriginalReformer, OriginalFlowformer):
        model = adapter(config()).eval()
        decoder = torch.randn(2, 8, 8)
        decoder_marks = torch.randn(2, 8, 4)
        torch.manual_seed(5)
        a = model(x, marks, decoder, decoder_marks, phase)
        torch.manual_seed(5)
        b = model(x, marks, decoder, decoder_marks, torch.zeros_like(phase))
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert a.shape == (2, 4, 8)
    report["checks"].append({"name": "four temporal adapters ignore dataset phase and preserve output shape", "passed": True})
    report["passed"] = True
    output = ROOT / "outputs" / "extension_model_checks.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
