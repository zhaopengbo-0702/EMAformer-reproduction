"""Armor ablations and explicitly documented interpretations of paper variants.

Import with both the project root and the official EMAformer directory on
sys.path. Fully enabled Transformer variants preserve upstream parameter keys,
initialization order, and forecasts. New variants are reproduction additions.
"""

import copy
from types import SimpleNamespace

import torch
import torch.nn as nn

from model.EMAformer import Model as OfficialEMAformer
from model.iInformer import Model as OfficialIInformer
from model.iReformer import Model as OfficialIReformer
from model.iFlowformer import Model as OfficialIFlowformer
from model.Transformer import Model as OfficialTransformer
from model.Informer import Model as OfficialInformer
from model.Reformer import Model as OfficialReformer
from model.Flowformer import Model as OfficialFlowformer


ABLATIONS = {
    "token_only": (0, 0, 0),
    "channel": (1, 0, 0),
    "phase": (0, 1, 0),
    "joint": (0, 0, 1),
    "channel_phase": (1, 1, 0),
    "channel_joint": (1, 0, 1),
    "phase_joint": (0, 1, 1),
    "full": (1, 1, 1),
}


def _get(config, field, default):
    value = getattr(config, field, default)
    return default if value is None else value


def _binary(value, field):
    if isinstance(value, str):
        if value not in ("0", "1"):
            raise ValueError("{} must be 0 or 1".format(field))
        value = int(value)
    if value not in (0, 1):
        raise ValueError("{} must be 0 or 1".format(field))
    return bool(value)


class ArmorMixin:
    """Mix into an official enhanced model without changing default weights."""

    _official_kind = "EMAformer"

    def __init__(self, config):
        # Always consume the original initialization RNG stream first. Disabled
        # modules are removed afterwards so they neither train nor inflate the
        # actual parameter count, while surviving initial weights stay matched.
        super().__init__(config)
        variant = _get(config, "armor_variant", None)
        if variant is not None and variant not in ABLATIONS:
            raise ValueError("Unknown armor_variant: {}".format(variant))
        flags = ABLATIONS.get(variant, (1, 1, 1))
        self.armor_channel = _binary(_get(config, "armor_channel", flags[0]), "armor_channel")
        self.armor_phase = _binary(_get(config, "armor_phase", flags[1]), "armor_phase")
        self.armor_joint = _binary(_get(config, "armor_joint", flags[2]), "armor_joint")
        self.backbone = _get(config, "backbone", "transformer")
        if self.backbone not in ("transformer", "mlp"):
            raise ValueError("backbone must be transformer or mlp")
        self.mean_input = _binary(_get(config, "mean_input", 0), "mean_input")
        self.mean_input_strategy = _get(config, "mean_input_strategy", "zero")
        if self.mean_input_strategy not in ("zero", "window_mean"):
            raise ValueError("mean_input_strategy must be zero or window_mean")
        self.mean_input_apply = _get(config, "mean_input_apply", "both")
        if self.mean_input_apply not in ("both", "train", "eval"):
            raise ValueError("mean_input_apply must be both, train or eval")
        self.phase_offset = int(_get(config, "phase_offset", 0))
        self._phase_name = "phase_embedding" if self._official_kind == "EMAformer" else "temporal_embedding"
        self._joint_name = "joint_embedding" if self._official_kind == "EMAformer" else "cc_embedding"
        if not self.armor_channel:
            self.register_parameter("channel_embedding", None)
        if not self.armor_phase:
            setattr(self, self._phase_name, None)
        if not self.armor_joint:
            setattr(self, self._joint_name, None)
        if self.backbone == "mlp":
            # Paper: remove Transformer, retain only prediction head. In
            # particular do not apply Identity(tokens)+tokens, which doubles
            # the input and is not a pure prediction-head intervention.
            self.encoder = nn.Identity()

    def extension_spec(self):
        return {
            "official_kind": self._official_kind,
            "armor_channel": int(self.armor_channel),
            "armor_phase": int(self.armor_phase),
            "armor_joint": int(self.armor_joint),
            "backbone": self.backbone,
            "mean_input": int(self.mean_input),
            "mean_input_strategy": self.mean_input_strategy,
            "mean_input_apply": self.mean_input_apply,
            "phase_offset": self.phase_offset,
            "mean_input_original_instance_statistics_retained": True,
            "mlp_skip_residual": "projector(tokens), no doubled residual",
        }

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, phase):
        use_norm = self.use_norm if self._official_kind == "EMAformer" else True
        if use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        intervene = self.mean_input and (
            self.mean_input_apply == "both"
            or (self.mean_input_apply == "train" and self.training)
            or (self.mean_input_apply == "eval" and not self.training)
        )
        if intervene:
            if self.mean_input_strategy == "zero":
                x_enc = torch.zeros_like(x_enc)
            else:
                x_enc = x_enc.mean(1, keepdim=True).expand_as(x_enc)

        batch, _, channels = x_enc.shape
        if channels != self.enc_in:
            raise ValueError("enc_in={} but input has {} channels".format(self.enc_in, channels))
        phase = phase.reshape(-1)
        if phase.numel() != batch:
            raise ValueError("phase requires one index per batch item")
        if self.phase_offset:
            phase = (phase + self.phase_offset) % self.cycle_len
        tokens = self.enc_embedding(x_enc, x_mark_enc)[:, :channels, :]
        if self.armor_channel:
            tokens = tokens + self.channel_embedding.expand(batch, channels, -1)
        if self.armor_phase:
            tokens = tokens + getattr(self, self._phase_name)(phase.view(-1, 1).expand(batch, channels))
        if self.armor_joint:
            tokens = tokens + getattr(self, self._joint_name)(phase).reshape(batch, self.enc_in, self.d_model)

        if self.backbone == "mlp":
            projected = self.projector(tokens)
            attns = []
        else:
            encoded, attns = self.encoder(tokens, attn_mask=None)
            if self._official_kind == "EMAformer":
                encoded = encoded + tokens
            projected = self.projector(encoded)
        output = projected.permute(0, 2, 1)[:, :, :channels]
        if use_norm:
            output = output * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            output = output + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return output, attns


class EMAformer(ArmorMixin, OfficialEMAformer):
    _official_kind = "EMAformer"


class IInformer(ArmorMixin, OfficialIInformer):
    _official_kind = "iInformer"


class IReformer(ArmorMixin, OfficialIReformer):
    _official_kind = "iReformer"


class IFlowformer(ArmorMixin, OfficialIFlowformer):
    _official_kind = "iFlowformer"


class EMAformerMLP(EMAformer):
    def __init__(self, config):
        config = copy.copy(config)
        config.backbone = "mlp"
        super().__init__(config)


class EMAformerMeanInput(EMAformer):
    def __init__(self, config):
        config = copy.copy(config)
        config.mean_input = 1
        super().__init__(config)


class IgnoreDatasetPhaseMixin:
    """Adapt temporal baselines to the five-argument forecasting engine."""

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, cycle_index=None, mask=None):
        return super().forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=mask)


class OriginalTransformer(IgnoreDatasetPhaseMixin, OfficialTransformer):
    pass


class OriginalInformer(IgnoreDatasetPhaseMixin, OfficialInformer):
    pass


class OriginalReformer(IgnoreDatasetPhaseMixin, OfficialReformer):
    pass


class OriginalFlowformer(IgnoreDatasetPhaseMixin, OfficialFlowformer):
    pass


def register_models(model_dict):
    """Mutate the official module-valued registry and return it for convenience."""
    for name, model in {
        "EMAformer": EMAformer,
        "iInformer": IInformer,
        "iReformer": IReformer,
        "iFlowformer": IFlowformer,
        "EMAformerMLP": EMAformerMLP,
        "EMAformerMeanInput": EMAformerMeanInput,
        "OriginalTransformer": OriginalTransformer,
        "OriginalInformer": OriginalInformer,
        "OriginalReformer": OriginalReformer,
        "OriginalFlowformer": OriginalFlowformer,
    }.items():
        model_dict[name] = SimpleNamespace(Model=model)
    return model_dict


def add_extension_args(parser):
    """Optional CLI helper; wrappers may inject the same fields via config JSON."""
    parser.add_argument("--armor_variant", choices=tuple(ABLATIONS), default=None)
    for field in ("armor_channel", "armor_phase", "armor_joint"):
        parser.add_argument("--" + field, type=int, choices=(0, 1), default=None)
    parser.add_argument("--backbone", choices=("transformer", "mlp"), default="transformer")
    parser.add_argument("--mean_input", type=int, choices=(0, 1), default=0)
    parser.add_argument("--mean_input_strategy", choices=("zero", "window_mean"), default="zero")
    parser.add_argument("--mean_input_apply", choices=("both", "train", "eval"), default="both")
    parser.add_argument("--phase_offset", type=int, default=0)
    return parser
