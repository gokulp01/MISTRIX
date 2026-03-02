from __future__ import annotations

import logging
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional

import torch
import torch.nn as nn
import transformers

LOGGER = logging.getLogger(__name__)

PIXTRAL_MODEL_ALIASES = {
    # Mistral-hosted Pixtral checkpoints are released in Mistral format.
    # For local HF Transformers fine-tuning we use the community-converted
    # Pixtral checkpoint that preserves the original weights architecture.
    "pixtral-12b-latest": "mistral-community/pixtral-12b",
    "mistralai/Pixtral-12B-2409": "mistral-community/pixtral-12b",
    "mistralai/Pixtral-12B-Base-2409": "mistral-community/pixtral-12b",
}


@dataclass
class ModeSetupResult:
    trainable_params: int
    total_params: int


def resolve_model_id(model_id: str) -> str:
    resolved = PIXTRAL_MODEL_ALIASES.get(model_id, model_id)
    if resolved != model_id:
        LOGGER.info("Resolved model id alias %s -> %s", model_id, resolved)
    return resolved


def _pick_pixtral_auto_class():
    for cls_name in ("LlavaForConditionalGeneration", "AutoModelForImageTextToText", "AutoModelForVision2Seq"):
        cls = getattr(transformers, cls_name, None)
        if cls is not None:
            return cls
    raise RuntimeError(
        "Your transformers version does not expose an image-text generation auto-model class. "
        "Upgrade transformers to a newer version."
    )


def load_pixtral_model(
    model_id: str,
    gradient_checkpointing: bool = False,
    use_4bit: bool = False,
) -> nn.Module:
    resolved_model_id = resolve_model_id(model_id)
    model_cls = _pick_pixtral_auto_class()

    kwargs = {"trust_remote_code": True}
    if torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        # Pixtral vision blocks currently require eager attention in Transformers.
        kwargs["attn_implementation"] = "eager"

    if use_4bit:
        bnb_cls = getattr(transformers, "BitsAndBytesConfig", None)
        if bnb_cls is None:
            raise RuntimeError("4-bit quantization requested but BitsAndBytesConfig is unavailable in transformers.")
        kwargs["quantization_config"] = bnb_cls(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )

    model = model_cls.from_pretrained(resolved_model_id, **kwargs)

    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        LOGGER.info("Enabled gradient checkpointing.")

    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    return model


def _set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


def _freeze_by_name_contains(model: nn.Module, name_fragments: list[str]) -> int:
    n = 0
    for name, p in model.named_parameters():
        if any(frag in name for frag in name_fragments):
            p.requires_grad = False
            n += p.numel()
    return n


def setup_training_mode(
    model: nn.Module,
    mode: str,
    freeze_vision_tower: bool = False,
) -> ModeSetupResult:
    _set_requires_grad(model, True)

    if mode not in {"sft_full", "sft_lora"}:
        raise ValueError(f"Unknown mode: {mode}")

    if freeze_vision_tower:
        n = _freeze_by_name_contains(
            model,
            ["vision_tower", "vision_model", "vision_encoder", "vision_backbone"],
        )
        LOGGER.info("Vision tower frozen by request (%s params).", f"{n:,}")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOGGER.info("Trainable params: %s / %s", f"{trainable:,}", f"{total:,}")
    return ModeSetupResult(trainable_params=trainable, total_params=total)


def maybe_prepare_for_kbit_training(model: nn.Module) -> nn.Module:
    try:
        from peft import prepare_model_for_kbit_training
    except Exception as exc:
        raise RuntimeError(
            "peft is required for 4-bit training preparation. Install peft or disable --use_4bit."
        ) from exc
    return prepare_model_for_kbit_training(model)


def maybe_apply_lora(
    model: nn.Module,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Optional[list[str]] = None,
) -> nn.Module:
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except Exception as exc:
        raise RuntimeError(
            "peft is not installed but LoRA mode was requested. Install peft or use --mode sft_full."
        ) from exc

    if target_modules is None:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    return model


def normalize_prediction_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def rank_labels_from_prediction(prediction: str, labels: list[str], top_k: int = 5) -> list[tuple[str, float]]:
    pred = normalize_prediction_text(prediction).lower()

    scored: list[tuple[str, float]] = []
    for label in labels:
        lbl = normalize_prediction_text(label).lower()
        if not lbl:
            continue

        if pred == lbl:
            score = 1.0
        elif lbl and lbl in pred:
            score = 0.95
        elif pred and pred in lbl:
            score = 0.90
        else:
            score = 0.80 * SequenceMatcher(a=pred, b=lbl).ratio()

        scored.append((label, float(score)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[: max(1, min(top_k, len(scored)))]


def map_prediction_to_label(
    prediction: str,
    labels: list[str],
    confidence_cutoff: float = 0.55,
) -> tuple[str, float]:
    ranked = rank_labels_from_prediction(prediction, labels, top_k=1)
    if not ranked:
        return "UNKNOWN", 0.0
    label, score = ranked[0]
    if score < confidence_cutoff:
        return "UNKNOWN", score
    return label, score
