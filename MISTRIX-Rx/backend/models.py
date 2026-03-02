"""Pixtral model loading and prediction-mapping helpers for inference."""

from __future__ import annotations

import logging
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import torch
import transformers

LOGGER = logging.getLogger(__name__)

PIXTRAL_MODEL_ALIASES = {
    "pixtral-12b-latest": "mistral-community/pixtral-12b",
    "mistralai/Pixtral-12B-2409": "mistral-community/pixtral-12b",
    "mistralai/Pixtral-12B-Base-2409": "mistral-community/pixtral-12b",
}


def resolve_model_id(model_id: str) -> str:
    """Resolve friendly/default Pixtral aliases into concrete HF model IDs."""
    resolved = PIXTRAL_MODEL_ALIASES.get(model_id, model_id)
    if resolved != model_id:
        LOGGER.info("Resolved model id alias %s -> %s", model_id, resolved)
    return resolved


def _iter_pixtral_auto_classes() -> list[Any]:
    classes = []
    for cls_name in (
        "LlavaForConditionalGeneration",
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
    ):
        cls = getattr(transformers, cls_name, None)
        if cls is not None:
            classes.append(cls)
    return classes


def _pixtral_load_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        # Pixtral vision blocks currently require eager attention in Transformers.
        kwargs["attn_implementation"] = "eager"
    return kwargs


def load_pixtral_generation_model(model_ref: str | Path):
    """Load Pixtral generation model from HF id/path across compatible auto classes."""
    kwargs = _pixtral_load_kwargs()
    errors: list[str] = []
    for cls in _iter_pixtral_auto_classes():
        try:
            return cls.from_pretrained(model_ref, **kwargs)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{cls.__name__}: {exc}")

    joined = " | ".join(errors) if errors else "No compatible model class found in transformers."
    raise RuntimeError(
        "Could not load Pixtral model with any supported class "
        "(LlavaForConditionalGeneration, AutoModelForImageTextToText, AutoModelForVision2Seq). "
        f"Details: {joined}"
    )


def normalize_prediction_text(text: str) -> str:
    """Normalize free-form generation output before label matching."""
    return " ".join((text or "").strip().split())


def rank_labels_from_prediction(prediction: str, labels: list[str], top_k: int = 5) -> list[tuple[str, float]]:
    """Rank candidate labels against generated text with lexical similarity heuristics."""
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
    """Map generated text to best label with optional abstain threshold."""
    ranked = rank_labels_from_prediction(prediction, labels, top_k=1)
    if not ranked:
        return "UNKNOWN", 0.0
    label, score = ranked[0]
    if score < confidence_cutoff:
        return "UNKNOWN", score
    return label, score
