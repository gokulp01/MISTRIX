"""Shared helper logic for candidate labels and mode resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from backend.utils import canonicalize_label_text, load_json

MODE_ALIASES = {
    "contrastive": "pixtral",
    "partial_unfreeze": "pixtral",
    "lora_optional": "pixtral",
    "linear_probe": "pixtral",
    "pixtral": "pixtral",
}


def load_prescription_candidates(
    prescription_file: Optional[Path], raw_items: Optional[list[str]]
) -> list[str]:
    """Merge prescription sources into a de-duplicated, canonicalized candidate list."""
    labels: list[str] = []
    seen = set()

    if prescription_file is not None:
        if not prescription_file.exists():
            raise FileNotFoundError(f"Prescription file not found: {prescription_file}")
        for line in prescription_file.read_text(encoding="utf-8").splitlines():
            label = canonicalize_label_text(line)
            if label and label not in seen:
                labels.append(label)
                seen.add(label)

    for item in raw_items or []:
        label = canonicalize_label_text(item)
        if label and label not in seen:
            labels.append(label)
            seen.add(label)

    if not labels:
        raise ValueError("No prescription candidates provided. Use --prescription_file and/or --prescription.")

    return labels


def resolve_mode(run_dir: Path, explicit_mode: Optional[str]) -> str:
    """Choose inference mode from explicit arg first, then ``config.json`` fallback."""
    raw_mode = explicit_mode
    if raw_mode is None:
        cfg_path = run_dir / "config.json"
        if cfg_path.exists():
            raw_mode = str(load_json(cfg_path).get("mode", "pixtral"))
        else:
            raw_mode = "pixtral"

    return MODE_ALIASES.get(str(raw_mode), "pixtral")
