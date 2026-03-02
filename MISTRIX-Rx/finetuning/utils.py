import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> None:
    """Configure root logger for both console and optional file logging."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def canonicalize_label_text(text: str) -> str:
    """Normalize labels for stable class mapping and prompts."""
    return " ".join((text or "").strip().split())


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Dict[str, Any], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: Path | str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(records: list[Dict[str, Any]], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def pick_mixed_precision(arg: str) -> str:
    """Resolve precision mode, defaulting to bf16 when supported else fp16 on CUDA."""
    if arg != "auto":
        return arg
    if not torch.cuda.is_available():
        return "no"
    if torch.cuda.is_bf16_supported():
        return "bf16"
    return "fp16"
