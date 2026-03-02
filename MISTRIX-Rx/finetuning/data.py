from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from utils import canonicalize_label_text

LOGGER = logging.getLogger(__name__)
KNOWN_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
DEFAULT_TASK_PROMPT = "Identify the medicine shown in the image. Respond with only the medicine name."


def _ensure_image_token_in_prompt(prompt: str, image_token: str) -> str:
    p = canonicalize_label_text(prompt)
    if image_token in p:
        return p
    return f"{image_token} {p}".strip()


@dataclass
class SplitFrames:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


class DataConfigError(RuntimeError):
    pass


def detect_metadata_path(data_dir: Path, metadata_path: Optional[str]) -> Path:
    if metadata_path:
        p = Path(metadata_path)
        if not p.exists():
            raise DataConfigError(f"metadata_path does not exist: {p}")
        return p

    candidates = [data_dir / "data.csv", data_dir / "metadata.csv", data_dir / "train.csv"]
    for p in candidates:
        if p.exists():
            return p

    found = sorted([p.name for p in data_dir.glob("*.csv")])
    raise DataConfigError(
        "Could not auto-detect metadata CSV. Looked for data.csv/metadata.csv/train.csv. "
        f"CSV files found: {found}"
    )


def detect_image_dir(data_dir: Path) -> Path:
    candidates = [data_dir / "data-images", data_dir / "images", data_dir / "img"]
    for p in candidates:
        if p.exists() and p.is_dir():
            return p
    raise DataConfigError(
        "Could not find image directory. Tried data-images/, images/, img/. "
        f"Directories in {data_dir}: {[p.name for p in data_dir.iterdir() if p.is_dir()]}"
    )


def infer_columns(df: pd.DataFrame, image_col: Optional[str], text_col: Optional[str]) -> tuple[str, str]:
    cols = [str(c).strip() for c in df.columns]

    if image_col and image_col not in cols:
        raise DataConfigError(f"image_col '{image_col}' not in columns: {cols}")
    if text_col and text_col not in cols:
        raise DataConfigError(f"text_col '{text_col}' not in columns: {cols}")

    if image_col is None:
        image_keywords = ["image", "img", "file", "filename", "path", "spl"]
        image_matches = [c for c in cols if any(k in c.lower() for k in image_keywords)]
        image_col = image_matches[0] if image_matches else cols[0]

    if text_col is None:
        text_keywords = ["label", "class", "name", "medicine", "drug", "pill", "text", "caption"]
        text_matches = [c for c in cols if any(k in c.lower() for k in text_keywords) and c != image_col]
        text_col = text_matches[0] if text_matches else (cols[1] if len(cols) > 1 else cols[0])

    if image_col == text_col:
        raise DataConfigError(
            f"Resolved identical image/text columns '{image_col}'. Columns available: {cols}. "
            "Pass --image_col and --text_col explicitly."
        )

    return image_col, text_col


def _build_casefold_index(image_dir: Path) -> dict[str, Path]:
    return {p.name.casefold(): p for p in image_dir.iterdir() if p.is_file()}


def resolve_image_path(raw_value: str, image_dir: Path, index: Optional[dict[str, Path]] = None) -> Optional[Path]:
    raw = str(raw_value or "").strip()
    if not raw:
        return None

    index = index or _build_casefold_index(image_dir)
    candidate = image_dir / raw
    if candidate.exists():
        return candidate

    candidate_from_index = index.get(candidate.name.casefold())
    if candidate_from_index is not None:
        return candidate_from_index

    suffix = Path(raw).suffix.lower()
    if suffix not in KNOWN_IMAGE_EXTS:
        for ext in KNOWN_IMAGE_EXTS:
            c = image_dir / f"{raw}{ext}"
            if c.exists():
                return c
            c_idx = index.get(c.name.casefold())
            if c_idx is not None:
                return c_idx

    return None


def load_dataset_frame(
    data_dir: Path,
    metadata_path: Optional[str] = None,
    image_col: Optional[str] = None,
    text_col: Optional[str] = None,
) -> tuple[pd.DataFrame, str, str, Path, Path]:
    metadata = detect_metadata_path(data_dir, metadata_path)
    image_dir = detect_image_dir(data_dir)

    try:
        df = pd.read_csv(metadata, encoding="utf-8-sig")
    except Exception as exc:
        csvs = sorted([p.name for p in data_dir.glob("*.csv")])
        raise DataConfigError(
            f"Failed to parse metadata CSV at {metadata}: {exc}. CSV candidates: {csvs}"
        ) from exc

    if df.empty:
        raise DataConfigError(f"Metadata file has no rows: {metadata}")

    image_col, text_col = infer_columns(df, image_col, text_col)
    idx = _build_casefold_index(image_dir)

    rows = []
    missing = 0
    for _, row in df.iterrows():
        raw_label = str(row[text_col] if pd.notna(row[text_col]) else "").strip()
        raw_image = str(row[image_col] if pd.notna(row[image_col]) else "").strip()
        if not raw_label or not raw_image:
            continue

        img_path = resolve_image_path(raw_image, image_dir, idx)
        if img_path is None:
            missing += 1
            continue

        label_norm = canonicalize_label_text(raw_label)
        if not label_norm:
            continue

        rows.append(
            {
                "image_path": str(img_path),
                "label_original": raw_label,
                "label_text": label_norm,
            }
        )

    if not rows:
        raise DataConfigError(
            "No valid image-text rows found after parsing metadata. "
            f"Metadata columns: {list(df.columns)}; image_col={image_col}; text_col={text_col}; "
            f"image_dir={image_dir}"
        )

    out = pd.DataFrame(rows).drop_duplicates(subset=["image_path", "label_text"]).reset_index(drop=True)
    LOGGER.info(
        "Loaded %d valid rows from %s (dropped %d missing-image rows).",
        len(out),
        metadata,
        missing,
    )
    return out, image_col, text_col, metadata, image_dir


def build_label_mapping(df: pd.DataFrame) -> tuple[list[str], dict[str, int]]:
    labels = sorted(df["label_text"].unique().tolist())
    mapping = {lbl: i for i, lbl in enumerate(labels)}
    return labels, mapping


def add_label_indices(df: pd.DataFrame, label_to_idx: dict[str, int]) -> pd.DataFrame:
    out = df.copy()
    out["label_idx"] = out["label_text"].map(label_to_idx)
    if out["label_idx"].isna().any():
        missing = out[out["label_idx"].isna()]["label_text"].unique().tolist()
        raise DataConfigError(f"Missing labels in mapping: {missing[:10]}")
    out["label_idx"] = out["label_idx"].astype(int)
    return out


def split_dataset(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> SplitFrames:
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise DataConfigError(
            f"train_frac + val_frac + test_frac must sum to 1.0, got {total:.4f}"
        )

    try:
        train_df, tmp_df = train_test_split(
            df,
            test_size=(1.0 - train_frac),
            random_state=seed,
            stratify=df["label_idx"],
        )
        rel_test = test_frac / (val_frac + test_frac)
        val_df, test_df = train_test_split(
            tmp_df,
            test_size=rel_test,
            random_state=seed,
            stratify=tmp_df["label_idx"],
        )
    except Exception as exc:
        LOGGER.warning(
            "Stratified split failed (%s). Falling back to random split without stratification.",
            exc,
        )
        train_df, tmp_df = train_test_split(
            df,
            test_size=(1.0 - train_frac),
            random_state=seed,
            shuffle=True,
        )
        rel_test = test_frac / (val_frac + test_frac)
        val_df, test_df = train_test_split(
            tmp_df,
            test_size=rel_test,
            random_state=seed,
            shuffle=True,
        )

    return SplitFrames(
        train=train_df.reset_index(drop=True),
        val=val_df.reset_index(drop=True),
        test=test_df.reset_index(drop=True),
    )


class RandomPadAndCrop:
    """Mimic small background/color jitter by padding with random RGB then cropping back."""

    def __init__(self, pad: int = 16):
        self.pad = pad

    def __call__(self, img: Image.Image) -> Image.Image:
        import random

        fill = tuple(random.randint(0, 255) for _ in range(3))
        padded = transforms.functional.pad(img, padding=self.pad, fill=fill)
        return transforms.RandomCrop(img.size[::-1])(padded)


def build_image_transform(augment: bool, image_size: int = 768) -> Callable[[Image.Image], Image.Image]:
    if augment:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomRotation(degrees=7),
                transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.06, hue=0.01),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
                RandomPadAndCrop(pad=10),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        ]
    )


class PillDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, augment: bool = False, image_size: int = 768):
        self.frame = frame.reset_index(drop=True)
        self.transform = build_image_transform(augment=augment, image_size=image_size)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict:
        row = self.frame.iloc[idx]
        image_path = Path(row["image_path"])
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Failed to open image {image_path}: {exc}") from exc

        image = self.transform(image)
        return {
            "image": image,
            "text": row["label_text"],
            "label_idx": int(row["label_idx"]),
            "image_path": str(image_path),
            "label_text": row["label_text"],
        }


def make_pixtral_sft_collate_fn(
    processor,
    task_prompt: str = DEFAULT_TASK_PROMPT,
    max_seq_len: int = 512,
    response_prefix: str = "Answer:",
):
    task_prompt = canonicalize_label_text(task_prompt) or DEFAULT_TASK_PROMPT
    image_token = getattr(processor, "image_token", "[IMG]")
    task_prompt = _ensure_image_token_in_prompt(task_prompt, image_token=image_token)

    def collate(batch: list[dict]) -> dict:
        images = [x["image"] for x in batch]
        prompts = [f"{task_prompt}\n{response_prefix}" for _ in batch]
        full_texts = [f"{task_prompt}\n{response_prefix} {x['text']}" for x in batch]

        prompt_enc = processor(
            images=images,
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        )
        full_enc = processor(
            images=images,
            text=full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        )

        if "input_ids" not in full_enc:
            raise RuntimeError("Processor output is missing input_ids required for SFT labels.")

        labels = full_enc["input_ids"].clone()
        prompt_lens = prompt_enc["attention_mask"].sum(dim=1)
        for i in range(labels.size(0)):
            labels[i, : int(prompt_lens[i].item())] = -100

        if "attention_mask" in full_enc:
            labels[full_enc["attention_mask"] == 0] = -100

        out = dict(full_enc)
        out["labels"] = labels
        out["label_indices"] = torch.tensor([x["label_idx"] for x in batch], dtype=torch.long)
        out["label_texts"] = [x["label_text"] for x in batch]
        out["image_paths"] = [x["image_path"] for x in batch]
        out["prompt_lengths"] = prompt_lens.to(torch.long)
        return out

    return collate


def make_pixtral_infer_collate_fn(
    processor,
    task_prompt: str = DEFAULT_TASK_PROMPT,
    max_seq_len: int = 512,
):
    task_prompt = canonicalize_label_text(task_prompt) or DEFAULT_TASK_PROMPT
    image_token = getattr(processor, "image_token", "[IMG]")
    task_prompt = _ensure_image_token_in_prompt(task_prompt, image_token=image_token)

    def collate(batch: list[dict]) -> dict:
        images = [x["image"] for x in batch]
        prompts = [task_prompt for _ in batch]

        enc = processor(
            images=images,
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        )
        if "attention_mask" not in enc:
            raise RuntimeError("Processor output is missing attention_mask required for generation decoding.")

        out = dict(enc)
        out["prompt_lengths"] = enc["attention_mask"].sum(dim=1).to(torch.long)
        out["prompts"] = prompts
        out["label_indices"] = torch.tensor([x["label_idx"] for x in batch], dtype=torch.long)
        out["label_texts"] = [x["label_text"] for x in batch]
        out["image_paths"] = [x["image_path"] for x in batch]
        return out

    return collate
