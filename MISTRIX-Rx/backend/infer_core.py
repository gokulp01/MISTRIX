"""Core inference utilities shared by labeler and direct CLI execution."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from transformers import AutoProcessor

from backend.models import load_pixtral_generation_model
from backend.models import map_prediction_to_label
from backend.models import rank_labels_from_prediction
from backend.models import resolve_model_id
from backend.utils import canonicalize_label_text, load_json

LOGGER = logging.getLogger(__name__)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_TASK_PROMPT = "Identify the medicine shown in the image. Respond with only the medicine name."
DEFAULT_PIXTRAL_MODEL_ID = "pixtral-12b-latest"

# Legacy MISTRIX-Rx mode names are accepted and normalized to Pixtral generation inference.
MODE_ALIASES = {
    "contrastive": "pixtral",
    "partial_unfreeze": "pixtral",
    "lora_optional": "pixtral",
    "linear_probe": "pixtral",
}


def normalize_mode(mode: str) -> str:
    """Map legacy mode names to the Pixtral inference mode."""
    return MODE_ALIASES.get(mode, mode)


def ensure_text_tokenizer_deps_available(mode: str) -> None:
    """Validate tokenizer/runtime deps required for Pixtral generation inference."""
    _ = normalize_mode(mode)
    try:
        import sentencepiece  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Pixtral inference requires sentencepiece. Install with: pip install sentencepiece"
        ) from exc
    try:
        import google.protobuf  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Pixtral inference requires protobuf. Install with: pip install protobuf"
        ) from exc


def parse_args() -> argparse.Namespace:
    """Parse CLI args for standalone inference runs."""
    parser = argparse.ArgumentParser(description="Inference for fine-tuned Pixtral 12B pill classifier")
    parser.add_argument("--run_dir", type=str, default=str(PACKAGE_ROOT))
    parser.add_argument("--input_path", type=str, required=True, help="Image file or directory")
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["pixtral", "contrastive", "linear_probe", "partial_unfreeze", "lora_optional"],
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="Optional override for Pixtral base model id.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--candidate_meds_file", type=str, default=None)
    parser.add_argument("--output_jsonl", type=str, default=None)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=12)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--task_prompt", type=str, default=DEFAULT_TASK_PROMPT)
    parser.add_argument("--include_label_space_in_prompt", action="store_true")
    parser.add_argument("--max_labels_in_prompt", type=int, default=64)
    parser.add_argument(
        "--allow_cpu",
        action="store_true",
        help="Allow CPU fallback. By default inference expects CUDA.",
    )
    return parser.parse_args()


def _iter_layout_candidates(base_dir: Path) -> list[tuple[Path, Path]]:
    return [
        (base_dir / "final" / "model", base_dir / "final" / "processor"),
        (base_dir / "checkpoints" / "final" / "model", base_dir / "checkpoints" / "final" / "processor"),
        (base_dir / "checkpoints" / "best" / "model", base_dir / "checkpoints" / "best" / "processor"),
        (base_dir / "models", base_dir / "processor"),
        (base_dir / "inference" / "models", base_dir / "inference" / "processor"),
    ]


def _find_finetuning_artifacts(repo_root: Path) -> list[tuple[Path, Path]]:
    roots = [repo_root / "finetuning" / "outputs", repo_root / "finetuning"]
    patterns = ["**/final/model", "**/checkpoints/final/model", "**/checkpoints/best/model"]

    pairs: list[tuple[Path, Path]] = []
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            for model_dir in root.glob(pattern):
                proc_dir = model_dir.parent / "processor"
                if model_dir.exists() and proc_dir.exists():
                    pairs.append((model_dir, proc_dir))

    # Prefer most recently modified model dir.
    pairs.sort(key=lambda item: item[0].stat().st_mtime, reverse=True)
    return pairs


def discover_model_processor_dirs(run_dir: Path) -> tuple[Path, Path]:
    """Locate model and processor artifacts across supported checkpoint layouts."""
    for model_dir, proc_dir in _iter_layout_candidates(run_dir):
        if model_dir.exists() and proc_dir.exists():
            return model_dir, proc_dir

    repo_root = PACKAGE_ROOT.parent
    for model_dir, proc_dir in _find_finetuning_artifacts(repo_root=repo_root):
        return model_dir, proc_dir

    raise FileNotFoundError(
        (
            f"Could not find Pixtral model/processor artifacts for run_dir={run_dir}. "
            "Expected one of: final/, checkpoints/{best,final}, models/processor, "
            "inference/models+inference/processor, or finetuning/outputs/*/{final,checkpoints/*}."
        )
    )


def resolve_inference_model_id(run_dir: Path, cli_model_id: Optional[str]) -> str:
    """Resolve model id from CLI override or run config with Pixtral alias support."""
    if cli_model_id:
        resolved = resolve_model_id(cli_model_id)
        source = "CLI"
    else:
        cfg_path = run_dir / "config.json"
        if cfg_path.exists():
            cfg = load_json(cfg_path)
            resolved = resolve_model_id(str(cfg.get("model_id", DEFAULT_PIXTRAL_MODEL_ID)))
            source = str(cfg_path)
        else:
            resolved = resolve_model_id(DEFAULT_PIXTRAL_MODEL_ID)
            source = "default alias"

    if "pixtral" not in resolved.lower():
        raise RuntimeError(
            f"Inference requires a Pixtral model, but resolved model_id='{resolved}' from {source}."
        )

    LOGGER.info("Using Pixtral model id for inference: %s (source: %s)", resolved, source)
    return resolved


def load_labels(run_dir: Path) -> list[str]:
    """Load class labels from ``labels.json`` supporting two schema variants."""
    labels_obj = load_json(run_dir / "labels.json")
    if "idx_to_label" in labels_obj:
        pairs = sorted(((int(k), v) for k, v in labels_obj["idx_to_label"].items()), key=lambda x: x[0])
        return [v for _, v in pairs]
    if "labels" in labels_obj:
        return list(labels_obj["labels"])
    raise ValueError("labels.json missing idx_to_label or labels.")


def list_images(input_path: Path) -> list[Path]:
    """Resolve input into image paths (single file or directory tree)."""
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"input_path not found: {input_path}")
    imgs = [p for p in sorted(input_path.rglob("*")) if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if not imgs:
        raise FileNotFoundError(f"No images found in {input_path} with extensions {sorted(IMG_EXTS)}")
    return imgs


def load_candidate_labels(candidate_file: Optional[str], default_labels: list[str]) -> list[str]:
    """Load optional candidate-label file, otherwise fall back to full label list."""
    if not candidate_file:
        return default_labels
    labels = []
    seen = set()
    with Path(candidate_file).open("r", encoding="utf-8") as f:
        for line in f:
            t = canonicalize_label_text(line)
            if t and t not in seen:
                labels.append(t)
                seen.add(t)
    if not labels:
        raise ValueError("candidate_meds_file contained no valid labels.")
    return labels


def batched(items: list, n: int):
    """Yield fixed-size chunks from ``items``."""
    for i in range(0, len(items), n):
        yield items[i : i + n]


def _build_task_prompt(base_prompt: str, labels: list[str], include_label_space: bool, max_labels: int) -> str:
    prompt = canonicalize_label_text(base_prompt) or DEFAULT_TASK_PROMPT
    if include_label_space and len(labels) <= max(1, max_labels):
        prompt = f"{prompt} Choose exactly one from: {', '.join(labels)}"
    return prompt


def _ensure_image_token(prompt: str, processor) -> str:
    image_token = getattr(processor, "image_token", "[IMG]")
    p = canonicalize_label_text(prompt)
    if image_token in p:
        return p
    return f"{image_token} {p}".strip()


def _ensure_processor_padding(processor, model=None) -> None:
    decoder = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    if getattr(decoder, "pad_token", None) is None:
        if getattr(decoder, "eos_token", None) is None:
            raise RuntimeError("Tokenizer has neither pad_token nor eos_token; cannot configure padding.")
        decoder.pad_token = decoder.eos_token
    if getattr(decoder, "pad_token_id", None) is None and getattr(decoder, "eos_token_id", None) is not None:
        decoder.pad_token_id = decoder.eos_token_id

    if model is not None and hasattr(model, "config"):
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = getattr(decoder, "pad_token_id", None)


def _load_model_and_processor(model_dir: Path, proc_dir: Path, allow_cpu: bool, expected_model_id: str):
    processor = AutoProcessor.from_pretrained(proc_dir, trust_remote_code=True)

    adapter_cfg_path = model_dir / "adapter_config.json"
    if adapter_cfg_path.exists():
        adapter_cfg = load_json(adapter_cfg_path)
        adapter_base = resolve_model_id(str(adapter_cfg.get("base_model_name_or_path", expected_model_id)))
        model = load_pixtral_generation_model(adapter_base)
        try:
            from peft import PeftModel
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("peft is required to run inference from LoRA adapter checkpoints.") from exc
        model = PeftModel.from_pretrained(model, model_dir)
    else:
        model = load_pixtral_generation_model(model_dir)

    _ensure_processor_padding(processor, model=model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not allow_cpu:
        raise RuntimeError(
            f"Loaded inference device={device}, but CUDA is required. "
            "Run in your CUDA environment or pass --allow_cpu to override."
        )

    model.to(device)
    model.eval()
    return model, processor, device


def _strip_prompt_prefix(text: str, prompt: Optional[str]) -> str:
    if not prompt:
        return text
    t = " ".join((text or "").split())
    p = " ".join((prompt or "").split())
    if not p:
        return t
    if t.startswith(p):
        return t[len(p) :].strip()
    return t


def _decode_tail(
    processor,
    generated_ids: torch.Tensor,
    prompt_lengths: torch.Tensor,
    prompts: Optional[list[str]] = None,
) -> list[str]:
    decoder = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    out = []
    for i in range(generated_ids.size(0)):
        ids = generated_ids[i].detach().cpu().tolist()
        pl = int(prompt_lengths[i].item())
        if pl < len(ids):
            decode_ids = ids[pl:]
        else:
            decode_ids = ids
        txt = decoder.decode(decode_ids, skip_special_tokens=True).strip()
        prompt_text = prompts[i] if prompts is not None and i < len(prompts) else None
        txt = _strip_prompt_prefix(txt, prompt_text)
        out.append(" ".join(txt.split()))
    return out


def build_output_record(path: str, labels: list[str], prediction_text: str, top_k: int, threshold: float) -> dict:
    """Build one prediction record with top-k outputs and abstain logic."""
    ranked = rank_labels_from_prediction(prediction_text, labels, top_k=top_k)
    top_labels = [lbl for lbl, _ in ranked]
    top_scores = [float(score) for _, score in ranked]

    pred_label, conf = map_prediction_to_label(prediction_text, labels, confidence_cutoff=threshold)
    abstain = pred_label == "UNKNOWN"

    return {
        "image_path": path,
        "raw_generation": prediction_text,
        "top_k_labels": top_labels,
        "top_k_scores": top_scores,
        "predicted_label": pred_label,
        "confidence": float(conf),
        "abstain_flag": bool(abstain),
    }


def run_pixtral_inference(
    run_dir: Path,
    image_paths: list[Path],
    candidate_labels: list[str],
    top_k: int,
    threshold: float,
    batch_size: int,
    model_id: Optional[str] = None,
    max_seq_len: int = 512,
    max_new_tokens: int = 12,
    num_beams: int = 1,
    task_prompt: str = DEFAULT_TASK_PROMPT,
    include_label_space_in_prompt: bool = False,
    max_labels_in_prompt: int = 64,
    allow_cpu: bool = True,
) -> list[dict]:
    """Run Pixtral generation inference and rank candidate labels from generated text."""
    model_dir, proc_dir = discover_model_processor_dirs(run_dir)
    expected_model_id = resolve_model_id(model_id or DEFAULT_PIXTRAL_MODEL_ID)

    model, processor, device = _load_model_and_processor(
        model_dir=model_dir,
        proc_dir=proc_dir,
        allow_cpu=allow_cpu,
        expected_model_id=expected_model_id,
    )

    prompt = _build_task_prompt(task_prompt, candidate_labels, include_label_space_in_prompt, max_labels_in_prompt)
    prompt = _ensure_image_token(prompt, processor)

    records = []
    with torch.no_grad():
        for chunk in batched(image_paths, max(1, batch_size)):
            pil_imgs = [Image.open(p).convert("RGB") for p in chunk]
            texts = [prompt for _ in chunk]

            enc = processor(
                images=pil_imgs,
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_len,
            )
            prompt_lengths = enc["attention_mask"].sum(dim=1)

            model_inputs = {}
            for k, v in enc.items():
                if torch.is_tensor(v):
                    model_inputs[k] = v.to(device)

            decoder = processor.tokenizer if hasattr(processor, "tokenizer") else processor
            pad_token_id = getattr(decoder, "pad_token_id", None)
            eos_token_id = getattr(decoder, "eos_token_id", None)
            if pad_token_id is None:
                pad_token_id = eos_token_id

            generated = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=False,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            decoded = _decode_tail(processor, generated, prompt_lengths.to(device), prompts=texts)

            for i, p in enumerate(chunk):
                records.append(
                    build_output_record(
                        str(p),
                        candidate_labels,
                        decoded[i],
                        top_k=top_k,
                        threshold=threshold,
                    )
                )

    return records


def run_contrastive_inference(
    run_dir: Path,
    image_paths: list[Path],
    candidate_labels: list[str],
    top_k: int,
    threshold: float,
    batch_size: int,
) -> list[dict]:
    """Backward-compatible wrapper that routes legacy contrastive mode to Pixtral inference."""
    return run_pixtral_inference(
        run_dir=run_dir,
        image_paths=image_paths,
        candidate_labels=candidate_labels,
        top_k=top_k,
        threshold=threshold,
        batch_size=batch_size,
    )


def run_linear_probe_inference(
    run_dir: Path,
    image_paths: list[Path],
    all_labels: list[str],
    candidate_labels: list[str],
    top_k: int,
    threshold: float,
    batch_size: int,
) -> list[dict]:
    """Backward-compatible wrapper that routes legacy linear-probe mode to Pixtral inference."""
    labels_for_infer = candidate_labels or all_labels
    return run_pixtral_inference(
        run_dir=run_dir,
        image_paths=image_paths,
        candidate_labels=labels_for_infer,
        top_k=top_k,
        threshold=threshold,
        batch_size=batch_size,
    )


def main() -> None:
    """CLI entry point for standalone inference."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    run_dir = Path(args.run_dir)
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        cfg = load_json(cfg_path)
        mode_raw = args.mode or str(cfg.get("mode", "pixtral"))
    else:
        mode_raw = args.mode or "pixtral"

    mode = normalize_mode(mode_raw)
    ensure_text_tokenizer_deps_available(mode)

    input_paths = list_images(Path(args.input_path))
    labels = load_labels(run_dir)
    candidate_labels = load_candidate_labels(args.candidate_meds_file, labels)
    model_id = resolve_inference_model_id(run_dir=run_dir, cli_model_id=args.model_id)

    records = run_pixtral_inference(
        run_dir=run_dir,
        image_paths=input_paths,
        candidate_labels=candidate_labels,
        top_k=args.top_k,
        threshold=args.threshold,
        batch_size=args.batch_size,
        model_id=model_id,
        max_seq_len=args.max_seq_len,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        task_prompt=args.task_prompt,
        include_label_space_in_prompt=bool(args.include_label_space_in_prompt),
        max_labels_in_prompt=args.max_labels_in_prompt,
        allow_cpu=bool(args.allow_cpu),
    )

    if args.output_jsonl:
        out = Path(args.output_jsonl)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        for r in records:
            print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
