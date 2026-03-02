from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from data import DEFAULT_TASK_PROMPT
from models import map_prediction_to_label, rank_labels_from_prediction, resolve_model_id
from utils import canonicalize_label_text, load_json

LOGGER = logging.getLogger(__name__)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_PIXTRAL_MODEL_ID = "mistral-community/pixtral-12b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for fine-tuned Pixtral pill classifier")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--input_path", type=str, required=True, help="Image file or directory")
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="Optional override for Pixtral base model id. Defaults to run config model_id or pixtral-12b-latest.",
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
        help="Allow CPU fallback. By default inference requires CUDA (recommended: A100).",
    )
    return parser.parse_args()


def ensure_cuda_or_raise(allow_cpu: bool) -> None:
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        names = [torch.cuda.get_device_name(i) for i in range(count)]
        LOGGER.info("CUDA is available. Visible GPUs: %d | %s", count, names)
        return

    if allow_cpu:
        LOGGER.warning("CUDA is not available. Proceeding on CPU because --allow_cpu was provided.")
        return

    raise RuntimeError(
        "CUDA is not available. This Pixtral inference script is configured to require GPU. "
        "On your A100 machine, verify CUDA visibility and retry, or pass --allow_cpu to override."
    )


def discover_model_processor_dirs(run_dir: Path) -> tuple[Path, Path]:
    candidates = [
        (run_dir / "final" / "model", run_dir / "final" / "processor"),
        (run_dir / "checkpoints" / "final" / "model", run_dir / "checkpoints" / "final" / "processor"),
        (run_dir / "checkpoints" / "best" / "model", run_dir / "checkpoints" / "best" / "processor"),
    ]
    for model_dir, proc_dir in candidates:
        if model_dir.exists() and proc_dir.exists():
            return model_dir, proc_dir
    raise FileNotFoundError(
        f"Could not find model/processor artifacts in {run_dir}. Expected final/ or checkpoints/{{best,final}}."
    )


def resolve_inference_model_id(run_dir: Path, cli_model_id: Optional[str]) -> str:
    if cli_model_id:
        resolved = resolve_model_id(cli_model_id)
        source = "CLI"
    else:
        cfg_path = run_dir / "config.json"
        if cfg_path.exists():
            cfg = load_json(cfg_path)
            resolved = resolve_model_id(str(cfg.get("model_id", "pixtral-12b-latest")))
            source = str(cfg_path)
        else:
            resolved = resolve_model_id("pixtral-12b-latest")
            source = "default alias"

    if "pixtral" not in resolved.lower():
        raise RuntimeError(
            f"Inference requires a Pixtral model, but resolved model_id='{resolved}' from {source}."
        )

    LOGGER.info("Using Pixtral model id for inference: %s (source: %s)", resolved, source)
    return resolved


def load_labels(run_dir: Path) -> list[str]:
    labels_obj = load_json(run_dir / "labels.json")
    if "idx_to_label" in labels_obj:
        pairs = sorted(((int(k), v) for k, v in labels_obj["idx_to_label"].items()), key=lambda x: x[0])
        return [v for _, v in pairs]
    if "labels" in labels_obj:
        return list(labels_obj["labels"])
    raise ValueError("labels.json missing idx_to_label or labels.")


def list_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"input_path not found: {input_path}")
    imgs = [p for p in sorted(input_path.iterdir()) if p.suffix.lower() in IMG_EXTS]
    if not imgs:
        raise FileNotFoundError(f"No images found in {input_path} with extensions {sorted(IMG_EXTS)}")
    return imgs


def load_candidate_labels(candidate_file: Optional[str], default_labels: list[str]) -> list[str]:
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


def _build_model_load_kwargs() -> dict:
    kwargs = {"trust_remote_code": True}
    if torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        kwargs["attn_implementation"] = "eager"
    return kwargs


def _load_with_supported_classes(model_ref: str | Path):
    model = None
    kwargs = _build_model_load_kwargs()
    for cls_name in ("LlavaForConditionalGeneration", "AutoModelForImageTextToText", "AutoModelForVision2Seq"):
        try:
            import transformers

            cls = getattr(transformers, cls_name, None)
            if cls is None:
                continue
            model = cls.from_pretrained(model_ref, **kwargs)
            break
        except Exception:
            continue
    return model


def _load_model_and_processor(model_dir: Path, proc_dir: Path, allow_cpu: bool, expected_model_id: str):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(proc_dir, trust_remote_code=True)
    adapter_cfg_path = model_dir / "adapter_config.json"
    if adapter_cfg_path.exists():
        adapter_cfg = load_json(adapter_cfg_path)
        adapter_base = resolve_model_id(str(adapter_cfg.get("base_model_name_or_path", expected_model_id)))
        if adapter_base != expected_model_id:
            LOGGER.warning(
                "Adapter base model '%s' differs from expected '%s'. Loading adapter base.",
                adapter_base,
                expected_model_id,
            )
        model = _load_with_supported_classes(adapter_base)
        if model is None:
            raise RuntimeError(
                f"Could not load Pixtral base model '{adapter_base}' for adapter inference."
            )
        try:
            from peft import PeftModel
        except Exception as exc:
            raise RuntimeError("peft is required to run inference from LoRA adapter checkpoints.") from exc
        model = PeftModel.from_pretrained(model, model_dir)
    else:
        model = _load_with_supported_classes(model_dir)

    if model is None:
        raise RuntimeError(
            "Could not load fine-tuned model with Llava/AutoModelForImageTextToText/AutoModelForVision2Seq. "
            "Upgrade transformers or verify saved model artifacts."
        )
    _ensure_processor_padding(processor, model=model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not allow_cpu:
        raise RuntimeError(
            f"Loaded inference device={device}, but CUDA is required. Use your A100 CUDA environment "
            "or pass --allow_cpu to override."
        )
    if device.type == "cuda":
        LOGGER.info("Active CUDA device name: %s", torch.cuda.get_device_name(torch.cuda.current_device()))
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


def run_inference(
    run_dir: Path,
    image_paths: list[Path],
    candidate_labels: list[str],
    top_k: int,
    threshold: float,
    batch_size: int,
    max_seq_len: int,
    max_new_tokens: int,
    num_beams: int,
    task_prompt: str,
    include_label_space_in_prompt: bool,
    max_labels_in_prompt: int,
    allow_cpu: bool,
    model_id: str,
) -> list[dict]:
    model_dir, proc_dir = discover_model_processor_dirs(run_dir)
    model, processor, device = _load_model_and_processor(
        model_dir,
        proc_dir,
        allow_cpu=allow_cpu,
        expected_model_id=model_id,
    )

    prompt = _build_task_prompt(task_prompt, candidate_labels, include_label_space_in_prompt, max_labels_in_prompt)
    prompt = _ensure_image_token(prompt, processor)
    records = []

    with torch.no_grad():
        for chunk in batched(image_paths, batch_size):
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
            decoded = _decode_tail(
                processor,
                generated,
                prompt_lengths.to(device),
                prompts=texts,
            )

            for i, p in enumerate(chunk):
                rec = build_output_record(str(p), candidate_labels, decoded[i], top_k=top_k, threshold=threshold)
                records.append(rec)

    return records


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    ensure_cuda_or_raise(allow_cpu=args.allow_cpu)

    run_dir = Path(args.run_dir)
    resolved_model_id = resolve_inference_model_id(run_dir, args.model_id)
    input_paths = list_images(Path(args.input_path))
    labels = load_labels(run_dir)
    candidate_labels = load_candidate_labels(args.candidate_meds_file, labels)

    records = run_inference(
        run_dir=run_dir,
        image_paths=input_paths,
        candidate_labels=candidate_labels,
        top_k=args.top_k,
        threshold=args.threshold,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        task_prompt=args.task_prompt,
        include_label_space_in_prompt=args.include_label_space_in_prompt,
        max_labels_in_prompt=args.max_labels_in_prompt,
        allow_cpu=args.allow_cpu,
        model_id=resolved_model_id,
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
