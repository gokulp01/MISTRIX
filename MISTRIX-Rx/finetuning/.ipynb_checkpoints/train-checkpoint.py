from __future__ import annotations

import argparse
import logging
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from accelerate import Accelerator
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoProcessor, get_cosine_schedule_with_warmup

from data import (
    DEFAULT_TASK_PROMPT,
    DataConfigError,
    PillDataset,
    add_label_indices,
    build_label_mapping,
    load_dataset_frame,
    make_pixtral_infer_collate_fn,
    make_pixtral_sft_collate_fn,
    split_dataset,
)
from models import (
    load_pixtral_model,
    map_prediction_to_label,
    maybe_apply_lora,
    maybe_prepare_for_kbit_training,
    resolve_model_id,
    setup_training_mode,
)
from utils import ensure_dir, get_logger, load_json, pick_mixed_precision, save_json, save_jsonl, set_seed, setup_logging

LOGGER = get_logger(__name__)
MODEL_ID = "pixtral-12b-latest"
MODE_ALIASES = {
    "contrastive": "sft_full",
    "partial_unfreeze": "sft_full",
    "lora_optional": "sft_lora",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Pixtral on pill image classification via generation")
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--run_name", type=str, required=True)

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["sft_full", "sft_lora", "contrastive", "partial_unfreeze", "lora_optional"],
        help="Use sft_full or sft_lora. Legacy modes are accepted as aliases.",
    )
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    parser.add_argument("--task_prompt", type=str, default=DEFAULT_TASK_PROMPT)
    parser.add_argument("--include_label_space_in_prompt", action="store_true")
    parser.add_argument("--max_labels_in_prompt", type=int, default=64)

    parser.add_argument("--metadata_path", type=str, default=None)
    parser.add_argument("--image_col", type=str, default=None)
    parser.add_argument("--text_col", type=str, default=None)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--image_size", type=int, default=768)

    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=12)
    parser.add_argument("--num_beams", type=int, default=1)

    parser.add_argument("--mixed_precision", choices=["auto", "no", "fp16", "bf16"], default="auto")
    parser.add_argument("--save_every_n_steps", type=int, default=200)
    parser.add_argument("--eval_every_n_steps", type=int, default=200)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--freeze_vision_tower", action="store_true")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--smoke_test", action="store_true")

    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--allow_cpu",
        action="store_true",
        help="Allow CPU fallback. By default this script requires CUDA (recommended: A100).",
    )

    return parser.parse_args()


def normalize_mode(mode: str) -> str:
    out = MODE_ALIASES.get(mode, mode)
    if out != mode:
        LOGGER.warning("Mode '%s' is legacy; remapped to '%s' for Pixtral training.", mode, out)
    return out


def ensure_cuda_or_raise(allow_cpu: bool) -> None:
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        names = [torch.cuda.get_device_name(i) for i in range(count)]
        LOGGER.info("CUDA is available. Visible GPUs: %d | %s", count, names)
        LOGGER.info("CUDA bf16 supported: %s", torch.cuda.is_bf16_supported())
        return

    if allow_cpu:
        LOGGER.warning("CUDA is not available. Proceeding on CPU because --allow_cpu was provided.")
        return

    raise RuntimeError(
        "CUDA is not available. This Pixtral training script is configured to require GPU. "
        "On your A100 machine, verify CUDA visibility (nvidia-smi / CUDA env) and retry, "
        "or pass --allow_cpu to override."
    )


def maybe_smoke_subset(frame, size: int, seed: int):
    if len(frame) <= size:
        return frame
    return frame.sample(n=size, random_state=seed).reset_index(drop=True)


def write_labels_file(run_dir: Path, labels: list[str]) -> None:
    obj = {
        "num_labels": len(labels),
        "idx_to_label": {str(i): lbl for i, lbl in enumerate(labels)},
        "label_to_idx": {lbl: i for i, lbl in enumerate(labels)},
    }
    save_json(obj, run_dir / "labels.json")


def prepare_data(args: argparse.Namespace):
    data_dir = Path(args.data_dir)
    frame, image_col, text_col, metadata_file, image_dir = load_dataset_frame(
        data_dir,
        metadata_path=args.metadata_path,
        image_col=args.image_col,
        text_col=args.text_col,
    )

    labels, label_to_idx = build_label_mapping(frame)
    frame = add_label_indices(frame, label_to_idx)
    splits = split_dataset(frame, args.train_frac, args.val_frac, args.test_frac, args.seed)

    if args.smoke_test:
        splits.train = maybe_smoke_subset(splits.train, size=96, seed=args.seed)
        splits.val = maybe_smoke_subset(splits.val, size=32, seed=args.seed)
        splits.test = maybe_smoke_subset(splits.test, size=32, seed=args.seed)

    metadata = {
        "resolved_metadata_path": str(metadata_file),
        "resolved_image_dir": str(image_dir),
        "resolved_image_col": image_col,
        "resolved_text_col": text_col,
        "dataset_size": len(frame),
        "train_size": len(splits.train),
        "val_size": len(splits.val),
        "test_size": len(splits.test),
        "num_labels": len(labels),
    }
    return splits, labels, metadata


def build_task_prompt(base_prompt: str, labels: list[str], include_label_space: bool, max_labels: int) -> str:
    base = " ".join((base_prompt or "").split()) or DEFAULT_TASK_PROMPT
    if include_label_space and len(labels) <= max(1, max_labels):
        return f"{base} Choose exactly one from: {', '.join(labels)}"
    return base


def build_dataloaders(args: argparse.Namespace, processor, splits, task_prompt: str):
    train_ds = PillDataset(splits.train, augment=args.augment, image_size=args.image_size)
    eval_ds_val = PillDataset(splits.val, augment=False, image_size=args.image_size)
    eval_ds_test = PillDataset(splits.test, augment=False, image_size=args.image_size)

    train_collate = make_pixtral_sft_collate_fn(
        processor=processor,
        task_prompt=task_prompt,
        max_seq_len=args.max_seq_len,
    )
    eval_collate = make_pixtral_infer_collate_fn(
        processor=processor,
        task_prompt=task_prompt,
        max_seq_len=args.max_seq_len,
    )

    eval_workers = 0 if args.num_workers <= 0 else max(1, args.num_workers // 2)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=train_collate,
    )
    val_loader = DataLoader(
        eval_ds_val,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=eval_workers,
        pin_memory=True,
        collate_fn=eval_collate,
    )
    test_loader = DataLoader(
        eval_ds_test,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=eval_workers,
        pin_memory=True,
        collate_fn=eval_collate,
    )
    return train_loader, val_loader, test_loader


def _get_text_decoder(processor):
    return processor.tokenizer if hasattr(processor, "tokenizer") else processor


def _decode_generation_tails(processor, generated_ids: torch.Tensor, prompt_lengths: torch.Tensor) -> list[str]:
    decoder = _get_text_decoder(processor)
    texts: list[str] = []
    for i in range(generated_ids.size(0)):
        prompt_len = int(prompt_lengths[i].item())
        tail_ids = generated_ids[i, prompt_len:].detach().cpu().tolist()
        text = decoder.decode(tail_ids, skip_special_tokens=True).strip()
        texts.append(" ".join(text.split()))
    return texts


def _gather_paths_compat(accelerator: Accelerator, local_paths: list[str]) -> list[str]:
    if accelerator.num_processes == 1:
        return list(local_paths)

    if hasattr(accelerator, "gather_object"):
        gathered = accelerator.gather_object(local_paths)
    else:
        try:
            from accelerate.utils import gather_object
        except Exception:
            LOGGER.warning("Could not import accelerate.utils.gather_object; keeping local path list only.")
            return list(local_paths)
        gathered = gather_object(local_paths)

    if gathered and isinstance(gathered[0], list):
        flat = []
        for item in gathered:
            flat.extend(item)
        return [str(x) for x in flat]
    return [str(x) for x in gathered]


def evaluate_generation(
    model,
    processor,
    dataloader,
    labels: list[str],
    accelerator: Accelerator,
    split_name: str,
    run_dir: Path,
    max_new_tokens: int,
    num_beams: int,
) -> Optional[dict]:
    model.eval()
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}

    y_true_chunks = []
    y_pred_chunks = []
    records = []

    pbar = tqdm(dataloader, disable=not accelerator.is_local_main_process, desc=f"eval-{split_name}")

    for batch in pbar:
        prompt_lengths = batch["prompt_lengths"].to(accelerator.device)
        y_local = batch["label_indices"].to(accelerator.device)

        gen_inputs = {}
        for k, v in batch.items():
            if k in {"label_indices", "label_texts", "image_paths", "prompt_lengths", "labels"}:
                continue
            if torch.is_tensor(v):
                gen_inputs[k] = v.to(accelerator.device, non_blocking=True)

        with torch.no_grad():
            decoder = _get_text_decoder(processor)
            pad_token_id = getattr(decoder, "pad_token_id", None)
            eos_token_id = getattr(decoder, "eos_token_id", None)
            if pad_token_id is None:
                pad_token_id = eos_token_id
            with accelerator.autocast():
                generated = model.generate(
                    **gen_inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    do_sample=False,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                )

        decoded = _decode_generation_tails(processor, generated, prompt_lengths)

        pred_idx_local = []
        pred_label_local = []
        pred_score_local = []
        for txt in decoded:
            pred_lbl, score = map_prediction_to_label(txt, labels)
            pred_label_local.append(pred_lbl)
            pred_score_local.append(float(score))
            pred_idx_local.append(int(label_to_idx[pred_lbl]) if pred_lbl in label_to_idx else -1)

        pred_t = torch.tensor(pred_idx_local, dtype=torch.long, device=accelerator.device)
        gathered_pred = accelerator.gather_for_metrics(pred_t)
        gathered_true = accelerator.gather_for_metrics(y_local)

        if accelerator.is_main_process:
            y_pred_chunks.append(gathered_pred.detach().cpu().numpy())
            y_true_chunks.append(gathered_true.detach().cpu().numpy())

        gathered_paths = _gather_paths_compat(accelerator, [str(p) for p in batch["image_paths"]])
        gathered_true_text = _gather_paths_compat(accelerator, [str(t) for t in batch["label_texts"]])
        gathered_raw_pred = _gather_paths_compat(accelerator, decoded)
        gathered_pred_lbl = _gather_paths_compat(accelerator, pred_label_local)
        gathered_pred_score = _gather_paths_compat(accelerator, [f"{x:.6f}" for x in pred_score_local])

        if accelerator.is_main_process:
            for i in range(min(len(gathered_paths), len(gathered_pred_lbl), len(gathered_true_text), len(gathered_raw_pred))):
                records.append(
                    {
                        "image_path": gathered_paths[i],
                        "true_label": gathered_true_text[i],
                        "raw_generation": gathered_raw_pred[i],
                        "predicted_label": gathered_pred_lbl[i],
                        "match_score": float(gathered_pred_score[i]),
                        "confidence": float(gathered_pred_score[i]),
                        "correct": bool(gathered_pred_lbl[i] == gathered_true_text[i]),
                    }
                )

    if not accelerator.is_main_process:
        return None

    if not y_true_chunks:
        return None

    y_true = np.concatenate(y_true_chunks, axis=0)
    y_pred = np.concatenate(y_pred_chunks, axis=0)

    top1 = float(np.mean(y_pred == y_true))
    unknown_rate = float(np.mean(y_pred < 0))

    y_pred_f1 = y_pred.copy()
    y_pred_f1[y_pred_f1 < 0] = len(labels)
    macro_f1 = float(
        f1_score(
            y_true,
            y_pred_f1,
            labels=list(range(len(labels))),
            average="macro",
            zero_division=0,
        )
    )

    metrics = {
        "top1": top1,
        "macro_f1": macro_f1,
        "unknown_rate": unknown_rate,
        "num_examples": int(y_true.shape[0]),
    }

    save_json(metrics, run_dir / "eval" / f"{split_name}_metrics.json")
    save_jsonl(records, run_dir / f"{split_name}_predictions.jsonl")
    return {"metrics": metrics, "records": records}


def save_checkpoint(
    accelerator: Accelerator,
    model,
    processor,
    run_dir: Path,
    tag: str,
    epoch: int,
    global_step: int,
    best_val_top1: float,
) -> Path:
    ckpt = ensure_dir(run_dir / "checkpoints" / tag)
    state_dir = ensure_dir(ckpt / "accelerate_state")
    accelerator.save_state(str(state_dir))

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        model_dir = ensure_dir(ckpt / "model")
        processor_dir = ensure_dir(ckpt / "processor")
        unwrapped.save_pretrained(model_dir)
        processor.save_pretrained(processor_dir)
        save_json(
            {
                "epoch": epoch,
                "global_step": global_step,
                "best_val_top1": best_val_top1,
            },
            ckpt / "training_state.json",
        )
    accelerator.wait_for_everyone()
    return ckpt


def maybe_resume(
    args: argparse.Namespace,
    accelerator: Accelerator,
) -> tuple[int, int, float]:
    if not args.resume_from:
        return 0, 0, -1.0

    ckpt = Path(args.resume_from)
    state_dir = ckpt / "accelerate_state"
    state_file = ckpt / "training_state.json"
    if not state_dir.exists() or not state_file.exists():
        raise RuntimeError(
            f"--resume_from must point to a checkpoint dir containing accelerate_state/ and training_state.json. Got: {ckpt}"
        )

    accelerator.load_state(str(state_dir))
    state = load_json(state_file)
    start_epoch = int(state.get("epoch", 0))
    global_step = int(state.get("global_step", 0))
    best_val_top1 = float(state.get("best_val_top1", -1.0))
    LOGGER.info("Resumed from %s at epoch=%d step=%d", ckpt, start_epoch, global_step)
    return start_epoch, global_step, best_val_top1


def run_training(args: argparse.Namespace, run_dir: Path, splits, labels: list[str]) -> None:
    mp = pick_mixed_precision(args.mixed_precision)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision=mp,
    )
    if accelerator.device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError(
            f"Accelerator selected device={accelerator.device}, but CUDA is required. "
            "Use your A100 CUDA environment or pass --allow_cpu to override."
        )
    if accelerator.is_local_main_process:
        LOGGER.info("Accelerator device: %s", accelerator.device)
        if accelerator.device.type == "cuda":
            LOGGER.info("Active CUDA device name: %s", torch.cuda.get_device_name(torch.cuda.current_device()))

    resolved_model_id = resolve_model_id(args.model_id)
    mode = normalize_mode(args.mode)

    processor = AutoProcessor.from_pretrained(resolved_model_id, trust_remote_code=True)
    model = load_pixtral_model(
        model_id=resolved_model_id,
        gradient_checkpointing=args.gradient_checkpointing,
        use_4bit=args.use_4bit,
    )

    if args.use_4bit:
        model = maybe_prepare_for_kbit_training(model)

    if mode == "sft_lora":
        model = maybe_apply_lora(
            model,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )

    setup_training_mode(
        model,
        mode=mode,
        freeze_vision_tower=args.freeze_vision_tower,
    )

    task_prompt = build_task_prompt(
        base_prompt=args.task_prompt,
        labels=labels,
        include_label_space=args.include_label_space_in_prompt,
        max_labels=args.max_labels_in_prompt,
    )

    train_loader, val_loader, test_loader = build_dataloaders(args, processor, splits, task_prompt)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters available after mode setup.")

    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    num_updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = max(1, args.epochs * num_updates_per_epoch)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(args.warmup_steps, total_steps // 2),
        num_training_steps=total_steps,
    )

    model, optimizer, train_loader, val_loader, test_loader, scheduler = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        scheduler,
    )

    start_epoch, global_step, best_val_top1 = maybe_resume(args, accelerator)

    LOGGER.info(
        "Starting Pixtral training | mode=%s | model_id=%s | epochs=%d | total_steps=%d | mixed_precision=%s",
        mode,
        resolved_model_id,
        args.epochs,
        total_steps,
        mp,
    )

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_losses = []
        pbar = tqdm(
            train_loader,
            disable=not accelerator.is_local_main_process,
            desc=f"train epoch {epoch + 1}/{args.epochs}",
        )

        for _, batch in enumerate(pbar):
            with accelerator.accumulate(model):
                model_inputs = {}
                for k, v in batch.items():
                    if k in {"label_indices", "label_texts", "image_paths", "prompt_lengths"}:
                        continue
                    if torch.is_tensor(v):
                        model_inputs[k] = v.to(accelerator.device, non_blocking=True)

                with accelerator.autocast():
                    out = model(**model_inputs, return_dict=True)
                    loss = out.loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            loss_val = float(accelerator.gather_for_metrics(loss.detach()).mean().cpu().item())
            epoch_losses.append(loss_val)

            if accelerator.sync_gradients:
                global_step += 1

            if accelerator.is_main_process:
                pbar.set_postfix({"loss": f"{loss_val:.4f}", "step": global_step})

            if accelerator.sync_gradients and args.save_every_n_steps > 0 and global_step % args.save_every_n_steps == 0:
                save_checkpoint(
                    accelerator,
                    model,
                    processor,
                    run_dir,
                    tag=f"step_{global_step}",
                    epoch=epoch,
                    global_step=global_step,
                    best_val_top1=best_val_top1,
                )

            if accelerator.sync_gradients and args.eval_every_n_steps > 0 and global_step % args.eval_every_n_steps == 0:
                val_out = evaluate_generation(
                    model=accelerator.unwrap_model(model),
                    processor=processor,
                    dataloader=val_loader,
                    labels=labels,
                    accelerator=accelerator,
                    split_name="val",
                    run_dir=run_dir,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                )
                if accelerator.is_main_process and val_out is not None:
                    val_top1 = float(val_out["metrics"]["top1"])
                    LOGGER.info("step=%d val metrics: %s", global_step, val_out["metrics"])
                    if val_top1 > best_val_top1:
                        best_val_top1 = val_top1
                        save_checkpoint(
                            accelerator,
                            model,
                            processor,
                            run_dir,
                            tag="best",
                            epoch=epoch,
                            global_step=global_step,
                            best_val_top1=best_val_top1,
                        )

        if accelerator.is_main_process and epoch_losses:
            LOGGER.info("epoch=%d mean_train_loss=%.5f", epoch + 1, float(np.mean(epoch_losses)))

        val_out = evaluate_generation(
            model=accelerator.unwrap_model(model),
            processor=processor,
            dataloader=val_loader,
            labels=labels,
            accelerator=accelerator,
            split_name="val",
            run_dir=run_dir,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
        )
        if accelerator.is_main_process and val_out is not None:
            val_top1 = float(val_out["metrics"]["top1"])
            LOGGER.info("epoch=%d val metrics: %s", epoch + 1, val_out["metrics"])
            if val_top1 > best_val_top1:
                best_val_top1 = val_top1
                save_checkpoint(
                    accelerator,
                    model,
                    processor,
                    run_dir,
                    tag="best",
                    epoch=epoch + 1,
                    global_step=global_step,
                    best_val_top1=best_val_top1,
                )

    save_checkpoint(
        accelerator,
        model,
        processor,
        run_dir,
        tag="final",
        epoch=args.epochs,
        global_step=global_step,
        best_val_top1=best_val_top1,
    )

    if accelerator.is_main_process:
        final_dir = ensure_dir(run_dir / "final")
        ensure_dir(final_dir / "model")
        ensure_dir(final_dir / "processor")
        accelerator.unwrap_model(model).save_pretrained(final_dir / "model")
        processor.save_pretrained(final_dir / "processor")

    test_out = evaluate_generation(
        model=accelerator.unwrap_model(model),
        processor=processor,
        dataloader=test_loader,
        labels=labels,
        accelerator=accelerator,
        split_name="test",
        run_dir=run_dir,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
    )

    if accelerator.is_main_process and test_out is not None:
        save_json(test_out["metrics"], run_dir / "test_metrics.json")
        LOGGER.info("Test metrics: %s", test_out["metrics"])


def main() -> None:
    args = parse_args()
    run_dir = ensure_dir(Path(args.output_dir) / args.run_name)
    setup_logging(run_dir / "train.log")

    ensure_cuda_or_raise(allow_cpu=args.allow_cpu)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    set_seed(args.seed)

    args.mode = normalize_mode(args.mode)
    args.model_id = resolve_model_id(args.model_id)

    LOGGER.info("Args: %s", vars(args))
    try:
        splits, labels, metadata = prepare_data(args)
    except DataConfigError as exc:
        LOGGER.error("Data configuration error: %s", exc)
        raise SystemExit(2)

    write_labels_file(run_dir, labels)
    save_json(vars(args), run_dir / "config.json")
    save_json(metadata, run_dir / "data_summary.json")

    run_training(args, run_dir, splits, labels)
    LOGGER.info("Training completed. Artifacts saved to %s", run_dir)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
