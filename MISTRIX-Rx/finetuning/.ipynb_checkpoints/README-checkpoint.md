# Pixtral Pill Classification (SFT)

This folder fine-tunes **Pixtral 12B** for pill-image medicine identification using your local Pillbox-style dataset.

Default model id in code is `pixtral-12b-latest`, which is resolved to a Transformers-compatible Pixtral checkpoint:
- `mistral-community/pixtral-12b`

## GPU Requirement

- Training and inference are CUDA-first and will fail fast if CUDA is not visible.
- This is intended for your A100 setup.
- To override and run on CPU intentionally, add `--allow_cpu`.

## Supported Modes

- `sft_full`: full-parameter supervised fine-tuning
- `sft_lora`: LoRA supervised fine-tuning (recommended for limited GPU memory)

Legacy mode names are accepted and remapped:
- `contrastive` -> `sft_full`
- `partial_unfreeze` -> `sft_full`
- `lora_optional` -> `sft_lora`

## 1) Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional for QLoRA (`--use_4bit`):
```bash
pip install bitsandbytes
```

## 2) Dataset Layout

Expected by default:
- `data.csv` (or `metadata.csv` / `train.csv`)
- `data-images/` (or `images/` / `img/`)

The loader auto-detects image/text columns and resolves common image extensions.

## 3) Smoke Test

```bash
python train.py \
  --run_name pixtral_smoke \
  --mode sft_lora \
  --data_dir . \
  --output_dir ./outputs \
  --smoke_test \
  --batch_size 1 \
  --eval_batch_size 1 \
  --grad_accum 4 \
  --epochs 1 \
  --num_workers 0
```

## 4) Training Commands

### A) LoRA training (recommended)

```bash
python train.py \
  --run_name pixtral_lora \
  --mode sft_lora \
  --data_dir . \
  --output_dir ./outputs \
  --batch_size 1 \
  --eval_batch_size 1 \
  --grad_accum 8 \
  --epochs 3 \
  --lr 2e-5 \
  --max_seq_len 512 \
  --max_new_tokens 12 \
  --augment \
  --gradient_checkpointing \
  --mixed_precision bf16
```

### B) Full fine-tune

```bash
python train.py \
  --run_name pixtral_full \
  --mode sft_full \
  --data_dir . \
  --output_dir ./outputs \
  --batch_size 1 \
  --eval_batch_size 1 \
  --grad_accum 8 \
  --epochs 3 \
  --lr 1e-5 \
  --max_seq_len 512 \
  --max_new_tokens 12 \
  --gradient_checkpointing \
  --mixed_precision bf16
```

### C) QLoRA-style run (optional)

```bash
python train.py \
  --run_name pixtral_qlora \
  --mode sft_lora \
  --use_4bit \
  --data_dir . \
  --output_dir ./outputs \
  --batch_size 1 \
  --grad_accum 16 \
  --epochs 3
```

## 5) Resume

```bash
python train.py \
  --run_name pixtral_lora_resume \
  --mode sft_lora \
  --data_dir . \
  --output_dir ./outputs \
  --resume_from ./outputs/pixtral_lora/checkpoints/step_200
```

## 6) Inference

```bash
python infer.py \
  --run_dir ./outputs/pixtral_lora \
  --input_path ./data-images \
  --top_k 5 \
  --threshold 0.55 \
  --output_jsonl ./outputs/pixtral_lora/infer.jsonl
```

With prescription-constrained candidates:

```bash
python infer.py \
  --run_dir ./outputs/pixtral_lora \
  --input_path ./data-images \
  --candidate_meds_file ./candidates.txt \
  --top_k 5 \
  --threshold 0.55
```

## 7) Outputs

For each run (`outputs/<run_name>/`):
- `config.json`
- `data_summary.json`
- `labels.json`
- `train.log`
- `val_predictions.jsonl`, `test_predictions.jsonl`
- `eval/val_metrics.json`, `eval/test_metrics.json`
- `checkpoints/`
- `final/model`, `final/processor`

## 8) Notes

- Pixtral is a generative VLM, so validation/inference uses generated label text mapped to dataset labels.
- If generation quality is weak, tune:
  - `--task_prompt`
  - `--include_label_space_in_prompt` (only recommended for smaller label sets)
  - `--max_new_tokens`
- For large datasets/models, start with `sft_lora` + `--gradient_checkpointing`.
