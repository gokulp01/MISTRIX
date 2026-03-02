# Finetuning Folder Usage (Pixtral)

## Included files

- `finetuning/train.py`
- `finetuning/data.py`
- `finetuning/models.py`
- `finetuning/eval.py`
- `finetuning/infer.py`
- `finetuning/utils.py`
- `finetuning/requirements.txt`
- `finetuning/README.md`

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r finetuning/requirements.txt
```

## Train (LoRA example)

```bash
python finetuning/train.py \
  --run_name pixtral_lora \
  --mode sft_lora \
  --data_dir /path/to/data_root \
  --output_dir /path/to/outputs \
  --batch_size 1 \
  --eval_batch_size 1 \
  --grad_accum 8 \
  --epochs 3 \
  --lr 2e-5 \
  --max_seq_len 512 \
  --max_new_tokens 12 \
  --gradient_checkpointing \
  --mixed_precision bf16 \
  --eval_every_n_steps 200 \
  --save_every_n_steps 200 \
  --num_workers 2
```

## Resume training

```bash
python finetuning/train.py \
  --run_name pixtral_lora_resume \
  --mode sft_lora \
  --data_dir /path/to/data_root \
  --output_dir /path/to/outputs \
  --epochs 5 \
  --resume_from /path/to/outputs/pixtral_lora/checkpoints/step_200
```

## Inference

```bash
python finetuning/infer.py \
  --run_dir /path/to/trained_run_dir \
  --input_path /path/to/image_or_folder \
  --top_k 5 \
  --threshold 0.55
```
