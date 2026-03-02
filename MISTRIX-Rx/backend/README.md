# MISTRIX-Rx Backend Pipeline

This folder contains the modular backend pipeline:

- `prescriber.py`: collects candidate medicines + optional target medicine.
- `bounder.py`: detects medicine bounding boxes in an image.
- `cropper.py`: crops every detected bounding box.
- `labeler.py`: runs Pixtral 12B inference on crops.
- `center.py`: computes bbox centers and target-medicine matches.
- `pipeline.py`: orchestrates all stages end to end.

Shared support code:

- `utils.py`, `models.py`, `infer_core.py`, `pipeline_helpers.py`, `common.py`
- `boundingbox/` (ROI + bbox extraction helpers)
- `labels.json`, `meds.txt`

## Setup

From repository root (`MISTRIX-Rx/`):

```bash
uv sync
```

## Model Artifacts

Inference expects Pixtral fine-tuning artifacts in a run directory with one of these layouts:

- `<run_dir>/final/model` and `<run_dir>/final/processor`
- `<run_dir>/checkpoints/best/model` and `<run_dir>/checkpoints/best/processor`
- `<run_dir>/checkpoints/final/model` and `<run_dir>/checkpoints/final/processor`

Default behavior:

- `--run_dir backend/` is used first.
- If artifacts are not found there, backend auto-searches `finetuning/outputs/` for the newest compatible run.

## Run Full Pipeline

```bash
uv run --project . --no-sync python -m backend.pipeline \
  --image_path image-input/photo.jpg \
  --prescription_file backend/meds.txt \
  --target_medicine Amoxicillin \
  --mode pixtral \
  --top_k 4 \
  --threshold 0.55
```

Artifacts are written to `backend/runs/<image>_<timestamp>/`.

## Run Individual Stages

```bash
uv run --project . --no-sync python -m backend.prescriber --prescription_file backend/meds.txt --output_json backend/runs/manual/prescriber_output.json
uv run --project . --no-sync python -m backend.bounder --image_path image-input/photo.jpg --inputs_dir backend/runs/manual/inputs --bbox_output_dir backend/runs/manual/bbox_outputs --output_json backend/runs/manual/bounder_result.json
uv run --project . --no-sync python -m backend.cropper --input_dir backend/runs/manual/inputs --json_dir backend/runs/manual/bbox_outputs/json --output_dir backend/runs/manual/crops --output_json backend/runs/manual/cropper_result.json
uv run --project . --no-sync python -m backend.labeler --run_dir finetuning/outputs/pixtral_lora --input_path backend/runs/manual/crops --prescriber_json backend/runs/manual/prescriber_output.json --bbox_json backend/runs/manual/bbox_outputs/json/photo.json --source_image photo.jpg --mode pixtral --top_k 4 --threshold 0.55 --output_jsonl backend/runs/manual/predictions.jsonl --output_summary_json backend/runs/manual/labeler_result.json
uv run --project . --no-sync python -m backend.center --bbox_json backend/runs/manual/bbox_outputs/json/photo.json --predictions_jsonl backend/runs/manual/predictions.jsonl --target_medicine Amoxicillin --output_json backend/runs/manual/centers.json
```
