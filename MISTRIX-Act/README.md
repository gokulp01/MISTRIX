# MISTRIX-Act

The system predicts robot action trajectories by generating discretized action tokens from multimodal context (images + instruction), then decoding them back to continuous control values.

## Core Design

- **Backbone**: Pixtral causal multimodal language model.
- **Action representation**: continuous actions are discretized into integer bins and serialized as text.
- **Training objective**: causal next-token prediction on assistant action tokens only.
- **Inference**: constrained numeric generation followed by de-quantization to continuous actions.

## Repository Layout

- `rv_train/models/pixtral/model.py`: policy orchestrator (`PixtralPolicy` / `PixtralActor`).
- `rv_train/pipelines/pixtral/action_codec.py`: action binning and de-binning.
- `rv_train/pipelines/pixtral/vision_adapter.py`: RGB batch validation and image conversion.
- `rv_train/pipelines/pixtral/chat_io.py`: dialog construction and processor input assembly.
- `rv_train/pipelines/pixtral/training_ops.py`: label masking and causal LM loss prep.
- `rv_train/pipelines/pixtral/model_loader.py`: HF model/processor loading and checkpoint restore.
- `rv_train/train.py`: training entrypoint.
- `eval/eval_libero.py`: evaluation entrypoint.
- `rv_train/deploy/service.py`: FastAPI inference service.

## End-to-End Architecture

```text
Dataset sample
  -> vision adapter (history/camera formatting)
  -> action codec (continuous -> binned integer text)
  -> chat I/O (system/user/assistant message build)
  -> Pixtral processor (tokenization + image features)
  -> Pixtral LM forward
  -> label builder (mask prompt/pad, keep action supervision)
  -> causal LM loss

Inference
  -> same encoding path
  -> constrained numeric generation
  -> action codec decode (text -> continuous horizon actions)
```

## Environment Setup

```bash
conda create -y -n pixtral_policy python=3.10
conda activate pixtral_policy
PIP_REQ_EXTRAS=pixtral,libero pip install --no-build-isolation -e ".[pixtral,libero]"
cd libs/RoboVerse
PIP_REQ_EXTRAS=lerobot pip install --no-build-isolation -e ".[lerobot]"
cd ../..
```

## Configuration

Primary experiment config: `configs/pixtral_12b.yaml`

Important fields:

- `EXP.MODEL: "pixtral"`
- `MODEL.PIXTRAL.pixtral_model_id`
- `MODEL.PIXTRAL.horizon`
- `MODEL.PIXTRAL.original_action_dim`
- `MODEL.PIXTRAL.num_bins_actions`
- `DATALOADER.ROBOVERSE.cfg_path`

## Training

```bash
python -m rv_train.train --exp-config ./configs/pixtral_12b.yaml
```

## Evaluation

```bash
python eval/eval_libero.py \
  --model_path ./runs/pixtral/model_last.pth \
  --task_suite_name libero_goal \
  --task_name put_the_wine_bottle_on_top_of_the_cabinet
```

## Deployment API

```bash
ROBOVERSE_DEPLOY_CHECKPOINT=./runs/pixtral/model_last.pth python rv_train/deploy/service.py
```

API docs are available at `http://<server_ip>:10000/docs`.

## Local Mac Smoke Test

Use this to validate wiring (forward/backward/generation) without full-scale training:

```bash
uv run --with torch python -m rv_train.models.pixtral.smoke_test
```

## Engineering Invariants

- Action normalization/de-normalization must always use dataset stats from `dataset_stats.pkl`.
- Prompt/user/system tokens must not contribute to loss.
- Generated text must decode into exactly one action horizon (pad/trim behavior is explicit in codec).
- Model-selection logic is centralized in `rv_train/model_specs.py`.

## Acknowledgment

This project is modified from the original implementation at: https://github.com/NVlabs/vla0
