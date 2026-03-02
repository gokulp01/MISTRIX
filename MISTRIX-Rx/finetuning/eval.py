from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from utils import ensure_dir, save_json

LOGGER = logging.getLogger(__name__)


def topk_accuracy_from_probs(probs: np.ndarray, y_true: np.ndarray, k: int = 1) -> float:
    k = min(k, probs.shape[1])
    topk = np.argpartition(-probs, kth=k - 1, axis=1)[:, :k]
    return float(np.mean([y_true[i] in topk[i] for i in range(len(y_true))]))


def macro_f1_from_probs(probs: np.ndarray, y_true: np.ndarray) -> float:
    y_pred = probs.argmax(axis=1)
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def confidence_stats(probs: np.ndarray) -> dict:
    conf = probs.max(axis=1)
    return {
        "conf_min": float(conf.min()),
        "conf_mean": float(conf.mean()),
        "conf_median": float(np.median(conf)),
        "conf_p90": float(np.percentile(conf, 90)),
        "conf_max": float(conf.max()),
    }


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == y_true).astype(np.float32)

    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf >= lo) & (conf < hi if i < bins - 1 else conf <= hi)
        if not np.any(mask):
            continue
        bin_acc = acc[mask].mean()
        bin_conf = conf[mask].mean()
        ece += np.abs(bin_acc - bin_conf) * mask.mean()
    return float(ece)


def save_confusion_artifacts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    output_dir: Path,
    prefix: str,
) -> tuple[Path, Path]:
    output_dir = ensure_dir(output_dir)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))

    csv_path = output_dir / f"{prefix}_confusion_matrix.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + labels)
        for i, row in enumerate(cm):
            w.writerow([labels[i]] + row.tolist())

    png_path = output_dir / f"{prefix}_confusion_matrix.png"
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_title(f"Confusion Matrix ({prefix})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if len(labels) <= 40:
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    return csv_path, png_path


def evaluate_classification(
    probs: np.ndarray,
    y_true: np.ndarray,
    labels: list[str],
    output_dir: Path,
    prefix: str,
) -> dict:
    y_pred = probs.argmax(axis=1)
    metrics = {
        "top1": topk_accuracy_from_probs(probs, y_true, k=1),
        "top5": topk_accuracy_from_probs(probs, y_true, k=5),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "ece": expected_calibration_error(probs, y_true),
    }
    metrics.update(confidence_stats(probs))
    csv_path, png_path = save_confusion_artifacts(y_true, y_pred, labels, output_dir, prefix)
    metrics["confusion_csv"] = str(csv_path)
    metrics["confusion_png"] = str(png_path)
    return metrics


def retrieval_topk_from_similarity(
    sim: np.ndarray,
    true_idx: np.ndarray,
    ks: Iterable[int] = (1, 5),
) -> dict[str, float]:
    out: dict[str, float] = {}
    n = sim.shape[0]
    for k in ks:
        k = min(k, sim.shape[1])
        topk = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
        out[f"top{k}"] = float(np.mean([true_idx[i] in topk[i] for i in range(n)]))
    return out


def build_prediction_records(
    image_paths: list[str],
    probs: np.ndarray,
    labels: list[str],
    true_indices: Optional[np.ndarray] = None,
    top_k: int = 5,
) -> list[dict]:
    top_k = min(top_k, probs.shape[1])
    records = []
    for i in range(len(image_paths)):
        order = np.argsort(-probs[i])[:top_k]
        top_labels = [labels[j] for j in order]
        top_scores = [float(probs[i, j]) for j in order]
        rec = {
            "image_path": image_paths[i],
            "top_k_labels": top_labels,
            "top_k_scores": top_scores,
            "predicted_label": top_labels[0],
            "confidence": top_scores[0],
        }
        if true_indices is not None:
            rec["true_label"] = labels[int(true_indices[i])]
        records.append(rec)
    return records


def sweep_abstain_threshold_from_records(
    records: list[dict],
    labels: list[str],
    num_thresholds: int = 101,
) -> dict:
    if not records or "true_label" not in records[0]:
        raise ValueError("Records must contain true_label for threshold sweep.")

    y_true = np.array([labels.index(r["true_label"]) for r in records], dtype=np.int64)
    y_pred = np.array([labels.index(r["predicted_label"]) for r in records], dtype=np.int64)
    conf = np.array([float(r["confidence"]) for r in records], dtype=np.float32)

    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    best = None
    rows = []

    for t in thresholds:
        pred_t = y_pred.copy()
        abstain_mask = conf < t
        pred_t[abstain_mask] = -1

        top1 = float(np.mean(pred_t == y_true))
        coverage = float(np.mean(~abstain_mask))

        pred_for_f1 = pred_t.copy()
        pred_for_f1[pred_for_f1 == -1] = len(labels)
        y_for_f1 = y_true.copy()
        macro_f1 = float(
            f1_score(
                y_for_f1,
                pred_for_f1,
                labels=list(range(len(labels))),
                average="macro",
                zero_division=0,
            )
        )
        score = macro_f1 * coverage

        row = {
            "threshold": float(t),
            "top1": top1,
            "coverage": coverage,
            "macro_f1": macro_f1,
            "selection_score": score,
        }
        rows.append(row)
        if best is None or row["selection_score"] > best["selection_score"]:
            best = row

    return {"best": best, "grid": rows}


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def cli_sweep_threshold(args: argparse.Namespace) -> None:
    records = _read_jsonl(Path(args.predictions_jsonl))
    labels = args.labels
    if labels is None:
        labels = sorted({r.get("true_label") for r in records if "true_label" in r})
    result = sweep_abstain_threshold_from_records(records, labels, args.num_thresholds)
    save_json(result, args.output_json)
    print(json.dumps(result["best"], indent=2))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluation and threshold sweep utilities.")
    sub = parser.add_subparsers(dest="command", required=True)

    sweep = sub.add_parser("sweep-threshold", help="Pick abstain threshold from validation predictions JSONL")
    sweep.add_argument("--predictions_jsonl", required=True, type=str)
    sweep.add_argument("--output_json", required=True, type=str)
    sweep.add_argument("--num_thresholds", type=int, default=101)
    sweep.add_argument(
        "--labels",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit label list in canonical order.",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = build_argparser()
    args = parser.parse_args()

    if args.command == "sweep-threshold":
        cli_sweep_threshold(args)


if __name__ == "__main__":
    main()
