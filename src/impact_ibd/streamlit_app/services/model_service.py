"""Pretrained model loading, evaluate adapter, and neighbor lookup."""

from __future__ import annotations

import copy
import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve

from impact_ibd import amgscr

from impact_ibd.streamlit_app.constants import DISTANCE_METRICS, MAX_K, TASK_LABELS
from impact_ibd.streamlit_app.services.preprocessing import maybe_apply_tss
from impact_ibd.streamlit_app.services.validation import (
    ValidationError,
    expected_model_features,
    normalize_weights,
    validate_abundance,
    validate_inference_params,
    validate_metadata,
)


def _cache_resource(func):
    try:
        import streamlit as st

        return st.cache_resource(show_spinner=False)(func)
    except Exception:
        return func


@_cache_resource
def get_cached_model(task: str):
    if task not in TASK_LABELS:
        raise ValidationError(f"Unknown task: {task}")
    return amgscr.load_pretrained_model(task=task)


def copy_model(task: str):
    return copy.deepcopy(get_cached_model(task))


def model_defaults(task: str) -> dict:
    model = get_cached_model(task)
    labels = _reference_labels(model)
    return {
        "task": task,
        "label": TASK_LABELS[task],
        "target_col": model.target_col_train_,
        "pos_label": model.pos_label_,
        "classes": labels,
        "threshold": float(model.threshold_),
        "k": int(model.k),
        "metric": str(model.metric),
        "weights": {
            "His": float(model.w[0]),
            "PTS": float(model.w[1]),
            "Fla": float(model.w[2]),
        },
        "n_features": len(expected_model_features(model)),
    }


def _reference_labels(model) -> list:
    col = model.target_col_train_
    values = model.meta_ref[col].dropna().astype(str).unique().tolist()
    pos = str(model.pos_label_) if model.pos_label_ is not None else None
    neg = [v for v in values if v != pos]
    if pos and pos in values:
        return neg + [pos]
    return sorted(values)


def run_signature(
    abundance_bytes: bytes,
    metadata_bytes: Optional[bytes],
    task: str,
    apply_tss: bool,
    k: int,
    metric: str,
    weights: Sequence[float],
    threshold: float,
) -> str:
    h = hashlib.sha256()
    h.update(abundance_bytes)
    h.update(b"|meta|")
    h.update(metadata_bytes or b"")
    payload = f"|{task}|{int(apply_tss)}|{k}|{metric}|{tuple(float(w) for w in weights)}|{float(threshold)}"
    h.update(payload.encode("utf-8"))
    return h.hexdigest()


def filter_neighbors(neighbors: pd.DataFrame, query_sample) -> pd.DataFrame:
    return neighbors.loc[neighbors["Query_Sample"] == query_sample].copy()


def neighbors_per_query(neighbors: pd.DataFrame) -> pd.Series:
    return neighbors.groupby("Query_Sample").size()


@dataclass
class ModelRunResult:
    mode: str
    prediction_df: pd.DataFrame
    neighbors_df: pd.DataFrame
    metrics: dict
    applied_tss: bool
    true_labels: Optional[pd.Series]
    pos_label: Any
    class_order: list
    confusion: Optional[np.ndarray]
    roc: Optional[dict]


def run_model(
    *,
    task: str,
    X_raw: pd.DataFrame,
    meta: Optional[pd.DataFrame],
    target_col: Optional[str],
    apply_tss: bool,
    k: int,
    metric: str,
    weight_his: float,
    weight_pts: float,
    weight_fla: float,
    threshold: float,
) -> ModelRunResult:
    prototype = get_cached_model(task)
    expected = expected_model_features(prototype)
    abd_summary = validate_abundance(X_raw, expected)
    if abd_summary.errors:
        raise ValidationError(" ".join(abd_summary.errors))
    X_checked = abd_summary.df
    assert X_checked is not None

    classes = _reference_labels(prototype)
    meta_valid = None
    mode = "predict"
    if meta is not None:
        target = target_col or prototype.target_col_train_
        meta_summary = validate_metadata(
            meta,
            X_checked.index,
            allowed_labels=classes,
            target_col=target,
        )
        if meta_summary.errors:
            raise ValidationError(" ".join(meta_summary.errors))
        meta_valid = meta_summary.df
        mode = "evaluate"
        target_col = target

    validate_inference_params(k, metric, threshold, DISTANCE_METRICS, MAX_K)
    weights = normalize_weights(weight_his, weight_pts, weight_fla)

    if apply_tss and abd_summary.zero_sum_ids:
        raise ValidationError(
            "TSS normalization cannot be applied because these samples have a row sum of 0 across all uploaded features: "
            + ", ".join(abd_summary.zero_sum_ids)
        )

    X_for_model = maybe_apply_tss(X_checked, apply_tss)

    model = copy.deepcopy(prototype)
    snapshot = (int(prototype.k), str(prototype.metric), tuple(float(x) for x in prototype.w), float(prototype.threshold_))
    model.k = int(k)
    model.metric = metric
    model.w = weights

    tmp_dir = tempfile.TemporaryDirectory()
    try:
        csv_path = Path(tmp_dir.name) / "predictions.csv"
        eval_kwargs = {
            "X_test": X_for_model,
            "threshold": float(threshold),
            "save_csv": str(csv_path),
            "save_pdf": None,
        }
        true_labels = None
        if mode == "evaluate":
            eval_kwargs["meta_test"] = meta_valid
            eval_kwargs["target_col"] = target_col
        metrics = model.evaluate(**eval_kwargs)
        prediction_df = pd.read_csv(csv_path)
        neighbors_df = model.get_nearest_neighbors(X_query=X_for_model, topk=int(k))
    finally:
        tmp_dir.cleanup()

    after = (int(prototype.k), str(prototype.metric), tuple(float(x) for x in prototype.w), float(prototype.threshold_))
    if snapshot != after:
        raise RuntimeError("The cached model prototype was modified; the run was aborted to protect shared state.")

    pos_label = metrics.get("pos_label")
    class_order = classes
    confusion = None
    roc = None
    if mode == "evaluate":
        aligned_meta = meta_valid.reindex(X_for_model.index)
        true_labels = aligned_meta[target_col]
        merged = prediction_df.copy()
        merged["_true"] = true_labels.to_numpy()
        if pos_label in class_order and len(class_order) == 2:
            neg = [c for c in class_order if c != pos_label][0]
            class_order = [neg, pos_label]
        confusion = confusion_matrix(
            merged["_true"],
            merged["Predicted_Class"],
            labels=class_order,
        )
        y_true_bin = (merged["_true"] == pos_label).astype(int)
        prob_col = f"Probability_{pos_label}"
        if prob_col in merged.columns and y_true_bin.nunique() == 2:
            fpr, tpr, _ = roc_curve(y_true_bin, merged[prob_col], pos_label=1)
            roc = {"fpr": fpr, "tpr": tpr, "auc": metrics.get("auc_roc")}
        else:
            roc = {"single_class": True}

    counts = neighbors_per_query(neighbors_df)
    if not counts.empty and int(counts.min()) != int(k):
        raise RuntimeError("The number of nearest neighbors does not match k for this prediction.")

    return ModelRunResult(
        mode=mode,
        prediction_df=prediction_df,
        neighbors_df=neighbors_df,
        metrics=metrics,
        applied_tss=bool(apply_tss),
        true_labels=true_labels,
        pos_label=pos_label,
        class_order=class_order,
        confusion=confusion,
        roc=roc,
    )
