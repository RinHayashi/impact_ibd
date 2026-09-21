"""Upload validation for abundance and metadata CSVs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from impact_ibd.streamlit_app.services.preprocessing import (
    looks_like_tss,
    row_sums,
    zero_row_sum_sample_ids,
)


class ValidationError(ValueError):
    """Blocking input validation error with a user-facing message."""


@dataclass
class AbundanceSummary:
    n_samples: int
    n_features: int
    n_overlap: int
    n_missing: int
    n_extra: int
    row_sums: pd.Series
    looks_normalized: bool
    zero_sum_ids: List[str]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    df: Optional[pd.DataFrame] = None


@dataclass
class MetadataSummary:
    n_rows: int
    columns: List[str]
    extra_samples: List[str]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    df: Optional[pd.DataFrame] = None


def parse_csv_bytes(data: bytes) -> pd.DataFrame:
    import io

    if not data:
        raise ValidationError("The uploaded CSV is empty.")
    try:
        df = pd.read_csv(io.BytesIO(data), index_col=0)
    except Exception:
        raise ValidationError(
            "Could not parse the CSV. Use a comma-separated file with sample IDs in the first column."
        ) from None
    if df.empty:
        raise ValidationError("The CSV does not contain any data rows.")
    return df


def expected_model_features(model) -> List[str]:
    return list(dict.fromkeys(model.genes_aa + model.genes_carb + model.genes_flag))


def validate_abundance(df: pd.DataFrame, expected_features: Sequence[str]) -> AbundanceSummary:
    errors: List[str] = []
    warnings: List[str] = []

    if df.index.hasnans:
        errors.append("Sample IDs (first column) contain missing values.")
    if not df.index.is_unique:
        errors.append("Sample IDs must be unique.")
    if df.columns.hasnans:
        errors.append("Feature names contain missing values.")
    if pd.Index(df.columns).duplicated().any():
        errors.append("Feature names are duplicated.")

    numeric = df.copy()
    try:
        numeric = numeric.apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError):
        errors.append("The abundance matrix contains cells that cannot be converted to numbers.")
        return AbundanceSummary(
            n_samples=len(df),
            n_features=df.shape[1],
            n_overlap=0,
            n_missing=len(expected_features),
            n_extra=df.shape[1],
            row_sums=pd.Series(dtype=float),
            looks_normalized=False,
            zero_sum_ids=[],
            errors=errors,
            warnings=warnings,
        )

    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        errors.append("The abundance matrix contains NaN or infinite values.")
    if np.nanmin(values) < 0:
        errors.append("The abundance matrix contains negative values.")

    provided = set(map(str, numeric.columns))
    expected = set(map(str, expected_features))
    overlap = provided & expected
    missing = expected - provided
    extra = provided - expected
    if len(overlap) == 0:
        errors.append("Uploaded features do not overlap the AMG features required by the model.")

    sums = row_sums(numeric)
    zeros = zero_row_sum_sample_ids(numeric)
    normalized = looks_like_tss(numeric) if not errors else False
    if not normalized:
        warnings.append(
            "Row sums may not reflect TSS normalization. If the values are still raw counts or unnormalized relative abundance, select TSS normalization."
        )

    return AbundanceSummary(
        n_samples=len(numeric),
        n_features=numeric.shape[1],
        n_overlap=len(overlap),
        n_missing=len(missing),
        n_extra=len(extra),
        row_sums=sums,
        looks_normalized=normalized,
        zero_sum_ids=zeros,
        errors=errors,
        warnings=warnings,
        df=numeric if not errors else None,
    )


def validate_metadata(
    meta: pd.DataFrame,
    sample_ids: Iterable[object],
    allowed_labels: Optional[Sequence[object]] = None,
    target_col: Optional[str] = None,
) -> MetadataSummary:
    errors: List[str] = []
    warnings: List[str] = []
    sample_ids = pd.Index(list(sample_ids))

    if meta.index.hasnans:
        errors.append("Metadata sample IDs contain missing values.")
    if not meta.index.is_unique:
        errors.append("Metadata sample IDs must be unique.")

    missing = [str(s) for s in sample_ids if s not in meta.index]
    extra = [str(s) for s in meta.index if s not in sample_ids]
    if missing:
        errors.append(
            "Metadata does not cover all abundance samples: " + ", ".join(missing[:20])
            + ("…" if len(missing) > 20 else "")
        )
    if extra:
        warnings.append(
            f"Metadata includes {len(extra)} extra samples; they will be ignored during evaluation."
        )

    if target_col is not None:
        if target_col not in meta.columns:
            errors.append(f"Target column {target_col} was not found in metadata.")
        elif allowed_labels is not None and missing == []:
            aligned = meta.reindex(sample_ids)
            labels = aligned[target_col]
            if labels.isna().any():
                errors.append(f"Target column {target_col} is missing for some samples.")
            illegal = sorted(
                {
                    str(v)
                    for v in labels.dropna().unique()
                    if v not in set(allowed_labels)
                }
            )
            if illegal:
                errors.append(
                    "The target column contains labels not allowed for this model: "
                    + ", ".join(illegal)
                    + f". Allowed labels: {', '.join(map(str, allowed_labels))}."
                )

    return MetadataSummary(
        n_rows=len(meta),
        columns=list(map(str, meta.columns)),
        extra_samples=extra,
        errors=errors,
        warnings=warnings,
        df=meta if not errors else None,
    )


def infer_mode(metadata_present: bool, metadata_valid: bool) -> str:
    if metadata_present and metadata_valid:
        return "evaluate"
    return "predict"


def normalize_weights(his: float, pts: float, fla: float) -> np.ndarray:
    if min(his, pts, fla) < 0:
        raise ValidationError("Module weights must be non-negative.")
    vec = np.array([his, pts, fla], dtype=float)
    total = float(vec.sum())
    if total <= 0:
        raise ValidationError("The His / PTS / Fla weights must sum to more than 0.")
    return vec / total


def validate_inference_params(k: int, metric: str, threshold: float, allowed_metrics: Sequence[str], max_k: int) -> None:
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or k < 1:
        raise ValidationError("k must be a positive integer.")
    if k > max_k:
        raise ValidationError(
            f"k cannot be greater than {max_k}, so neighbor-graph nodes do not overlap."
        )
    if metric not in allowed_metrics:
        raise ValidationError(f"Unsupported distance metric: {metric}.")
    if not np.isfinite(threshold) or threshold < 0 or threshold > 1:
        raise ValidationError("The decision threshold must be between 0 and 1.")
