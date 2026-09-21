"""Optional TSS normalization on the full uploaded feature set."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

TSS_ROW_SUM_TOLERANCE = 1e-6


class TSSError(ValueError):
    """Raised when TSS cannot be applied."""


def apply_tss(X_raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize each sample across *all* uploaded numeric features.

    Must run before any model feature alignment or filtering.
    Does not modify ``X_raw``.
    """
    X_for_model = X_raw.copy()
    row_sums = X_for_model.sum(axis=1)
    zero_ids = [str(idx) for idx, total in row_sums.items() if float(total) == 0.0]
    if zero_ids:
        raise TSSError(
            "TSS normalization cannot be applied because these samples have a row sum of 0 across all uploaded features: "
            + ", ".join(zero_ids)
        )
    X_for_model = X_for_model.div(row_sums, axis=0)
    _assert_unit_row_sums(X_for_model)
    return X_for_model


def maybe_apply_tss(X_raw: pd.DataFrame, apply: bool) -> pd.DataFrame:
    if not apply:
        return X_raw.copy()
    return apply_tss(X_raw)


def row_sums(X: pd.DataFrame) -> pd.Series:
    return X.sum(axis=1)


def zero_row_sum_sample_ids(X: pd.DataFrame) -> List[str]:
    totals = row_sums(X)
    return [str(idx) for idx, total in totals.items() if float(total) == 0.0]


def looks_like_tss(X: pd.DataFrame) -> bool:
    totals = row_sums(X)
    if totals.empty:
        return False
    return bool(np.all(np.abs(totals.to_numpy(dtype=float) - 1.0) <= 1e-3))


def _assert_unit_row_sums(X: pd.DataFrame) -> None:
    totals = row_sums(X)
    nonzero = totals[totals != 0]
    if nonzero.empty:
        return
    if not np.all(np.abs(nonzero.to_numpy(dtype=float) - 1.0) <= TSS_ROW_SUM_TOLERANCE):
        raise TSSError(
            "After TSS normalization, some samples do not have row sums equal to 1 within floating-point tolerance."
        )


def prepare_model_matrix(
    X_raw: pd.DataFrame, apply_tss_flag: bool
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Return X_for_model without aligning to model features."""
    if apply_tss_flag:
        return apply_tss(X_raw), "tss"
    return X_raw.copy(), "raw"
