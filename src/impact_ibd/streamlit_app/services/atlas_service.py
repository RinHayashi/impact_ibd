"""Reference atlas coordinates, pre-generated HTML, and recoding helpers."""

from __future__ import annotations

from typing import Mapping, Optional, Tuple

import pandas as pd

import impact_ibd as imibd
from impact_ibd.atlas import (
    add_spherical_coordinates,
    compute_amg_atlas_coordinates,
    plot_spherical_amg_atlas,
)

from impact_ibd.streamlit_app.constants import (
    ATLAS_HTML_PATH,
    GROUP_COLORS,
    HOST_ATLAS_COLOR_FIELDS,
    NA_COLOR,
)


def _cache_resource(func):
    try:
        import streamlit as st

        return st.cache_resource(show_spinner=False)(func)
    except Exception:
        return func


def read_pregenerated_atlas_html() -> str:
    if not ATLAS_HTML_PATH.is_file():
        raise FileNotFoundError(
            f"Pregenerated atlas HTML was not found: {ATLAS_HTML_PATH}"
        )
    return ATLAS_HTML_PATH.read_text(encoding="utf-8")


def _indexed_copy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    return out.set_index(out.columns[0])


@_cache_resource
def get_reference_atlas() -> pd.DataFrame:
    """Compute spherical coordinates once (seed=42) for Host-page recoloring."""
    abd = _indexed_copy(imibd.load_amg_abd())
    host = _indexed_copy(imibd.load_host_profile())
    atlas = compute_amg_atlas_coordinates(
        abd,
        host,
        seed=42,
        extra_cols=list(HOST_ATLAS_COLOR_FIELDS),
    )
    return add_spherical_coordinates(atlas)


def fill_color_na(values: pd.Series) -> pd.Series:
    filled = values.copy()
    mask = filled.isna() | (filled.astype("string").str.strip() == "") | (filled.astype("string") == "<NA>")
    filled = filled.astype("object")
    filled.loc[mask] = "NA"
    return filled.astype(str)


def prepare_colored_atlas(
    atlas: pd.DataFrame,
    color_col: str,
    host_profile: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, str, dict]:
    work = atlas.copy()
    host = host_profile
    if host is not None and color_col not in work.columns:
        host = host.copy()
        if host.columns[0].lower() in {"sample", "sampleid", "sample_id"} or work.index.name == host.columns[0]:
            host = host.set_index(host.columns[0])
        host.index = host.index.astype(str)
        resolved = color_col
        lowered = {str(c).lower(): c for c in host.columns}
        if color_col not in host.columns and color_col.lower() in lowered:
            resolved = lowered[color_col.lower()]
        work[resolved] = host.reindex(work.index.astype(str))[resolved].to_numpy()
        color_col = resolved
    if color_col not in work.columns:
        lowered = {str(c).lower(): c for c in work.columns}
        if color_col.lower() in lowered:
            color_col = lowered[color_col.lower()]
        else:
            raise KeyError(color_col)
    work[color_col] = fill_color_na(work[color_col])
    color_map = {**GROUP_COLORS, "NA": NA_COLOR}
    return work, color_col, color_map


def plot_colored_atlas(
    atlas: pd.DataFrame,
    color_col: str,
    host_profile: Optional[pd.DataFrame] = None,
    color_map: Optional[Mapping[str, str]] = None,
):
    prepared, resolved, default_map = prepare_colored_atlas(atlas, color_col, host_profile)
    return plot_spherical_amg_atlas(
        prepared,
        color_col=resolved,
        host_profile=None,
        color_map=dict(default_map if color_map is None else color_map),
        title="<b>Spherical Disease-aware AMG Functional Atlas</b>",
        fig_width=900,
        fig_height=720,
    )
