"""Spherical disease-aware AMG functional atlas (3D visualization).

Ports the reference notebook ``3dumap_viz_part.ipynb`` into a reusable API.
Coordinate computation is independent of plotting so a later Streamlit /
CLI layer can cache embeddings and only re-color the figure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import umap
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

PathLike = Union[str, Path]

# ---------------------------------------------------------------------------
# Default AMG modules (same membership as the reference notebook)
# ---------------------------------------------------------------------------

PTS_AMGS = [
    "K02777", "K02779", "K02791", "K20108", "K02804", "K02810",
    "K02753", "K02757", "K02819", "K11192", "K02795", "K19507",
    "K19509", "K02746", "K02747", "K17465", "K02774", "K02798",
    "K02759", "K02760", "K02761",
]

HIS_AMGS = [
    "K00013", "K00765", "K00817", "K01089", "K01496", "K01693",
    "K01814", "K02500", "K02502", "K04486", "K11755",
]

FLA_AMGS = [
    "K02389", "K02390", "K02391", "K02396", "K02397", "K02399",
    "K02407", "K02412", "K02416", "K02417", "K02422", "K02424",
    "K02556", "K02557",
]

DEFAULT_KEEP_GROUPS = ("Control", "UC", "CD")
DEFAULT_GROUP_ORDER = ["Control", "UC", "CD"]

DEFAULT_GROUP_COLORS = {
    "Control": "#9E9E9E",
    "UC": "#4C78A8",
    "CD": "#E45756",
}

POINT_SIZE_NEAR = {"Control": 4.8, "UC": 5.3, "CD": 5.3}
POINT_SIZE_FAR = {"Control": 5.0, "UC": 2.7, "CD": 2.7}
POINT_OPACITY_NEAR = {"Control": 0.60, "UC": 0.88, "CD": 0.88}
POINT_OPACITY_FAR = {"Control": 0.18, "UC": 0.22, "CD": 0.22}
FAR_WHITE_MIX = {"Control": 0.62, "UC": 0.60, "CD": 0.60}

# Fallbacks used when color_col is not the disease Group column
_FALLBACK_SIZE_NEAR = 5.3
_FALLBACK_SIZE_FAR = 2.7
_FALLBACK_OPACITY_NEAR = 0.88
_FALLBACK_OPACITY_FAR = 0.22
_FALLBACK_WHITE_MIX = 0.60

_QUALITATIVE_PALETTE = [
    "#4C78A8", "#E45756", "#72B7B2", "#F58518", "#54A24B",
    "#EECA3B", "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC",
    "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
    "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
]

_ID_COLUMN_ALIASES = {
    "amg_function", "sample", "sampleid", "sample_id",
    "votu_id", "taxid", "amg_orf", "votu_rep_seq",
}


@dataclass
class AMGAtlasResult:
    """Container returned by :func:`visualize_amg_atlas`."""

    atlas: pd.DataFrame
    figure: go.Figure
    paths: Dict[str, Path] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _match_col(df: pd.DataFrame, name: str) -> str:
    if name in df.columns:
        return name
    lowered = {str(c).lower(): c for c in df.columns}
    key = str(name).lower()
    if key in lowered:
        return lowered[key]
    raise KeyError(
        f"Column '{name}' not found. Available columns: {list(df.columns)}"
    )


def _ensure_indexed(df: pd.DataFrame, id_candidates: Sequence[str] = ()) -> pd.DataFrame:
    """Set the identifier column as index when the caller has not done so."""
    if df is None:
        raise ValueError("DataFrame is None")
    out = df.copy()
    for cand in id_candidates:
        if cand in out.columns:
            return out.set_index(cand)
        lowered = {str(c).lower(): c for c in out.columns}
        if cand.lower() in lowered:
            return out.set_index(lowered[cand.lower()])
    if isinstance(out.index, pd.RangeIndex) or (
        out.index.name is None and pd.api.types.is_integer_dtype(out.index)
    ):
        first = out.columns[0]
        if str(first).lower() in _ID_COLUMN_ALIASES or not pd.api.types.is_numeric_dtype(out[first]):
            return out.set_index(first)
    return out


def _align_abd_host(
    amg_abd: pd.DataFrame,
    host_profile: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return abundance with samples as columns and host with samples as index."""
    abd = _ensure_indexed(amg_abd, id_candidates=("AMG_function",))
    host = _ensure_indexed(host_profile, id_candidates=("sample",))

    abd.columns = abd.columns.astype(str)
    abd.index = abd.index.astype(str)
    host.index = host.index.astype(str)

    host_ids = set(host.index)
    col_overlap = len(host_ids.intersection(abd.columns.astype(str)))
    idx_overlap = len(host_ids.intersection(abd.index.astype(str)))
    if idx_overlap > col_overlap:
        abd = abd.T
        abd.columns = abd.columns.astype(str)
        abd.index = abd.index.astype(str)

    return abd, host


def _resolve_group_col(host: pd.DataFrame, group_col: Optional[str]) -> str:
    if group_col is not None:
        return _match_col(host, group_col)
    for cand in ("Group", "group"):
        if cand in host.columns:
            return cand
    raise KeyError(
        "Cannot find a disease-group column. Pass group_col=... "
        f"(available: {list(host.columns)})"
    )


def _present(genes: Iterable[str], columns: Iterable[str]) -> list:
    colset = set(columns)
    return [g for g in genes if g in colset]


def _module_pca(
    X_raw: pd.DataFrame,
    genes: Sequence[str],
    n_pc: int,
    seed: int,
) -> np.ndarray:
    genes = _present(genes, X_raw.columns)
    if not genes:
        return np.zeros((X_raw.shape[0], 0))
    Xm = np.log1p(X_raw.loc[:, genes].copy())
    sd = Xm.std(axis=0)
    keep = sd[sd > 0].index
    if len(keep) == 0:
        return np.zeros((X_raw.shape[0], 0))
    Xm = Xm.loc[:, keep]
    Xm_z = StandardScaler().fit_transform(Xm)
    n_comp = min(n_pc, Xm_z.shape[1])
    if n_comp < 1:
        return np.zeros((X_raw.shape[0], 0))
    pca = PCA(n_components=n_comp, random_state=seed)
    return pca.fit_transform(Xm_z)


def _make_ball(XYZ, percentile: float = 98) -> Tuple[np.ndarray, np.ndarray]:
    """Rescale 3D coordinates into a ball without forcing them onto the surface."""
    XYZ = np.asarray(XYZ, dtype=float)
    XYZ = XYZ - XYZ.mean(axis=0, keepdims=True)
    XYZ = StandardScaler().fit_transform(XYZ)

    radius = np.linalg.norm(XYZ, axis=1)
    scale = np.percentile(radius, percentile)
    XYZ_ball = XYZ / (scale + 1e-12)

    radius = np.linalg.norm(XYZ_ball, axis=1)
    mask = radius > 1
    if mask.any():
        XYZ_ball[mask] = XYZ_ball[mask] / radius[mask, None]

    radius = np.linalg.norm(XYZ_ball, axis=1)
    return XYZ_ball, radius


def _rank_uniform(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    r = rankdata(x, method="average")
    return (r - 0.5) / len(r)


def _robust_01(x: np.ndarray, low_q: float = 0.02, high_q: float = 0.98) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo = np.quantile(x, low_q)
    hi = np.quantile(x, high_q)
    xc = np.clip(x, lo, hi)
    return (xc - lo) / (hi - lo + 1e-12)


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def _mix_with_white(hex_color: str, amount: float) -> str:
    amount = float(np.clip(amount, 0, 1))
    r, g, b = _hex_to_rgb(hex_color)
    r2 = int(r + (255 - r) * amount)
    g2 = int(g + (255 - g) * amount)
    b2 = int(b + (255 - b) * amount)
    return f"rgb({r2},{g2},{b2})"


def _style_lookup(mapping: Mapping[str, float], key: str, fallback: float) -> float:
    return float(mapping.get(key, fallback))


def _colors_for(categories: Sequence[str], color_map: Optional[Mapping[str, str]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    palette_i = 0
    for cat in categories:
        if color_map and cat in color_map:
            out[cat] = color_map[cat]
        elif cat in DEFAULT_GROUP_COLORS:
            out[cat] = DEFAULT_GROUP_COLORS[cat]
        else:
            out[cat] = _QUALITATIVE_PALETTE[palette_i % len(_QUALITATIVE_PALETTE)]
            palette_i += 1
    return out


def _category_order(values: Iterable[object]) -> list:
    seen = []
    for v in values:
        if pd.isna(v):
            continue
        s = str(v)
        if s not in seen:
            seen.append(s)
    ordered = [g for g in DEFAULT_GROUP_ORDER if g in seen]
    ordered.extend([g for g in seen if g not in ordered])
    return ordered


def _effective_splits(y: np.ndarray, n_splits: int) -> int:
    counts = pd.Series(y).value_counts()
    min_class = int(counts.min()) if len(counts) else 0
    if min_class < 2:
        raise ValueError(
            "Need at least 2 samples in each class for out-of-fold disease models."
        )
    return max(2, min(int(n_splits), min_class))


def _oof_proba(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    seed: int,
    cv_seed: Optional[int] = None,
) -> np.ndarray:
    oof = np.zeros(len(y), dtype=float)
    n_splits_eff = _effective_splits(y, n_splits)
    if cv_seed is None:
        cv_seed = seed
    cv = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=cv_seed)
    for train_idx, val_idx in cv.split(X, y):
        clf = LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="liblinear",
            max_iter=2000,
            random_state=seed,
        )
        clf.fit(X[train_idx], y[train_idx])
        oof[val_idx] = clf.predict_proba(X[val_idx])[:, 1]
    return oof


# ---------------------------------------------------------------------------
# Stage 1 — global + functional + OOF disease-aware coordinates
# ---------------------------------------------------------------------------

def compute_amg_atlas_coordinates(
    amg_abd: pd.DataFrame,
    host_profile: pd.DataFrame,
    *,
    group_col: Optional[str] = None,
    extra_cols: Optional[Sequence[str]] = None,
    seed: int = 42,
    min_prevalence: float = 0.02,
    n_global_pc: int = 20,
    n_module_pc: int = 3,
    n_splits: int = 5,
    oof_ibd_weight: float = 2.0,
    oof_subtype_weight: float = 1.5,
    umap_n_neighbors: int = 50,
    umap_min_dist: float = 0.15,
    keep_groups: Sequence[str] = DEFAULT_KEEP_GROUPS,
    his_amgs: Sequence[str] = HIS_AMGS,
    pts_amgs: Sequence[str] = PTS_AMGS,
    fla_amgs: Sequence[str] = FLA_AMGS,
    output_path: Optional[PathLike] = None,
) -> pd.DataFrame:
    """Build the unsupervised + disease-aware 3D atlas coordinate table.

    Parameters
    ----------
    amg_abd
        AMG abundance. After setting the first column as the index, rows are
        AMG functions and columns are sample names (``load_amg_abd()`` layout).
        A samples × features orientation is accepted and auto-transposed.
    host_profile
        Sample metadata. After setting the first column as the index, rows are
        sample names (``load_host_profile()`` layout).
    group_col
        Metadata column used for disease-aware OOF labels. Defaults to
        ``Group`` / ``group``. Samples are restricted to *keep_groups*.
    extra_cols
        Additional host columns copied into the atlas (e.g. the colouring
        column). ``Group`` is always written.
    seed
        Random seed for PCA, UMAP and stratified OOF splits.
    """
    np.random.seed(seed)

    abd, host = _align_abd_host(amg_abd, host_profile)
    group_col = _resolve_group_col(host, group_col)

    common_samples = abd.columns.intersection(host.index)
    host = host.loc[common_samples].copy()
    host = host.loc[host[group_col].isin(list(keep_groups))].copy()
    if host.empty:
        raise ValueError(
            f"No samples left after intersecting abundance/metadata and "
            f"keeping groups {tuple(keep_groups)} in column '{group_col}'."
        )
    abd = abd.loc[:, host.index].copy()

    X_all_raw = abd.T.astype(float)
    X_all_raw = X_all_raw.replace([np.inf, -np.inf], np.nan).fillna(0)

    prevalence = (X_all_raw > 0).mean(axis=0)
    global_genes = prevalence[prevalence >= min_prevalence].index
    X_global_raw = X_all_raw.loc[:, global_genes].copy()

    X_global_log = np.log1p(X_global_raw)
    sd = X_global_log.std(axis=0)
    X_global_log = X_global_log.loc[:, sd[sd > 0].index]
    if X_global_log.shape[1] < 1:
        raise ValueError("No AMG features remaining after prevalence / variance filters.")

    X_global_z = StandardScaler().fit_transform(X_global_log)
    n_pc = min(n_global_pc, X_global_z.shape[1])
    Z_global = PCA(n_components=n_pc, random_state=seed).fit_transform(X_global_z)

    Z_his = _module_pca(X_all_raw, his_amgs, n_module_pc, seed)
    Z_pts = _module_pca(X_all_raw, pts_amgs, n_module_pc, seed)
    Z_fla = _module_pca(X_all_raw, fla_amgs, n_module_pc, seed)

    X_unsup = np.column_stack([Z_global, Z_his, Z_pts, Z_fla])
    X_unsup_z = StandardScaler().fit_transform(X_unsup)

    group = host[group_col].astype(str).to_numpy()
    y_ibd = (group != "Control").astype(int)
    ibd_mask = group != "Control"
    y_cd = (group == "CD").astype(int)

    if len(np.unique(y_ibd)) < 2:
        raise ValueError(
            "Disease-aware atlas requires both Control and IBD (UC/CD) samples."
        )

    oof_ibd = _oof_proba(X_unsup_z, y_ibd, n_splits=n_splits, seed=seed)

    oof_subtype = np.full(len(host), np.nan)
    y_ibd_subtype = y_cd[ibd_mask]
    if ibd_mask.sum() >= 4 and len(np.unique(y_ibd_subtype)) >= 2:
        local_oof = _oof_proba(
            X_unsup_z[ibd_mask],
            y_ibd_subtype,
            n_splits=n_splits,
            seed=seed,
            cv_seed=seed + 1,
        )
        oof_subtype[ibd_mask] = local_oof

    oof_subtype_filled = oof_subtype.copy()
    oof_subtype_filled[~np.isfinite(oof_subtype_filled)] = 0.5

    oof_ibd_z = StandardScaler().fit_transform(oof_ibd.reshape(-1, 1)).ravel()
    oof_subtype_z = StandardScaler().fit_transform(
        oof_subtype_filled.reshape(-1, 1)
    ).ravel()

    X_disease_aware = np.column_stack(
        [
            X_unsup_z,
            oof_ibd_weight * oof_ibd_z,
            oof_subtype_weight * oof_subtype_z,
        ]
    )

    n_samples = X_unsup_z.shape[0]
    n_neighbors = max(2, min(int(umap_n_neighbors), n_samples - 1))

    def _umap3(X: np.ndarray) -> np.ndarray:
        return umap.UMAP(
            n_components=3,
            n_neighbors=n_neighbors,
            min_dist=umap_min_dist,
            metric="euclidean",
            random_state=seed,
        ).fit_transform(X)

    XYZ_pca_unsup_ball, radius_pca_unsup = _make_ball(
        PCA(n_components=3, random_state=seed).fit_transform(X_unsup_z)
    )
    XYZ_pca_disease_ball, radius_pca_disease = _make_ball(
        PCA(n_components=3, random_state=seed).fit_transform(X_disease_aware)
    )
    XYZ_umap_unsup_ball, radius_umap_unsup = _make_ball(_umap3(X_unsup_z))
    XYZ_umap_disease_ball, radius_umap_disease = _make_ball(_umap3(X_disease_aware))

    atlas = pd.DataFrame(index=host.index)
    atlas["UPCA1"] = XYZ_pca_unsup_ball[:, 0]
    atlas["UPCA2"] = XYZ_pca_unsup_ball[:, 1]
    atlas["UPCA3"] = XYZ_pca_unsup_ball[:, 2]
    atlas["UPCA_R"] = radius_pca_unsup
    atlas["DPCA1"] = XYZ_pca_disease_ball[:, 0]
    atlas["DPCA2"] = XYZ_pca_disease_ball[:, 1]
    atlas["DPCA3"] = XYZ_pca_disease_ball[:, 2]
    atlas["DPCA_R"] = radius_pca_disease
    atlas["UUMAP1"] = XYZ_umap_unsup_ball[:, 0]
    atlas["UUMAP2"] = XYZ_umap_unsup_ball[:, 1]
    atlas["UUMAP3"] = XYZ_umap_unsup_ball[:, 2]
    atlas["UUMAP_R"] = radius_umap_unsup
    atlas["DUMAP1"] = XYZ_umap_disease_ball[:, 0]
    atlas["DUMAP2"] = XYZ_umap_disease_ball[:, 1]
    atlas["DUMAP3"] = XYZ_umap_disease_ball[:, 2]
    atlas["DUMAP_R"] = radius_umap_disease
    atlas["OOF_IBD"] = oof_ibd
    atlas["OOF_CD"] = oof_subtype_filled
    atlas["Group"] = group

    if extra_cols:
        for col in extra_cols:
            resolved = _match_col(host, col)
            if resolved == group_col:
                continue
            atlas[resolved] = host[resolved].to_numpy()

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        atlas.to_csv(out)
        print(f"Saved {atlas.shape[0]} samples to {out}")

    return atlas


# ---------------------------------------------------------------------------
# Stage 2 — spherical disease-aware coordinates
# ---------------------------------------------------------------------------

def add_spherical_coordinates(
    atlas: pd.DataFrame,
    *,
    lat_max_deg: float = 72,
    min_radius: float = 0.35,
    max_radius: float = 1.00,
    umap3_radial_weight: float = 0.40,
    ibd_radial_weight: float = 0.45,
    subtype_radial_weight: float = 0.15,
    low_q: float = 0.02,
    high_q: float = 0.98,
    output_path: Optional[PathLike] = None,
) -> pd.DataFrame:
    """Map disease-aware UMAP axes onto spherical coordinates.

    Angular position uses DUMAP1/DUMAP2 only. Radius uses DUMAP3 plus OOF
    disease scores. Samples are not forced onto the sphere surface.
    """
    atlas = atlas.copy()
    required = ["DUMAP1", "DUMAP2", "DUMAP3", "OOF_IBD", "OOF_CD"]
    missing = [c for c in required if c not in atlas.columns]
    if missing:
        raise ValueError(f"Atlas is missing columns required for the sphere: {missing}")

    u_lon = _rank_uniform(atlas["DUMAP1"].values)
    u_lat = _rank_uniform(atlas["DUMAP2"].values)

    longitude = -np.pi + 2 * np.pi * u_lon
    lat_max = np.deg2rad(lat_max_deg)
    sin_lat_max = np.sin(lat_max)
    sin_lat = -sin_lat_max + 2 * sin_lat_max * u_lat
    latitude = np.arcsin(sin_lat)

    umap3_01 = _robust_01(atlas["DUMAP3"].values, low_q, high_q)
    ibd_01 = _robust_01(atlas["OOF_IBD"].values, low_q, high_q)

    subtype_raw = atlas["OOF_CD"].to_numpy(dtype=float, copy=True)
    subtype_strength = np.abs(subtype_raw - 0.5) * 2
    subtype_01 = _robust_01(subtype_strength, low_q, high_q)

    radial_score = (
        umap3_radial_weight * umap3_01
        + ibd_radial_weight * ibd_01
        + subtype_radial_weight * subtype_01
    )
    radial_score = _robust_01(radial_score, 0.01, 0.99)
    radius = min_radius + radial_score * (max_radius - min_radius)

    atlas["SphereLongitude"] = longitude
    atlas["SphereLatitude"] = latitude
    atlas["SphereRadius"] = radius
    atlas["SphereX"] = radius * np.cos(latitude) * np.cos(longitude)
    atlas["SphereY"] = radius * np.cos(latitude) * np.sin(longitude)
    atlas["SphereZ"] = radius * np.sin(latitude)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        atlas.to_csv(out)
        print(f"Saved {atlas.shape[0]} samples to {out}")

    return atlas


# ---------------------------------------------------------------------------
# Stage 3 — depth-aware Plotly figure
# ---------------------------------------------------------------------------

def plot_spherical_amg_atlas(
    atlas: pd.DataFrame,
    *,
    color_col: str = "Group",
    host_profile: Optional[pd.DataFrame] = None,
    color_map: Optional[Mapping[str, str]] = None,
    n_depth_layers: int = 8,
    camera_eye: Sequence[float] = (-18, 3, 8),
    show_sphere: bool = True,
    sphere_opacity: float = 0.058,
    show_centroids: bool = True,
    centroid_size: float = 6,
    fig_width: int = 900,
    fig_height: int = 820,
    title: Optional[str] = None,
    html_path: Optional[PathLike] = None,
    show: bool = False,
) -> go.Figure:
    """Render the spherical atlas with perspective depth cues.

    *color_col* only controls marker colours / legend / centroids. Disease-aware
    coordinates themselves are not recomputed.
    """
    atlas = atlas.copy()
    if host_profile is not None and color_col not in atlas.columns:
        host = _ensure_indexed(host_profile, id_candidates=("sample",))
        host.index = host.index.astype(str)
        resolved = _match_col(host, color_col)
        atlas[resolved] = host.reindex(atlas.index.astype(str))[resolved].to_numpy()
        color_col = resolved

    try:
        color_col = _match_col(atlas, color_col)
    except KeyError as exc:
        raise KeyError(
            f"Colour column '{color_col}' is not in the atlas. Pass host_profile "
            "or include it via extra_cols= when computing coordinates."
        ) from exc

    required_cols = ["SphereX", "SphereY", "SphereZ", "SphereRadius", "OOF_IBD", "OOF_CD"]
    missing_cols = [c for c in required_cols if c not in atlas.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    camera_eye = np.asarray(camera_eye, dtype=float)
    xyz = atlas[["SphereX", "SphereY", "SphereZ"]].to_numpy(dtype=float)
    camera_distance = np.linalg.norm(xyz - camera_eye[None, :], axis=1)
    atlas["CameraDistance"] = camera_distance

    d_low = np.percentile(camera_distance, 1)
    d_high = np.percentile(camera_distance, 99)
    depth01 = (np.clip(camera_distance, d_low, d_high) - d_low) / (d_high - d_low + 1e-12)
    atlas["Depth"] = depth01

    depth_layer = np.floor(depth01 * n_depth_layers).astype(int)
    depth_layer = np.clip(depth_layer, 0, n_depth_layers - 1)
    atlas["DepthLayer"] = depth_layer

    labels = atlas[color_col].astype("string")
    group_order = _category_order(labels.dropna())
    colors = _colors_for(group_order, color_map)

    legend_title = "Group" if str(color_col).lower() == "group" else str(color_col)
    if title is None:
        title = "<b>Spherical Disease-aware AMG Functional Atlas</b>"

    fig = go.Figure()
    legend_shown = set()

    for layer in reversed(range(n_depth_layers)):
        layer_depth = layer / (n_depth_layers - 1) if n_depth_layers > 1 else 0.0
        visual_depth = layer_depth ** 1.25

        for group in group_order:
            sub = atlas.loc[(labels == group) & (atlas["DepthLayer"] == layer)].copy()
            if len(sub) == 0:
                continue

            marker_size = (
                _style_lookup(POINT_SIZE_NEAR, group, _FALLBACK_SIZE_NEAR)
                - visual_depth
                * (
                    _style_lookup(POINT_SIZE_NEAR, group, _FALLBACK_SIZE_NEAR)
                    - _style_lookup(POINT_SIZE_FAR, group, _FALLBACK_SIZE_FAR)
                )
            )
            marker_opacity = (
                _style_lookup(POINT_OPACITY_NEAR, group, _FALLBACK_OPACITY_NEAR)
                - visual_depth
                * (
                    _style_lookup(POINT_OPACITY_NEAR, group, _FALLBACK_OPACITY_NEAR)
                    - _style_lookup(POINT_OPACITY_FAR, group, _FALLBACK_OPACITY_FAR)
                )
            )
            white_mix = visual_depth * _style_lookup(FAR_WHITE_MIX, group, _FALLBACK_WHITE_MIX)
            marker_color = _mix_with_white(colors[group], white_mix)

            custom_data = np.column_stack(
                [
                    sub["OOF_IBD"].to_numpy(),
                    sub["OOF_CD"].to_numpy(),
                    sub["SphereRadius"].to_numpy(),
                    sub["Depth"].to_numpy(),
                ]
            )

            # Reproduce the notebook: legend on the nearest layer that has points.
            show_legend = group not in legend_shown and layer == 0
            if show_legend:
                legend_shown.add(group)

            fig.add_trace(
                go.Scatter3d(
                    x=sub["SphereX"],
                    y=sub["SphereY"],
                    z=sub["SphereZ"],
                    mode="markers",
                    name=group,
                    legendgroup=group,
                    showlegend=show_legend,
                    text=sub.index.astype(str),
                    customdata=custom_data,
                    marker=dict(
                        size=marker_size,
                        color=marker_color,
                        opacity=marker_opacity,
                        line=dict(width=0),
                    ),
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        f"{legend_title}: {group}<br>"
                        "IBD probability: %{customdata[0]:.3f}<br>"
                        "CD probability: %{customdata[1]:.3f}<br>"
                        "Radial position: %{customdata[2]:.3f}"
                        "<extra></extra>"
                    ),
                )
            )

    if show_sphere:
        u = np.linspace(0, 2 * np.pi, 80)
        v = np.linspace(0, np.pi, 55)
        shell_x = np.outer(np.cos(u), np.sin(v))
        shell_y = np.outer(np.sin(u), np.sin(v))
        shell_z = np.outer(np.ones_like(u), np.cos(v))
        fig.add_trace(
            go.Surface(
                x=shell_x,
                y=shell_y,
                z=shell_z,
                surfacecolor=np.ones_like(shell_x),
                colorscale=[[0, "#DCE3EA"], [1, "#DCE3EA"]],
                opacity=sphere_opacity,
                showscale=False,
                hoverinfo="skip",
                showlegend=False,
                name="Atlas boundary",
            )
        )

    if show_centroids:
        for group in group_order:
            sub = atlas.loc[labels == group]
            if sub.empty:
                continue
            fig.add_trace(
                go.Scatter3d(
                    x=[sub["SphereX"].mean()],
                    y=[sub["SphereY"].mean()],
                    z=[sub["SphereZ"].mean()],
                    mode="markers",
                    name=f"{group} centroid",
                    showlegend=False,
                    marker=dict(
                        size=centroid_size,
                        color=colors[group],
                        symbol="diamond",
                        opacity=1,
                        line=dict(color="white", width=1.8),
                    ),
                    hovertemplate=f"<b>{group} centroid</b><extra></extra>",
                )
            )

    hidden_axis = dict(
        visible=False,
        showbackground=False,
        showgrid=False,
        zeroline=False,
        showline=False,
        showticklabels=False,
        ticks="",
        title="",
    )

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            y=0.97,
            font=dict(size=21, color="#222222"),
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        width=fig_width,
        height=fig_height,
        scene=dict(
            xaxis=hidden_axis,
            yaxis=hidden_axis,
            zaxis=hidden_axis,
            bgcolor="rgba(0,0,0,0)",
            aspectmode="cube",
            camera=dict(
                eye=dict(
                    x=float(camera_eye[0]),
                    y=float(camera_eye[1]),
                    z=float(camera_eye[2]),
                ),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1),
                projection=dict(type="perspective"),
            ),
        ),
        legend=dict(
            title=dict(text=legend_title, font=dict(size=14)),
            x=0.86,
            y=0.94,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0)",
            borderwidth=0,
            font=dict(size=13),
            itemsizing="constant",
        ),
        margin=dict(l=0, r=0, b=0, t=60),
    )

    config = {
        "displaylogo": False,
        "responsive": True,
        "scrollZoom": True,
        "modeBarButtonsToRemove": ["select2d", "lasso2d"],
        "toImageButtonOptions": {
            "format": "png",
            "filename": "spherical_AMG_atlas_depth",
            "height": 1600,
            "width": 1800,
            "scale": 2,
        },
    }

    if show:
        fig.show(config=config)

    if html_path is not None:
        out = Path(html_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out), config=config, include_plotlyjs=True, full_html=True)
        print(f"\nSaved:\n{out}")

    return fig


# ---------------------------------------------------------------------------
# One-shot entry matching the notebook outputs
# ---------------------------------------------------------------------------

def visualize_amg_atlas(
    amg_abd: Optional[pd.DataFrame] = None,
    host_profile: Optional[pd.DataFrame] = None,
    *,
    color_col: str = "Group",
    seed: int = 42,
    output_dir: Optional[PathLike] = "amg_global_functional_atlas",
    group_col: Optional[str] = None,
    show: bool = False,
    save_csv: bool = True,
    save_html: bool = True,
    color_map: Optional[Mapping[str, str]] = None,
    min_prevalence: float = 0.02,
    n_global_pc: int = 20,
    n_module_pc: int = 3,
    n_splits: int = 5,
    oof_ibd_weight: float = 2.0,
    oof_subtype_weight: float = 1.5,
    umap_n_neighbors: int = 50,
    umap_min_dist: float = 0.15,
    keep_groups: Sequence[str] = DEFAULT_KEEP_GROUPS,
    lat_max_deg: float = 72,
    min_radius: float = 0.35,
    max_radius: float = 1.00,
    umap3_radial_weight: float = 0.40,
    ibd_radial_weight: float = 0.45,
    subtype_radial_weight: float = 0.15,
    n_depth_layers: int = 8,
    camera_eye: Sequence[float] = (-18, 3, 8),
    show_sphere: bool = True,
    show_centroids: bool = True,
) -> AMGAtlasResult:
    """End-to-end spherical disease-aware AMG atlas (notebook-equivalent).

    Standard IMPACT-IBD tables can be loaded with :func:`impact_ibd.load_amg_abd`
    and :func:`impact_ibd.load_host_profile`. After ``set_index`` on the first
    column they match the layouts expected here. When *amg_abd* / *host_profile*
    are omitted, those bundled tables are loaded automatically.

    Output files (when *output_dir* is set and saving is enabled)::

        {output_dir}/global_functional_AMG_atlas_coordinates.csv
        {output_dir}/spherical_diseaseaware/spherical_diseaseaware_AMG_coordinates.csv
        {output_dir}/spherical_diseaseaware/final_depth_plot/{Color}_color_AMG_atlas_depth.html
    """
    if amg_abd is None or host_profile is None:
        from . import load_amg_abd, load_host_profile

        if amg_abd is None:
            amg_abd = load_amg_abd()
        if host_profile is None:
            host_profile = load_host_profile()

    extra_cols = None if color_col is None else [color_col]

    coord_path = None
    sphere_path = None
    html_path = None
    paths: Dict[str, Path] = {}

    if output_dir is not None:
        out_root = Path(output_dir)
        sphere_dir = out_root / "spherical_diseaseaware"
        plot_dir = sphere_dir / "final_depth_plot"
        if save_csv:
            coord_path = out_root / "global_functional_AMG_atlas_coordinates.csv"
            sphere_path = sphere_dir / "spherical_diseaseaware_AMG_coordinates.csv"
        if save_html:
            color_label = "Group" if str(color_col).lower() == "group" else str(color_col)
            html_path = plot_dir / f"{color_label}_color_AMG_atlas_depth.html"

    atlas = compute_amg_atlas_coordinates(
        amg_abd,
        host_profile,
        group_col=group_col,
        extra_cols=extra_cols,
        seed=seed,
        min_prevalence=min_prevalence,
        n_global_pc=n_global_pc,
        n_module_pc=n_module_pc,
        n_splits=n_splits,
        oof_ibd_weight=oof_ibd_weight,
        oof_subtype_weight=oof_subtype_weight,
        umap_n_neighbors=umap_n_neighbors,
        umap_min_dist=umap_min_dist,
        keep_groups=keep_groups,
        output_path=coord_path,
    )
    atlas = add_spherical_coordinates(
        atlas,
        lat_max_deg=lat_max_deg,
        min_radius=min_radius,
        max_radius=max_radius,
        umap3_radial_weight=umap3_radial_weight,
        ibd_radial_weight=ibd_radial_weight,
        subtype_radial_weight=subtype_radial_weight,
        output_path=sphere_path,
    )
    figure = plot_spherical_amg_atlas(
        atlas,
        color_col=color_col,
        host_profile=host_profile,
        color_map=color_map,
        n_depth_layers=n_depth_layers,
        camera_eye=camera_eye,
        show_sphere=show_sphere,
        show_centroids=show_centroids,
        html_path=html_path,
        show=show,
    )

    if coord_path is not None:
        paths["coordinates"] = Path(coord_path)
    if sphere_path is not None:
        paths["spherical"] = Path(sphere_path)
    if html_path is not None:
        paths["html"] = Path(html_path)

    return AMGAtlasResult(atlas=atlas, figure=figure, paths=paths)
