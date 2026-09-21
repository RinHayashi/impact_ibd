"""Read-only bundled profile loaders and cached descriptive summaries."""

from __future__ import annotations

from typing import Callable, Dict

import pandas as pd

import impact_ibd as imibd

from impact_ibd.streamlit_app.constants import DEFAULT_PROFILE, PROFILE_OPTIONS


def _as_na_string(series: pd.Series) -> pd.Series:
    return series.astype("object").where(series.notna(), "NA").astype(str)


PROFILE_LOADERS: Dict[str, Callable[[], pd.DataFrame]] = {
    "Host profile": imibd.load_host_profile,
    "AMG profile": imibd.load_amg_profile,
    "vOTU profile": imibd.load_votu_profile,
    "Prokaryote profile": imibd.load_prok_profile,
}

assert tuple(PROFILE_LOADERS) == PROFILE_OPTIONS
assert next(iter(PROFILE_LOADERS)) == DEFAULT_PROFILE


def _cache_resource(func):
    try:
        import streamlit as st

        return st.cache_resource(show_spinner=False)(func)
    except Exception:
        return func


def _cache_data(func):
    try:
        import streamlit as st

        return st.cache_data(show_spinner=False)(func)
    except Exception:
        return func


@_cache_resource
def load_profile(name: str) -> pd.DataFrame:
    if name not in PROFILE_LOADERS:
        raise KeyError(f"Unknown profile: {name}")
    return PROFILE_LOADERS[name]()


@_cache_resource
def load_host_profile() -> pd.DataFrame:
    return imibd.load_host_profile()


@_cache_resource
def load_amg_profile() -> pd.DataFrame:
    return imibd.load_amg_profile()


@_cache_resource
def load_votu_profile() -> pd.DataFrame:
    return imibd.load_votu_profile()


@_cache_resource
def load_amg_abd() -> pd.DataFrame:
    return imibd.load_amg_abd()


def copy_profile(name: str) -> pd.DataFrame:
    return load_profile(name).copy()


@_cache_data
def home_scale_metrics() -> dict:
    amg = load_amg_profile()
    votu = load_votu_profile()
    host = load_host_profile()
    return {
        "amg_orfs": int(amg["AMG_ORF"].nunique()) if "AMG_ORF" in amg.columns else len(amg),
        "amg_functions": int(amg["AMG_function"].nunique()) if "AMG_function" in amg.columns else 0,
        "votu": int(len(votu)),
        "samples": int(len(host)),
    }


@_cache_data
def value_counts_table(profile_name: str, column: str) -> pd.DataFrame:
    df = load_profile(profile_name)
    counts = df[column].value_counts(dropna=False)
    out = counts.rename_axis(column).reset_index(name="count")
    out[column] = _as_na_string(out[column])
    total = int(out["count"].sum())
    out["percent"] = out["count"] / total * 100 if total else 0.0
    return out


@_cache_data
def top_n_with_other(profile_name: str, column: str, n: int = 15) -> pd.DataFrame:
    table = value_counts_table(profile_name, column)
    if len(table) <= n:
        return table
    head = table.iloc[:n].copy()
    other_count = int(table.iloc[n:]["count"].sum())
    other = pd.DataFrame(
        {column: [f"Other ({len(table) - n} more)"], "count": [other_count], "percent": [other_count / table["count"].sum() * 100]}
    )
    return pd.concat([head, other], ignore_index=True)


@_cache_data
def host_overview() -> dict:
    host = load_host_profile()
    group = host["group"] if "group" in host.columns else pd.Series(dtype=object)
    return {
        "n_samples": int(len(host)),
        "n_cohorts": int(host["cohort"].nunique(dropna=True)) if "cohort" in host.columns else 0,
        "n_regions": int(host["region"].nunique(dropna=True)) if "region" in host.columns else 0,
        "n_groups": int(group.nunique(dropna=True)),
        "group_counts": group.value_counts(dropna=False).rename_axis("group").reset_index(name="count"),
        "missing_rates": (
            host.isna().mean().sort_values(ascending=False).rename("missing_rate").reset_index().rename(columns={"index": "field"})
        ),
    }


@_cache_data
def host_cohort_group_crosstab() -> pd.DataFrame:
    host = load_host_profile()
    work = host.copy()
    work["cohort"] = work["cohort"].astype("object").where(work["cohort"].notna(), "NA").astype(str)
    work["group"] = _as_na_string(work["group"])
    ct = pd.crosstab(work["cohort"], work["group"])
    ct["total"] = ct.sum(axis=1)
    return ct.sort_values("total", ascending=False)


@_cache_data
def host_numeric_by_group(column: str) -> pd.DataFrame:
    host = load_host_profile()
    work = host[[column, "group"]].copy()
    work = work.dropna(subset=[column])
    work["group"] = _as_na_string(work["group"])
    return work


@_cache_data
def amg_profile_metrics() -> dict:
    df = load_amg_profile()
    return {
        "n_rows": int(len(df)),
        "n_orfs": int(df["AMG_ORF"].nunique()),
        "n_functions": int(df["AMG_function"].nunique()),
        "n_votu": int(df["vOTU(No.)"].nunique()) if "vOTU(No.)" in df.columns else 0,
    }


@_cache_data
def votu_profile_metrics() -> dict:
    df = load_votu_profile()
    has_host = df["host"].notna() if "host" in df.columns else pd.Series(False, index=df.index)
    return {
        "n_votu": int(len(df)),
        "n_hosts": int(df["host"].nunique(dropna=True)) if "host" in df.columns else 0,
        "n_lifestyle": int(df["lifestyle"].nunique(dropna=True)) if "lifestyle" in df.columns else 0,
        "host_annotated_frac": float(has_host.mean()) if len(df) else 0.0,
    }


@_cache_data
def prok_profile_metrics() -> dict:
    df = load_profile("Prokaryote profile")
    return {
        "n_taxid": int(len(df)),
        "n_phylum": int(df["phylum"].nunique(dropna=True)),
        "n_genus": int(df["genus"].nunique(dropna=True)),
        "n_species": int(df["species"].nunique(dropna=True)),
    }


@_cache_data
def prok_taxonomy_levels(max_phylum: int = 12, max_nodes: int = 60) -> pd.DataFrame:
    df = load_profile("Prokaryote profile").copy()
    for col in ("phylum", "class", "order"):
        df[col] = df[col].fillna("NA").astype(str)
    top_phyla = df["phylum"].value_counts().head(max_phylum).index
    df.loc[~df["phylum"].isin(top_phyla), "phylum"] = "Other"
    grouped = (
        df.groupby(["phylum", "class", "order"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    if len(grouped) > max_nodes:
        keep = grouped.iloc[: max_nodes - 1]
        rest = grouped.iloc[max_nodes - 1 :]
        other = pd.DataFrame(
            {
                "phylum": ["Other"],
                "class": ["Other"],
                "order": [f"Other ({len(rest)} more taxa)"],
                "count": [int(rest["count"].sum())],
            }
        )
        grouped = pd.concat([keep, other], ignore_index=True)
    return grouped
