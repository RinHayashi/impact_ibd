"""Data explorer page."""

from __future__ import annotations

import streamlit as st

from impact_ibd.streamlit_app.components import atlas as atlas_ui
from impact_ibd.streamlit_app.components import charts
from impact_ibd.streamlit_app.components.layout import (
    card,
    card_title,
    metric_row,
    page_header,
    render_sidebar,
    section_gap,
)
from impact_ibd.streamlit_app.components.tables import render_paged_table
from impact_ibd.streamlit_app.constants import HOST_ATLAS_COLOR_FIELDS, PLOTLY_CONFIG, PROFILE_OPTIONS
from impact_ibd.streamlit_app.services import data_service
from impact_ibd.streamlit_app.state import SS

PROFILE_TABLE_WIDTH = 720
PROFILE_CHART_WIDTH = 330
PROFILE_CARD_HEIGHT = 620


def _on_profile_change() -> None:
    st.session_state[SS.data_page_number] = 1


def render() -> None:
    render_sidebar()
    page_header(
        "Data explorer",
        "Browse the bundled profile tables and descriptive charts. Abundance matrices are not shown on this page.",
        eyebrow="Atlas reference data",
    )

    with card("card_data_profile"):
        st.selectbox(
            "Select profile",
            options=list(PROFILE_OPTIONS),
            key=SS.data_profile_name,
            on_change=_on_profile_change,
        )
    profile_name = st.session_state[SS.data_profile_name]
    df = data_service.load_profile(profile_name)

    section_gap()
    if profile_name == "Host profile":
        _render_host(df)
    elif profile_name == "AMG profile":
        _render_amg(df)
    elif profile_name == "vOTU profile":
        _render_votu(df)
    else:
        _render_prok(df)


def _chart(fig, key: str) -> None:
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, key=key)


def _top_bar(table, column: str, key: str) -> None:
    labels = table[column].astype(str)
    other_mask = labels.str.lower().str.startswith(("other (", "others ("))
    other_count = int(table.loc[other_mask, "count"].sum())
    visible = table.loc[~other_mask]
    _chart(charts.bar_counts(visible, column), key)
    if other_count:
        st.caption(f"Others (not shown in the chart): {other_count:,} records.")


def _render_host(df) -> None:
    overview = data_service.host_overview()
    metric_row(
        [
            ("Samples", f"{overview['n_samples']:,}"),
            ("Cohorts", overview["n_cohorts"]),
            ("Regions", overview["n_regions"]),
            ("Groups", overview["n_groups"]),
        ],
        key_prefix="host_metrics",
    )
    with card("card_host_atlas"):
        st.selectbox(
            "Atlas color field",
            options=list(HOST_ATLAS_COLOR_FIELDS),
            key=SS.host_atlas_color_col,
        )
        atlas_ui.render_host_atlas(
            st.session_state[SS.host_atlas_color_col],
            key=f"host_atlas_{st.session_state[SS.host_atlas_color_col]}",
        )
        st.caption("Missing values are shown as NA (gray). Changing the color field does not recompute UMAP coordinates.")
    section_gap()
    with st.container(horizontal=True, gap="medium"):
        render_paged_table(
            df,
            title="Host profile table",
            card_key="card_host_table",
            width=PROFILE_TABLE_WIDTH,
            height=PROFILE_CARD_HEIGHT,
        )
        with card("card_host_statistics", width=PROFILE_CHART_WIDTH, height=PROFILE_CARD_HEIGHT):
            field = st.selectbox(
                "Field statistics",
                options=list(HOST_ATLAS_COLOR_FIELDS) + ["age", "BMI"],
                key="host_statistics_field",
            )
            card_title(f"{field} distribution")
            if field in ("age", "BMI"):
                _chart(charts.histogram(df[field], None, field), f"host_{field}_hist")
            else:
                counts = data_service.value_counts_table("Host profile", field)
                _chart(charts.bar_counts(counts, field, yaxis="Samples"), f"host_{field}_bar")


def _render_amg(df) -> None:
    metrics = data_service.amg_profile_metrics()
    metric_row(
        [
            ("Rows", f"{metrics['n_rows']:,}"),
            ("Unique ORFs", f"{metrics['n_orfs']:,}"),
            ("Unique AMG functions", f"{metrics['n_functions']:,}"),
            ("Linked vOTUs", f"{metrics['n_votu']:,}"),
        ],
        key_prefix="amg_metrics",
    )
    st.caption("The charts below count annotation records, not abundance.")
    section_gap()
    with st.container(horizontal=True, gap="medium"):
        render_paged_table(
            df,
            title="AMG profile table",
            card_key="card_amg_table",
            width=PROFILE_TABLE_WIDTH,
            height=PROFILE_CARD_HEIGHT,
        )
        with card("card_amg_statistics", width=PROFILE_CHART_WIDTH, height=PROFILE_CARD_HEIGHT):
            view = st.selectbox(
                "Field statistics",
                options=[
                    "Top AMG functions",
                    "Annotation source composition",
                    "Auxiliary score distribution",
                    "AMG flag distribution",
                ],
                key="amg_statistics_field",
            )
            card_title(view)
            if view == "Top AMG functions":
                top_fn = data_service.top_n_with_other("AMG profile", "AMG_function", n=20)
                _top_bar(top_fn, "AMG_function", "amg_top_functions")
            elif view == "Annotation source composition":
                sources = data_service.value_counts_table("AMG profile", "anno_source")
                _chart(charts.bar_counts(sources, "anno_source"), "amg_anno_source")
            elif view == "Auxiliary score distribution":
                scores = data_service.value_counts_table("AMG profile", "auxiliary_score").sort_values(
                    "auxiliary_score"
                )
                _chart(charts.bar_counts(scores, "auxiliary_score"), "amg_aux_bar")
            else:
                flags = data_service.value_counts_table("AMG profile", "AMG_flag")
                _chart(charts.bar_counts(flags, "AMG_flag"), "amg_flag_bar")


def _render_votu(df) -> None:
    metrics = data_service.votu_profile_metrics()
    metric_row(
        [
            ("vOTUs", f"{metrics['n_votu']:,}"),
            ("Unique predicted hosts", f"{metrics['n_hosts']:,}"),
            ("Lifestyle categories", metrics["n_lifestyle"]),
            ("Host annotation rate", f"{metrics['host_annotated_frac'] * 100:.1f}%"),
        ],
        key_prefix="votu_metrics",
    )
    section_gap()
    with st.container(horizontal=True, gap="medium"):
        render_paged_table(
            df,
            title="vOTU profile table",
            card_key="card_votu_table",
            width=PROFILE_TABLE_WIDTH,
            height=PROFILE_CARD_HEIGHT,
        )
        with card("card_votu_statistics", width=PROFILE_CHART_WIDTH, height=PROFILE_CARD_HEIGHT):
            view = st.selectbox(
                "Field statistics",
                options=[
                    "Top predicted hosts",
                    "Lifestyle composition",
                    "AMG gene count distribution",
                    "AMG density distribution",
                ],
                key="votu_statistics_field",
            )
            card_title(view)
            if view == "Top predicted hosts":
                hosts = data_service.top_n_with_other("vOTU profile", "host", n=15)
                _top_bar(hosts, "host", "votu_hosts")
            elif view == "Lifestyle composition":
                life = data_service.value_counts_table("vOTU profile", "lifestyle")
                _chart(charts.bar_counts(life, "lifestyle"), "votu_lifestyle")
            elif view == "AMG gene count distribution":
                _chart(charts.histogram(df["AMG_gene_count"], None, "AMG_gene_count"), "votu_amg_count")
            else:
                _chart(charts.histogram(df["AMG_density"], None, "AMG_density"), "votu_amg_density")


def _render_prok(df) -> None:
    metrics = data_service.prok_profile_metrics()
    metric_row(
        [
            ("taxid count", metrics["n_taxid"]),
            ("phylum count", metrics["n_phylum"]),
            ("genus count", metrics["n_genus"]),
            ("species count", metrics["n_species"]),
        ],
        key_prefix="prok_metrics",
    )
    section_gap()
    render_paged_table(df, title="Prokaryote profile table", card_key="card_prok_table")
    section_gap()
    with card("card_prok_statistics"):
        view = st.selectbox(
            "Field statistics",
            options=["Taxonomy treemap", "Top phylum records", "Top genus records"],
            key="prok_statistics_field",
        )
        card_title(view)
        if view == "Taxonomy treemap":
            tree = data_service.prok_taxonomy_levels()
            st.caption("phylum → class → order; remaining taxa are grouped as Other.")
            _chart(charts.treemap(tree), "prok_treemap")
        elif view == "Top phylum records":
            phylum = data_service.top_n_with_other("Prokaryote profile", "phylum", n=15)
            _top_bar(phylum, "phylum", "prok_phylum")
        else:
            genus = data_service.top_n_with_other("Prokaryote profile", "genus", n=15)
            _top_bar(genus, "genus", "prok_genus")
