"""Explicitly paginated profile tables."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from impact_ibd.streamlit_app.components.layout import card, card_title
from impact_ibd.streamlit_app.constants import PAGE_SIZES
from impact_ibd.streamlit_app.services.pagination import paginate
from impact_ibd.streamlit_app.state import SS


SEARCH_MODES = ("Exact value", "Contains")


def filter_exact_rows(df: pd.DataFrame, column: str, value: str) -> pd.DataFrame:
    """Return rows whose selected-column value exactly matches the search text."""
    query = value.strip()
    if not query:
        return df
    return df.loc[df[column].astype("string").eq(query).fillna(False)]


def filter_contains_rows(df: pd.DataFrame, column: str, value: str) -> pd.DataFrame:
    """Return rows whose selected-column text contains the keyword (case-insensitive)."""
    query = value.strip()
    if not query:
        return df
    return df.loc[df[column].astype("string").str.contains(query, case=False, na=False, regex=False)]


def filter_search_rows(df: pd.DataFrame, column: str, value: str, mode: str) -> pd.DataFrame:
    if mode == "Contains":
        return filter_contains_rows(df, column, value)
    return filter_exact_rows(df, column, value)


def _reset_page() -> None:
    st.session_state[SS.data_page_number] = 1


def render_paged_table(
    df,
    *,
    title: str,
    card_key: str = "card_profile_table",
    width="stretch",
    height="content",
) -> None:
    with card(card_key, width=width, height=height):
        card_title(title)
        search_controls = st.columns([2, 2, 3])
        with search_controls[0]:
            search_column = st.selectbox(
                "Search column",
                options=list(df.columns),
                key=f"{card_key}_search_column",
                on_change=_reset_page,
            )
        with search_controls[1]:
            search_mode = st.selectbox(
                "Match mode",
                options=list(SEARCH_MODES),
                key=f"{card_key}_search_mode",
                on_change=_reset_page,
            )
        with search_controls[2]:
            search_value = st.text_input(
                "Keyword",
                key=f"{card_key}_search_value",
                placeholder="Enter a keyword to filter rows",
                on_change=_reset_page,
            )

        filtered_df = filter_search_rows(df, search_column, search_value, search_mode)
        controls = st.columns([1, 1])
        with controls[0]:
            page_size = st.selectbox(
                "Rows per page",
                options=list(PAGE_SIZES),
                key=SS.data_page_size,
            )
        preview = paginate(filtered_df, st.session_state.get(SS.data_page_number, 1), int(page_size))
        if preview.page_number != st.session_state.get(SS.data_page_number):
            st.session_state[SS.data_page_number] = preview.page_number

        with controls[1]:
            st.number_input(
                "Page",
                min_value=1,
                max_value=int(preview.total_pages),
                step=1,
                key=SS.data_page_number,
            )
        preview = paginate(filtered_df, int(st.session_state[SS.data_page_number]), int(page_size))
        start_display = preview.start + 1 if preview.total_rows else 0
        row_summary = (
            f"{preview.total_rows} of {len(df)} matching rows"
            if search_value.strip()
            else f"{preview.total_rows} rows"
        )
        st.caption(
            f"{row_summary} × {df.shape[1]} columns; "
            f"showing {start_display}–{preview.end}, "
            f"page {preview.page_number} of {preview.total_pages}."
        )
        st.dataframe(preview.page_df, width="stretch", hide_index=True, height=350)
