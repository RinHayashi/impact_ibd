"""Shared layout helpers."""

from __future__ import annotations

from typing import Optional

import streamlit as st

from impact_ibd.streamlit_app.constants import LOGO_PATH
from impact_ibd.streamlit_app.state import MODEL_RESULT_KEYS, SS

_TILE_ACCENTS = ("blue", "mint", "lavender", "peach")


def brand_expansion(class_name: str) -> None:
    st.markdown(
        f"""
        <p class="{class_name}">
          <span class="imibd-brand-accent">I</span>ntegrated
          <span class="imibd-brand-accent">M</span>icrobiome-<span class="imibd-brand-accent">P</span>hage-<span class="imibd-brand-accent">A</span>uxiliary
          metabolic gene <span class="imibd-brand-accent">C</span>ar<span class="imibd-brand-accent">T</span>ography
          for <span class="imibd-brand-accent">IBD</span>
        </p>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    with st.sidebar:
        with st.container(key="sidebar_brand"):
            if LOGO_PATH.is_file():
                st.image(str(LOGO_PATH), width=150)
            brand_expansion("imibd-brand-tag")


def page_header(title: str, caption: Optional[str] = None, *, eyebrow: Optional[str] = None) -> None:
    if eyebrow:
        st.markdown(f'<p class="imibd-eyebrow">{eyebrow}</p>', unsafe_allow_html=True)
    st.title(title)
    if caption:
        st.markdown(f'<p class="imibd-page-caption">{caption}</p>', unsafe_allow_html=True)


def section_title(text: str) -> None:
    st.markdown(f'<h2 class="imibd-section-title">{text}</h2>', unsafe_allow_html=True)


def card_title(text: str, caption: Optional[str] = None) -> None:
    st.markdown(f'<h3 class="imibd-card-title">{text}</h3>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<p class="imibd-card-copy">{caption}</p>', unsafe_allow_html=True)


def card(key: str, *, width="stretch", height="content"):
    return st.container(border=True, key=key, width=width, height=height)


def section_gap() -> None:
    st.markdown('<div class="imibd-section-gap"></div>', unsafe_allow_html=True)


def metric_tile(label: str, value, key: str, accent: str = "blue") -> None:
    with st.container(border=False, key=f"tile_{accent}_{key}"):
        st.metric(label, value)


def metric_row(items, *, key_prefix: str) -> None:
    cols = st.columns(len(items))
    for i, (col, (label, value)) in enumerate(zip(cols, items)):
        with col:
            metric_tile(label, value, key=f"{key_prefix}_{i}", accent=_TILE_ACCENTS[i % len(_TILE_ACCENTS)])


def show_errors(messages) -> None:
    for msg in messages:
        st.error(msg)


def show_warnings(messages) -> None:
    for msg in messages:
        st.warning(msg)


def clear_model_results() -> None:
    for key in MODEL_RESULT_KEYS:
        st.session_state.pop(key, None)


def init_defaults() -> None:
    st.session_state.setdefault(SS.data_profile_name, "Host profile")
    st.session_state.setdefault(SS.data_page_size, 50)
    st.session_state.setdefault(SS.data_page_number, 1)
    st.session_state.setdefault(SS.host_atlas_color_col, "group")
    st.session_state.setdefault(SS.model_task, "ibd_control")
    st.session_state.setdefault(SS.model_apply_tss, False)
