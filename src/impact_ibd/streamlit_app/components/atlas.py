"""Atlas HTML embed and Plotly recoding."""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

from impact_ibd.streamlit_app.constants import ATLAS_HTML_HEIGHT, PLOTLY_CONFIG
from impact_ibd.streamlit_app.services import atlas_service


def render_pregenerated_atlas() -> None:
    try:
        html = atlas_service.read_pregenerated_atlas_html()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return
    components.html(html, height=ATLAS_HTML_HEIGHT, scrolling=False)


def render_host_atlas(color_col: str, key: str) -> None:
    with st.spinner("Loading reference atlas coordinates (UMAP is computed on the first visit)…"):
        atlas = atlas_service.get_reference_atlas()
    fig = atlas_service.plot_colored_atlas(atlas, color_col=color_col)
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, key=key)
