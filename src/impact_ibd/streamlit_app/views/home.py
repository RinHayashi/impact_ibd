"""Home page."""

from __future__ import annotations

import streamlit as st

from impact_ibd.streamlit_app import routes
from impact_ibd.streamlit_app.components.layout import (
    brand_expansion,
    render_sidebar,
    section_gap,
    section_title,
)
from impact_ibd.streamlit_app.constants import GITHUB_REPOSITORY_URL, LOGO_PATH
from impact_ibd.streamlit_app.services import data_service
from impact_ibd.streamlit_app.state import SS


def _open_profile(profile_name: str) -> None:
    st.session_state[SS.data_profile_name] = profile_name
    st.session_state[SS.data_page_number] = 1
    st.switch_page(routes.DATA)


def _bubble(label: str, value: int, key: str, profile_name: str) -> None:
    if st.button(
        f"**{value:,}**  \n{label}",
        key=f"home_bubble_{key}",
        help=f"Open {profile_name} in Data explorer",
        use_container_width=True,
    ):
        _open_profile(profile_name)


def _render_feature_cards() -> None:
    section_title("✨What can we do✨")
    st.markdown(
        """
        <div class="imibd-feature-list">
          <article class="imibd-feature-card imibd-feature-data">
            <h3>Reference tables</h3>
            <p>Browse curated host, AMG, vOTU, and prokaryote profiles.</p>
          </article>
          <article class="imibd-feature-card imibd-feature-atlas">
            <h3>Atlas views</h3>
            <p>Explore sample patterns and recolor the atlas by host metadata.</p>
          </article>
          <article class="imibd-feature-card imibd-feature-model">
            <h3>Predictive models</h3>
            <p>Classify IBD vs Control or CD vs UC with pretrained models.</p>
          </article>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render() -> None:
    render_sidebar()
    scale = data_service.home_scale_metrics()

    hero_left, hero_right = st.columns([1.35, 0.65], gap="large")
    with hero_left:
        with st.container(key="home_hero_copy"):
            if LOGO_PATH.is_file():
                st.image(str(LOGO_PATH), width=112)
            st.markdown('<p class="imibd-eyebrow">Explore the atlas</p>', unsafe_allow_html=True)
            st.markdown(
                '<h1 class="imibd-home-title">IMPACT<span class="imibd-title-accent">-IBD</span></h1>',
                unsafe_allow_html=True,
            )
            brand_expansion("imibd-home-caption")
            actions = st.columns(2, gap="small")
            with actions[0]:
                if st.button("Explore data", key="home_to_data", type="primary", use_container_width=True):
                    st.switch_page(routes.DATA)
            with actions[1]:
                if st.button("Run prediction", key="home_to_model", use_container_width=True):
                    st.switch_page(routes.MODEL)

        section_gap()
        _render_feature_cards()

    with hero_right:
        with st.container(key="home_bubble_stage"):
            st.markdown(
                '<p class="imibd-bubble-kicker">Explore atlas data</p>',
                unsafe_allow_html=True,
            )
            _bubble("AMG ORFs", scale["amg_orfs"], "amg_orfs", "AMG profile")
            _bubble("AMG Functions", scale["amg_functions"], "amg_functions", "AMG profile")
            _bubble("vOTU (with AMGs)", scale["votu"], "votu", "vOTU profile")
            _bubble("Samples", scale["samples"], "samples", "Host profile")

    section_gap()
    st.markdown(
        f"""
        <div class="imibd-dependency">
          <span>This application is powered by the <strong>impact-ibd</strong> Python package.</span>
          <a href="{GITHUB_REPOSITORY_URL}" target="_blank" rel="noopener noreferrer" aria-label="Open the impact-ibd GitHub repository">
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M12 .7a11.5 11.5 0 0 0-3.64 22.41c.58.1.79-.25.79-.56v-2.23c-3.22.7-3.9-1.37-3.9-1.37-.52-1.34-1.28-1.69-1.28-1.69-1.05-.72.08-.7.08-.7 1.16.08 1.77 1.19 1.77 1.19 1.03 1.77 2.7 1.26 3.36.96.1-.75.4-1.26.73-1.55-2.57-.29-5.27-1.28-5.27-5.69 0-1.26.45-2.28 1.19-3.09-.12-.29-.52-1.46.11-3.05 0 0 .97-.31 3.16 1.18a10.95 10.95 0 0 1 5.76 0c2.19-1.49 3.16-1.18 3.16-1.18.63 1.59.23 2.76.11 3.05.74.81 1.19 1.83 1.19 3.09 0 4.42-2.71 5.39-5.29 5.68.42.36.79 1.07.79 2.16v3.2c0 .31.21.67.8.56A11.5 11.5 0 0 0 12 .7Z"/>
            </svg>
            <span>GitHub</span>
          </a>
        </div>
        """,
        unsafe_allow_html=True,
    )
