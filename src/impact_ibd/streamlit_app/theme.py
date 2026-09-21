"""Load the shared visual stylesheet once per run."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

_CSS_PATH = Path(__file__).resolve().parent / "assets" / "styles.css"


def apply_theme() -> None:
    css = _CSS_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
