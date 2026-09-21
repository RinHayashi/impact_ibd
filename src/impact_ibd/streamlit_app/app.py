"""IMPACT-IBD Streamlit entrypoint and page registry."""

from __future__ import annotations

import streamlit as st

from impact_ibd.streamlit_app import routes
from impact_ibd.streamlit_app.components.layout import init_defaults
from impact_ibd.streamlit_app.constants import PAGE_ICON_PATH
from impact_ibd.streamlit_app.theme import apply_theme
from impact_ibd.streamlit_app.views import data_explorer, home, model_explorer

st.set_page_config(
    page_title="IMPACT-IBD",
    page_icon=str(PAGE_ICON_PATH) if PAGE_ICON_PATH.is_file() else "🦠",
    layout="wide",
)
apply_theme()
init_defaults()

routes.HOME = st.Page(home.render, title="Home", icon="🏠", default=True, url_path="home")
routes.DATA = st.Page(data_explorer.render, title="Data explorer", icon="📊", url_path="data")
routes.MODEL = st.Page(model_explorer.render, title="Model explorer", icon="💻", url_path="model")

pg = st.navigation(
    [routes.HOME, routes.DATA, routes.MODEL],
    position="sidebar",
)
pg.run()
