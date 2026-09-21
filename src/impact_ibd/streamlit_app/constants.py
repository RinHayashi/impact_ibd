"""Shared UI constants for the Streamlit frontend."""

from __future__ import annotations

from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = APP_ROOT / "assets"
ATLAS_HTML_PATH = ASSETS_DIR / "atlas" / "Group_color_AMG_atlas_depth.html"
LOGO_PATH = ASSETS_DIR / "logo.svg"
PAGE_ICON_PATH = ASSETS_DIR / "icon.svg"

# Replace this with the public repository URL once it is available.
GITHUB_REPOSITORY_URL = "https://github.com/RinHayashi/impact_ibd"

CANVAS = "#F5F7FB"
SURFACE = "#FFFFFF"
INK = "#24324A"
MUTED = "#68758A"
BORDER = "#E6EAF1"
GRID = "#E9EDF4"
MACARON_BLUE = "#79A8FF"
MACARON_MINT = "#71D6C1"
MACARON_LAVENDER = "#B9A7F7"
MACARON_CORAL = "#FF9B9B"
MACARON_PEACH = "#FFC98B"
MACARON_YELLOW = "#F3D77A"
PLOTLY_FONT = 'Inter, "Avenir Next", Avenir, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'

MACARON_SEQUENCE = (
    MACARON_BLUE,
    MACARON_MINT,
    MACARON_LAVENDER,
    MACARON_CORAL,
    MACARON_PEACH,
    MACARON_YELLOW,
)

# Control / UC / CD / IBD / NA stay consistent across pages.
GROUP_COLORS = {
    "Control": "#98A2B3",
    "UC": "#6C9FE8",
    "CD": "#EF7D86",
}
IBD_COLOR = "#9678D3"
NA_COLOR = "#A8B0BD"

TASK_LABELS = {
    "ibd_control": "IBD vs Control",
    "cd_uc": "CD vs UC",
}

TASK_PALETTES = {
    "ibd_control": {
        "Control": GROUP_COLORS["Control"],
        "IBD": IBD_COLOR,
        "NA": NA_COLOR,
    },
    "cd_uc": {
        "UC": GROUP_COLORS["UC"],
        "CD": GROUP_COLORS["CD"],
        "NA": NA_COLOR,
    },
}

PROFILE_OPTIONS = (
    "Host profile",
    "AMG profile",
    "vOTU profile",
    "Prokaryote profile",
)
DEFAULT_PROFILE = "Host profile"

PAGE_SIZES = (25, 50, 100)
DEFAULT_PAGE_SIZE = 50

HOST_ATLAS_COLOR_FIELDS = (
    "group",
    "cohort",
    "region",
    "detailed_region",
    "gender",
    "race",
    "smoking",
    "enterotype",
)

DISTANCE_METRICS = ("correlation", "cosine", "euclidean", "braycurtis")
MAX_K = 15
TSS_HELP = (
    "TSS normalization must run on all features in the uploaded matrix, by sample, "
    "before selecting the AMG features required by the model. "
    "Do not subset to the model AMGs and then apply TSS normalization."
)

HOME_SCALE_LABELS = {
    "amg_orfs": "AMG ORFs",
    "amg_functions": "AMG Functions",
    "votu": "vOTU (with AMGs)",
    "samples": "Samples / Individuals",
}

ATLAS_HTML_HEIGHT = 720
PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": True,
    "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
}
