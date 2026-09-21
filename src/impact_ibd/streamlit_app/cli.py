"""Console entry point for the packaged Streamlit application."""

from __future__ import annotations

import subprocess
import sys
from importlib.resources import as_file, files


def main() -> int:
    """Launch the Streamlit application from the installed package."""
    app = files("impact_ibd.streamlit_app").joinpath("app.py")
    with as_file(app) as app_path:
        return subprocess.call(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(app_path),
                "--theme.primaryColor=#79A8FF",
                "--theme.backgroundColor=#F5F7FB",
                "--theme.secondaryBackgroundColor=#EEF3FA",
                "--theme.textColor=#24324A",
                "--theme.font=sans serif",
                "--server.maxUploadSize=200",
            ]
        )
