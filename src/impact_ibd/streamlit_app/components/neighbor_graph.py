"""Deterministic Plotly neighbor graph (no NetworkX / Graphviz)."""

from __future__ import annotations

import math
from typing import Mapping, Optional

import pandas as pd
import plotly.graph_objects as go

from impact_ibd.streamlit_app.components.charts import apply_figure_style
from impact_ibd.streamlit_app.constants import (
    INK,
    MACARON_YELLOW,
    NA_COLOR,
    PLOTLY_CONFIG,
    TASK_PALETTES,
)

QUERY_NAME = "Query"


def neighbor_angle(rank: int, k: int) -> float:
    """Fixed polar angle for neighbor rank ``r`` (1-indexed). Rank 1 is at the top."""
    if k <= 0:
        raise ValueError("k must be positive")
    return math.pi / 2 + 2 * math.pi * (int(rank) - 1) / int(k)


def neighbor_xy(rank: int, k: int, radius: float = 1.0) -> tuple:
    theta = neighbor_angle(rank, k)
    return radius * math.cos(theta), radius * math.sin(theta)


def _label_color(label, palette: Mapping[str, str]) -> str:
    if label is None or (isinstance(label, float) and pd.isna(label)) or pd.isna(label):
        key = "NA"
    else:
        key = str(label)
        if key in {"", "nan", "None", "<NA>"}:
            key = "NA"
    return palette.get(key, palette.get("NA", NA_COLOR))


def _display_label(label) -> str:
    if label is None or pd.isna(label):
        return "NA"
    text = str(label)
    return "NA" if text in {"", "nan", "None", "<NA>"} else text


def build_neighbor_figure(
    neighbors: pd.DataFrame,
    query_sample,
    *,
    task: str,
    predicted_label=None,
    confidence=None,
    true_label=None,
    k: Optional[int] = None,
) -> go.Figure:
    subset = neighbors.loc[neighbors["Query_Sample"] == query_sample].sort_values("Rank")
    if subset.empty:
        fig = go.Figure()
        fig.update_layout(title=None)
        apply_figure_style(fig, titled=False, show_grid=False)
        fig.add_annotation(
            text="No nearest neighbors to display",
            showarrow=False,
            font=dict(size=14, color=INK),
        )
        return fig

    k_eff = int(k if k is not None else len(subset))
    palette = TASK_PALETTES.get(task, TASK_PALETTES["ibd_control"])
    fig = go.Figure()

    for _, row in subset.iterrows():
        nx, ny = neighbor_xy(int(row["Rank"]), k_eff)
        fig.add_trace(
            go.Scatter(
                x=[0, nx],
                y=[0, ny],
                mode="lines",
                line=dict(color="#C5CDD8", width=2),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[nx / 2],
                y=[ny / 2],
                mode="markers+text",
                text=[f"{float(row['Similarity']):.3f}"],
                textposition="middle center",
                textfont=dict(size=_font_size(k_eff), color=INK),
                marker=dict(size=1, opacity=0, color="rgba(0,0,0,0)"),
                customdata=[[int(row["Rank"]), float(row["Distance"]), float(row["Similarity"])]],
                hovertemplate=(
                    "Rank=%{customdata[0]}<br>"
                    "Distance=%{customdata[1]:.4f}<br>"
                    "Similarity=%{customdata[2]:.3f}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    query_hover = f"Query: {query_sample}<br>Predicted: {predicted_label}<br>Confidence: {confidence}"
    if true_label is not None:
        query_hover += f"<br>True label: {true_label}"
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            mode="markers+text",
            name=QUERY_NAME,
            text=[str(query_sample)],
            textposition="bottom center",
            textfont=dict(size=_font_size(k_eff)),
            marker=dict(
                size=22,
                symbol="diamond",
                color=INK,
                line=dict(width=2, color=MACARON_YELLOW),
            ),
            hovertemplate=query_hover + "<extra></extra>",
        )
    )

    grouped = {}
    for _, row in subset.iterrows():
        label = _display_label(row["Reference_Label"])
        grouped.setdefault(label, []).append(row)

    for label, rows in grouped.items():
        xs, ys, texts, custom = [], [], [], []
        for row in rows:
            x, y = neighbor_xy(int(row["Rank"]), k_eff)
            xs.append(x)
            ys.append(y)
            texts.append(str(row["Reference_Sample"]))
            custom.append(
                [
                    label,
                    int(row["Rank"]),
                    float(row["Distance"]),
                    float(row["Similarity"]),
                ]
            )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                name=str(label),
                text=texts,
                textposition="top center",
                textfont=dict(size=_font_size(k_eff)),
                marker=dict(
                    size=16,
                    symbol="circle",
                    color=_label_color(label, palette),
                    line=dict(width=1, color="#FFFFFF"),
                ),
                customdata=custom,
                hovertemplate=(
                    "Reference Label=%{customdata[0]}<br>"
                    "Rank=%{customdata[1]}<br>"
                    "Distance=%{customdata[2]:.4f}<br>"
                    "Similarity=%{customdata[3]:.3f}<extra></extra>"
                ),
            )
        )

    height = 520 + 24 * max(0, k_eff - 5)
    apply_figure_style(fig, titled=False, show_grid=False)
    fig.update_layout(
        title=None,
        height=height,
        xaxis=dict(visible=False, scaleanchor="y", scaleratio=1, range=[-1.55, 1.55]),
        yaxis=dict(visible=False, range=[-1.55, 1.55]),
        legend_title="Nodes",
        margin=dict(t=28, l=20, r=20, b=40),
        hovermode="closest",
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, visible=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, visible=False)
    return fig


def _font_size(k: int) -> int:
    return max(9, 14 - max(0, k - 5))


def plotly_config() -> dict:
    return dict(PLOTLY_CONFIG)
