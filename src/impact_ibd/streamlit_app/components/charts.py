"""Plotly descriptive charts used by the data explorer."""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go

from impact_ibd.streamlit_app.constants import (
    GROUP_COLORS,
    IBD_COLOR,
    INK,
    MACARON_BLUE,
    MACARON_CORAL,
    MACARON_SEQUENCE,
    NA_COLOR,
    PLOTLY_FONT,
)

_SEMANTIC = {**GROUP_COLORS, "IBD": IBD_COLOR, "NA": NA_COLOR}


def _color_for(label: str, index: int = 0) -> str:
    if label in _SEMANTIC:
        return _SEMANTIC[label]
    return MACARON_SEQUENCE[index % len(MACARON_SEQUENCE)]


def _evaluation_square_layout(fig: go.Figure, *, with_colorbar: bool = False) -> go.Figure:
    """Keep the evaluation plotting area square in a responsive container."""
    fig.update_layout(
        autosize=True,
        margin={"l": 72, "r": 86 if with_colorbar else 36, "t": 54, "b": 72},
    )
    fig.update_xaxes(constrain="domain")
    fig.update_yaxes(scaleanchor="x", scaleratio=1, constrain="domain")
    if with_colorbar:
        fig.update_traces(
            selector={"type": "heatmap"},
            colorbar={
                "len": 0.72,
                "lenmode": "fraction",
                "thickness": 12,
                "y": 0.5,
                "yanchor": "middle",
                "outlinewidth": 0,
                "tickfont": {"size": 11, "color": INK},
            },
        )
    return fig


def apply_figure_style(
    fig: go.Figure,
    *,
    show_grid: bool = True,
    titled: bool = False,
) -> go.Figure:
    """Apply the shared 2D Plotly visual template."""
    top = 56 if titled else 28
    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": PLOTLY_FONT, "color": INK},
        margin={"l": 48, "r": 24, "t": top, "b": 48},
        hoverlabel={"bgcolor": INK, "font_color": "#FFFFFF"},
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0, title_text=""),
    )
    if titled:
        layout["title_font"] = {"size": 18, "color": INK}
    else:
        layout["title"] = {"text": ""}
    fig.update_layout(**layout)
    axis_kwargs = dict(showline=False, zeroline=False)
    if show_grid:
        axis_kwargs["gridcolor"] = "#E9EDF4"
    else:
        axis_kwargs["showgrid"] = False
    fig.update_xaxes(**axis_kwargs)
    fig.update_yaxes(**axis_kwargs)
    return fig


def bar_counts(
    table: pd.DataFrame,
    category_col: str,
    title: Optional[str] = None,
    yaxis: str = "Count",
) -> go.Figure:
    labels = table[category_col].astype(str)
    colors = [_color_for(str(v), i) for i, v in enumerate(table[category_col])]
    horizontal = bool(labels.str.len().max() >= 16) if len(labels) else False
    hover = (
        "%{y}<br>Count: %{x}<br>Percent: %{customdata[0]:.1f}%<extra></extra>"
        if horizontal and "percent" in table.columns
        else "%{x}<br>Count: %{y}<br>Percent: %{customdata[0]:.1f}%<extra></extra>"
        if "percent" in table.columns
        else ("%{y}<br>Count: %{x}<extra></extra>" if horizontal else "%{x}<br>Count: %{y}<extra></extra>")
    )
    if horizontal:
        bar = go.Bar(
            y=labels,
            x=table["count"],
            orientation="h",
            marker_color=colors,
            customdata=table[["percent"]] if "percent" in table.columns else None,
            hovertemplate=hover,
        )
    else:
        bar = go.Bar(
            x=labels,
            y=table["count"],
            marker_color=colors,
            customdata=table[["percent"]] if "percent" in table.columns else None,
            hovertemplate=hover,
        )
    fig = go.Figure(bar)
    layout = dict(
        xaxis_title=yaxis if horizontal else category_col,
        yaxis_title=category_col if horizontal else yaxis,
        legend=dict(title=""),
    )
    if title:
        layout["title"] = title
    fig.update_layout(**layout)
    if not horizontal:
        fig.update_xaxes(tickangle=-35)
    else:
        fig.update_yaxes(autorange="reversed")
    return apply_figure_style(fig, titled=bool(title))


def histogram(series: pd.Series, title: Optional[str], xaxis: str, nbins: int = 40) -> go.Figure:
    clean = series.dropna()
    fig = go.Figure(
        go.Histogram(
            x=clean,
            nbinsx=nbins,
            marker_color=MACARON_BLUE,
            hovertemplate="Bin: %{x}<br>Count: %{y}<extra></extra>",
        )
    )
    shown_title = f"{title} (valid n = {len(clean)})" if title else None
    layout = dict(
        title=shown_title,
        xaxis_title=xaxis,
        yaxis_title="Count",
        bargap=0.05,
    )
    if not title:
        layout["annotations"] = [
            dict(
                text=f"Valid n = {len(clean)}",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=1,
                y=1.08,
                xanchor="right",
                font=dict(size=12, color="#68758A"),
            )
        ]
    fig.update_layout(**layout)
    return apply_figure_style(fig, titled=bool(title))


def stacked_cohort_group(crosstab: pd.DataFrame, title: Optional[str] = None) -> go.Figure:
    fig = go.Figure()
    order = [c for c in ("Control", "UC", "CD", "NA") if c in crosstab.columns]
    order.extend([c for c in crosstab.columns if c not in order and c != "total"])
    for i, group in enumerate(order):
        fig.add_trace(
            go.Bar(
                name=str(group),
                x=crosstab.index.astype(str),
                y=crosstab[group],
                marker_color=_color_for(str(group), i),
                hovertemplate="cohort=%{x}<br>" + str(group) + ": %{y}<extra></extra>",
            )
        )
    layout = dict(
        barmode="stack",
        xaxis_title="cohort",
        yaxis_title="Samples",
        legend_title="group",
    )
    if title:
        layout["title"] = title
    fig.update_layout(**layout)
    fig.update_xaxes(tickangle=-35)
    return apply_figure_style(fig, titled=bool(title))


def grouped_histogram(df: pd.DataFrame, value_col: str, group_col: str, title: Optional[str] = None) -> go.Figure:
    fig = go.Figure()
    for i, (group, sub) in enumerate(df.groupby(group_col)):
        fig.add_trace(
            go.Histogram(
                x=sub[value_col],
                name=str(group),
                marker_color=_color_for(str(group), i),
                opacity=0.7,
                hovertemplate=f"{group}<br>{value_col}=%{{x}}<br>Count: %{{y}}<extra></extra>",
            )
        )
    layout = dict(
        barmode="overlay",
        xaxis_title=value_col,
        yaxis_title="Samples",
        legend_title=group_col,
    )
    if title:
        layout["title"] = f"{title} (valid n = {len(df)})"
    if not title:
        layout["annotations"] = [
            dict(
                text=f"Valid n = {len(df)}",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=1,
                y=1.08,
                xanchor="right",
                font=dict(size=12, color="#68758A"),
            )
        ]
    fig.update_layout(**layout)
    return apply_figure_style(fig, titled=bool(title))


def treemap(df: pd.DataFrame, title: Optional[str] = None) -> go.Figure:
    ids, labels, parents, values = [], [], [], []
    for phylum, psub in df.groupby("phylum"):
        pid = f"p:{phylum}"
        ids.append(pid)
        labels.append(str(phylum))
        parents.append("")
        values.append(int(psub["count"].sum()))
        for cls, csub in psub.groupby("class"):
            cid = f"c:{phylum}/{cls}"
            ids.append(cid)
            labels.append(str(cls))
            parents.append(pid)
            values.append(int(csub["count"].sum()))
            for _, row in csub.iterrows():
                ids.append(f"o:{phylum}/{cls}/{row['order']}")
                labels.append(str(row["order"]))
                parents.append(cid)
                values.append(int(row["count"]))
    fig = go.Figure(
        go.Treemap(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            hovertemplate="%{label}<br>Records: %{value}<extra></extra>",
            branchvalues="total",
            marker=dict(colorscale=["#EEF3FA", "#79A8FF", "#B9A7F7"]),
        )
    )
    fig.update_layout(margin=dict(t=50 if title else 24, l=10, r=10, b=10))
    if title:
        fig.update_layout(title=title)
    return apply_figure_style(fig, titled=bool(title), show_grid=False)


def confusion_heatmap(matrix, labels, title: Optional[str] = None) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=[str(x) for x in labels],
            y=[str(x) for x in labels],
            colorscale=[[0, "#F8FAFD"], [1, "#6C9FE8"]],
            text=matrix,
            texttemplate="%{text:d}",
            hovertemplate="Actual=%{y}<br>Predicted=%{x}<br>Count=%{z}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title="Predicted",
        yaxis_title="Actual",
        yaxis=dict(autorange="reversed"),
    )
    if title:
        fig.update_layout(title=title)
    fig = apply_figure_style(fig, titled=bool(title), show_grid=False)
    return _evaluation_square_layout(fig, with_colorbar=True)


def roc_curve_figure(fpr, tpr, auc_value) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color=MACARON_CORAL, width=1.5),
            name="Chance",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            line=dict(color=MACARON_BLUE, width=2.5),
            name=f"AUC = {auc_value:.4f}" if auc_value is not None else "ROC",
        )
    )
    fig.update_layout(
        title=None,
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
        xaxis=dict(range=[-0.02, 1.02]),
        yaxis=dict(range=[-0.02, 1.02]),
        legend=dict(
            orientation="h",
            x=0.5,
            y=1.03,
            xanchor="center",
            yanchor="bottom",
        ),
    )
    fig = apply_figure_style(fig, titled=False)
    return _evaluation_square_layout(fig)
