"""Explicit DataFrame pagination (current page only)."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import pandas as pd


@dataclass(frozen=True)
class PageSlice:
    page_df: pd.DataFrame
    total_rows: int
    total_pages: int
    page_number: int
    page_size: int
    start: int
    end: int


def paginate(df: pd.DataFrame, page_number: int, page_size: int) -> PageSlice:
    total_rows = int(len(df))
    total_pages = max(1, ceil(total_rows / page_size) if page_size else 1)
    page_number = min(max(int(page_number), 1), total_pages)
    start = (page_number - 1) * page_size
    end = min(start + page_size, total_rows)
    page_df = df.iloc[start:end]
    return PageSlice(
        page_df=page_df,
        total_rows=total_rows,
        total_pages=total_pages,
        page_number=page_number,
        page_size=page_size,
        start=start,
        end=end,
    )
