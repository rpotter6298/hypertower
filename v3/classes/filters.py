from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Union
import re

import pandas as pd


@dataclass
class RegexFilter:
    pattern: str
    flags: int = 0

    def apply_paths(self, paths: Sequence[str]) -> Tuple[List[str], List[str]]:
        if not self.pattern:
            return list(paths), []
        try:
            regex = re.compile(self.pattern, self.flags)
        except re.error as err:
            return list(paths), [f'Invalid regex "{self.pattern}": {err}']
        filtered = [p for p in paths if regex.search(p)]
        return filtered, []


@dataclass
class ColumnFilter:
    column: str
    operator: str
    value: str
    case_insensitive: bool = True

    def apply_df(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        warnings: List[str] = []
        if not self.column:
            return df, ["Column filter missing column name."]
        columns = list(df.columns)
        col_index = _resolve_column_index(columns, self.column, warnings)
        if col_index is None:
            return df, warnings
        col_name = columns[col_index]
        if self.value is None or self.value == "":
            return df, [f'Column filter "{self.column}" missing value.']
        series = df[col_name]
        mask = series.apply(
            lambda cell: compare_cell(
                cell, self.value, self.operator, case_insensitive=self.case_insensitive
            )
        )
        return df[mask], warnings


FilterSpec = Union[RegexFilter, ColumnFilter]


def apply_regex_filters(paths: Sequence[str], filters: Iterable[RegexFilter]) -> Tuple[List[str], List[str]]:
    filtered = list(paths)
    warnings: List[str] = []
    for filt in filters:
        filtered, warn = filt.apply_paths(filtered)
        warnings.extend(warn)
    return filtered, warnings


def apply_column_filters(df: pd.DataFrame, filters: Iterable[ColumnFilter]) -> Tuple[pd.DataFrame, List[str]]:
    filtered = df
    warnings: List[str] = []
    for filt in filters:
        filtered, warn = filt.apply_df(filtered)
        warnings.extend(warn)
    return filtered, warnings


def compare_cell(cell, raw_value: str, operator: str, case_insensitive: bool = True) -> bool:
    cell_str = "" if cell is None else str(cell).strip()
    value_str = "" if raw_value is None else str(raw_value).strip()
    if case_insensitive:
        cell_str = cell_str.lower()
        value_str = value_str.lower()
    if operator == "=":
        return cell_str == value_str
    if operator == "!=":
        return cell_str != value_str
    cell_num = _to_float(cell_str)
    value_num = _to_float(value_str)
    if cell_num is None or value_num is None:
        return False
    if operator == ">":
        return cell_num > value_num
    if operator == ">=":
        return cell_num >= value_num
    if operator == "<":
        return cell_num < value_num
    if operator == "<=":
        return cell_num <= value_num
    return False


def _resolve_column_index(columns: Sequence[str], column: str, warnings: List[str]) -> int | None:
    try:
        return columns.index(column)
    except ValueError:
        lower = column.lower()
        matches = [idx for idx, col in enumerate(columns) if str(col).lower() == lower]
        if matches:
            if len(matches) > 1:
                warnings.append(
                    f'Column "{column}" matched multiple headers; using "{columns[matches[0]]}".'
                )
            return matches[0]
    warnings.append(f'Column "{column}" not found.')
    return None


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
