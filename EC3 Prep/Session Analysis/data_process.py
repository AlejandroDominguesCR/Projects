from __future__ import annotations

import os
import re
import warnings
from typing import Dict, List

import pandas as pd


def _read_csv(path: str) -> pd.DataFrame:
    """Read a CSV file using ``;`` as separator and normalise columns."""

    df = pd.read_csv(
        path,
        sep=";",
        engine="python",
        encoding="utf-8-sig",
    )

    # Drop unnamed columns that usually appear due to trailing separators
    df = df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")], errors="ignore")

    # Normalise column names: strip spaces, replace spaces by underscore and
    # lower case everything
    df.columns = [re.sub(r"\s+", "_", c.strip()).lower() for c in df.columns]
    return df

def load_session_data(folder_path: str) -> Dict[str, pd.DataFrame]:
    """Load all CSV files contained in ``folder_path``.

    Parameters
    ----------
    folder_path:
        Path to a folder that contains the session CSV files.

    Returns
    -------
    dict
        A mapping ``name -> DataFrame`` where *name* is the base file name
        without extension.
    """

    data: Dict[str, pd.DataFrame] = {}
    for fname in os.listdir(folder_path):
        if not fname.lower().endswith(".csv"):
            continue
        path = os.path.join(folder_path, fname)
        key = os.path.splitext(fname)[0]
        data[key] = _read_csv(path)
    return data

def export_raw_session(folder_path: str, output_path: str) -> None:
    """Combine all session CSVs into a single Excel workbook."""

    data = load_session_data(folder_path)
    with pd.ExcelWriter(output_path) as writer:
        for name, df in data.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)

def parse_time_to_seconds(value: str | float | int) -> float:
    """Parse a ``mm:ss.xxx`` or ``hh:mm:ss.xxx`` time string to seconds."""

    if pd.isna(value):
        return float("nan")
    text = str(value).strip()

    match = re.match(r"^(\d+):(\d{2})\.(\d+)$", text)
    if match:
        minutes, seconds, millis = match.groups()
        return int(minutes) * 60 + float(f"{seconds}.{millis}")

    match = re.match(r"^(\d+):(\d{2}):(\d{2})\.(\d+)$", text)
    if match:
        hours, minutes, seconds, millis = match.groups()
        total = int(hours) * 3600
        total += int(minutes) * 60
        total += float(f"{seconds}.{millis}")
        return total

    try:
        return float(text)
    except ValueError:
        return float("nan")

def convert_time_column(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Convert ``time_col`` of ``df`` to seconds if present."""

    if time_col not in df.columns:
        return df

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        df = df.copy()
        df[time_col] = df[time_col].apply(parse_time_to_seconds)
    df = df.sort_values(time_col).reset_index(drop=True)
    return df

def unify_timestamps(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    """Convert ``time_col`` to ``datetime`` if it exists and sort by it."""

    if time_col not in df.columns:
        return df

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.sort_values(time_col).reset_index(drop=True)
    return df

def build_driver_matrix(
    df_analysis: pd.DataFrame,
    tracklimits_df: pd.DataFrame | None = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Return a nested dict ``{team: {driver: laps_df}}``.

    ``df_analysis`` must contain at least the columns ``team``, ``driver`` and
    ``lap_number``.  ``tracklimits_df`` is optional and, when provided, a column
    ``track_limits`` with the count of infractions per lap is merged.
    """

    required = {"team", "driver", "lap_number"}
    if not required.issubset(df_analysis.columns):
        raise KeyError(f"df_analysis missing columns: {required - set(df_analysis.columns)}")

    df = df_analysis.copy()
    df = df.sort_values(["driver", "lap_number"])  # ensure order

    driver_matrix: Dict[str, Dict[str, pd.DataFrame]] = {}
    for (team, driver), group in df.groupby(["team", "driver"]):
        driver_matrix.setdefault(team, {})[driver] = group.reset_index(drop=True)

    if tracklimits_df is not None and not tracklimits_df.empty:
        tl = tracklimits_df.rename(columns=lambda c: c.strip().lower())
        lap_col = "lap" if "lap" in tl.columns else "lap_number"
        tl_counts = tl.groupby(["team", "driver", lap_col]).size().reset_index(name="track_limits")
        for _, row in tl_counts.iterrows():
            team = row["team"]
            driver = row["driver"]
            lap = row[lap_col]
            if team in driver_matrix and driver in driver_matrix[team]:
                df_driver = driver_matrix[team][driver]
                mask = df_driver["lap_number"] == lap
                if "track_limits" not in df_driver.columns:
                    df_driver["track_limits"] = 0
                df_driver.loc[mask, "track_limits"] = row["track_limits"]
                driver_matrix[team][driver] = df_driver

    return driver_matrix

def seconds_to_mmss(t: float | int | str) -> str:
    """Convierte segundos (float) → 'm:ss.mmm'.  NaN → ''."""
    try:
        t = float(t)
    except (TypeError, ValueError):
        return ""
    m, s = divmod(t, 60)
    return f"{int(m):d}:{s:06.3f}"


def compute_gap_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute gap-related columns if not already present.

    Parameters
    ----------
    df:
        DataFrame that should contain at least a ``hour`` column.  The
        function adds ``timestamp`` (if missing), ``GapAhead`` and, when the
        three sector times are available, ``GapAhead_S1``, ``GapAhead_S2`` and
        ``GapAhead_S3``.

    Returns
    -------
    pd.DataFrame
        The DataFrame with the computed columns.
    """

    df = df.copy()

    # ── timestamp global + GapAhead (meta) ─────────────────────────────
    if "timestamp" not in df.columns:
        if "hour" in df.columns:
            df["timestamp"] = pd.to_datetime(df["hour"], errors="coerce")
        else:
            return df

    df = df.sort_values("timestamp").reset_index(drop=True)

    if "GapAhead" not in df.columns:
        df["GapAhead"] = df["timestamp"].diff().dt.total_seconds().abs()

    # ── gaps por sector (S1/S2/S3) si existen ─────────────────────────
    need_cols = {"GapAhead_S1", "GapAhead_S2", "GapAhead_S3"}
    if need_cols - set(df.columns):
        sector_aliases = {
            "sector1": ["sector1", "s1", "sector1_seconds", "s1_seconds"],
            "sector2": ["sector2", "s2", "sector2_seconds", "s2_seconds"],
            "sector3": ["sector3", "s3", "sector3_seconds", "s3_seconds"],
        }
        sector_cols = {
            std: next((c for c in aliases if c in df.columns), None)
            for std, aliases in sector_aliases.items()
        }

        if all(sector_cols.values()):
            for col in sector_cols.values():
                if not pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].apply(parse_time_to_seconds)

            df["T3"] = df["timestamp"]
            df["T2"] = df["T3"] - pd.to_timedelta(df[sector_cols["sector3"]], unit="s")
            df["T1"] = df["T2"] - pd.to_timedelta(df[sector_cols["sector2"]], unit="s")

            df["GapAhead_S1"] = (
                df.loc[df["T1"].sort_values().index, "T1"]
                  .diff().dt.total_seconds().abs()
                  .reindex(df.index)
            )
            df["GapAhead_S2"] = (
                df.loc[df["T2"].sort_values().index, "T2"]
                  .diff().dt.total_seconds().abs()
                  .reindex(df.index)
            )
            df["GapAhead_S3"] = (
                df.loc[df["T3"].sort_values().index, "T3"]
                  .diff().dt.total_seconds().abs()
                  .reindex(df.index)
            )

    return df