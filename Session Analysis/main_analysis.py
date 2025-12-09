import os
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import plotly.express as px
from dash import dash_table  
import plotly.graph_objects as go
import argparse
from plotly.offline import plot
import tkinter as tk
from tkinter import filedialog
import webbrowser
import logging
import random 
from plotly.subplots import make_subplots 
from collections import OrderedDict
import numpy as np

try:
    import statsmodels.api as sm  # noqa:F401
    HAS_STATSMODELS = True
except Exception:  # statsmodels not available
    HAS_STATSMODELS = False

logging.basicConfig(level=logging.INFO)

from data_process import (
    load_session_data,
    export_raw_session,
    unify_timestamps,
    convert_time_column,
    compute_gap_columns,
    seconds_to_mmss,
    parse_time_to_seconds,
    convert_jerez_timing,
)

from KPI_builder import (
    compute_top_speeds,
    track_limit_rate,
    team_ranking,
    best_sector_times,
    ideal_lap_gap,
    lap_time_history,
    lap_time_consistency,
    extract_session_summary,
    slipstream_stats,
    sector_slipstream_stats,
    build_driver_tables,
    build_fastest_lap_table,
    slipstream_gap_gain,
    slipstream_sector_gap_gain,
)

def get_team_colors() -> dict[str, str]:
    """Devuelve el color principal (hex) para cada equipo."""
    return {
        # Amarillo
        "GriffinCore"  : "#000000",
        "Griffin Core" : "#000000",
        "Campos Racing": "#000000",
        "Campos Racing": "#000000",

        # Naranja
        "MP Motorsport"    : "#FD9800",
        "KCL MP Motorsport": "#FD9800",
        # Rojo
        "Drivex": "#C59E9E",
        "DXR"   : "#C59E9E",

        #Marron 
        "TC Racing": "#8B4513",

        # Verde
        "GRS Team": "#65FD00",

        #Azul claro
        "G4 Racing": "#049C04",

        #Oscuro 
        "Rodin Motorsport": "#0080FF",

        #Rojizo 
        "Monlau Motorsport": "#FD0000",

        #violeta 
        "T-Code by Amtog": "#8400FF",
        "Tecnicar by Amtog": "#8400FF",
        "Tecnicar":    "#8400FF",

       # Azul-violeta
        "Palou Motorsport": "#2E2B9B",
        "Sparco Palou MS": "#2E2B9B",
        # Azul muy oscuro
        "Sainteloc Racing": "#FF02DD",
        "Sainteloc Racing":   "#FF02DD",
        "Saintéloc Racing": "#FF02DD",
        "Sainteloc Raccing": "#FF02DD",
        "Saintéloc Raccing": "#FF02DD",
    }

TEAM_COLOR = get_team_colors()

# ---------------------------------------------------------------------------
# Control panel utilities
# ---------------------------------------------------------------------------

DEFAULT_KPI_VALUES: dict[str, float] = {
    "fast_threshold": 0.02,
    "dt_min": 0.20,
    "dt_max": 2.50,
    "topspeed_delta": 6.0,
    "ref_gap": 5.0,
    "min_laps": 3,
    "consistency_threshold": 0.08,
    "consistency_trim": 0.10,
    "consistency_min_laps": 3,
}

def get_grid_order(df_analysis: pd.DataFrame, df_class: pd.DataFrame) -> list[str]:
    if not df_class.empty and "position" in df_class.columns:
        drv_col = next(
            (c for c in [
                "number",
                "driver_name",
                "driver_shortname",
                "driver_number",
            ] if c in df_class.columns),
            "number",
        )
        return df_class.sort_values("position")[drv_col].dropna().tolist()

    df = df_analysis.copy()
    if not pd.api.types.is_numeric_dtype(df["lap_time"]):
        df["lap_time"] = df["lap_time"].apply(parse_time_to_seconds)
    return (
        df.groupby("number")["lap_time"]
        .min()
        .sort_values()
        .index
        .tolist()
    )

def make_gap_table(
    df_tbl: pd.DataFrame,
    number: str,
    include_sector_gaps: bool = False,
    include_sectors: bool = True,
) -> html.Div:
    """Devuelve un Div con título + DataTable coloreado según GapAhead."""
    df_tbl = df_tbl.copy()

    time_cols = [c for c in df_tbl.columns if c.startswith("Sector") or c == "LapTime"]
    for col in time_cols:
        df_tbl[col] = df_tbl[col].apply(seconds_to_mmss)

    # formateo numérico antes de construir la DataTable
    # ── 1· redondeo numérico dinámico ─────────────────────────────
    round_specs = {"GapStart": 1, "TopSpeed": 0}
    #   ▸ cualquier columna cuyo nombre empiece por “GapAhead_” o “GapSector”
    for col in df_tbl.columns:
        if col.startswith(("GapAhead_", "GapSector")):
            round_specs[col] = 1

    for col, decimals in round_specs.items():
        if col in df_tbl.columns and pd.api.types.is_numeric_dtype(df_tbl[col]):
            df_tbl[col] = df_tbl[col].round(decimals)

    # ── 2· columnas de gap a colorear ──────────────────────────────
    gap_cols = ["GapStart"]
    if include_sector_gaps:
        # añade cualquier columna que empiece por GapAhead_ o GapSector
        gap_cols += [
            c for c in df_tbl.columns
            if c.startswith(("GapAhead_", "GapSector"))
        ]

    style_cond = []
    for col in gap_cols:
        style_cond.extend([
            {
                "if": {"filter_query": f"{{{col}}} <= 1", "column_id": col},
                "backgroundColor": "rgb(255, 102, 102)",
                "color": "white",
            },
            {
                "if": {"filter_query": f"{{{col}}} > 1 && {{{col}}} < 3", "column_id": col},
                "backgroundColor": "rgb(255, 178, 102)",
            },
            {
                "if": {"filter_query": f"{{{col}}} >= 3 && {{{col}}} <= 5", "column_id": col},
                "backgroundColor": "rgb(102, 178, 255)",
            },
        ])
        # ② gradiente de TopSpeed  (azul claro → azul oscuro)  – versión sensible
        if "TopSpeed" in df_tbl.columns and pd.api.types.is_numeric_dtype(df_tbl["TopSpeed"]):
            vmin, vmax = df_tbl["TopSpeed"].min(), df_tbl["TopSpeed"].max()
            vrange = vmax - vmin if vmax > vmin else 1

            p      = 0.4      # ← cuanto menor (<1), más contraste
            base   = 0.15     # opacidad mínima
            span   = 0.85     # base + span ≤ 1

            for i, v in enumerate(df_tbl["TopSpeed"]):
                pct    = (v - vmin) / vrange      # 0 … 1 lineal
                pct_t  = pct ** p                 # ↗️ realza valores medios-altos
                alpha  = base + span * pct_t      # 0.15 → 1.0 (no lineal)

                style_cond.append({
                    "if": {"row_index": i, "column_id": "TopSpeed"},
                    "backgroundColor": f"rgba(0, 123, 255, {alpha:.2f})",
                    "color": "white" if pct_t > 0.5 else "black",
                })
    if not include_sectors:
        df_tbl.drop(columns=[c for c in df_tbl.columns if c.startswith("Sector")],
                    inplace=True, errors="ignore")

    return html.Div(
        children=[
            html.H4(number),
            dash_table.DataTable(
                data=df_tbl.to_dict("records"),
                columns=[{"name": c, "id": c} for c in df_tbl.columns],
                sort_action="native",
                style_data_conditional=style_cond,
                style_cell = {
                "textAlign": "center",
                "padding": "4px",
                "fontSize": "12px",
                "height": "22px",
            },
                style_header={"fontWeight": "bold"},
            ),
        ],
        style={"flex": "1 0 30%", "padding": "5px"},
    )

def build_alias_map_from_folder(folder: str) -> dict[str, str]:
    """
    Busca en la carpeta de sesión un Excel/CSV de 'entry list' y construye
    un mapa {alias -> nombre real}. Si no se encuentra nada válido, devuelve {}.

    Ejemplo: 'CR_2' -> 'Vivek Kanthan', etc.
    """

    try:
        files = os.listdir(folder)
    except Exception:
        return {}

    for fname in files:
        lower = fname.lower()

        # Solo Excel/CSV
        if not lower.endswith((".xls", ".xlsx", ".csv")):
            continue

        # Que el nombre del archivo sugiera que es una entry list
        if "entry" not in lower and "lista" not in lower and "list" not in lower:
            continue

        path = os.path.join(folder, fname)

        # Leemos el archivo
        try:
            if lower.endswith((".xls", ".xlsx")):
                df = pd.read_excel(path)
            else:
                # CSV genérico: que autodetecte el separador
                df = pd.read_csv(path, sep=None, engine="python")
        except Exception:
            continue

        # --- Caso específico: formato F4_25T21JAR_Entry_List.xlsx ---
        if "Unnamed: 3" in df.columns and "Unnamed: 6" in df.columns:
            df2 = df.copy()
            # Fila 0 suele tener 'Driver' / 'Screen'
            mask_header = (
                df2["Unnamed: 3"].astype(str).str.lower().eq("driver")
                | df2["Unnamed: 6"].astype(str).str.lower().eq("screen")
            )
            df2 = df2[~mask_header]

            alias_series = df2["Unnamed: 6"]   # CR_2, MP_3, TEC_76, etc.
            name_series  = df2["Unnamed: 3"]   # Nombre completo
        else:
            # --- Caso genérico: intentamos localizar columnas de alias y nombre ---
            df2 = df.copy()
            cols_lower = {str(c).strip().lower(): c for c in df2.columns}

            alias_col = next(
                (
                    cols_lower[c]
                    for c in ["screen", "alias", "driver_shortname", "driver_code", "driver_name"]
                    if c in cols_lower
                ),
                None,
            )
            name_col = next(
                (
                    cols_lower[c]
                    for c in ["driver", "driver_name", "name", "nombre"]
                    if c in cols_lower
                ),
                None,
            )

            if not alias_col or not name_col:
                continue

            alias_series = df2[alias_col]
            name_series  = df2[name_col]

        # Construimos el dict alias -> nombre
        mapping: dict[str, str] = {}
        for alias, name in zip(alias_series, name_series):
            if pd.isna(alias) or pd.isna(name):
                continue
            a = str(alias).strip()
            n = str(name).strip()
            if not a or not n:
                continue
            mapping[a] = n

        if mapping:
            return mapping

    return {}

def load_data(folder: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and preprocess session CSV files."""
    data = load_session_data(folder)
    for key, df in data.items():
        # si algún CSV trae una columna "time" (no es el caso de Analysis/Weather nuevos),
        # se normaliza a datetime y se ordena
        data[key] = unify_timestamps(df, "time")

    

    # Identificamos cada tipo por nombre de fichero
    analysis_key = next((k for k in data if "analysis" in k.lower()), None)
    class_key    = next((k for k in data if "classification" in k.lower()), None)
    weather_key  = next((k for k in data if "weather" in k.lower()), None)
    tracklimits_key = next((k for k in data if "tracklimits" in k.lower()), None)

    df_analysis    = data.get(analysis_key, list(data.values())[0])
    df_class       = data.get(class_key, pd.DataFrame())
    weather_df     = data.get(weather_key, pd.DataFrame())
    tracklimits_df = data.get(tracklimits_key, pd.DataFrame())

    # ──────────────────────────────────────────────────────────────────────
    # Normalización rápida para timing CSV F4 (SectorAnalysis + clasificación)
    # ──────────────────────────────────────────────────────────────────────
    # --- Normalización específica: timing tipo Jerez -------------------------
    # CSV con FL / VMIn / IP1 / IP2 / VM2In / IN y columnas:
    # lap, ip, number, name, time_lap, time_sector_fl->ip, ...
    jerez_required = {"lap", "ip", "number", "name", "time_lap", "time_sector_fl->ip"}
    if jerez_required.issubset(set(df_analysis.columns)) and "lap_number" not in df_analysis.columns:
        df_analysis = convert_jerez_timing(df_analysis)


    # 1) Si no hay df_class detectado por nombre, intenta detectarlo por columnas
    if df_class.empty:
        for k, df in data.items():
            if k == analysis_key:
                continue
            cols = set(df.columns)
            # patrón de la tabla de clasificación F4 (Pos, Nr, Name, Entrant, time, laps, kmh)
            if {"pos", "nr", "name", "laps"}.issubset(cols):
                df_class = df.copy()
                break

    # 2) Normalizar el CSV de SectorAnalysis (vueltas + sectores + speedtraps)
    cols_analysis = set(df_analysis.columns)
    if {"lapnumber", "laptime", "sector_1_time"}.issubset(cols_analysis):
        # Renombrar a esquema estándar del análisis
        rename_map = {}
        if "name" in cols_analysis and "driver_name" not in cols_analysis:
            rename_map["name"] = "driver_name"
        if "nr" in cols_analysis and "driver_number" not in cols_analysis:
            rename_map["nr"] = "driver_number"
        if "lapnumber" in cols_analysis and "lap_number" not in cols_analysis:
            rename_map["lapnumber"] = "lap_number"
        if "laptime" in cols_analysis and "lap_time" not in cols_analysis:
            rename_map["laptime"] = "lap_time"

        df_analysis = df_analysis.rename(columns=rename_map)
        cols_analysis = set(df_analysis.columns)

        # Sectores -> s1, s2, s3 en float (ya vienen como segundos tipo 40.130)
        for src, dst in [
            ("sector_1_time", "s1"),
            ("sector_2_time", "s2"),
            ("sector_3_time", "s3"),
        ]:
            if src in df_analysis.columns:
                df_analysis[dst] = pd.to_numeric(df_analysis[src], errors="coerce")

        # Columna "hour" a partir de time_of_day (09:16:09.318)
        if "hour" not in df_analysis.columns and "time_of_day" in df_analysis.columns:
            df_analysis["hour"] = df_analysis["time_of_day"].astype(str).str.strip()

        # top_speed = máximo de los speedtraps de la vuelta
        speed_cols = [c for c in df_analysis.columns if c.startswith("speedtrap")]
        if speed_cols and "top_speed" not in df_analysis.columns:
            def _row_max_speed(row):
                vals = []
                for c in speed_cols:
                    v = row.get(c)
                    if pd.isna(v):
                        continue
                    s = str(v).split()[0].replace(",", ".")  # "164.89 Kmh" -> "164.89"
                    try:
                        vals.append(float(s))
                    except ValueError:
                        continue
                return max(vals) if vals else None

            df_analysis["top_speed"] = df_analysis.apply(_row_max_speed, axis=1)

    # 3) Normalizar la tabla de clasificación F4 al esquema estándar
    if not df_class.empty:
        cols_class = set(df_class.columns)
        rename_map_cls = {}
        if "nr" in cols_class and "driver_number" not in cols_class:
            rename_map_cls["nr"] = "driver_number"
        if "name" in cols_class and "driver_name" not in cols_class:
            rename_map_cls["name"] = "driver_name"
        if "entrant" in cols_class and "team" not in cols_class:
            rename_map_cls["entrant"] = "team"
        if "pos" in cols_class and "position" not in cols_class:
            rename_map_cls["pos"] = "position"
        if "time" in cols_class and "best_lap_time" not in cols_class:
            rename_map_cls["time"] = "best_lap_time"
        if "kmh" in cols_class and "best_speed" not in cols_class:
            rename_map_cls["kmh"] = "best_speed"

        df_class = df_class.rename(columns=rename_map_cls)


    # ──────────────────────────────────────────────────────────────────────
    # Mapeo opcional alias -> nombre real usando la Entry List de la carpeta
    # ──────────────────────────────────────────────────────────────────────
    alias_map = build_alias_map_from_folder(folder)
    if alias_map:
        for col in ["driver_name", "driver_shortname", "driver_abbname"]:
            if col in df_analysis.columns:
                df_analysis[col] = (
                    df_analysis[col].astype(str).str.strip().replace(alias_map)
                )
            if col in df_class.columns:
                df_class[col] = (
                    df_class[col].astype(str).str.strip().replace(alias_map)
                )

    # ──────────────────────────────────────────────────────────────────────
    # ANALYSIS: construir identificador de coche y tiempos en segundos
    # ──────────────────────────────────────────────────────────────────────
    driver_col = next(
        c for c in ["driver_shortname", "driver_name", "driver_number"]
        if c in df_analysis.columns
    )

    # (1) identificador único: "nombre_corto #dorsal"
    # Buscamos una columna que pueda actuar como dorsal / car number
    dorsal_col = next(
        (
            c
            for c in ["number", "dorsal", "no", "car_number", "driver_number"]
            if c in df_analysis.columns
        ),
        None,
    )

    if dorsal_col is not None:
        df_analysis["number"] = (
            df_analysis[driver_col].astype(str).str.strip()
            + " #" + df_analysis[dorsal_col].astype(str)
        )
    else:
        # Fallback: solo alias
        df_analysis["number"] = df_analysis[driver_col].astype(str).str.strip()

    # (2) columna legible de piloto
    df_analysis["driver"] = df_analysis[driver_col]

    # (3) lap_time a segundos y columnas de gaps
    df_analysis = convert_time_column(df_analysis, "lap_time")
    df_analysis = compute_gap_columns(df_analysis)


    # Asegurar sectores como float
    for col in ["s1", "s2", "s3"]:
        if col in df_analysis.columns:
            df_analysis[col] = (
                pd.to_numeric(df_analysis[col], errors="coerce")
            )


    # ──────────────────────────────────────────────────────────────────────
    # CLASSIFICATION: opcional, solo si existe
    # ──────────────────────────────────────────────────────────────────────
    if not df_class.empty:
        class_driver = next(
            (c for c in ["driver_name", "driver_shortname", "driver_number"]
             if c in df_class.columns),
            None,
        )
        if class_driver:
            df_class = df_class.copy()

            # localizar columna con el dorsal
            dorsal_col = next(
                (c for c in ["number", "dorsal", "no", "car_number", "driver_number"]
                 if c in df_class.columns),
                None,
            )

            if dorsal_col:
                df_class["number"] = (
                    df_class[class_driver].astype(str).str.strip()
                    + " #" + df_class[dorsal_col].astype(str)
                )
            else:
                df_class["number"] = df_class[class_driver].astype(str).str.strip()

    return df_analysis, df_class, weather_df, tracklimits_df

def build_figures(
    df_analysis,
    df_class,
    weather_df,
    tracklimits_df,
    teams=None,
    *,
    filter_fast=True,
    include_sectors=True,
    include_sector_gaps=False,
    kpi_params: dict[str, float] | None = None,
    timemap_order: str | None = None,
    red_flags: list[dict[str, str]] | None = None,
):
    """Compute a minimal set of KPI figures and a lap table.

    Parameters
    ----------
    df_analysis : pd.DataFrame
        Analysis dataframe.
    df_class : pd.DataFrame
    weather_df : pd.DataFrame
    tracklimits_df : pd.DataFrame
    teams : list[str] | None
        Optional list of team names to filter ``df_analysis``.

    Returns
    -------
    tuple[dict, OrderedDict, pd.DataFrame, list[str]]
        The figures dictionary, a mapping of number names to tables, the summary
        table of fastest laps, and the preferred number order.
    """

    cfg = kpi_params or {}

    # 1) calculamos ALWAYS sobre todo el df
    figs = {}
    driver_tables = build_driver_tables(
        df_analysis,
        teams=None,  # aquí pasamos None para que calcule gaps con TODOS
        filter_fast=filter_fast,
        include_sectors=include_sectors,
        include_sector_gaps=include_sector_gaps,
    )
    # 2) filtramos sólo la presentación de tablas
    if teams:
        driver_tables = OrderedDict(
            (drv, tbl)
            for drv, tbl in driver_tables.items()
            if tbl["Team"].iloc[0] in teams
        )

    fastest_table = build_fastest_lap_table(
        df_analysis,
        df_class,
        teams=teams,                  # <— aquí indicamos qué equipos queremos ver
        filter_fast=filter_fast,
        include_sectors=include_sectors,
        include_sector_gaps=include_sector_gaps,
    )

    grid_order = list(driver_tables.keys())
    if not df_class.empty and "position" in df_class.columns:
        grid_order = (
            df_class.sort_values("position")["number"].tolist()
        )
    grid_order = [d for d in grid_order if d in driver_tables]
    for drv in driver_tables:
        if drv not in grid_order:
            grid_order.append(drv)
    grid_order = get_grid_order(df_analysis, df_class)

    ordered = OrderedDict()
    for drv in grid_order:
        if drv in driver_tables:
            ordered[drv] = driver_tables[drv]
    for drv, tbl in driver_tables.items():
        if drv not in ordered:
            ordered[drv] = tbl
    driver_tables = ordered
    # Top Speeds for each number & generar colores por piloto
    ts = compute_top_speeds(df_analysis)
    # Mapa de colores por equipo (igual que Team Ranking)
    team_colors = get_team_colors()

    for team in ts['team'].unique():
        team_colors.setdefault(team, f'#{random.randint(0, 0xFFFFFF):06x}')

    ss = slipstream_stats(
        df_analysis,
        fast_threshold=cfg.get("fast_threshold", 0.02),
        dt_min=cfg.get("dt_min", 0.20),
        dt_max=cfg.get("dt_max", 2.50),
        topspeed_delta=cfg.get("topspeed_delta", 6.0),
    )

    ss_sector = sector_slipstream_stats(
        df_analysis,
        fast_threshold=cfg.get("fast_threshold", 0.02),   # usa el mismo slider
        dt_min=cfg.get("dt_min", 0.20),
        dt_max=cfg.get("dt_max", 2.50),
        topspeed_delta=cfg.get("topspeed_delta", 6.0),
    ).rename(columns={
        "min_time_with_slip_s1": "min_s1_with_slip",
        "min_time_no_slip_s1": "min_s1_no_slip",
        "min_time_with_slip_s2": "min_s2_with_slip",
        "min_time_no_slip_s2": "min_s2_no_slip",
    })

    if not ss.empty:
        # 1) Lap-time mínimo (orden asc.)
        ss_lap   = ss.sort_values("min_lap_time_with_slip")
        orderLap = ss_lap["number"].tolist()
        fig_slip_lap = px.bar(
            ss_lap, x="number",
            y=["min_lap_time_no_slip", "min_lap_time_with_slip"],
            barmode="group", title="Lap-time mínimo – Con vs Sin rebufo",
        )
        first_val = ss_lap.iloc[0][["min_lap_time_no_slip",
                                    "min_lap_time_with_slip"]].min()
        last_val  = ss_lap.iloc[-1][["min_lap_time_no_slip",
                                    "min_lap_time_with_slip"]].max()
        fig_slip_lap.update_layout(
            xaxis={"categoryorder": "array", "categoryarray": orderLap},
            yaxis={"range": [first_val*0.90, last_val*1.05]}
        )

        # 2) Top-speed máximo (orden desc.)
        ss_spd   = ss.sort_values("max_top_speed_with_slip", ascending=False)
        orderSpd = ss_spd["number"].tolist()
        fig_slip_speed = px.bar(
            ss_spd, x="number",
            y=["max_top_speed_no_slip", "max_top_speed_with_slip"],
            barmode="group", title="Top Speed máxima – Con vs Sin rebufo",
        )
        first_val = ss_spd.iloc[0][["max_top_speed_no_slip",
                                    "max_top_speed_with_slip"]].max()
        last_val  = ss_spd.iloc[-1][["max_top_speed_no_slip",
                                    "max_top_speed_with_slip"]].min()
        fig_slip_speed.update_layout(
            xaxis={"categoryorder": "array", "categoryarray": orderSpd},
            yaxis={"range": [last_val*0.90, first_val*1.05]}
        )

        # 3) Registrar las figuras ⇣
        figs["Slipstream Lap-time (min)"] = fig_slip_lap
        figs["Slipstream TopSpeed (max)"] = fig_slip_speed

    if not ss_sector.empty:
        s1 = ss_sector.sort_values("min_s1_with_slip")
        order_s1 = s1["number"].tolist()
        fig_s1 = px.bar(
            s1,
            x="number",
            y=["min_s1_no_slip", "min_s1_with_slip"],
            barmode="group",
            title="Slipstream Lap-time S1",
        )
        first_val = s1.iloc[0][["min_s1_no_slip", "min_s1_with_slip"]].min()
        last_val = s1.iloc[-1][["min_s1_no_slip", "min_s1_with_slip"]].max()
        fig_s1.update_layout(
            xaxis={"categoryorder": "array", "categoryarray": order_s1},
            yaxis={"range": [first_val * 0.90, last_val * 1.05]},
        )

        s2 = ss_sector.sort_values("min_s2_with_slip")
        order_s2 = s2["number"].tolist()
        fig_s2 = px.bar(
            s2,
            x="number",
            y=["min_s2_no_slip", "min_s2_with_slip"],
            barmode="group",
            title="Slipstream Lap-time S2",
        )
        first_val = s2.iloc[0][["min_s2_no_slip", "min_s2_with_slip"]].min()
        last_val = s2.iloc[-1][["min_s2_no_slip", "min_s2_with_slip"]].max()
        fig_s2.update_layout(
            xaxis={"categoryorder": "array", "categoryarray": order_s2},
            yaxis={"range": [first_val * 0.90, last_val * 1.05]},
        )

        figs["Slipstream Lap-time S1"] = fig_s1
        figs["Slipstream Lap-time S2"] = fig_s2

    df_tmp = df_analysis.copy()
    df_tmp["slipstream"] = False


    gain_df = slipstream_gap_gain(
        df_analysis,
        gap_col="GapAhead",
        fast_threshold=cfg.get("fast_threshold", 0.03),
        ref_gap=cfg.get("ref_gap", 5.0),
        min_laps=int(cfg.get("min_laps", 3)),
    )

    if not gain_df.empty:
        fig_gain = px.scatter(
            gain_df, x="gap_range", y="mean_gain",
            error_y="std_gain", size="laps",
            title="Ganancia media de lap-time vs Gap",
            labels={"mean_gain": "Δt medio (s)", "gap_range": "Gap en meta (s)"},
        )
        figs["Slipstream gain vs gap"] = fig_gain
    
    # asegurar columnas Sector1/2/3 en segundos si existen
    sector_map = {
        "Sector1": {"s1_seconds", "s1", "sector1"},
        "Sector2": {"s2_seconds", "s2", "sector2"},
        "Sector3": {"s3_seconds", "s3", "sector3"},
    }
    for std, aliases in sector_map.items():
        raw = next((c for c in df_analysis.columns if c.lower() in aliases), None)
        if raw is None:
            continue
        if not pd.api.types.is_numeric_dtype(df_analysis[raw]):
            df_analysis[raw] = df_analysis[raw].apply(parse_time_to_seconds)
        df_analysis[std] = df_analysis[raw]


    sector_gain = slipstream_sector_gap_gain(
        df_analysis,
        fast_threshold=cfg.get("fast_threshold", 0.10),
        ref_gap=cfg.get("ref_gap", 5.0),
        min_laps=int(cfg.get("min_laps", 3)),
    )

    if not sector_gain.empty:
        fig_sector = px.line(
            sector_gain, x="gap_range", y="mean_gain",
            color="sector",  markers=True, error_y="std_gain",
            labels={
                "gap_range": "Gap (s)",
                "mean_gain": "Δsector-time (s)",
                "sector": "Sector",
            },
            title="Ganancia por sector vs Gap",
        )
        fig_sector.update_layout(
            xaxis=dict(
                categoryorder="array",
                categoryarray=[
                    "[0.0, 0.5)", "[0.5, 1.0)", "[1.0, 1.5)", "[1.5, 2.0)",
                    "[2.0, 2.5)", "[2.5, 3.5)", "[3.5, 5.0)", "[5.0, inf)"
                ]
            )
        )
        figs["Slipstream gain by sector"] = fig_sector
               

    # --- Top Speeds ---------------------------------------------------------
    if not ts.empty:
        fig_ts = px.bar(
            ts,
            x="number", y="max_top_speed",
            color="team", color_discrete_map=team_colors,
            title="Top Speeds (ordenado desc)"
        )
        fig_ts.update_layout(
            xaxis={'categoryorder': 'array',
                   'categoryarray': ts['number'].tolist()}
        )
        # Límite dinámico Y
        ymin, ymax = ts['max_top_speed'].min(), ts['max_top_speed'].max()
        delta = ymax - ymin
        fig_ts.update_layout(
            yaxis=dict(
                range=[ymin - 0.05*delta, ymax + 0.10*delta],
                title="Velocidad Máxima (km/h)"
            )
        )
        figs["Top Speeds"] = fig_ts



    # Track limits rate per lap if data available
    if not tracklimits_df.empty:
        rate_df = track_limit_rate(tracklimits_df, df_analysis)
        figs["Track Limits per Lap"] = px.bar(
            rate_df,
            x="number",
            y="rate",
            title="Track Limits per Lap",
        )

    # 3) Team Ranking
    team_df = team_ranking(df_analysis)
    if not team_df.empty:
        fig_team = px.bar(
            team_df,
            x="team", y="mean_top_speed",
            color="team", color_discrete_map=team_colors,
            title="Team Ranking (mean top speed)"
        )
        ymin, ymax = team_df['mean_top_speed'].min(), team_df['mean_top_speed'].max()
        delta = ymax - ymin
        fig_team.update_layout(
            yaxis=dict(
                range=[ymin - 0.05*delta, ymax + 0.10*delta],
                title="Velocidad Media por Equipo"
            )
        )
        figs["Team Ranking"] = fig_team

    # 4) Gap a Vuelta Ideal vs Mejor Vuelta Real

    ig = ideal_lap_gap(df_analysis)
    if not ig.empty:
        ig_sorted = ig.sort_values("best_lap", ascending=True).reset_index(drop=True)
        drivers = ig_sorted["number"].tolist()

        # Colores robustos aunque falte 'team' o haya equipos desconocidos
        if "team" in ig_sorted.columns:
            cols = []
            for t in ig_sorted["team"]:
                # Tratamos NaN o cadena vacía como "sin equipo"
                if pd.isna(t) or str(t).strip() == "":
                    cols.append("#888888")  # gris por defecto
                else:
                    t_clean = str(t).strip()
                    # Si el equipo no está aún en el mapa, le asignamos un color nuevo
                    if t_clean not in team_colors:
                        team_colors[t_clean] = f"#{random.randint(0, 0xFFFFFF):06x}"
                    cols.append(team_colors[t_clean])
        else:
            # Si no hay columna de equipo, todos en gris
            cols = ["#888888"] * len(drivers)

        fig_gap = go.Figure()
        fig_gap.add_trace(go.Bar(
            x=drivers, y=ig_sorted["ideal_time"],
            marker_color=cols, name="Vuelta Ideal"
        ))
        fig_gap.add_trace(go.Scatter(
            x=drivers, y=ig_sorted["best_lap"],
            mode="markers", marker=dict(size=10, color=cols),
            name="Mejor Vuelta Real"
        ))
        for i in range(len(drivers)):
            fig_gap.add_shape(
                type="line",
                x0=drivers[i],
                x1=drivers[i],
                y0=ig_sorted.loc[i, "ideal_time"],
                y1=ig_sorted.loc[i, "best_lap"],
                line=dict(color=cols[i], dash="dash"),
            )
        fig_gap.update_layout(
            title="Gap a Vuelta Ideal vs Mejor Vuelta Real",
            xaxis={"categoryorder": "array", "categoryarray": drivers},
        )
        # Límite dinámico Y usando tiempos ideal y real
        ymin = min(ig_sorted["ideal_time"].min(), ig_sorted["best_lap"].min())
        ymax = max(ig_sorted["ideal_time"].max(), ig_sorted["best_lap"].max())
        delta = ymax - ymin
        fig_gap.update_layout(
            yaxis=dict(
                range=[ymin - 0.05 * delta, ymax + 0.10 * delta],
                title="Tiempo (s)",
            )
        )
        figs["Gap a Vuelta Ideal"] = fig_gap


    hist_df = lap_time_history(df_analysis)
    if not hist_df.empty:
        lap_col = "lap_number" if "lap_number" in hist_df.columns else "lap"

        has_team_hist = "team" in hist_df.columns
        fig_hist = px.line(
            hist_df.sort_values([lap_col, "number"]),
            x=lap_col, y="lap_time",
            color="team" if has_team_hist else None,
            line_group="number",
            color_discrete_map=team_colors if has_team_hist else None,
            title="Histórico de tiempos por vuelta",
        )
        fig_hist.update_layout(
            xaxis_title="Vuelta",
            yaxis_title="Tiempo (s)",
            legend_title="Piloto",
        )
        # Ajuste dinámico de eje Y (–5% … +10%)
        ymin, ymax = hist_df["lap_time"].min(), hist_df["lap_time"].max()
        delta = ymax - ymin
        fig_hist.update_layout(
            yaxis=dict(
                range=[ymin - 0.05 * delta, ymax + 0.10 * delta]
            )
        )
        figs["Lap Time History"] = fig_hist


    bst = best_sector_times(df_analysis)
    if not bst.empty:
        has_team = "team" in bst.columns

        for sec in ["sector1", "sector2", "sector3"]:
            # Si ese sector no existe en bst, lo saltamos
            if sec not in bst.columns:
                continue

            # Columnas base: siempre number, opcionalmente team
            base_cols = ["number"] + (["team"] if has_team else [])
            df_sec = bst[base_cols + [sec]].copy()

            # Asegurarnos de trabajar con una Serie (no DataFrame con columnas duplicadas)
            col = df_sec[sec]
            if isinstance(col, pd.DataFrame):
                # Si hubiera columnas duplicadas 'sec', cogemos la primera
                col = col.iloc[:, 0]

            # Convertimos a numérico por si acaso
            col = pd.to_numeric(col, errors="coerce")

            # Diferencia respecto al mejor sector (más rápido = 0.0)
            best_time = col.min()
            df_sec["diff"] = col - best_time

            # Ordenamos por diff
            df_sec = df_sec.sort_values("diff", ascending=True).reset_index(drop=True)

            # Gráfico
            fig = px.bar(
                df_sec,
                x="number",
                y="diff",
                color="team" if has_team else None,
                color_discrete_map=team_colors if has_team else None,
                title=f"Diferencia en {sec.upper()} vs mejor",
            )

            fig.update_layout(
                xaxis={
                    "categoryorder": "array",
                    "categoryarray": df_sec["number"].tolist(),
                }
            )

            ymin, ymax = df_sec["diff"].min(), df_sec["diff"].max()
            delta = ymax - ymin if ymax > ymin else 1.0
            fig.update_layout(
                yaxis=dict(
                    range=[ymin - 0.05 * delta, ymax + 0.10 * delta],
                    title="Diferencia (s)",
                )
            )

            figs[f"{sec.upper()} Diff"] = fig


    
    cons_df = lap_time_consistency(
        df_analysis,
        threshold=cfg.get("consistency_threshold", 0.08),
        trim=cfg.get("consistency_trim", 0.10),
        min_laps=int(cfg.get("consistency_min_laps", 3)),
    )

    if "team" in df_analysis.columns:
        cons_df = cons_df.merge(
            df_analysis[["number", "team"]].drop_duplicates(),
            on="number",
            how="left",
        )
    if not cons_df.empty:
        cons_df = cons_df.sort_values('lap_time_std', ascending=True)
        fig_cons = px.bar(
            cons_df,
            x="number", y="lap_time_std",
            color="team", color_discrete_map=TEAM_COLOR,
            title="Lap Time Consistency",
        )
        ymin, ymax = cons_df['lap_time_std'].min(), cons_df['lap_time_std'].max()
        delta = ymax - ymin
        fig_cons.update_layout(
            yaxis=dict(
                range=[ymin - 0.05 * delta, ymax + 0.10 * delta],
                title='Desviación (s)'
            )
        )
        figs['Lap Time Consistency'] = fig_cons

    # ───────────────── Time Map: Vueltas vs tiempo de sesión ──────────────────
    required_cols = {"number", "lap_time", "timestamp"}
    if required_cols.issubset(df_analysis.columns):
        tm = df_analysis.copy()

        # Filtro por equipos (si aplica)
        if teams:
            tm = tm[tm["team"].isin(teams)]

        # Limpiar
        tm = tm.dropna(subset=["timestamp", "lap_time", "number"])
        if not pd.api.types.is_numeric_dtype(tm["lap_time"]):
            tm["lap_time"] = tm["lap_time"].apply(parse_time_to_seconds)

        if not tm.empty:
            # Columna de vuelta
            lap_col = (
                "lap_number"
                if "lap_number" in tm.columns
                else ("lap" if "lap" in tm.columns else None)
            )

            # Texto con formato m:ss.mmm dentro del recuadro
            tm["lap_label"] = tm["lap_time"].apply(seconds_to_mmss)

            # ── Clasificación por tipo de vuelta ───────────────────────────
            # Referencia global de la sesión (solo para marcar la Fastest)
            best_global = tm["lap_time"].min()

            # Umbral de “push” relativo a la mejor vuelta de CADA piloto
            push_thr = 0.03  # 3 %

            tm["lap_type"] = "Other"

            # Mejor vuelta de cada piloto (referencia local)
            best_by_driver = tm.groupby("number")["lap_time"].transform("min")

            # Candidatas a push: vueltas dentro del % sobre la best lap propia
            tm.loc[
                tm["lap_time"] <= best_by_driver * (1 + push_thr), "lap_type"
            ] = "Push"
            # Ojo: la fastest global se marca más abajo,
            # en el bloque "Best lap de cada piloto + fastest de la sesión"


            if lap_col is not None:
                # Orden por coche + nº de vuelta
                tm = tm.sort_values(["number", lap_col])

                # Out lap = primera vuelta de cada coche
                first_idx = tm.groupby("number")[lap_col].idxmin()
                tm.loc[first_idx, "lap_type"] = "Out"

                # In lap = última vuelta de cada coche
                last_idx = tm.groupby("number")[lap_col].idxmax()
                tm.loc[last_idx, "lap_type"] = "In"

                # Warm lap = vuelta anterior a la primera Push/Fastest de cada coche
                first_push_idx = (
                    tm[tm["lap_type"].isin(["Push", "Fastest"])]
                    .groupby("number")[lap_col]
                    .idxmin()
                )
                for idx in first_push_idx:
                    num = tm.loc[idx, "number"]
                    this_lap = tm.loc[idx, lap_col]
                    prev = tm[
                        (tm["number"] == num) & (tm[lap_col] == this_lap - 1)
                    ]
                    if not prev.empty:
                        prev_idx = prev.index[0]
                        if tm.loc[prev_idx, "lap_type"] == "Other":
                            tm.loc[prev_idx, "lap_type"] = "Warm"

                # Reset = vuelta lenta entre dos push/fastest
                tm = tm.sort_values(["number", lap_col])
                for num, grp in tm.groupby("number"):
                    idxs = grp.index.to_list()
                    types = grp["lap_type"].to_list()
                    for i in range(1, len(grp) - 1):
                        if (
                            types[i] == "Other"
                            and types[i - 1] in ("Push", "Fastest")
                            and types[i + 1] in ("Push", "Fastest")
                        ):
                            tm.loc[idxs[i], "lap_type"] = "Reset"

            # ── Best lap de cada piloto + fastest de la sesión ─────────────
            best_idx = tm.groupby("number")["lap_time"].idxmin()
            tm.loc[best_idx, "lap_type"] = "BestLap"
            tm.loc[tm["lap_time"] == best_global, "lap_type"] = "Fastest"

            # ── Orden eje X (pilotos) según selección -----------------------
            nums_available = tm["number"].dropna().astype(str).unique().tolist()
            if not timemap_order:
                timemap_order = "bestlap"

            drivers_order = nums_available

            # helper para ordenar por dorsal
            def _car_number_key(label: str) -> tuple[int, str]:
                s = str(label)
                if "#" in s:
                    part = s.split("#")[-1].strip()
                else:
                    part = s
                try:
                    n = int(part)
                except ValueError:
                    n = 9999
                return (n, s)

            if timemap_order == "number":
                drivers_order = sorted(nums_available, key=_car_number_key)

            elif timemap_order == "team" and "team" in tm.columns:
                drivers_order = (
                    tm.sort_values(["team", "number"])
                    .drop_duplicates("number")["number"]
                    .tolist()
                )

            elif (
                timemap_order == "position"
                and df_class is not None
                and not df_class.empty
                and {"position", "number"} <= set(df_class.columns)
            ):
                ordered = (
                    df_class.dropna(subset=["number", "position"])
                    .sort_values("position")["number"]
                    .astype(str)
                    .tolist()
                )
                s = set(nums_available)
                drivers_order = [d for d in ordered if d in s] + [
                    d for d in nums_available if d not in ordered
                ]

            elif timemap_order == "bestlap":
                drivers_order = (
                    tm.groupby("number")["lap_time"]
                    .min()
                    .sort_values()
                    .index.astype(str)
                    .tolist()
                )

            if not drivers_order:
                drivers_order = nums_available

            # ── Colores por tipo de vuelta ─────────────────────────────────
            lap_type_colors = {
                "Fastest": "#a953ff",  # morado → best lap de la sesión
                "BestLap": "#5eff00",  # verde lima → best lap por piloto
                "Push": "#00cc44",  # verde → vueltas rápidas
                "Warm": "#ffd700",  # amarillo "warm"
                "Out": "#ffe680",  # amarillo claro / out lap
                "Reset": "#a0a0a0",  # gris
                "In": "#1f77b4",  # azul para in lap
                "Other": "#cccccc",  # resto
            }
            for lt in tm["lap_type"].unique():
                lap_type_colors.setdefault(lt, "#cccccc")

            # ── Figura base: cuadrados con texto ───────────────────────────
            fig_tm = px.scatter(
                tm,
                x="number",
                y="timestamp",
                color="lap_type",
                color_discrete_map=lap_type_colors,
                text="lap_label",
            )

            fig_tm.update_traces(
                marker=dict(
                    symbol="square",
                    size=46,  # un pelín más grande
                    line=dict(width=1, color="black"),
                    opacity=0.7,
                ),
                textposition="middle center",
                textfont=dict(size=10),
            )

            # ── Líneas de inicio/fin de sesión + franja bandera roja ───────
            session_start = tm["timestamp"].min()
            session_end = tm["timestamp"].max()

            shapes = [
                # inicio sesión
                dict(
                    type="line",
                    xref="paper",
                    x0=0,
                    x1=1,
                    yref="y",
                    y0=session_start,
                    y1=session_start,
                    line=dict(color="#444", width=1, dash="dot"),
                ),
                # fin sesión
                dict(
                    type="line",
                    xref="paper",
                    x0=0,
                    x1=1,
                    yref="y",
                    y0=session_end,
                    y1=session_end,
                    line=dict(color="#444", width=1, dash="dot"),
                ),
            ]
            # Banderas rojas múltiples (si hay horas válidas)
            if red_flags:
                base_day = tm["timestamp"].dropna().iloc[0].normalize()
                for rf in red_flags:
                    s = (rf.get("start") or "").strip()
                    e = (rf.get("end") or "").strip()
                    if not s or not e:
                        continue
                    try:
                        rf_start_dt = pd.to_datetime(
                            f"{base_day.date()} {s}",
                            errors="coerce",
                        )
                        rf_end_dt = pd.to_datetime(
                            f"{base_day.date()} {e}",
                            errors="coerce",
                        )
                    except Exception:
                        continue

                    if pd.isna(rf_start_dt) or pd.isna(rf_end_dt):
                        continue

                    y0 = min(rf_start_dt, rf_end_dt)
                    y1 = max(rf_start_dt, rf_end_dt)
                    shapes.append(
                        dict(
                            type="rect",
                            xref="paper",
                            x0=0,
                            x1=1,
                            yref="y",
                            y0=y0,
                            y1=y1,
                            fillcolor="rgba(255,0,0,0.18)",
                            line={"width": 0},
                            layer="below",
                        )
                    )

            # ── Franjas verticales para los coches del equipo Tecnicar ─────
            if "team" in tm.columns:
                tecnicar_mask = tm["team"].astype(str).str.contains("Tecnicar", case=False, na=False)
                tecnicar_numbers = (
                    tm.loc[tecnicar_mask, "number"]
                      .dropna()
                      .astype(str)
                      .unique()
                      .tolist()
                )

                if drivers_order:
                    # ancho relativo de la franja respecto al “slot” de cada coche
                    # 0.5 = 50 % del ancho → ±25 % a cada lado
                    width_frac = 0.50
                    half_width = 0.5 * width_frac   # en unidades de índice de categoría

                    for num in tecnicar_numbers:
                        if num not in drivers_order:
                            continue

                        # índice entero del coche en el eje X categórico
                        idx = drivers_order.index(num)

                        # usamos coordenadas del eje X (categoría ~ índice entero)
                        x_center = float(idx)
                        x_left = x_center - half_width
                        x_right = x_center + half_width

                        shapes.append(
                            dict(
                                type="rect",
                                xref="x",      # ahora en coordenadas del eje X
                                x0=x_left,
                                x1=x_right,
                                yref="paper",
                                y0=0,
                                y1=1,
                                fillcolor="rgba(194,153,231, 0.6)",
                                line={"width": 0},
                                layer="below",
                            )
                        )



            # Eje Y invertido: abajo el final de la sesión, arriba el inicio
            fig_tm.update_layout(
                title="Time Map – Vueltas por coche vs tiempo de sesión",
                xaxis_title="Coche",
                yaxis_title="Hora de paso por meta",
                xaxis=dict(
                    type="category",
                    tickangle=60,
                    categoryorder="array",
                    categoryarray=drivers_order,
                ),
                yaxis=dict(autorange="reversed"),
                legend_title_text="Tipo de vuelta",
                height=1600,
                width=2200,  # más ancho para que quepa el texto
                margin=dict(l=80, r=40, t=80, b=160),
                shapes=shapes,
            )

            figs["Time Map"] = fig_tm

    # --- Wind + Track-Temp plots ----------------------------------------------
    ts_cols = [
        "time_utc",
        "time_local",
        "time",
        "time_utc_str",
        "TIME_UTC_STR",
        "TIME_UTC_SECONDS",
    ]  # posibles timestamps

    # Columna de dirección de viento: aceptamos varios nombres
    wind_dir_col = next(
        (
            c
            for c in [
                "wind_direction",
                "wind_dir",
                "windDir",
                "WindDirection",
                "WIND_DIRECTION",
            ]
            if c in weather_df.columns
        ),
        None,
    )

    if (
        not weather_df.empty
        and wind_dir_col is not None
        and any(c in weather_df.columns for c in ts_cols)
    ):
        # ① Limpieza rápida -----------------------------------------------------
        wdf = weather_df.copy()

        # Normalizamos el nombre a 'wind_direction'
        if wind_dir_col != "wind_direction":
            wdf = wdf.rename(columns={wind_dir_col: "wind_direction"})

        wdf["wind_direction"] = (
            wdf["wind_direction"]
            .astype(str)
            .str.replace(r"[^0-9,.-]", "", regex=True)
            .str.replace(",", ".")
            .astype(float)
        )

        # Velocidad de viento: varios nombres posibles
        speed_col = next(
            (
                c
                for c in [
                    "wind_speed_kph",
                    "wind_speed_kmh",
                    "wind_speed",
                    "wind_velocity",
                    "wind_speed_ms",
                    "WIND_SPEED",
                ]
                if c in wdf.columns
            ),
            None,
        )
        if speed_col:
            wdf[speed_col] = (
                wdf[speed_col]
                .astype(str)
                .str.replace(r"[^0-9,.-]", "", regex=True)
                .str.replace(",", ".")
                .astype(float)
            )

        # Nombres posibles de temperatura de pista
        temp_candidates = [
            "track_temp",
            "track_temperature",
            "track_temp_c",
            "track_temperature_c",
            "TrackTemp",
            "Track Temp",
            "track temp",
            "TRACK_TEMP",
        ]
        temp_col = next((c for c in temp_candidates if c in wdf.columns), None)
        if temp_col:
            wdf[temp_col] = (
                wdf[temp_col]
                .astype(str)
                .str.replace(r"[^0-9,.-]", "", regex=True)
                .str.replace(",", ".")
                .astype(float)
            )

        # ② Timestamp uniforme --------------------------------------------------
        time_col = next((c for c in ts_cols if c in wdf.columns), None)
        wdf["timestamp"] = pd.to_datetime(wdf[time_col], errors="coerce")

        # ③ Alineamos meteo con vueltas ----------------------------------------
        ldf = df_analysis[["lap_number", "lap_time", "hour", "number", "team"]].copy()
        session_day = wdf["timestamp"].dropna().iloc[0].normalize()
        ldf["timestamp"] = (
            pd.to_datetime(
                session_day.strftime("%Y-%m-%d ") + ldf["hour"],
                format="%Y-%m-%d %H:%M:%S.%f",
                errors="coerce",
            )
            - pd.Timedelta(hours=2)  # ajusta si ≠ UTC-2
        )

        merged = (
            pd.merge_asof(
                ldf.dropna(subset=["timestamp"]).sort_values("timestamp"),
                wdf.dropna(subset=["timestamp"]).sort_values("timestamp"),
                on="timestamp",
                direction="nearest",
                tolerance=pd.Timedelta("5min"),
            )
            .dropna(subset=["wind_direction", "lap_time"])
        )
        merged["color"] = merged["team"].map(TEAM_COLOR).fillna("#888")

        if not merged.empty:
            # ───── Gráfico 1: Track-Temp vs Time ───────────────────────────────
            if temp_col:
                fig_temp_time = px.line(
                    wdf,
                    x="timestamp",
                    y=temp_col,
                    title="Track Temperature vs Time",
                    labels={temp_col: "Temp (°C)", "timestamp": "Hora"},
                )
                fig_temp_time.update_traces(showlegend=False)
            else:
                fig_temp_time = go.Figure()
                fig_temp_time.update_layout(
                    title="Track Temperature vs Time (sin columna de track temp en Weather CSV)"
                )

            # ───── Gráfico 2: Wind evolution (dir + speed) ────────────────────
            fig_wind_evo = make_subplots(specs=[[{"secondary_y": True}]])
            fig_wind_evo.add_trace(
                go.Scatter(
                    x=wdf["timestamp"],
                    y=wdf["wind_direction"],
                    mode="lines",
                    name="Dirección (°)",
                ),
                secondary_y=False,
            )
            if speed_col:
                fig_wind_evo.add_trace(
                    go.Scatter(
                        x=wdf["timestamp"],
                        y=wdf[speed_col],
                        mode="lines",
                        name="Velocidad (km/h)",
                    ),
                    secondary_y=True,
                )
            fig_wind_evo.update_layout(
                title="Wind Direction & Speed vs Time",
                xaxis_title="Hora",
                yaxis_title="Dirección (°)",
            )
            if speed_col:
                fig_wind_evo.update_yaxes(
                    title_text="Velocidad (km/h)", secondary_y=True
                )

            # ───── Gráfico 3: Lap-Time vs Wind-Dir ─────────────────────────────
            color_kw = (
                dict(color=speed_col, color_continuous_scale="Viridis")
                if speed_col
                else {}
            )
            fig_scatter_rot = px.scatter(
                merged,
                x="lap_time",
                y="wind_direction",
                title="Lap Time vs Wind Direction",
                **color_kw,
                trendline="ols" if HAS_STATSMODELS else None,
            )
            fig_scatter_rot.update_layout(
                xaxis_title="Lap Time (s)", yaxis_title="Dirección (°)"
            )
            fig_scatter_rot.update_traces(marker_color=merged["color"])

            # ───── Combina Track-Temp + Wind evolution en 1×2 ─────────────────
            combo = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=(
                    "Track Temp vs Time",
                    "Wind Direction & Speed vs Time",
                ),
                specs=[[{"type": "xy"}, {"type": "xy", "secondary_y": True}]],
                column_widths=[0.45, 0.55],
            )

            # Col-1: temperatura
            for tr in fig_temp_time.data:
                combo.add_trace(tr, row=1, col=1)

            # Col-2: viento (dir + vel)
            for tr in fig_wind_evo.data:
                combo.add_trace(
                    tr,
                    row=1,
                    col=2,
                    secondary_y=("Velocidad" in tr.name),
                )

            combo.update_layout(
                height=550,
                title_text="Track Temperature & Wind Evolution",
                legend=dict(x=0.78, y=1.00, yanchor="top", xanchor="left"),
            )

            # ───── Gráfico 3-A: Track-Temp vs Lap-Time ─────────────────────────
            if temp_col:
                fig_temp_vs_lt = px.scatter(
                    merged,
                    x="lap_time",
                    y=temp_col,
                    title="Track Temp vs Lap Time",
                    labels={"lap_time": "Lap Time (s)", temp_col: "Temp (°C)"},
                )
                fig_temp_vs_lt.update_traces(
                    marker_color=merged["color"],
                    showlegend=False,
                )
            else:
                fig_temp_vs_lt = go.Figure()
                fig_temp_vs_lt.update_layout(
                    title="Track Temp vs Lap Time (sin columna de track temp)"
                )

            # ───── Figura apilada (2 filas, 1 columna) ────────────────────────
            stack = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.07,
                subplot_titles=(
                    "Track Temp vs LapTime",
                    "Wind Direction vs LapTime",
                ),
            )

            # Row-1 → Track Temp
            for tr in fig_temp_vs_lt.data:
                stack.add_trace(tr, row=1, col=1)

            # Row-2 → Wind Direction
            for tr in fig_scatter_rot.data:
                stack.add_trace(tr, row=2, col=1)

            # Etiquetas
            stack.update_yaxes(title_text="Temp (°C)", row=1, col=1)
            stack.update_yaxes(title_text="Dirección (°)", row=2, col=1)
            stack.update_xaxes(title_text="Lap Time (s)", row=2, col=1)

            # Ajuste opcional de la barra de color si hay velocidad
            if speed_col:
                stack.update_layout(
                    coloraxis_colorbar=dict(
                        title="Velocidad (km/h)",
                        y=0.15,
                        len=0.35,
                    )
                )

            stack.update_layout(
                height=650,
                title_text="Lap Time vs Track Temp & Wind Direction",
                showlegend=False,
            )

            # ───── Registrar figuras en el diccionario ────────────────────────
            figs["Track & Wind Evolution"] = combo
            figs["LapTime vs TrackTemp & WindDir"] = stack



    return figs, driver_tables, fastest_table, grid_order

def _gap_color(val: float) -> str:
    """Color de fondo según el gap en segundos."""
    if pd.isna(val):
        return ""
    if val <= 1:
        return "background-color: rgb(255,102,102); color:white"
    if val < 3:
        return "background-color: rgb(255,178,102)"
    return "background-color: rgb(102,178,255)"

def export_report(
    figs: dict,
    path: str,
    driver_tables: "OrderedDict[str, pd.DataFrame] | None" = None,
    fast_table: pd.DataFrame | None = None,
    standard_table: pd.DataFrame | None = None,
    ideal_table: pd.DataFrame | None = None,
):

    """
    Escribe un HTML con una tabla (opcional) al principio y luego las figuras.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write("<html><head>\n<meta charset='utf-8'>\n")
        f.write("<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>\n")
        f.write("<title>Session Report</title></head><body>\n")

        if fast_table is not None and not fast_table.empty:
            ft = fast_table.copy()
            if "BestLap" in ft.columns:
                ft["BestLap"] = ft["BestLap"].apply(seconds_to_mmss)
            f.write("<h2>Fastest Laps</h2>\n")
            def _team_style(row):
                col = TEAM_COLOR.get(row['team'], '#FFFFFF')
                txt = 'black' if col.lower() in ('#ffff00', '#fe9900') else 'white'
                return [f'background-color: {col}; color: {txt}'] * len(row)

            styler = (
                ft.style
                .apply(_team_style, axis=1)
                .format({'BestLap': seconds_to_mmss})
                .set_table_attributes("border='1' cellspacing='0' cellpadding='3' "
                                    "style='text-align:center; font-family:Arial; font-size:13px'")
)

            f.write(styler.to_html(index=False))
            f.write("<hr>")
            
        # --- Clasificación por mejor vuelta real ---------------------------
        if standard_table is not None and not standard_table.empty:
            st = standard_table.copy()
            f.write("<h2>Clasificación – Mejor vuelta real</h2>\n")
            styler_std = (
                st.style
                .set_table_attributes(
                    "border='1' cellspacing='0' cellpadding='3' "
                    "style='text-align:center; font-family:Arial; font-size:13px'"
                )
            )
            f.write(styler_std.to_html(index=False))
            f.write("<hr>")

        # --- Clasificación por vuelta ideal --------------------------------
        if ideal_table is not None and not ideal_table.empty:
            it = ideal_table.copy()
            f.write("<h2>Clasificación – Vuelta ideal teórica</h2>\n")
            styler_ideal = (
                it.style
                .set_table_attributes(
                    "border='1' cellspacing='0' cellpadding='3' "
                    "style='text-align:center; font-family:Arial; font-size:13px'"
                )
            )
            f.write(styler_ideal.to_html(index=False))
            f.write("<hr>")

        if driver_tables:
            # --- encabezado + CSS de rejilla 3xN -----------------------------
            f.write("""
            <h2>Vueltas – Griffin Core & Campos Racing</h2>
            <style>
            .number-grid  { display:flex; flex-wrap:wrap; gap:12px; }
            .number-item  { flex:1 1 calc(33.333% - 12px); box-sizing:border-box; }
            .number-item h3 { text-align:center; margin:4px 0; }
            .number-item table { width:100%; border-collapse:collapse; }
            </style>
            <div class="number-grid">
            """)

            # --- tablas ------------------------------------------------------
            for drv, rows in driver_tables.items():
                if rows.empty:
                    continue
                df_tbl = pd.DataFrame(rows)
                time_cols = [c for c in ["LapTime", "Sector1", "Sector2", "Sector3"] if c in df_tbl.columns]
                format_map = {c: seconds_to_mmss for c in time_cols}
                gap_cols = [c for c in ["GapAhead", "GapAhead_S1", "GapAhead_S2", "GapAhead_S3"] if c in df_tbl.columns]
                styler = (
                    df_tbl.style
                    .format(format_map, na_rep="")
                    .applymap(_gap_color, subset=gap_cols)
                    .set_table_attributes(
                        "border='1' cellspacing='0' cellpadding='3' "
                        "style='text-align:center; font-family:Arial; font-size:13px'"
                    )
                    .set_table_styles([
                        {"selector": "th", "props": [("background", "#F2F2F2"), ("font-weight", "bold")]},
                        {"selector": "td", "props": [("border", "1px solid #E0E0E0")]},
                    ])
                )
                f.write(f"<div class='number-item'><h3>{drv}</h3>")
                f.write(styler.to_html(index=False))
                f.write("</div>")                           

            
            f.write("</div>")                               
            f.write("<hr>")

            # --- GRÁFICAS ----------------------------------------------------
            first = True                                     # solo la 1ª incluye JS
            for title, fig in figs.items():
                f.write(f"<h2>{title}</h2>\n")
                div = plot(
                    fig,
                    include_plotlyjs="cdn" if first else False,
                    output_type="div",
                )
                f.write(div)
                first = False

            # --- cierre del documento ---------------------------------------
            f.write("</body></html>")

def build_session_package(
    folder: str,
    kpi_cfg: dict | None,
    teams: list[str] | None,
    drivers: list[str] | None,
    lap_toggle,
    sec_toggle,
    gap_toggle,
    timemap_order: str | None = None,
    red_flags: list[dict[str, str]] | None = None,
) -> dict:
    """Load session data and return a serializable package."""

    df_analysis, df_class, weather_df, tracklimits_df = load_data(folder)

    if not teams:
        teams = sorted(df_analysis["team"].dropna().unique().tolist())
    if not drivers:
        drivers = sorted(df_analysis["number"].dropna().unique().tolist())

    df_analysis = df_analysis[df_analysis["number"].isin(drivers)]

    filter_fast = "ALL" not in (lap_toggle or [])
    include_sects = "SECT" in (sec_toggle or [])
    include_gap = "GAP" in (gap_toggle or [])
    cfg = kpi_cfg or {}

    figs, driver_tables, fast_table, grid_order = build_figures(
        df_analysis,
        df_class,
        weather_df,
        tracklimits_df,
        teams,
        filter_fast=filter_fast,
        include_sectors=include_sects,
        include_sector_gaps=include_gap,
        kpi_params=cfg,
        timemap_order=timemap_order,
        red_flags=red_flags,
    )

    grid_order = list(driver_tables.keys())
    if not df_class.empty and "position" in df_class.columns:
        grid_order = df_class.sort_values("position")["number"].tolist()

    grid_order = [d for d in grid_order if d in drivers]
    for drv in drivers:
        if drv not in grid_order:
            grid_order.append(drv)

    # -----------------------------
    # Tablas resumen de tiempos
    # -----------------------------
    fast_rows: list[dict] = [] if fast_table is None else fast_table.to_dict(orient="records")

    # Tabla estándar (mejor vuelta real)
    standard_rows: list[dict] = []
    ideal_rows: list[dict] = []

    if fast_table is not None and not fast_table.empty:
        # Build a lookup mapping from race number to the corresponding driver name.
        # Some timing files do not include a column called 'driver'; they may
        # provide 'driver_name', 'driver_shortname' or even only a numeric
        # identifier.  We detect the best available column and normalise it
        # under the name 'driver'.  If no suitable column is found we fall
        # back to using the number itself as the driver label.
        driver_col = next(
            (c for c in ["driver", "driver_name", "driver_shortname", "driver_number"]
             if c in df_analysis.columns),
            None,
        )
        if driver_col is not None:
            driver_lookup = (
                df_analysis[["number", driver_col]]
                .drop_duplicates()
                .rename(columns={driver_col: "driver"})
            )
        else:
            # Fallback: duplicate the number into a 'driver' column
            driver_lookup = (
                df_analysis[["number"]]
                .drop_duplicates()
                .assign(driver=df_analysis["number"])
            )

        std = fast_table.copy()
        std = std.merge(driver_lookup, on="number", how="left")
        # Extraer solo el número (parte después de '#') si está concatenado
        if std["number"].astype(str).str.contains("#").any():
            std["number"] = std["number"].astype(str).str.extract(r"#(\d+)", expand=False)


        # Ordenar por mejor vuelta real
        if "BestLap" in std.columns:
            std = std.sort_values("BestLap").reset_index(drop=True)
            std["Pos"] = range(1, len(std) + 1)

            best_ref = std["BestLap"].min()
            std["GapToLeader"] = std["BestLap"] - best_ref
            std["GapToPrev"] = std["BestLap"].diff().fillna(0.0)

            # Formatear tiempos
            std["BestLap"] = std["BestLap"].apply(seconds_to_mmss)
            std["GapToLeader"] = std["GapToLeader"].apply(
                lambda x: "" if pd.isna(x) or x <= 0 else f"+{x:.3f}"
            )
            std["GapToPrev"] = std["GapToPrev"].apply(
                lambda x: "" if pd.isna(x) or x <= 0 else f"+{x:.3f}"
            )

        cols_std = ["Pos", "number", "driver", "team", "BestLap"]

        if include_sects:
            for c in ["Sector1", "Sector2", "Sector3"]:
                if c in std.columns:
                    cols_std.append(c)

                    
        if "GapToLeader" in std.columns:
            cols_std.append("GapToLeader")
        if "GapToPrev" in std.columns:
            cols_std.append("GapToPrev")

        standard_rows = std[cols_std].to_dict(orient="records")

        # Tabla por vuelta ideal
        ideal_df = ideal_lap_gap(df_analysis)
        if ideal_df is not None and not ideal_df.empty:
            ideal_df = ideal_df.merge(driver_lookup, on="number", how="left")
            if ideal_df["number"].astype(str).str.contains("#").any():
                ideal_df["number"] = ideal_df["number"].astype(str).str.extract(r"#(\d+)", expand=False)

            ideal_df = ideal_df.sort_values("ideal_time").reset_index(drop=True)
            ideal_df["Pos"] = range(1, len(ideal_df) + 1)

            base_ideal = ideal_df["ideal_time"].min()
            ideal_df["GapToLeader"] = ideal_df["ideal_time"] - base_ideal
            ideal_df["GapToPrev"] = ideal_df["ideal_time"].diff().fillna(0.0)

            ideal_df["ideal_time"] = ideal_df["ideal_time"].apply(seconds_to_mmss)
            ideal_df["best_lap"] = ideal_df["best_lap"].apply(seconds_to_mmss)
            ideal_df["ideal_gap"] = ideal_df["ideal_gap"].apply(
                lambda x: "" if pd.isna(x) or abs(x) < 1e-4 else f"{x:+.3f}"
            )
            ideal_df["GapToLeader"] = ideal_df["GapToLeader"].apply(
                lambda x: "" if pd.isna(x) or x <= 0 else f"+{x:.3f}"
            )
            ideal_df["GapToPrev"] = ideal_df["GapToPrev"].apply(
                lambda x: "" if pd.isna(x) or x <= 0 else f"+{x:.3f}"
            )

            cols_ideal = [
                "Pos",
                "number",
                "driver",
                "team",
                "ideal_time",
                "best_lap",
                "ideal_gap",
                "GapToLeader",
                "GapToPrev",
            ]
            ideal_rows = ideal_df[cols_ideal].to_dict(orient="records")

    return {
        "figs": {name: fig.to_dict() for name, fig in figs.items()},
        "tables": {
            drv: tbl.to_dict(orient="records") for drv, tbl in driver_tables.items()
        },
        "fast_table": fast_rows,          # Backwards compatible
        "standard_table": standard_rows,  # Nueva tabla estándar
        "ideal_table": ideal_rows,        # Nueva tabla ideal
        "include_gap": include_gap,
        "grid_order": grid_order,
        "red_flags": red_flags or [],
    }

def build_control_panel(defaults: dict[str, float]) -> html.Div:
    """Return a 1-column control panel (label + input per row)."""

    def row(label: str, component) -> dbc.Row:
        return dbc.Row(
            [
                dbc.Col(html.Label(label), width=2, style={"fontWeight": 600}),
                dbc.Col(component,        width=12),
            ],
            className="mb-2",
        )

    panel = html.Div(
        [
            row(
                "Teams",
                dcc.Dropdown(
                    id="team-filter",
                    multi=True,
                    placeholder="Select teams…",
                    style={"width": "100%"}
                ),
            ),

            row(
                "Drivers",
                dcc.Dropdown(
                    id="number-filter",
                    multi=True,
                    placeholder="Select drivers…",
                    style={"width": "100%"}
                ),
            ),

            html.Hr(),

            row(
                "% vuelta competitiva",
                dbc.Input(
                    id="kpi-fast-threshold",
                    type="number",
                    step=0.01,
                    value=defaults.get("fast_threshold", 0.02),
                ),
            ),
            row(
                "Δ\u202F t mínimo meta (s)",
                dbc.Input(
                    id="kpi-dt-min",
                    type="number",
                    step=0.01,
                    value=defaults.get("dt_min", 0.20),
                ),
            ),
            row(
                "Δ\u202F t máximo meta (s)",
                dbc.Input(
                    id="kpi-dt-max",
                    type="number",
                    step=0.01,
                    value=defaults.get("dt_max", 2.50),
                ),
            ),
            row(
                "+Top-Speed rebufo (km/h)",
                dbc.Input(
                    id="kpi-topspeed-delta",
                    type="number",
                    step=0.1,
                    value=defaults.get("topspeed_delta", 6.0),
                ),
            ),
            row(
                "Gap ref. gain (s)",
                dbc.Input(
                    id="kpi-ref-gap",
                    type="number",
                    step=0.5,
                    value=defaults.get("ref_gap", 5.0),
                ),
            ),
            row(
                "Laps mín. gain",
                dbc.Input(
                    id="kpi-min-laps",
                    type="number",
                    step=1,
                    min=1,
                    value=defaults.get("min_laps", 3),
                ),
            ),
            row(
                "% trim consistencia",
                dbc.Input(
                    id="kpi-consistency-trim",
                    type="number",
                    step=0.01,
                    value=defaults.get("consistency_trim", 0.10),
                ),
            ),

            html.Hr(),
            dbc.Button("Apply",  id="apply-kpi-btn", color="primary", className="me-2"),
            dbc.Button("Reset",  id="reset-kpi-btn", color="secondary"),
        ],
        style={"maxWidth": "1600px", "width": "100vw",  
               "padding": "15px"}
    )

    return panel

def _discover_session_folders(event_dir: str) -> list[str]:
    """Devuelve subcarpetas dentro de event_dir que parezcan sesiones."""
    if not event_dir or not os.path.isdir(event_dir):
        return []
    subs = []
    for name in sorted(os.listdir(event_dir)):
        p = os.path.join(event_dir, name)
        if os.path.isdir(p):
            subs.append(p)
    return subs

def _session_label_from_path(path: str) -> str:
    """Etiqueta legible de sesión a partir del nombre de carpeta."""
    base = os.path.basename(path.rstrip(os.sep))
    # intenta limpiar prefijos largos típicos: 23_Analysis_*, 21_Classification_*
    return base

def _extract_positions_from_class(df_class: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza clasificación a columnas:
      - number
      - team (opcional)
      - position (Int)
      - best_lap (s) si la clasificación trae una columna de mejor vuelta
    """
    if df_class is None or df_class.empty:
        return pd.DataFrame(columns=["number", "team", "position", "best_lap"])

    df = df_class.copy()
    df.columns = [c.lower() for c in df.columns]

    # --- detectar columnas de piloto y dorsal ---
    drv_col = next(
        (c for c in ["driver_name", "driver_shortname", "driver_abbname", "driver"]
         if c in df.columns),
        None,
    )
    nr_col = next(
        (c for c in ["number", "dorsal", "no", "car_number", "driver_number", "nr"]
         if c in df.columns),
        None,
    )

    if drv_col and nr_col:
        df["number"] = (
            df[drv_col].astype(str).str.strip()
            + " #" + df[nr_col].astype(str).str.strip()
        )
    elif drv_col:
        df["number"] = df[drv_col].astype(str).str.strip()
    elif "number" not in df.columns:
        # sin identificador usable → nada que hacer
        return pd.DataFrame(columns=["number", "team", "position", "best_lap"])

    # --- columna de posición ---
    pos_col = next(
        (c for c in ["position", "pos", "rank", "clas", "class_pos", "gridpos"]
         if c in df.columns),
        None,
    )
    if not pos_col:
        return pd.DataFrame(columns=["number", "team", "position", "best_lap"])

    cols = ["number", pos_col]
    if "team" in df.columns:
        cols.insert(1, "team")

    out = df[cols].copy().rename(columns={pos_col: "position"})

    # --- intentar detectar columna de mejor vuelta en la clasificación ---
    best_candidates = [
        c for c in df.columns
        if (
            ("best" in c and "lap" in c)
            or ("fastest" in c and "lap" in c)
            or c in {"bestlap", "best_lap", "best lap"}
        )
    ]
    if best_candidates:
        bl_col = best_candidates[0]
        best_series = df[bl_col]

        # Normalizar a segundos si no es ya numérico
        if not pd.api.types.is_numeric_dtype(best_series):
            best_series = best_series.apply(parse_time_to_seconds)

        out["best_lap"] = pd.to_numeric(best_series, errors="coerce")

    # Posición como entero
    out["position"] = pd.to_numeric(out["position"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["position"]).reset_index(drop=True)

    if "team" not in out.columns:
        out["team"] = None

    cols_out = ["number", "team", "position"]
    if "best_lap" in out.columns:
        cols_out.append("best_lap")

    return out[cols_out]

def _extract_positions_from_analysis(df_analysis: pd.DataFrame) -> pd.DataFrame:
    """
    Si no hay Classification utilizable, infiere posición por mejor vuelta.

    Devuelve columnas: number, team, position, best_lap
    """
    if df_analysis is None or df_analysis.empty:
        return pd.DataFrame(columns=["number", "team", "position", "best_lap"])

    df = df_analysis.copy()
    df.columns = [c.lower() for c in df.columns]

    # Asegurar columna 'number'
    if "number" not in df.columns:
        drv_col = next(
            (c for c in ["driver_shortname", "driver_name", "driver_abbname", "driver"]
             if c in df.columns),
            None,
        )
        if drv_col and "driver_number" in df.columns:
            df["number"] = (
                df[drv_col].astype(str).str.strip()
                + " #" + df["driver_number"].astype(str).str.strip()
            )
        elif drv_col:
            df["number"] = df[drv_col].astype(str).str.strip()
        else:
            return pd.DataFrame(columns=["number", "team", "position", "best_lap"])

    if "lap_time" not in df.columns:
        return pd.DataFrame(columns=["number", "team", "position", "best_lap"])

    # lap_time → segundos si hace falta
    if not pd.api.types.is_numeric_dtype(df["lap_time"]):
        df["lap_time"] = df["lap_time"].apply(parse_time_to_seconds)

    best = (
        df.dropna(subset=["lap_time"])
          .groupby("number", as_index=False)["lap_time"].min()
          .rename(columns={"lap_time": "best_lap"})
    )
    if best.empty:
        return pd.DataFrame(columns=["number", "team", "position", "best_lap"])

    if "team" in df.columns:
        best = best.merge(
            df[["number", "team"]].drop_duplicates(),
            on="number",
            how="left",
        )
    else:
        best["team"] = None

    best = best.sort_values("best_lap", ascending=True).reset_index(drop=True)
    best["position"] = (best.index + 1).astype(int)

    return best[["number", "team", "position", "best_lap"]]

def build_event_positions_df(event_dir: str) -> pd.DataFrame:
    """
    Construye un DataFrame con la evolución de posiciones y mejores vueltas
    a lo largo del evento.

    Columnas devueltas (cuando existen):
        - session  : nombre/etiqueta de la sesión (FP1, Q, R1, ...).
        - number   : identificador de coche/piloto (ej: 'CR_5 #5').
        - team     : equipo.
        - position : posición de la clasificación (cuando hay df_class).
        - best_lap : mejor lap_time (s) de esa sesión para ese coche.
    """
    rows: list[pd.DataFrame] = []

    # Seguridad por si el directorio no existe o está vacío
    if not event_dir or not os.path.isdir(event_dir):
        return pd.DataFrame(columns=["session", "number", "team", "position", "best_lap"])

    # Usamos el mismo descubridor de sesiones que el resto de la pestaña
    for sess_path in _discover_session_folders(event_dir):
        label = _session_label_from_path(sess_path)

        try:
            # load_data ya aplica toda la normalización F4 / alias, etc.
            df_analysis, df_class, _, _ = load_data(sess_path)
        except Exception as exc:
            logging.exception(
                "Error cargando sesión '%s' para Event Positions: %s",
                sess_path, exc
            )
            continue

        # Si no hay nada útil, saltamos
        if (df_analysis is None or df_analysis.empty) and (df_class is None or df_class.empty):
            continue

        pos = pd.DataFrame()
        best = pd.DataFrame()

        # 1) Posiciones desde clasificación (si existe)
        if df_class is not None and not df_class.empty:
            pos = _extract_positions_from_class(df_class)  # number, team, position

        # 2) Best lap desde analysis (si existe)
        if df_analysis is not None and not df_analysis.empty:
            best = _extract_positions_from_analysis(df_analysis)  # number, team, position?, best_lap

        # 3) Combinar según lo que haya
        if (pos is None or pos.empty) and (best is not None and not best.empty):
            # No hay clasificación → usamos directamente ranking por best lap
            pos = best.copy()

        elif (pos is not None and not pos.empty) and (best is not None and not best.empty):
            # Hay clasificación → añadimos best_lap a las posiciones
            if "best_lap" in best.columns:
                best_slim = (
                    best[["number", "best_lap"]]
                    .drop_duplicates(subset=["number"])
                )
                pos = pos.merge(best_slim, on="number", how="left")

        # Si seguimos sin nada, saltamos sesión
        if pos is None or pos.empty:
            continue

        pos = pos.copy()
        pos["session"] = label

        # Nos quedamos con columnas estándar si existen
        keep_cols = [c for c in ["session", "number", "team", "position", "best_lap"] if c in pos.columns]
        pos = pos[keep_cols]

        rows.append(pos)

    if not rows:
        return pd.DataFrame(columns=["session", "number", "team", "position", "best_lap"])

    df_pos = pd.concat(rows, ignore_index=True)

    if "position" in df_pos.columns:
        df_pos["position"] = pd.to_numeric(df_pos["position"], errors="coerce").astype("Int64")

    df_pos["session"] = df_pos["session"].astype(str)

    return df_pos

def build_event_positions_fig(
    df_pos: pd.DataFrame,
    team_filter: list[str] | None = None,
    drivers_filter: list[str] | None = None,
) -> go.Figure:
    """Líneas/markers con Y invertida. Cada 'number' es una serie, con dorsal dentro del punto."""
    if df_pos is None or df_pos.empty:
        return go.Figure()

    df = df_pos.copy()
    if team_filter:
        df = df[df["team"].isin(team_filter)]
    if drivers_filter:
        df = df[df["number"].isin(drivers_filter)]

    if df.empty:
        return go.Figure()

    # Orden de sesiones en el eje X
    session_order = df["session"].dropna().unique().tolist()

    # Colores de equipo
    team_colors = get_team_colors()  # :contentReference[oaicite:0]{index=0}
    fallback = "#888888"

    # Conteo de pilotos por equipo
    if "team" in df.columns:
        team_driver_counts = df.groupby("team")["number"].nunique().to_dict()
    else:
        team_driver_counts = {}

    team_dash_index: dict[str, int] = {}
    dash_cycle = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]

    def extract_car_number(label: str) -> str:
        """De 'Nombre Apellido #23' -> '23'."""
        if not isinstance(label, str):
            return str(label)
        if "#" in label:
            return label.split("#")[-1].strip()
        return label.strip()

    fig = go.Figure()

    for number, g in df.groupby("number", sort=False):
        team = (
            g["team"].dropna().iloc[0]
            if "team" in g.columns and not g["team"].isna().all()
            else None
        )
        color = team_colors.get(team, fallback) if team else fallback

        # estilo de línea si hay varios pilotos en el mismo equipo
        if team and team_driver_counts.get(team, 1) > 1:
            idx = team_dash_index.get(team, 0)
            dash_style = dash_cycle[idx % len(dash_cycle)]
            team_dash_index[team] = idx + 1
        else:
            dash_style = "solid"

        # asegura orden de sesiones
        g = g.set_index("session").reindex(session_order).reset_index()

        # dorsal en TODOS los puntos
        car_nr = extract_car_number(number)
        text_vals = [car_nr] * len(g)

        fig.add_trace(
            go.Scatter(
                x=g["session"],
                y=g["position"],
                mode="lines+markers+text",
                name=number,
                line=dict(
                    shape="linear",   # líneas en diagonal entre sesiones
                    width=3,
                    color=color,
                    dash=dash_style,
                ),
                marker=dict(
                    size=20,
                    color=color,
                    line=dict(width=1, color="black"),
                ),
                text=text_vals,
                textposition="middle center",      # texto centrado en el marker
                textfont=dict(size=11, color="white"),
                hovertext=[number] * len(g),       # tooltip con nombre completo
                hovertemplate="<b>%{hovertext}</b><br>Sesión: %{x}<br>Pos: %{y}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Evolución de posiciones por sesión",
        xaxis=dict(
            title="Sesión",
            categoryorder="array",
            categoryarray=session_order,
        ),
        yaxis=dict(
            title="Posición",
            autorange="reversed",
            dtick=1,
        ),
        legend_title="Piloto",
        hovermode="x unified",
        margin=dict(l=40, r=10, t=60, b=40),
        height=900,
    )
    return fig

def build_event_bestlap_fig(
    event_dir: str,
    team_filter: list[str] | None = None,
    drivers_filter: list[str] | None = None,
) -> go.Figure:
    """
    Best lap por sesión, calculado directamente a partir de los análisis de cada
    sesión del evento.

    X: sesión del evento (T01, T02, Q, R1, ...).
    Y: best lap en segundos por coche.
    """

    if not event_dir or not os.path.isdir(event_dir):
        fig = go.Figure()
        fig.update_layout(title="Sin tiempos de vuelta disponibles para este evento")
        return fig

    rows: list[pd.DataFrame] = []

    for sess_path in _discover_session_folders(event_dir):
        label = _session_label_from_path(sess_path)

        try:
            df_analysis, _, _, _ = load_data(sess_path)
        except Exception as exc:
            logging.exception(
                "Error cargando sesión '%s' para Event Best Laps: %s",
                sess_path, exc
            )
            continue

        if df_analysis is None or df_analysis.empty:
            continue

        df = df_analysis.copy()
        df.columns = [c.lower() for c in df.columns]

        # Detectar columna de tiempo de vuelta / best lap
        time_candidates = [
            "lap_time", "laptime", "lap time",
            "best_lap", "best_lap_time", "bestlap", "fastest_lap",
        ]
        time_col = next((c for c in time_candidates if c in df.columns), None)
        if not time_col:
            continue

        # Aseguramos que sea numérica (s)
        if not pd.api.types.is_numeric_dtype(df[time_col]):
            df[time_col] = df[time_col].apply(parse_time_to_seconds)

        df = df.dropna(subset=[time_col])
        if df.empty:
            continue

        # Aseguramos columna 'number'
        if "number" not in df.columns:
            drv_col = next(
                (c for c in ["driver_shortname", "driver_name", "driver_abbname", "driver"]
                 if c in df.columns),
                None,
            )
            if drv_col and "driver_number" in df.columns:
                df["number"] = (
                    df[drv_col].astype(str).str.strip()
                    + " #" + df["driver_number"].astype(str).str.strip()
                )
            elif drv_col:
                df["number"] = df[drv_col].astype(str).str.strip()
            else:
                # No podemos identificar el coche
                continue

        if "team" not in df.columns:
            df["team"] = None

        # Best lap por coche en esa sesión
        tmp = (
            df[["number", "team", time_col]]
            .groupby(["number", "team"], as_index=False)[time_col]
            .min()
            .rename(columns={time_col: "best_lap"})
        )

        if tmp.empty:
            continue

        tmp["session"] = label
        rows.append(tmp)

    if not rows:
        fig = go.Figure()
        fig.update_layout(title="Sin tiempos de vuelta disponibles para este evento")
        return fig

    df = pd.concat(rows, ignore_index=True)

    # Filtros
    if team_filter:
        df = df[df["team"].isin(team_filter)]
    if drivers_filter:
        df = df[df["number"].isin(drivers_filter)]

    df = df.dropna(subset=["best_lap"])
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="Sin tiempos de vuelta válidos tras aplicar filtros")
        return fig

    session_order = df["session"].dropna().unique().tolist()

    team_colors = get_team_colors()
    fallback = "#888888"

    if "team" in df.columns:
        team_driver_counts = df.groupby("team")["number"].nunique().to_dict()
    else:
        team_driver_counts = {}

    team_dash_index: dict[str, int] = {}
    dash_cycle = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]

    def extract_car_number(label: str) -> str:
        if not isinstance(label, str):
            return str(label)
        if "#" in label:
            return label.split("#")[-1].strip()
        return label.strip()

    fig = go.Figure()

    for number, g in df.groupby("number", sort=False):
        team = (
            g["team"].dropna().iloc[0]
            if "team" in g.columns and not g["team"].isna().all()
            else None
        )
        color = team_colors.get(team, fallback) if team else fallback

        if team and team_driver_counts.get(team, 1) > 1:
            idx = team_dash_index.get(team, 0)
            dash_style = dash_cycle[idx % len(dash_cycle)]
            team_dash_index[team] = idx + 1
        else:
            dash_style = "solid"

        # Orden de sesiones
        g = g.set_index("session").reindex(session_order).reset_index()

        car_nr = extract_car_number(number)
        text_vals = [car_nr] * len(g)

        fig.add_trace(
            go.Scatter(
                x=g["session"],
                y=g["best_lap"],
                mode="lines+markers+text",
                name=number,
                line=dict(
                    shape="linear",
                    width=3,
                    color=color,
                    dash=dash_style,
                ),
                marker=dict(
                    size=20,
                    color=color,
                    line=dict(width=1, color="black"),
                ),
                text=text_vals,
                textposition="middle center",
                textfont=dict(size=11, color="white"),
                hovertext=[number] * len(g),
                hovertemplate="<b>%{hovertext}</b><br>Sesión: %{x}"
                              "<br>Best lap: %{y:.3f} s<extra></extra>",
            )
        )

    fig.update_layout(
        title="Best lap (s) por sesión",
        xaxis=dict(
            title="Sesión",
            categoryorder="array",
            categoryarray=session_order,
        ),
        yaxis=dict(
            title="Best lap (s)",
            autorange=True,      # si quieres que el más rápido quede arriba: autorange="reversed"
        ),
        legend_title="Piloto",
        hovermode="x unified",
        margin=dict(l=40, r=10, t=60, b=40),
        height=900,
    )

    return fig

def build_event_topspeed_fig(
    event_dir: str,
    team_filter: list[str] | None = None,
    drivers_filter: list[str] | None = None,
) -> go.Figure:
    """Top speed medio por sesión y piloto, con barras de desviación estándar.

    - Usa todas las vueltas del evento.
    - Para cada (sesión, piloto) calcula media, σ y nº de vueltas.
    - X: sesión del evento (T01, T02, Q, R1, ...).
    - Y: top_speed media (km/h) por piloto.
    - Cada punto: un piloto, con error bar = σ de sus top speeds en esa sesión.
    """

    if not event_dir or not os.path.isdir(event_dir):
        fig = go.Figure()
        fig.update_layout(title="Evento no válido para top speed")
        return fig

    rows: list[pd.DataFrame] = []

    for sess_path in _discover_session_folders(event_dir):
        label = _session_label_from_path(sess_path)

        try:
            df_analysis, _, _, _ = load_data(sess_path)
        except Exception as exc:
            logging.exception(
                "Error cargando sesión '%s' para Event TopSpeed: %s",
                sess_path, exc
            )
            continue

        if df_analysis is None or df_analysis.empty:
            continue

        df = df_analysis.copy()
        df.columns = [c.lower() for c in df.columns]

        # Columna de top_speed (intentamos variantes si no existe)
        if "top_speed" not in df.columns:
            candidates = [c for c in df.columns if "top" in c and "speed" in c]
            if not candidates:
                continue
            df["top_speed"] = df[candidates[0]]

        # Identificador de coche
        if "number" not in df.columns:
            drv_col = next(
                (c for c in ["driver_shortname", "driver_name", "driver_abbname", "driver"]
                 if c in df.columns),
                None,
            )
            if drv_col and "driver_number" in df.columns:
                df["number"] = (
                    df[drv_col].astype(str).str.strip()
                    + " #" + df["driver_number"].astype(str).str.strip()
                )
            elif drv_col:
                df["number"] = df[drv_col].astype(str).str.strip()
            else:
                # No sabemos identificar el coche
                continue

        if "team" not in df.columns:
            df["team"] = None

        df["top_speed"] = pd.to_numeric(df["top_speed"], errors="coerce")
        df = df.dropna(subset=["top_speed"])
        df = df[df["top_speed"] > 0]

        if df.empty:
            continue

        # Estadísticos por piloto dentro de esta sesión
        stats_sess = (
            df.groupby(["number", "team"], dropna=False)["top_speed"]
            .agg(ts_mean="mean", ts_std="std", n_laps="count")
            .reset_index()
        )
        stats_sess["ts_std"] = stats_sess["ts_std"].fillna(0.0)
        stats_sess["session"] = label

        rows.append(stats_sess)

    if not rows:
        fig = go.Figure()
        fig.update_layout(title="Sin datos de top speed para este evento")
        return fig

    stats_all = pd.concat(rows, ignore_index=True)

    # Filtros
    if team_filter:
        stats_all = stats_all[stats_all["team"].isin(team_filter)]
    if drivers_filter:
        stats_all = stats_all[stats_all["number"].isin(drivers_filter)]

    if stats_all.empty:
        fig = go.Figure()
        fig.update_layout(title="Sin top speeds válidos tras aplicar filtros")
        return fig

    session_order = stats_all["session"].dropna().unique().tolist()

    team_colors = get_team_colors()
    fallback = "#888888"

    if "team" in stats_all.columns:
        team_driver_counts = stats_all.groupby("team")["number"].nunique().to_dict()
    else:
        team_driver_counts = {}

    team_dash_index: dict[str, int] = {}
    dash_cycle = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]

    def extract_car_number(label: str) -> str:
        if not isinstance(label, str):
            return str(label)
        if "#" in label:
            return label.split("#")[-1].strip()
        return label.strip()

    fig = go.Figure()

    for number, g in stats_all.groupby("number", sort=False):
        team = (
            g["team"].dropna().iloc[0]
            if "team" in g.columns and not g["team"].isna().all()
            else None
        )
        color = team_colors.get(team, fallback) if team else fallback

        # Estilo de línea distinto para compañeros de equipo
        if team and team_driver_counts.get(team, 1) > 1:
            idx = team_dash_index.get(team, 0)
            dash_style = dash_cycle[idx % len(dash_cycle)]
            team_dash_index[team] = idx + 1
        else:
            dash_style = "solid"

        # Ordenar por sesión siguiendo session_order
        g = g.set_index("session").reindex(session_order).reset_index()
        g = g.dropna(subset=["ts_mean"])
        if g.empty:
            continue

        car_nr = extract_car_number(number)

        # σ real de cada piloto en esa sesión
        err_array = g["ts_std"].to_numpy()
        n_laps = g["n_laps"].to_numpy()

        # Escala visual de la barra (por ejemplo, ±0.5σ)
        std_scale = 0.2   # ajusta aquí si quieres 0.3, 0.7, etc.
        err_plot = err_array * std_scale

        # En el hover seguimos mostrando la σ real
        custom = np.column_stack([err_array, n_laps])

        fig.add_trace(
            go.Scatter(
                x=g["session"],
                y=g["ts_mean"],
                mode="lines+markers+text",
                name=number,
                line=dict(
                    shape="linear",
                    width=3,
                    color=color,
                    dash=dash_style,
                ),
                marker=dict(
                    size=16,
                    color=color,
                    line=dict(width=1, color="black"),
                ),
                error_y=dict(
                    type="data",
                    array=err_plot,   # usamos la σ escalada para la barra
                    visible=True,
                ),
                text=[car_nr] * len(g),
                textposition="middle center",
                textfont=dict(size=11, color="white"),
                customdata=custom,
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Sesión: %{x}<br>"
                    "Top speed medio: %{y:.1f} km/h<br>"
                    "σ top speed: %{customdata[0]:.2f} km/h<br>"
                    "Vueltas usadas: %{customdata[1]}<extra></extra>"
                ),
            )
        )

    ymin = float(stats_all["ts_mean"].min())
    ymax = float(stats_all["ts_mean"].max())
    delta = ymax - ymin if ymax > ymin else 1.0

    fig.update_layout(
        title="Top speed medio por sesión y piloto ",
        xaxis=dict(
            title="Sesión",
            categoryorder="array",
            categoryarray=session_order,
        ),
        yaxis=dict(
            title="Top speed (km/h)",
            range=[ymin - 0.05 * delta, ymax + 0.10 * delta],
        ),
        legend_title="Piloto",
        hovermode="x unified",
        margin=dict(l=40, r=10, t=60, b=40),
        height=900,
    )

    return fig


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

DEFAULT_SESSION_FOLDER = 'D:\Racing Projects\2025\CAMPOS\Projects_040825\Projects-main\EC3 Prep\Data\CT1'

app.layout = html.Div(
    [
        html.H1("Session Analysis Dashboard"),
        dcc.Input(
            id="folder-input",
            type="text",
            value=DEFAULT_SESSION_FOLDER,              
            placeholder="Ruta a carpeta de sesión",   
        ),
        html.Button("Browse", id="browse-btn"),
        html.Button("Load Session", id="load-btn"),
        dcc.Input(id="folder-input-b", type="text", placeholder="C:/carpeta/de/sesión"),
        html.Button("Browse 2", id="browse-btn-b"),
        html.Button("Load Session B", id="load-btn-b"),
        html.Button("Export Report", id="export-btn"),
        dcc.Download(id="download-report"),
        dcc.Store(id="data-store"),
        dcc.Store(id="session-data"),
        dcc.Store(id="data-store-b"),
        dcc.Store(id="session-data-b"),
        dcc.Store(id="kpi-config", storage_type="session", data=DEFAULT_KPI_VALUES),
        dcc.Store(id="red-flag-store", data=[]),
        dcc.Checklist(id="lap-filter-toggle",
                  options=[{"label": "See all the laps", "value": "ALL"}],
                  value=[], style={"marginTop": "10px"}),
        dcc.Checklist(id="sector-toggle",
                  options=[{"label": "Show Sector Times", "value": "SECT"}],
                  value=[], style={"marginBottom": "10px"}),
        dcc.Checklist(id="gap-toggle",
                  options=[{"label": "Show Gaps in Each Sector", "value": "GAP"}],
                  value=[], style={"marginBottom": "10px"}),
        html.Div(
            [
                html.Label("Available Tables:"),
                dcc.RadioItems(
                    id="tables-mode",
                    options=[
                        {"label": "Standard time table", "value": "standard"},
                        {"label": "Ideal lap table", "value": "ideal"},
                        {"label": "Original detailed table", "value": "Slipstream"},
                    ],
                    value="standard",
                    labelStyle={"display": "block"},
                    style={"marginTop": "4px"},
                ),
            ],
            style={"marginBottom": "10px"},
        ),
        html.Hr(),
        # ─────────────────────────────
        # Controles específicos Time Map
        # ─────────────────────────────
        html.Div(
            [
                html.H4("Time Map – Controls", style={"marginBottom": "6px"}),

                html.Div(
                    [
                        html.Label("X Axis order (Drivers):"),
                        dcc.Dropdown(
                            id="timemap-order-mode",
                            options=[
                                {"label": "Best lap order", "value": "bestlap"},
                                {"label": "Grid position", "value": "position"},
                                {"label": "Per Number", "value": "number"},
                                {"label": "Per Team", "value": "team"},
                            ],
                            value="bestlap",       # por defecto como ahora
                            clearable=False,
                            style={"width": "260px"},
                        ),
                    ],
                    style={"marginBottom": "8px"},
                ),

                html.Div(
                    [
                        html.Label("Red Flag Timing (HH:MM[:SS]):"),
                        dcc.Input(
                            id="red-flag-start",
                            type="text",
                            placeholder="Beginning",
                            style={"width": "100px", "marginRight": "4px"},
                        ),
                        dcc.Input(
                            id="red-flag-end",
                            type="text",
                            placeholder="End",
                            style={"width": "100px", "marginRight": "8px"},
                        ),
                        html.Button("Apply Red Flag", id="red-flag-apply-btn"),
                        html.Span(
                            "  (Applied to time map)",
                            style={
                                "marginLeft": "6px",
                                "fontSize": "11px",
                                "color": "#555",
                            },
                        ),
                    ]
                ),
            ],
            style={
                "padding": "8px 0",
                "marginBottom": "8px",
                "borderBottom": "1px solid #ddd",
            },
        ),
        # ─────────────────────────────
        # Tabs + contenido
        # ─────────────────────────────
        dcc.Tabs(
            id="result-tabs",
            value="tab-tables",
            children=[
                dcc.Tab(label="Tables", value="tab-tables"),
                dcc.Tab(label="Figures", value="tab-figs"),
                dcc.Tab(label="Time Map", value="tab-timemap"),
                dcc.Tab(
                    label="Control Panel",
                    value="tab-control",
                    children=[
                        html.Div(
                            id="control-panel-body",
                            style={"padding": "15px", "maxWidth": "500px"},
                        )
                    ],
                ),
                dcc.Tab(label="Compare Sessions", value="tab-compare"),

                # ───────────────────────
                # NUEVA TAB: Event Positions
                # ───────────────────────
                dcc.Tab(
                    label="Event Positions",
                    value="tab-event-positions",
                    children=[
                        html.Div(
                            [
                                html.H4(
                                    "Evolución de posiciones en el evento",
                                    style={"marginBottom": "10px"},
                                ),

                                # Selector de carpeta de evento
                                html.Div(
                                    [
                                        dcc.Input(
                                            id="event_dir",
                                            type="text",
                                            placeholder="Carpeta del EVENTO (contiene las sesiones)",
                                            style={
                                                "width": "60%",
                                                "marginRight": "6px",
                                            },
                                        ),
                                        html.Button(
                                            "Browse Event",
                                            id="btn_pick_event",
                                            style={"marginRight": "6px"},
                                        ),
                                        html.Button(
                                            "Load positions",
                                            id="btn_load_event_positions",
                                        ),
                                    ],
                                    style={
                                        "display": "flex",
                                        "alignItems": "center",
                                        "flexWrap": "wrap",
                                        "gap": "6px",
                                        "marginBottom": "12px",
                                    },
                                ),

                                # Filtros de equipo / piloto
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Label("Filter by team:"),
                                                dcc.Dropdown(
                                                    id="event_pos_team_filter",
                                                    multi=True,
                                                    placeholder="All teams",
                                                ),
                                            ],
                                            style={"flex": 1, "minWidth": "220px"},
                                        ),
                                        html.Div(
                                            [
                                                html.Label("Filter by driver:"),
                                                dcc.Dropdown(
                                                    id="event_pos_driver_filter",
                                                    multi=True,
                                                    placeholder="All drivers",
                                                ),
                                            ],
                                            style={"flex": 1, "minWidth": "220px"},
                                        ),
                                    ],
                                    style={
                                        "display": "flex",
                                        "gap": "10px",
                                        "flexWrap": "wrap",
                                        "marginBottom": "12px",
                                    },
                                ),

                                # Selector de métrica (posición vs mejor vuelta)
                                html.Div(
                                    [
                                        html.Label("Y axis:"),
                                        dcc.RadioItems(
                                            id="event_pos_mode",
                                            options=[
                                                {"label": "Race position", "value": "position"},
                                                {"label": "Best lap time (s)", "value": "bestlap"},
                                                {"label": "Top speed (km/h)", "value": "topspeed"},
                                            ],
                                            value="position",
                                            labelStyle={
                                                "display": "inline-block",
                                                "marginRight": "12px",
                                            },
                                        ),
                                    ],
                                    style={"marginBottom": "8px"},
                                ),

                                # Gráfica de evolución
                                dcc.Graph(
                                    id="event_positions_fig",
                                    style={
                                        "width": "100%",
                                        "height": "1000px",
                                        "marginBottom": "12px",
                                    },
                                ),

                                # Tabla resumen
                                dash_table.DataTable(
                                    id="event_positions_table",
                                    columns=[],   # se rellenan desde el callback
                                    data=[],
                                    sort_action="native",
                                    page_size=20,
                                    style_table={
                                        "maxHeight": "400px",
                                        "overflowY": "auto",
                                    },
                                    style_cell={
                                        "textAlign": "center",
                                        "padding": "4px",
                                        "fontSize": 12,
                                    },
                                    style_header={"fontWeight": "bold"},
                                ),
                            ],
                            style={"padding": "10px"},
                        )
                    ],
                ),
            ],
        ),
        html.Div(id="tab-content"),
    ]
)

app.layout["control-panel-body"].children = build_control_panel(DEFAULT_KPI_VALUES)

@app.callback(
    Output("kpi-config", "data"),
    Output("kpi-fast-threshold", "value"),
    Output("kpi-dt-min", "value"),
    Output("kpi-dt-max", "value"),
    Output("kpi-topspeed-delta", "value"),
    Output("kpi-ref-gap", "value"),
    Output("kpi-min-laps", "value"),
    Output("kpi-consistency-trim", "value"),
    Input("apply-kpi-btn", "n_clicks"),
    Input("reset-kpi-btn", "n_clicks"),
    State("kpi-fast-threshold", "value"),
    State("kpi-dt-min", "value"),
    State("kpi-dt-max", "value"),
    State("kpi-topspeed-delta", "value"),
    State("kpi-ref-gap", "value"),
    State("kpi-min-laps", "value"),
    State("kpi-consistency-trim", "value"),
    prevent_initial_call=True,
)

def update_kpi_store(n_apply, n_reset,
                     fast_thr, dt_min, dt_max,
                     ts_delta, ref_gap, min_laps, cons_trim):
    """Update kpi-config store and optionally reset the panel inputs."""
    trig = dash.callback_context.triggered_id
    if trig == "reset-kpi-btn":
        cfg = DEFAULT_KPI_VALUES.copy()
    else:
        cfg = {
            "fast_threshold": fast_thr,
            "dt_min": dt_min,
            "dt_max": dt_max,
            "topspeed_delta": ts_delta,
            "ref_gap": ref_gap,
            "min_laps": min_laps,
            "consistency_threshold": DEFAULT_KPI_VALUES["consistency_threshold"],
            "consistency_trim": cons_trim,
            "consistency_min_laps": DEFAULT_KPI_VALUES["consistency_min_laps"],
        }

    return (
        cfg,
        cfg["fast_threshold"],
        cfg["dt_min"],
        cfg["dt_max"],
        cfg["topspeed_delta"],
        cfg["ref_gap"],
        cfg["min_laps"],
        cfg["consistency_trim"],
    )

@app.callback(
    Output("folder-input", "value"),
    Input("browse-btn", "n_clicks"),
    prevent_initial_call=True,
)

def on_browse(n_clicks):
    logging.info(f"[Browse] Click detectado, n_clicks={n_clicks}")
    root = tk.Tk()
    # que no se vea la ventana principal
    root.withdraw()
    # hacerla topmost para que el diálogo salga delante
    root.attributes("-topmost", True)
    root.update()
    folder = filedialog.askdirectory(parent=root)
    root.destroy()
    if not folder:
        raise PreventUpdate
    logging.info(f"[Browse] Carpeta seleccionada: {folder}")
    return folder

@app.callback(
    Output("folder-input-b", "value"),
    Input("browse-btn-b", "n_clicks"),
    prevent_initial_call=True,
)

def on_browse_b(_):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.update()
    folder = filedialog.askdirectory(parent=root)
    root.destroy()
    if not folder:
        raise PreventUpdate
    return folder

@app.callback(Output("data-store-b", "data"),
              Output("session-data-b", "data"),
              Input("load-btn-b", "n_clicks"),
              State("folder-input-b", "value"),
              State("kpi-config", "data"),
              State("lap-filter-toggle", "value"),
              State("sector-toggle", "value"),
              State("gap-toggle", "value"),
              prevent_initial_call=True)

def load_session_b(n_clicks, folder_b, kpi_cfg, lap_toggle, sec_toggle, gap_toggle):
    if not n_clicks or not folder_b:
        raise PreventUpdate
    pkg = build_session_package(folder_b, kpi_cfg, None, None,
                                lap_toggle, sec_toggle, gap_toggle)
    return pkg, pkg

@app.callback(
    Output("tab-content", "children"),
    Output("data-store", "data"),
    Output("session-data", "data"),
    Output("team-filter", "options"),
    Output("team-filter", "value"),
    Output("number-filter", "options"),
    Output("number-filter", "value"),
    Output("red-flag-store", "data"),
    Input("load-btn", "n_clicks"),
    Input("result-tabs", "value"),
    Input("kpi-config", "data"),
    Input("lap-filter-toggle", "value"),
    Input("sector-toggle", "value"),
    Input("gap-toggle", "value"),
    Input("tables-mode", "value"),
    Input("timemap-order-mode", "value"),
    Input("red-flag-apply-btn", "n_clicks"),
    State("folder-input", "value"),
    State("team-filter", "value"),
    State("number-filter", "value"),
    State("data-store", "data"),
    State("session-data-b", "data"),
    State("red-flag-start", "value"),
    State("red-flag-end", "value"),
    State("red-flag-store", "data"),
    prevent_initial_call=True,
)

def update_tab_content(
    n_clicks,
    tab,
    kpi_cfg,
    lap_toggle,
    sec_toggle,
    gap_toggle,
    tables_mode,
    timemap_order,
    red_flag_clicks,
    folder,
    teams,
    drivers,
    stored,
    sess_b,
    red_flag_start,
    red_flag_end,
    red_flags_store,
):
    ctx = dash.callback_context
    trig = ctx.triggered_id

    # Stored red flags
    red_flags = list(red_flags_store or [])

    # Normalize red-flag inputs
    rf_start = red_flag_start.strip() if isinstance(red_flag_start, str) else None
    if rf_start == "":
        rf_start = None
    rf_end = red_flag_end.strip() if isinstance(red_flag_end, str) else None
    if rf_end == "":
        rf_end = None

    if not tables_mode:
        tables_mode = "standard"

    # --- Only tab switch with cached data ---------------------------------------
    if trig == "result-tabs" and stored:
        data = stored
        team_opts = dash.no_update
        driver_opts = dash.no_update
        teams_out = dash.no_update
        drivers_out = dash.no_update
        red_flags_out = dash.no_update

    else:
        # --- Load Session A ------------------------------------------------------
        if not n_clicks:
            raise PreventUpdate

        if not folder:
            logging.warning("No session folder provided (folder-input is empty).")
            raise PreventUpdate

        folder = folder.strip().strip('"')

        if not os.path.isdir(folder):
            logging.warning(f"Provided folder does not exist or is invalid: {folder!r}")
            raise PreventUpdate

        df_analysis, _, _, _ = load_data(folder)
        raw_path = os.path.join(folder, "session_raw_data.xlsx")
        export_raw_session(folder, raw_path)
        logging.info(f"Raw data exported to {raw_path}")

        team_names = sorted(df_analysis["team"].dropna().unique().tolist())
        driver_names = sorted(df_analysis["number"].dropna().unique().tolist())

        if not teams:
            teams = team_names
        if not drivers:
            drivers = driver_names

        # Red-flag list update
        if trig == "load-btn":
            red_flags = []
        elif trig == "red-flag-apply-btn" and rf_start and rf_end:
            interval = {"start": rf_start, "end": rf_end}
            if interval not in red_flags:
                red_flags.append(interval)

        data = build_session_package(
            folder,
            kpi_cfg,
            teams,
            drivers,
            lap_toggle,
            sec_toggle,
            gap_toggle,
            timemap_order,
            red_flags,
        )

        team_opts = [{"label": t, "value": t} for t in team_names]
        driver_opts = [{"label": d, "value": d} for d in driver_names]
        teams_out = teams
        drivers_out = drivers
        red_flags_out = red_flags

    # ----- Rebuild figures & driver tables --------------------------------------
    figs_dict = data.get("figs", data)
    figs = {name: go.Figure(fig) if isinstance(fig, dict) else fig
            for name, fig in figs_dict.items()}

    tables_dict = data.get("tables", {})
    driver_tables = {drv: pd.DataFrame(rows) for drv, rows in tables_dict.items()}

    fast_rows = data.get("fast_table", [])
    standard_rows = data.get("standard_table", [])
    ideal_rows = data.get("ideal_table", [])

    include_gap = data.get("include_gap", False)

    # --- Build main summary dataframe depending on mode -------------------------
    std_df   = pd.DataFrame(standard_rows) if standard_rows else pd.DataFrame()
    ideal_df = pd.DataFrame(ideal_rows)    if ideal_rows    else pd.DataFrame()
    fast_df  = pd.DataFrame(fast_rows)     if fast_rows     else pd.DataFrame()

    summary_df = pd.DataFrame()

    if tables_mode == "ideal" and not ideal_df.empty:
        summary_df = ideal_df.copy()
        # Position change vs standard
        if (not std_df.empty
            and {"number", "Pos"} <= set(std_df.columns)
            and {"number", "Pos"} <= set(summary_df.columns)):
            base_pos = std_df.set_index("number")["Pos"]
            summary_df["PosChange"] = summary_df["number"].map(base_pos) - summary_df["Pos"]
            # signed string (+/-)
            summary_df["PosChange"] = summary_df["PosChange"].apply(
                lambda v: f"{int(v):+d}" if pd.notna(v) else ""
            )

    elif tables_mode in ("raw", "Slipstream") and not fast_df.empty:
        # Original detailed table from build_fastest_lap_table
        summary_df = fast_df

    elif not std_df.empty:
        summary_df = std_df

    else:
        summary_df = fast_df

    # Preferred column order for the summary table (only if present)
    if not summary_df.empty:
        preferred_order = []
        for col in ["Pos", "number", "driver", "team"]:
            if col in summary_df.columns:
                preferred_order.append(col)

        time_cols = [c for c in ["BestLap", "ideal_time", "best_lap", "ideal_gap"]
                     if c in summary_df.columns]
        sector_cols = [c for c in ["Sector1", "Sector2", "Sector3"]
                       if c in summary_df.columns]
        gap_cols_summary = [c for c in ["GapToLeader", "GapToPrev",
                                        "GapAhead_S1", "GapAhead_S2", "GapAhead_S3"]
                            if c in summary_df.columns]
        trailing = [c for c in ["PosChange"] if c in summary_df.columns]

        ordered = preferred_order + time_cols + sector_cols + gap_cols_summary + trailing
        # append any remaining columns that we did not classify
        ordered += [c for c in summary_df.columns if c not in ordered]
        summary_df = summary_df[ordered]

    grid_order = data.get("grid_order", [])
    table_divs = [
        make_gap_table(driver_tables[drv], drv, include_gap)
        for drv in grid_order
        if drv in driver_tables
    ]

    # ----- Summary DataTable (header table) -------------------------------------
    if not summary_df.empty:
        # Format time-like columns if they are numeric
        if "BestLap" in summary_df.columns and np.issubdtype(summary_df["BestLap"].dtype, np.number):
            summary_df["BestLap"] = summary_df["BestLap"].apply(seconds_to_mmss)
        if "ideal_time" in summary_df.columns and np.issubdtype(summary_df["ideal_time"].dtype, np.number):
            summary_df["ideal_time"] = summary_df["ideal_time"].apply(seconds_to_mmss)
        if "best_lap" in summary_df.columns and np.issubdtype(summary_df["best_lap"].dtype, np.number):
            summary_df["best_lap"] = summary_df["best_lap"].apply(seconds_to_mmss)

        # Team coloring on (number/driver/team)
        team_styles = []
        for team, color in TEAM_COLOR.items():
            text_color = "black" if color.lower() in ("#ffff00", "#fe9900") else "white"
            for col in ["number", "driver", "team"]:
                if col in summary_df.columns:
                    team_styles.append(
                        {
                            "if": {"filter_query": f'{{team}} = "{team}"', "column_id": col},
                            "backgroundColor": color,
                            "color": text_color,
                        }
                    )

        # Gap heatmap styles
        gap_cols_heat = ["GapStart"]
        if include_gap:
            gap_cols_heat += ["GapAhead_S1", "GapAhead_S2", "GapAhead_S3"]

        gap_styles = []
        for col in gap_cols_heat:
            gap_styles.extend([
                {"if": {"filter_query": f"{{{col}}} <= 1", "column_id": col},
                 "backgroundColor": "rgb(255,102,102)", "color": "white"},
                {"if": {"filter_query": f"{{{col}}} > 1 && {{{col}}} < 3", "column_id": col},
                 "backgroundColor": "rgb(255,178,102)"},
                {"if": {"filter_query": f"{{{col}}} >= 3 && {{{col}}} <= 5", "column_id": col},
                 "backgroundColor": "rgb(102,178,255)"},
                {"if": {"filter_query": f"{{{col}}} > 5", "column_id": col},
                 "backgroundColor": "rgb(0,102,204)", "color": "white"},
            ])

        # Narrower widths for number/driver/team
        style_cell_conditional = []
        if "number" in summary_df.columns:
            style_cell_conditional.append({"if": {"column_id": "number"}, "width": "80px"})
        if "driver" in summary_df.columns:
            style_cell_conditional.append({"if": {"column_id": "driver"}, "width": "160px"})
        if "team" in summary_df.columns:
            style_cell_conditional.append({"if": {"column_id": "team"}, "width": "140px"})

        summary = dash_table.DataTable(
            data=summary_df.to_dict("records"),
            columns=[{"name": c, "id": c} for c in summary_df.columns],
            sort_action="native",
            style_cell={"textAlign": "center"},
            style_header={"fontWeight": "bold"},
            style_cell_conditional=style_cell_conditional,
            style_data_conditional=team_styles + gap_styles,
        )
        table_divs.insert(0, html.Div(summary, style={"flex": "1 0 100%", "padding": "5px"}))

    # ==========================
    #   CONTENT PER TAB
    # ==========================
    if tab == "tab-tables":
        content = html.Div(table_divs, style={"display": "flex", "flexWrap": "wrap"})

    elif tab == "tab-figs":
        graphs = []
        sector_keys = ["SECTOR1 Diff", "SECTOR2 Diff", "SECTOR3 Diff"]
        gap_key = "Gap a Vuelta Ideal"  # figure key stays as produced upstream
        timemap_key = "Time Map"

        sector_figs = [dcc.Graph(figure=figs[k], style={"flex": 1})
                       for k in sector_keys if k in figs]
        if sector_figs:
            graphs.append(html.Div(sector_figs, style={"display": "flex", "gap": "10px"}))

        if gap_key in figs:
            graphs.append(dcc.Graph(figure=figs[gap_key], style={"width": "100%"}))

        excluded = sector_keys + [gap_key, timemap_key]
        remaining = [k for k in figs if k not in excluded]
        if remaining:
            grid_items = [dcc.Graph(figure=figs[k]) for k in remaining]
            graphs.append(
                html.Div(
                    grid_items,
                    style={"display": "grid", "gridTemplateColumns": "repeat(2, 1fr)", "gap": "10px"},
                )
            )

        content = html.Div(graphs)

    elif tab == "tab-timemap":
        if "Time Map" in figs:
            content = html.Div(
                dcc.Graph(
                    figure=figs["Time Map"],
                    style={"width": "100%", "height": "1500px"},
                ),
                style={"padding": "10px", "width": "100%", "maxWidth": "2300px", "margin": "0 auto"},
            )
        else:
            content = html.Div("Time Map not available for this session.", style={"padding": "10px"})

    elif tab == "tab-control":
        content = html.Div()

    elif tab == "tab-compare":
        sess_a = data
        sess_b_pkg = sess_b
        if not sess_a or not sess_b_pkg:
            content = html.Div(
                "Load 'Session A' and 'Session B' first using the buttons above.",
                style={"padding": "10px"},
            )
        else:
            content = render_compare(sess_a, sess_b_pkg)

    else:
        content = html.Div()

    return (content, data, data, team_opts, teams_out, driver_opts, drivers_out, red_flags_out)

def render_compare(sess_a: dict, sess_b: dict) -> html.Div:
    """Construye la vista de comparación A vs B.

    sess_a: paquete de la sesión A (data-store / session-data)
    sess_b: paquete de la sesión B (session-data-b)
    """
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    if not sess_a or not sess_b:
        return html.Div(
            "No hay datos suficientes para comparar sesiones.",
            style={"padding": "10px"}
        )

    # -------- Δ mejor vuelta (B - A) por dorsal -------------------------------
    fast_a = pd.DataFrame(sess_a.get("fast_table") or [])
    fast_b = pd.DataFrame(sess_b.get("fast_table") or [])

    fig_delta = go.Figure()
    if (not fast_a.empty and not fast_b.empty
            and {"number", "BestLap"}.issubset(fast_a.columns)
            and {"number", "BestLap"}.issubset(fast_b.columns)):

        merged = fast_a.merge(
            fast_b,
            on="number",
            suffixes=("_A", "_B")
        ).dropna(subset=["BestLap_A", "BestLap_B"])

        if not merged.empty:
            merged["DeltaBest"] = merged["BestLap_B"] - merged["BestLap_A"]
            fig_delta = px.bar(
                merged,
                x="number",
                y="DeltaBest",
                title="Δ mejor vuelta (B – A)",
                labels={"number": "Piloto", "DeltaBest": "Δ tiempo (s)"},
            )
            fig_delta.add_hline(y=0, line_color="black", line_width=1)
        else:
            fig_delta.update_layout(
                title="Δ mejor vuelta (B – A) – sin pilotos comunes"
            )
    else:
        fig_delta.update_layout(
            title="Δ mejor vuelta (B – A) – datos insuficientes"
        )

    # -------- Distribución de laptimes (violin) A vs B ------------------------
    tables_a = sess_a.get("tables") or {}
    tables_b = sess_b.get("tables") or {}

    df_list_a = [pd.DataFrame(rows) for rows in tables_a.values() if rows]
    df_list_b = [pd.DataFrame(rows) for rows in tables_b.values() if rows]

    fig_hist = go.Figure()
    if df_list_a and df_list_b:
        laps_a = pd.concat(df_list_a, ignore_index=True).assign(session="A")
        laps_b = pd.concat(df_list_b, ignore_index=True).assign(session="B")

        # Aseguramos LapTime numérico (segundos)
        for df_ in (laps_a, laps_b):
            if "LapTime" in df_.columns:
                df_["LapTime"] = pd.to_numeric(df_["LapTime"], errors="coerce")

        laps_all = pd.concat([laps_a, laps_b], ignore_index=True)
        laps_all = laps_all.dropna(subset=["LapTime"])

        if not laps_all.empty:
            fig_hist = px.violin(
                laps_all,
                x="session",
                y="LapTime",
                box=True,
                points="all",
                title="Distribución de tiempos por sesión",
            )
            fig_hist.update_layout(yaxis_title="LapTime (s)")
        else:
            fig_hist.update_layout(
                title="Distribución de tiempos – sin tiempos válidos"
            )
    else:
        fig_hist.update_layout(
            title="Distribución de tiempos – datos insuficientes"
        )

    return html.Div(
        [
            html.H3("Comparativa Session A vs Session B"),
            html.P(
                "A = primera sesión cargada · B = segunda sesión cargada",
                style={"fontSize": "12px", "color": "#555"},
            ),
            dcc.Graph(figure=fig_delta),
            dcc.Graph(figure=fig_hist),
        ],
        style={"padding": "10px"},
    )

@app.callback(
    Output("event_dir", "value"),
    Input("btn_pick_event", "n_clicks"),
    prevent_initial_call=True,
)

def _pick_event_folder(n):
    # selector nativo (igual patrón que usas en otros botones)
    root = tk.Tk(); root.withdraw()
    path = filedialog.askdirectory(title="Selecciona carpeta del EVENTO (contiene sesiones)")
    root.destroy()
    if not path:
        raise PreventUpdate
    return path

@app.callback(
    Output("event_positions_fig", "figure"),
    Output("event_positions_table", "data"),
    Output("event_pos_team_filter", "options"),
    Output("event_pos_driver_filter", "options"),
    Input("btn_load_event_positions", "n_clicks"),
    State("event_dir", "value"),
    State("event_pos_team_filter", "value"),
    State("event_pos_driver_filter", "value"),
    State("event_pos_mode", "value"),   # ← NUEVO
    prevent_initial_call=True,
)

def _build_event_positions(n_clicks, event_dir, team_filter, drv_filter, mode):
    if not event_dir or not os.path.isdir(event_dir):
        raise PreventUpdate

    try:
        df_pos = build_event_positions_df(event_dir)
    except Exception:
        logging.exception("Error building event positions")
        return go.Figure(), [], [], []

    if df_pos.empty:
        fig = go.Figure()
        fig.update_layout(title="Sin datos de posiciones para este evento")
        return fig, [], [], []

    team_opts = sorted(df_pos["team"].dropna().unique().tolist()) if "team" in df_pos.columns else []
    drv_opts  = sorted(df_pos["number"].dropna().unique().tolist())

    if mode == "bestlap":
        fig = build_event_bestlap_fig(event_dir, team_filter, drv_filter)
    elif mode == "topspeed":
        fig = build_event_topspeed_fig(event_dir, team_filter, drv_filter)
    else:
        fig = build_event_positions_fig(df_pos, team_filter, drv_filter)


    table_data = df_pos.sort_values(["session", "position"]).to_dict("records")

    return (
        fig,
        table_data,
        [{"label": t, "value": t} for t in team_opts],
        [{"label": d, "value": d} for d in drv_opts],
    )

@app.callback(
    Output("event_positions_fig", "figure", allow_duplicate=True),
    Input("event_pos_team_filter", "value"),
    Input("event_pos_driver_filter", "value"),
    Input("event_pos_mode", "value"),       # ← NUEVO
    State("event_dir", "value"),
    prevent_initial_call=True,
)

def _update_event_positions_on_filters(team_filter, drv_filter, mode, event_dir):
    if not event_dir or not os.path.isdir(event_dir):
        raise PreventUpdate

    if mode == "topspeed":
        # Top speed → tiramos directamente de los análisis por sesión
        fig = build_event_topspeed_fig(event_dir, team_filter, drv_filter)
    elif mode == "bestlap":
        # Best lap → también desde los análisis por sesión
        fig = build_event_bestlap_fig(event_dir, team_filter, drv_filter)
    else:
        # Posición → usamos el df_pos clásico
        df_pos = build_event_positions_df(event_dir)
        fig = build_event_positions_fig(df_pos, team_filter, drv_filter)

    return fig

@app.callback(
    Output("download-report", "data"),
    Input("export-btn", "n_clicks"),
    State("session-data", "data"),
    prevent_initial_call=True,
)

def on_export(n_clicks, session_data):
    data = session_data
    if not data:
        raise PreventUpdate


    # ----- reconstruir las figuras -------------------------------------------------
    figs_dict = data.get("figs", data)
    figs = {
        name: go.Figure(fig) if isinstance(fig, dict) else fig
        for name, fig in figs_dict.items()
    }

    # ----- reconstruir driver_tables (dict {number: DataFrame}) --------------------
    table_data = data.get("tables") 
    driver_tables = (
        {drv: pd.DataFrame(rows) for drv, rows in table_data.items()}
        if table_data else None
    )
    fast_table_data = data.get("fast_table")
    fast_table = pd.DataFrame(fast_table_data) if fast_table_data else None

    # ----- exportar el informe -----------------------------------------------------
    path = "session_report.html"
    export_report(figs, path, driver_tables, fast_table)

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    os.remove(path)

    return dict(content=content, filename="session_report.html")

def parse_args():
    parser = argparse.ArgumentParser(description="Session Analysis Dashboard")
    parser.add_argument(
        "--host",
        default=os.environ.get("DASH_HOST", "127.0.0.1"),
        help="Host donde se servirá el Dash (p.ej. 0.0.0.0 o 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("DASH_PORT", 8053)),
        help="Puerto en el que escuchará el Dash",
    )
    return parser.parse_args()

def run_app(host: str, port: int):
    """Lanza el Dash y abre el navegador en la URL correcta."""
    url = f"http://{host}:{port}/"
    webbrowser.open_new(url)
    # Nota: puedes usar app.run_server si prefieres. Aquí usamos Flask.run para seguir tu estilo.
    app.run(host=host, port=port, debug=False, threaded=False, use_reloader=False)

if __name__ == "__main__":
    args = parse_args()
    run_app(args.host, args.port)