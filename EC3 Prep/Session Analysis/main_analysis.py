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
        "GriffinCore": "#FFFF00",
        "Griffin Core": "#FFFF00",
        "Campos Racing": "#FFFF00",

        # Amarillo
        "Campos Racing": "#FFFF00",
        # Rojo
        "Drivex": "#FF0000",
        # Verde
        "GRS Team": "#008000",
        # Gris claro
        "Allay Racing": "#00FA2A",
        # Naranja
        "MP Motorsport": "#FE9900",
        "KCL MP Motorsport": "#FE9900",
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

from comparison_utils import build_comparison_figures
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

def load_data(folder: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and preprocess session CSV files."""
    data = load_session_data(folder)
    for key, df in data.items():
        data[key] = unify_timestamps(df, "time")

    analysis_key = next((k for k in data if "analysis" in k.lower()), None)
    class_key = next((k for k in data if "classification" in k.lower()), None)
    weather_key = next((k for k in data if "weather" in k.lower()), None)
    tracklimits_key = next((k for k in data if "tracklimits" in k.lower()), None)

    df_analysis = data.get(analysis_key, list(data.values())[0])
    df_class = data.get(class_key, pd.DataFrame())
    weather_df = data.get(weather_key, pd.DataFrame())
    tracklimits_df = data.get(tracklimits_key, pd.DataFrame())

    driver_col = next(
        c for c in ["driver_shortname", "driver_name", "driver_number"]
        if c in df_analysis.columns
    )

    # (1) identificador único: apellido + #dorsal
    if "number" in df_analysis.columns:
        df_analysis["number"] = (
            df_analysis[driver_col].astype(str).str.strip()
            + " #" + df_analysis["number"].astype(str)
        )
    else:
        df_analysis["number"] = df_analysis[driver_col]

    # (2) columna legible para mostrar
    df_analysis["driver"] = df_analysis[driver_col]
    df_analysis = convert_time_column(df_analysis, "lap_time")
    df_analysis = compute_gap_columns(df_analysis)

    if not df_class.empty:
        class_driver = next(
            (c for c in ["driver_name", "driver_shortname", "driver_number"]
            if c in df_class.columns),
            None,
        )
        if class_driver:
            df_class = df_class.copy()

            # 1) localizar columna con el dorsal ──> número único
            dorsal_col = next(
                (c for c in ["number", "dorsal", "no", "car_number", "driver_number"]
                if c in df_class.columns),
                None,
            )

            if dorsal_col:
                # "Smith #45", "García #21", …
                df_class["number"] = (
                    df_class[class_driver].astype(str).str.strip()
                    + " #" + df_class[dorsal_col].astype(str)
                )
            else:
                # no hay dorsal → al menos usa el nombre corto
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

    if not ss.empty or not ss_sector.empty:
        df_tmp = df_analysis.copy()
        df_tmp['slipstream'] = False


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
               
            
    # --- función auxiliar rápida para marcar las vueltas --------------------
    def _flag_slip(df):
        df = df.sort_values("hour")
        best = df["lap_time"].min()
        df["competitive"] = df["lap_time"] <= 1.05 * best

        delta = (
            pd.to_datetime(
                df["hour"], format="%H:%M:%S.%f", errors="coerce"
            )
            .diff()
            .dt.total_seconds()
        )

        df["slip_flag"] = (
            df["competitive"]
            & delta.between(0.4, 2.5)
            & (df["top_speed"] >= df["top_speed"].median() + 6)
        )

        # Propaga al lap siguiente
        df["slipstream"] = (
            df["slip_flag"]
            | df.groupby("number")["slip_flag"].shift(fill_value=False)
        )
        return df

    # ① agrupa y saca la punta máxima con y sin rebufo
    ts_split = (
        df_tmp
        .groupby(["number", "slipstream"])["top_speed"]
        .max()
        .reset_index()
    )

    ts_split["rebufo"] = ts_split["slipstream"].map(
        {True: "Con rebufo", False: "Sin rebufo"}
    )

    # Gráfico Top Speeds con colores por piloto
    fig_ts = px.bar(
        ts,
        x="number", y="max_top_speed",
        color="team", color_discrete_map=team_colors,
        title="Top Speeds (ordenado desc)"
    )
    fig_ts.update_layout(
        xaxis={'categoryorder':'array','categoryarray':ts['number'].tolist()}
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
        ig_sorted = ig.sort_values('best_lap', ascending=True).reset_index(drop=True)
        drivers = ig_sorted['number'].tolist()
        cols = [team_colors[t] for t in ig_sorted['team']]

        fig_gap = go.Figure()
        fig_gap.add_trace(go.Bar(
            x=drivers, y=ig_sorted['ideal_time'],
            marker_color=cols, name='Vuelta Ideal'
        ))
        fig_gap.add_trace(go.Scatter(
            x=drivers, y=ig_sorted['best_lap'],
            mode='markers', marker=dict(size=10, color=cols),
            name='Mejor Vuelta Real'
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
            xaxis={'categoryorder':'array','categoryarray':drivers}
        )
        # Límite dinámico Y usando tiempos ideal y real
        ymin = min(ig_sorted['ideal_time'].min(), ig_sorted['best_lap'].min())
        ymax = max(ig_sorted['ideal_time'].max(), ig_sorted['best_lap'].max())
        delta = ymax - ymin
        fig_gap.update_layout(
            yaxis=dict(
                range=[ymin - 0.05*delta, ymax + 0.10*delta],
                title="Tiempo (s)"
            )
        )
        figs["Gap a Vuelta Ideal"] = fig_gap

    hist_df = lap_time_history(df_analysis)
    if not hist_df.empty:
        lap_col = 'lap_number' if 'lap_number' in hist_df.columns else 'lap'
        fig_hist = px.line(
            hist_df.sort_values([lap_col, 'number']),
            x=lap_col, y='lap_time',
            color='team', line_group='number',
            color_discrete_map=team_colors,
            title="Histórico de tiempos por vuelta",
        )
        fig_hist.update_layout(
            xaxis_title="Vuelta",
            yaxis_title="Tiempo (s)",
            legend_title="Piloto",
        )
        # Ajuste dinámico de eje Y (–5% … +10%)
        ymin, ymax = hist_df['lap_time'].min(), hist_df['lap_time'].max()
        delta = ymax - ymin
        fig_hist.update_layout(
            yaxis=dict(
                range=[ymin - 0.05 * delta, ymax + 0.10 * delta]
            )
        )
        figs["Lap Time History"] = fig_hist

    bst = best_sector_times(df_analysis)
    if not bst.empty:
        for sec in ['sector1', 'sector2', 'sector3']:
            # 1) DataFrame con number, team y mejor tiempo de sector
            df_sec = bst[['number', 'team', sec]].copy()

            # 2) Calculamos la diferencia vs el tiempo mínimo (0.0 para el más rápido)
            best_time = df_sec[sec].min()
            df_sec['diff'] = df_sec[sec] - best_time

            # 3) Orden ascendente (0.0 primero)
            df_sec = df_sec.sort_values('diff', ascending=True).reset_index(drop=True)

            # 4) Gráfico de barras coloreado por equipo
            fig = px.bar(
                df_sec,
                x='number',
                y='diff',
                color='team',
                color_discrete_map=team_colors,
                title=f"Diferencia en {sec.upper()} vs mejor"
            )
            # Forzamos el orden de los pilotos
            fig.update_layout(
                xaxis={
                    'categoryorder': 'array',
                    'categoryarray': df_sec['number'].tolist()
                }
            )

            # 5) Ajustamos el rango Y (–5% … +10%)
            ymin, ymax = df_sec['diff'].min(), df_sec['diff'].max()
            delta = ymax - ymin
            fig.update_layout(
                yaxis=dict(
                    range=[ymin - 0.05 * delta, ymax + 0.10 * delta],
                    title="Diferencia (s)"
                )
            )

            # 6) Añadimos la figura al diccionario
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

    # --- Wind + Track-Temp plots ----------------------------------------------
    ts_cols = {"time_utc", "time_local", "time", "time_utc_str"}   # posibles timestamps
    if (not weather_df.empty
            and "wind_direction" in weather_df.columns
            and len(weather_df.columns.intersection(ts_cols)) > 0):

        # ① Limpieza rápida -----------------------------------------------------
        wdf = weather_df.copy()
        wdf["wind_direction"] = (
            wdf["wind_direction"].astype(str)
                .str.replace(r"[^0-9,.-]", "", regex=True)
                .str.replace(",", ".")
                .astype(float)
        )

        speed_col = next(
            (c for c in ["wind_speed_kph", "wind_speed_kmh",
                        "wind_speed", "wind_velocity", "wind_speed_ms"]
            if c in wdf.columns), None
        )
        if speed_col:
            wdf[speed_col] = (
                wdf[speed_col].astype(str)
                    .str.replace(r"[^0-9,.-]", "", regex=True)
                    .str.replace(",", ".")
                    .astype(float)
            )

        temp_col = next(
            (c for c in ["track_temp", "track_temperature",
                        "track_temp_c", "track_temperature_c"]
            if c in wdf.columns), None
        )
        if temp_col:
            wdf[temp_col] = (
                wdf[temp_col].astype(str)
                    .str.replace(r"[^0-9,.-]", "", regex=True)
                    .str.replace(",", ".")
                    .astype(float)
            )

        # ② Timestamp uniforme --------------------------------------------------
        time_col = next((c for c in ts_cols if c in wdf.columns), None)
        wdf["timestamp"] = pd.to_datetime(wdf[time_col], errors="coerce")

        # ③ Alineamos meteo con vueltas -----------------------------------------
        ldf = df_analysis[["lap_number", "lap_time", "hour", "number", "team"]].copy()
        session_day = wdf["timestamp"].dropna().iloc[0].normalize()
        ldf["timestamp"] = (
            pd.to_datetime(session_day.strftime("%Y-%m-%d ") + ldf["hour"],
                        format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
            - pd.Timedelta(hours=2)                 # ajusta si ≠ UTC-2
        )

        merged = (
            pd.merge_asof(ldf.sort_values("timestamp"),
                        wdf.sort_values("timestamp"),
                        on="timestamp",
                        direction="nearest",
                        tolerance=pd.Timedelta("5min"))
            .dropna(subset=["wind_direction", "lap_time"])
        )
        merged["color"] = merged["team"].map(TEAM_COLOR).fillna("#888")

        if not merged.empty and temp_col:
            # ───── Gráfico 1: Track-Temp vs Time ───────────────────────────────
            fig_temp_time = px.line(
                wdf, x="timestamp", y=temp_col,
                title="Track Temperature vs Time",
                labels={temp_col: "Temp (°C)", "timestamp": "Hora"},
            )
            fig_temp_time.update_traces(showlegend=False)

            # ───── Gráfico 2: Wind evolution (dir + speed) ────────────────────
            fig_wind_evo = make_subplots(specs=[[{"secondary_y": True}]])
            fig_wind_evo.add_trace(
                go.Scatter(x=wdf["timestamp"], y=wdf["wind_direction"],
                        mode="lines", name="Dirección (°)"),
                secondary_y=False,
            )
            if speed_col:
                fig_wind_evo.add_trace(
                    go.Scatter(x=wdf["timestamp"], y=wdf[speed_col],
                            mode="lines", name="Velocidad (km/h)"),
                    secondary_y=True,
                )
            fig_wind_evo.update_layout(
                title="Wind Direction & Speed vs Time",
                xaxis_title="Hora", yaxis_title="Dirección (°)",
            )
            if speed_col:
                fig_wind_evo.update_yaxes(title_text="Velocidad", secondary_y=True)

            # ───── Gráfico 3: Lap-Time vs Wind-Dir (eje invertido) ─────────────
            #     ← reemplaza al antiguo Track-Temp independiente
            color_kw = (
                dict(color=speed_col, color_continuous_scale="Viridis")
                if speed_col else {}
            )
            fig_scatter_rot = px.scatter(
                merged, x="lap_time", y="wind_direction",
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
                rows=1, cols=2,
                subplot_titles=("Track Temp vs Time",
                                "Wind Direction & Speed vs Time"),
                specs=[
                    [{"type": "xy"}, {"type": "xy", "secondary_y": True}]
                ],
                column_widths=[0.45, 0.55],
            )

            # Col-1: temperatura
            for tr in fig_temp_time.data:
                combo.add_trace(tr, row=1, col=1)

            # Col-2: viento (dir + vel)
            for tr in fig_wind_evo.data:
                combo.add_trace(
                    tr, row=1, col=2,
                    secondary_y=("Velocidad" in tr.name),
                )

            combo.update_layout(
                height=550,
                title_text="Track Temperature & Wind Evolution",
                legend=dict(x=0.78, y=1.00, yanchor="top", xanchor="left"),
            )
            # ───── Gráfico 3-A: Track-Temp vs Lap-Time ─────────────────────────
            fig_temp_vs_lt = px.scatter(
                merged, x="lap_time", y=temp_col,
                title="Track Temp vs Lap Time",
                labels={"lap_time": "Lap Time (s)", temp_col: "Temp (°C)"}
            )
            fig_temp_vs_lt.update_traces(marker_color=merged["color"], showlegend=False)

            # ───── Gráfico 3-B: Wind-Dir vs Lap-Time (ya creado arriba) ───────
            # (fig_scatter_rot)

            # ───── Figura apilada (2 filas, 1 columna) ────────────────────────
            stack = make_subplots(
                rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                subplot_titles=("Track Temp vs LapTime",
                                "Wind Direction vs LapTime")
            )

            # Row-1 → Track Temp
            for tr in fig_temp_vs_lt.data:
                stack.add_trace(tr, row=1, col=1)

            # Row-2 → Wind Direction
            for tr in fig_scatter_rot.data:
                # mantiene la barra de color (si existe) fuera del área de trazas
                stack.add_trace(tr, row=2, col=1)

            # Etiquetas
            stack.update_yaxes(title_text="Temp (°C)",          row=1, col=1)
            stack.update_yaxes(title_text="Dirección (°)",      row=2, col=1)
            stack.update_xaxes(title_text="Lap Time (s)",       row=2, col=1)
            # (solo la fila inferior muestra el eje-X)

            # Ajuste opcional de la barra de color si hay velocidad
            if speed_col:
                stack.update_layout(coloraxis_colorbar=dict(
                    title="Velocidad (km/h)",
                    y=0.15,           # posición vertical (ajusta si hace falta)
                    len=0.35
                ))

            stack.update_layout(
                height=650,
                title_text="Lap Time vs Track Temp & Wind Direction",
                showlegend=False
            )


            # ───── Registrar figuras en el diccionario ────────────────────────
            figs["Track & Wind Evolution"]   = combo          # nuevo subplot 1×2
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

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div(
    [
        html.H1("Session Analysis Dashboard"),
        dcc.Input(id="folder-input", type="text", placeholder="Session folder"),
        html.Button("Browse", id="browse-btn"),
        dcc.Input(id="folder-input-2", type="text", placeholder="Second session folder"),
        html.Button("Browse 2", id="browse-btn-2"),
        html.Button("Load Session", id="load-btn"),
        html.Button("Export Report", id="export-btn"),
        dcc.Download(id="download-report"),
        dcc.Store(id="data-store"),
        dcc.Store(id="session-data"),
        dcc.Store(id="sessionA-data"),
        dcc.Store(id="sessionB-data"),
        dcc.Store(id="kpi-config", storage_type="session", data=DEFAULT_KPI_VALUES),
        dcc.Checklist(id="lap-filter-toggle",
                  options=[{"label": "Ver todas las vueltas", "value": "ALL"}],
                  value=[], style={"marginTop": "10px"}),
        dcc.Checklist(id="sector-toggle",
                  options=[{"label": "Mostrar sectores", "value": "SECT"}],
                  value=[], style={"marginBottom": "10px"}),
        dcc.Checklist(id="gap-toggle",
                  options=[{"label": "Mostrar gap sectores", "value": "GAP"}],
                  value=[], style={"marginBottom": "10px"}),
        html.Hr(),
        dcc.Tabs(
            id="result-tabs",
            value="tab-tables",
            children=[
                dcc.Tab(label="Tables", value="tab-tables"),
                dcc.Tab(label="Figures", value="tab-figs"),
                dcc.Tab(label="Comparison", value="tab-compare"),
                dcc.Tab(label="Control Panel", value="tab-control",
                        children=[html.Div(id="control-panel-body",
                                           style={"padding": "15px", "maxWidth": "500px"})]),
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
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory()
    root.destroy()
    if not folder:
        raise PreventUpdate
    return folder

@app.callback(
    Output("folder-input-2", "value"),
    Input("browse-btn-2", "n_clicks"),
    prevent_initial_call=True,
)

def on_browse_2(n_clicks):
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory()
    root.destroy()
    if not folder:
        raise PreventUpdate
    return folder

@app.callback(
    Output("tab-content", "children"),
    Output("sessionA-data", "data"),
    Output("sessionB-data", "data"),
    Output("data-store", "data"),
    Output("session-data", "data"),
    Output("team-filter", "options"),
    Output("team-filter", "value"),
    Output("number-filter", "options"),
    Output("number-filter", "value"),
    Input("load-btn", "n_clicks"),
    Input("result-tabs", "value"),
    Input("kpi-config", "data"),
    Input("lap-filter-toggle", "value"),
    Input("sector-toggle", "value"),
    Input("gap-toggle", "value"),
    State("folder-input", "value"),
    State("folder-input-2", "value"),
    State("team-filter", "value"),
    State("number-filter", "value"),
    State("data-store", "data"),
    prevent_initial_call=True,
)

def update_single_tab(n_clicks, tab, kpi_cfg,
                      lap_toggle, sec_toggle, gap_toggle,
                      folderA, folderB, teams, drivers, stored):
    
    ctx = dash.callback_context
    trig = ctx.triggered_id

    if trig == "result-tabs" and stored:
        data = stored
        team_opts = dash.no_update
        driver_opts = dash.no_update
        teams_out = dash.no_update
        drivers_out = dash.no_update
        storeA = dash.no_update
        storeB = dash.no_update
    else:
        if not n_clicks or not folderA or not os.path.isdir(folderA):
            raise PreventUpdate
        df_analysis, df_class, weather_df, tracklimits_df = load_data(folderA)
        raw_path = os.path.join(folderA, "session_raw_data.xlsx")
        export_raw_session(folderA, raw_path)
        logging.info(f"Raw data exported to {raw_path}")

        if folderB and os.path.isdir(folderB):
            dfB_analysis, dfB_class, _, _ = load_data(folderB)
        else:
            dfB_analysis = pd.DataFrame()
            dfB_class = pd.DataFrame()

        storeA = {
            "name": os.path.basename(folderA) or "Session 1", 
            "analysis": df_analysis.to_dict("records"),
            "class": df_class.to_dict("records"),
            "weather": weather_df.to_dict("records"),
            "track": tracklimits_df.to_dict("records"),
        }
        storeB = {
            "name": os.path.basename(folderB) or "Session 2",
            "analysis": dfB_analysis.to_dict("records"),
            "class": dfB_class.to_dict("records"),
            "weather": weather_df.to_dict("records"),
            "track": tracklimits_df.to_dict("records"),
        }

        team_names = sorted(df_analysis["team"].dropna().unique().tolist())
        driver_names = sorted(df_analysis["number"].dropna().unique().tolist())

        if not teams:
            teams = team_names
        if not drivers:
            drivers = driver_names

        df_analysis = df_analysis[df_analysis["number"].isin(drivers)]

        filter_fast = "ALL" not in (lap_toggle or [])
        include_sects = "SECT" in (sec_toggle or [])
        include_gap = "GAP" in (gap_toggle or [])
        cfg = kpi_cfg or {}

        figs, driver_tables, fast_table, grid_order = build_figures(
            df_analysis, df_class, weather_df, tracklimits_df, teams,
            filter_fast=filter_fast,
            include_sectors=include_sects,
            include_sector_gaps=include_gap,
            kpi_params=cfg,
        )

        grid_order = list(driver_tables.keys())
        if not df_class.empty and "position" in df_class.columns:
            grid_order = (
                df_class.sort_values("position")["number"].tolist()
            )

        grid_order = [d for d in grid_order if d in drivers]
        for drv in drivers:
            if drv not in grid_order:
                grid_order.append(drv)

        fast_rows = [] if fast_table is None else fast_table.to_dict(orient="records")
        data = {
            "figs": {name: fig.to_dict() for name, fig in figs.items()},
            "tables": {drv: tbl.to_dict(orient="records") for drv, tbl in driver_tables.items()},
            "fast_table": fast_rows,
            "include_gap": include_gap,
            "grid_order": grid_order,
        }

        team_opts = [{"label": t, "value": t} for t in team_names]
        driver_opts = [{"label": d, "value": d} for d in driver_names]
        teams_out = teams
        drivers_out = drivers

    figs_dict = data.get("figs", {})
    figs = {
        name: go.Figure(fig) if isinstance(fig, dict) else fig
        for name, fig in figs_dict.items()
    }

    tables_dict = data.get("tables", {})
    driver_tables = {drv: pd.DataFrame(rows) for drv, rows in tables_dict.items()}
    fast_rows = data.get("fast_table", [])
    fast_df = pd.DataFrame(fast_rows)
    include_gap = data.get("include_gap", False)

    grid_order = data.get("grid_order", [])
    table_divs = [
        make_gap_table(driver_tables[drv], drv, include_gap)
        for drv in grid_order
        if drv in driver_tables
    ]

    if not fast_df.empty:
        if "BestLap" in fast_df.columns:
            fast_df["BestLap"] = fast_df["BestLap"].apply(seconds_to_mmss)
        team_styles = [
            {
                "if": {
                    "filter_query": f'{{team}} = "{team}"',
                    "column_id": ["number"]             
                },
                "backgroundColor": color,
                "color": "black" if color.lower() in ("#ffff00", "#fe9900") else "white",
            }
            for team, color in TEAM_COLOR.items()
        ]

        gap_cols = ["GapStart"]
        if include_gap:                                   # mismo flag que en las tablas individuales
            gap_cols += ["GapAhead_S1", "GapAhead_S2", "GapAhead_S3"]

        gap_styles = []
        for col in gap_cols:
            gap_styles.extend([
                {   # rojo  ≤ 1 s
                    "if": {"filter_query": f"{{{col}}} <= 1", "column_id": col},
                    "backgroundColor": "rgb(255,102,102)",
                    "color": "white",
                },
                {   # naranja 1–3 s
                    "if": {"filter_query": f"{{{col}}} > 1 && {{{col}}} < 3", "column_id": col},
                    "backgroundColor": "rgb(255,178,102)",
                },
                {   # azul claro 3–5 s
                    "if": {"filter_query": f"{{{col}}} >= 3 && {{{col}}} <= 5", "column_id": col},
                    "backgroundColor": "rgb(102,178,255)",
                },
                {   # azul oscuro  > 5 s   
                    "if": {"filter_query": f"{{{col}}} > 5", "column_id": col},
                    "backgroundColor": "rgb(0,102,204)",
                    "color": "white",
                },
            ])

        summary = dash_table.DataTable(
            data=fast_df.to_dict("records"),
            columns=[{"name": c, "id": c} for c in fast_df.columns],
            sort_action="native",
            style_cell={"textAlign": "center"},
            style_header={"fontWeight": "bold"},
            style_data_conditional=team_styles  + gap_styles    
        )
        table_divs.insert(0, html.Div(summary, style={"flex": "1 0 100%", "padding": "5px"}))

    if tab == "tab-compare":
        content = dash.no_update
    elif tab == "tab-tables":
        content = html.Div(table_divs, style={"display": "flex", "flexWrap": "wrap"})
    elif tab == "tab-figs":
        graphs = []

        sector_keys = ["SECTOR1 Diff", "SECTOR2 Diff", "SECTOR3 Diff"]
        gap_key = "Gap a Vuelta Ideal"

        sector_figs = [
            dcc.Graph(figure=figs[k], style={"flex": 1})
            for k in sector_keys if k in figs
        ]
        if sector_figs:
            graphs.append(html.Div(sector_figs, style={"display": "flex", "gap": "10px"}))

        if gap_key in figs:
            graphs.append(dcc.Graph(figure=figs[gap_key], style={"width": "100%"}))

        remaining = [k for k in figs if k not in sector_keys + [gap_key]]
        if remaining:
            grid_items = [dcc.Graph(figure=figs[k]) for k in remaining]
            graphs.append(
                html.Div(
                    grid_items,
                    style={"display": "grid", "gridTemplateColumns": "repeat(2, 1fr)", "gap": "10px"}
                )
            )

        content = html.Div(graphs)
    elif tab == "tab-control":
        content = html.Div()
    else:
        content = html.Div()

    return (
        content,
        storeA,
        storeB,
        data,
        data,
        team_opts,
        teams_out,
        driver_opts,
        drivers_out,
    )

@app.callback(
    Output("tab-content", "children"),
    Input("result-tabs", "value"),
    State("sessionA-data", "data"),
    State("sessionB-data", "data"),
    State("team-filter", "value"),
    State("number-filter", "value"),
    State("lap-filter-toggle", "value"),
    State("sector-toggle", "value"),
    State("gap-toggle", "value"),
    prevent_initial_call=True,
)

def update_compare_content(tab, dataA, dataB,
                           teams, drivers,
                           lap_toggle, sec_toggle, gap_toggle):
    if tab != "tab-compare":
        raise PreventUpdate

    if not dataA or not dataB:
        return html.Div("Load two sessions to compare")

    def df_from(store, key):
        rows = store.get(key, [])
        return pd.DataFrame(rows)

    dfA = df_from(dataA, "analysis")
    dfB = df_from(dataB, "analysis")
    classA = df_from(dataA, "class")
    classB = df_from(dataB, "class")

    if teams:
        dfA = dfA[dfA["team"].isin(teams)]
        dfB = dfB[dfB["team"].isin(teams)]
    if drivers:
        dfA = dfA[dfA["number"].isin(drivers)]
        dfB = dfB[dfB["number"].isin(drivers)]
    
    sessionA_name = dataA.get("name", "Session 1")
    sessionB_name = dataB.get("name", "Session 2")

    figs = build_comparison_figures(
    dfA, dfB,
    nameA=sessionA_name,          # o simplemente "Q1"
    nameB=sessionB_name,          #            "Q2"
    team_colors=TEAM_COLOR,       # ← aquí
)
    graphs = [dcc.Graph(figure=f) for f in figs.values()]
    return html.Div(graphs, style={
        "display": "grid",
        "gridTemplateColumns": "repeat(2,1fr)",
        "gap": "12px",
    })

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