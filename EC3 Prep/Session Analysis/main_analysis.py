import os
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.express as px
from dash import dash_table  
import plotly.graph_objects as go
from plotly.offline import plot
import tkinter as tk
from tkinter import filedialog
import webbrowser
import logging
import random 
from collections import OrderedDict

logging.basicConfig(level=logging.INFO)

from data_process import (
    load_session_data,
    export_raw_session,
    unify_timestamps,
    convert_time_column,
    seconds_to_mmss
)
from KPI_builder import (
    compute_top_speeds,
    track_limit_rate,
    team_ranking,
    best_sector_times,
    ideal_lap_gap,
    lap_time_history,
    pit_stop_summary,
    lap_time_consistency,
    extract_session_summary,
    slipstream_stats,
    sector_slipstream_stats,
    build_driver_tables,
)


def make_gap_table(df_tbl, driver, include_sector_gaps=False):
    """Devuelve un Div con título + DataTable coloreado según GapAhead."""
    df_tbl = df_tbl.copy()

    time_cols = [c for c in df_tbl.columns if c.startswith("Sector") or c == "LapTime"]
    for col in time_cols:
        df_tbl[col] = df_tbl[col].apply(seconds_to_mmss)

    gap_cols = ["GapAhead"]
    if include_sector_gaps:
        gap_cols += ["GapAhead_S1", "GapAhead_S2", "GapAhead_S3"]

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

    return html.Div(
        children=[
            html.H4(driver),
            dash_table.DataTable(
                data=df_tbl.to_dict("records"),
                columns=[{"name": c, "id": c} for c in df_tbl.columns],
                sort_action="native",
                style_data_conditional=style_cond,
                style_cell={"textAlign": "center"},
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
        (c for c in ["driver_name", "driver_shortname", "driver_number"] if c in df_analysis.columns),
        None,
    )
    if not driver_col:
        raise KeyError("No se encontró columna de piloto en Analysis")
    df_analysis = df_analysis.copy()
    df_analysis["driver"] = df_analysis[driver_col]
    df_analysis = convert_time_column(df_analysis, "lap_time")

    if not df_class.empty:
        class_driver = next(
            (c for c in ["driver_name", "driver_shortname", "driver_number"] if c in df_class.columns),
            None,
        )
        if class_driver:
            df_class = df_class.copy()
            df_class["driver"] = df_class[class_driver]

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
    tuple[dict, OrderedDict]
        The figures dictionary and a mapping of driver names to tables.
    """

    if teams:
        df_analysis = df_analysis[df_analysis["team"].isin(teams)]
    figs = {}
    driver_tables = build_driver_tables(
        df_analysis,
        teams=teams,
        filter_fast=filter_fast,
        include_sectors=include_sectors,
        include_sector_gaps=include_sector_gaps,
    )
    # Top Speeds for each driver & generar colores por piloto
    ts = compute_top_speeds(df_analysis)
    # Mapa de colores por equipo (igual que Team Ranking)
    team_colors = {}
    for team in ts['team'].unique():
        if team == 'Campos Racing':
            team_colors[team] = '#FF5733'
        elif team == 'Griffin Core':
            team_colors[team] = '#33C1FF'
        else:
            team_colors[team] = f'#{random.randint(0, 0xFFFFFF):06x}'

    ss = slipstream_stats(df_analysis)
    ss_sector = sector_slipstream_stats(df_analysis)

    if not ss.empty:
        # 1) Lap-time mínimo (orden asc.)
        ss_lap   = ss.sort_values("min_lap_time_with_slip")
        orderLap = ss_lap["driver"].tolist()
        fig_slip_lap = px.bar(
            ss_lap, x="driver",
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
        orderSpd = ss_spd["driver"].tolist()
        fig_slip_speed = px.bar(
            ss_spd, x="driver",
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
        order_s1 = s1["driver"].tolist()
        fig_s1 = px.bar(
            s1,
            x="driver",
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
        order_s2 = s2["driver"].tolist()
        fig_s2 = px.bar(
            s2,
            x="driver",
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
               
            


    # --- función auxiliar rápida para marcar las vueltas --------------------
    def _flag_slip(df):
        df = df.sort_values("hour")
        best = df["lap_time"].min()
        df["fast"] = df["lap_time"] <= 1.10 * best

        delta = pd.to_datetime(df["hour"], format="%H:%M:%S.%f").diff().dt.total_seconds()
        prev_fast = df["fast"].shift()

        df["slip_flag"] = (
            df["fast"]
            & prev_fast
            & delta.between(0.4, 2.5)
            & (df["top_speed"] >= df["top_speed"].median() + 6)
        )

        # Propaga al lap siguiente
        df["slipstream"] = (
            df["slip_flag"]
            | df.groupby("driver")["slip_flag"].shift(fill_value=False)
        )
        return df

    # ① agrupa y saca la punta máxima con y sin rebufo
    ts_split = (
        df_tmp
        .groupby(["driver", "slipstream"])["top_speed"]
        .max()
        .reset_index()
    )

    ts_split["rebufo"] = ts_split["slipstream"].map(
        {True: "Con rebufo", False: "Sin rebufo"}
    )

    # Gráfico Top Speeds con colores por piloto
    fig_ts = px.bar(
        ts,
        x="driver", y="max_top_speed",
        color="team", color_discrete_map=team_colors,
        title="Top Speeds (ordenado desc)"
    )
    fig_ts.update_layout(
        xaxis={'categoryorder':'array','categoryarray':ts['driver'].tolist()}
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
            x="driver",
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
        drivers = ig_sorted['driver'].tolist()
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
            hist_df.sort_values([lap_col, 'driver']),
            x=lap_col, y='lap_time',
            color='team', line_group='driver',
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
            # 1) DataFrame con driver, team y mejor tiempo de sector
            df_sec = bst[['driver', 'team', sec]].copy()

            # 2) Calculamos la diferencia vs el tiempo mínimo (0.0 para el más rápido)
            best_time = df_sec[sec].min()
            df_sec['diff'] = df_sec[sec] - best_time

            # 3) Orden ascendente (0.0 primero)
            df_sec = df_sec.sort_values('diff', ascending=True).reset_index(drop=True)

            # 4) Gráfico de barras coloreado por equipo
            fig = px.bar(
                df_sec,
                x='driver',
                y='diff',
                color='team',
                color_discrete_map=team_colors,
                title=f"Diferencia en {sec.upper()} vs mejor"
            )
            # Forzamos el orden de los pilotos
            fig.update_layout(
                xaxis={
                    'categoryorder': 'array',
                    'categoryarray': df_sec['driver'].tolist()
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

    pit_df = pit_stop_summary(df_analysis)
    if not pit_df.empty:
        pit_df = pit_df.sort_values('best_pit_time', ascending=True)
        drivers = pit_df['driver'].tolist()
        if 'team' in pit_df.columns:
            colors = [team_colors.get(t, '#333333') for t in pit_df['team']]
        else:
            colors = ['#333333'] * len(pit_df)
        fig_pit = go.Figure()
        fig_pit.add_trace(go.Bar(x=drivers, y=pit_df['best_pit_time'], marker_color=colors, name='Mejor Parada'))
        fig_pit.add_trace(go.Bar(x=drivers, y=pit_df['mean_pit_time'], marker_color=colors, opacity=0.6, name='Media Paradas'))
        fig_pit.update_layout(
            barmode='group',
            title='Pit Stop Summary',
            xaxis={'categoryorder':'array','categoryarray':drivers},
            yaxis_title='Tiempo (s)'
        )
        figs['Pit Stop Summary'] = fig_pit

    cons_df = lap_time_consistency(df_analysis)
    if not cons_df.empty:
        cons_df = cons_df.sort_values('lap_time_std', ascending=True)
        fig_cons = px.bar(
            cons_df,
            x='driver', y='lap_time_std',
            color='team' if 'team' in cons_df.columns else None,
            color_discrete_map=team_colors,
            title='Lap Time Consistency'
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

    return figs, driver_tables

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
):
    """
    Escribe un HTML con una tabla (opcional) al principio y luego las figuras.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write("<html><head>\n<meta charset='utf-8'>\n")
        f.write("<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>\n")
        f.write("<title>Session Report</title></head><body>\n")

        if driver_tables:
            # --- encabezado + CSS de rejilla 3xN -----------------------------
            f.write("""
            <h2>Vueltas – Griffin Core & Campos Racing</h2>
            <style>
            .driver-grid  { display:flex; flex-wrap:wrap; gap:12px; }
            .driver-item  { flex:1 1 calc(33.333% - 12px); box-sizing:border-box; }
            .driver-item h3 { text-align:center; margin:4px 0; }
            .driver-item table { width:100%; border-collapse:collapse; }
            </style>
            <div class="driver-grid">
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
                f.write(f"<div class='driver-item'><h3>{drv}</h3>")
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

app = dash.Dash(__name__)

app.layout = html.Div(
    [
        html.H1("Session Analysis Dashboard"),
        dcc.Input(id="folder-input", type="text", placeholder="Session folder"),
        html.Button("Browse", id="browse-btn"),
        html.Button("Load Session", id="load-btn"),
        html.Button("Export Report", id="export-btn"),
        dcc.Download(id="download-report"),
        dcc.Store(id="fig-store"),
        dcc.Dropdown(id="team-filter", multi=True),
        dcc.Dropdown(
            id="driver-filter",
            multi=True,
            placeholder="Selecciona pilotos",   # se rellenará tras cargar la sesión
        ),
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
        html.Div(id="graphs-container"),
    ]
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
    Output("graphs-container", "children"),
    Output("fig-store", "data"),
    Output("team-filter", "options"),
    Output("team-filter", "value"),
    Output("driver-filter", "options"),   
    Output("driver-filter", "value"),     
    Input("load-btn", "n_clicks"),
    Input("team-filter", "value"),
    State("folder-input", "value"),
    Input("lap-filter-toggle", "value"),
    Input("sector-toggle", "value"),
    Input("gap-toggle", "value"),
    Input("driver-filter", "value"),
)

def on_load(n_clicks, teams, folder, lap_toggle, sec_toggle, gap_toggle, drivers):
    if not n_clicks or not folder or not os.path.isdir(folder):
        raise PreventUpdate
    df_analysis, df_class, weather_df, tracklimits_df = load_data(folder)
    # 0) Extraemos la base “raw” en Excel
    raw_path = os.path.join(folder, "session_raw_data.xlsx")
    export_raw_session(folder, raw_path)
    logging.info(f"Raw data exported to {raw_path}")
    team_names   = sorted(df_analysis["team"].dropna().unique().tolist())
    driver_names = sorted(df_analysis["driver"].dropna().unique().tolist())
    if not teams:
        teams = team_names

    if not drivers:                      
        drivers = driver_names

    df_analysis = df_analysis[df_analysis["driver"].isin(drivers)]
    df_analysis = df_analysis[df_analysis["team"].isin(teams)]

    filter_fast   = "ALL" not in (lap_toggle or [])
    include_sects = "SECT" in (sec_toggle or [])
    include_gap   = "GAP" in (gap_toggle or [])
    figs, driver_tables = build_figures(
        df_analysis, df_class, weather_df, tracklimits_df, teams,
        filter_fast=filter_fast,
        include_sectors=include_sects,
        include_sector_gaps=include_gap,
    )

    table_divs = [
        make_gap_table(tbl, drv, include_gap) for drv, tbl in driver_tables.items()
    ]

    graphs = [html.Div(table_divs, style={"display": "flex", "flexWrap": "wrap"})]
    graphs += [
        element
        for name, fig in figs.items()
        for element in (html.H3(name), dcc.Graph(figure=fig))
    ]

    serialized = {
        "figs": {name: fig.to_dict() for name, fig in figs.items()},
        "tables": {drv: tbl.to_dict(orient="records") for drv, tbl in driver_tables.items()},
    }
    team_opts   = [{"label": t, "value": t} for t in team_names]
    driver_opts = [{"label": d, "value": d} for d in driver_names]

    return html.Div(graphs), serialized, team_opts, teams, driver_opts, drivers

@app.callback(
    Output("download-report", "data"),
    Input("export-btn", "n_clicks"),
    State("fig-store", "data"),
    prevent_initial_call=True,
)

def on_export(n_clicks, data):
    if not data:
        raise PreventUpdate

    # ----- reconstruir las figuras -------------------------------------------------
    figs_dict = data.get("figs", data)
    figs = {
        name: go.Figure(fig) if isinstance(fig, dict) else fig
        for name, fig in figs_dict.items()
    }

    # ----- reconstruir driver_tables (dict {driver: DataFrame}) --------------------
    table_data = data.get("tables")          # llega como JSON-serializable
    driver_tables = (
        {drv: pd.DataFrame(rows) for drv, rows in table_data.items()}
        if table_data else None
    )

    # ----- exportar el informe -----------------------------------------------------
    path = "session_report.html"
    export_report(figs, path, driver_tables)

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    os.remove(path)

    return dict(content=content, filename="session_report.html")

def run_app():
    """Launch the Dash application and open it in the browser."""
    webbrowser.open_new("http://127.0.0.1:8050/")
    app.run(debug=False, threaded=False, use_reloader=False)

if __name__ == "__main__":
    run_app()