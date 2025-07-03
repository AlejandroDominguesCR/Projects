import os
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
import tkinter as tk
from tkinter import filedialog
import webbrowser
import logging
import random   
logging.basicConfig(level=logging.INFO)

from data_process import (
    load_session_data,
    export_raw_session,
    unify_timestamps,
    convert_time_column,
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

def build_figures(df_analysis, df_class, weather_df, tracklimits_df, teams=None):
    """Compute a minimal set of KPI figures.

    Parameters
    ----------
    df_analysis : pd.DataFrame
        Analysis dataframe.
    df_class : pd.DataFrame
    weather_df : pd.DataFrame
    tracklimits_df : pd.DataFrame
    teams : list[str] | None
        Optional list of team names to filter ``df_analysis``.
    """

    if teams:
        df_analysis = df_analysis[df_analysis["team"].isin(teams)]
    figs = {}

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
    if not ss.empty:
        # ordenamos por la vuelta media con rebufo (ascendente)
        ss = ss.sort_values("avg_lap_time_with_slip")

        fig_slip_lap = px.bar(
            ss,
            x="driver",                       #  ← antes ponía "number"
            y=["avg_lap_time_no_slip", "avg_lap_time_with_slip"],
            barmode="group",
            title="Lap-time medio – Con vs Sin rebufo",
        )
        figs["Slipstream Lap-time"] = fig

        fig_slip_speed = px.bar(
            ss,
            x="driver",                       #  ← idem
            y=["avg_top_speed_no_slip", "avg_top_speed_with_slip"],
            barmode="group",
            title="Top-speed medio – Con vs Sin rebufo",
        )

        figs["Lap-time medio – con vs sin rebufo"] = fig_slip_lap
        figs["Top-speed medio – con vs sin rebufo"] = fig_slip_speed


    df_tmp = df_analysis.copy()
    df_tmp['slipstream'] = False     

    # --- función auxiliar rápida para marcar las vueltas --------------------
    def _flag_slip(df):
        df = df.sort_values("hour")                  # hora string 'HH:MM:SS.mmm'
        best = df["lap_time"].min()
        df["fast"] = df["lap_time"] <= 1.10 * best
        delta = pd.to_datetime(df["hour"], format="%H:%M:%S.%f").diff().dt.total_seconds()
        prev_fast = df["fast"].shift()
        df.loc[
            df["fast"]
            & prev_fast
            & delta.between(0.4, 2.5)
            & (df["top_speed"] >= df["top_speed"].median() + 6),
            "slipstream"
        ] = True
        return df
    

    df_tmp = df_analysis.copy()
    df_tmp['slipstream'] = False

    # … (define la función _flag_slip y agrupa por driver)
    df_tmp = df_tmp.groupby("driver", group_keys=False).apply(_flag_slip)

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

    fig_ts_split = px.bar(
        ts_split,
        x="driver",
        y="top_speed",
        color="rebufo",
        barmode="group",
        title="Top Speed máxima – Con vs Sin rebufo",
        color_discrete_map={"Con rebufo": "#FF5555", "Sin rebufo": "#AAAAAA"},
    )

    figs["Top Speed (rebufo)"] = fig_ts_split

    df_tmp = df_tmp.groupby("driver", group_keys=False).apply(_flag_slip)

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

    return figs

def export_report(figs: dict, path: str):
    """Export all figures to a single HTML file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("<html><head>")
        f.write('<meta charset="utf-8">')
        f.write('<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>')
        f.write("<title>Session Report</title></head><body>")
        for i, (name, fig) in enumerate(figs.items()):
            f.write(f"<h2>{name}</h2>")
            f.write(plot(fig, include_plotlyjs=i == 0, output_type="div"))
            f.write("<hr>")
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
    Input("load-btn", "n_clicks"),
    Input("team-filter", "value"),
    State("folder-input", "value"),
)
def on_load(n_clicks, teams, folder):
    if not n_clicks or not folder or not os.path.isdir(folder):
        raise PreventUpdate
    df_analysis, df_class, weather_df, tracklimits_df = load_data(folder)
    # 0) Extraemos la base “raw” en Excel
    raw_path = os.path.join(folder, "session_raw_data.xlsx")
    export_raw_session(folder, raw_path)
    logging.info(f"Raw data exported to {raw_path}")
    team_names = sorted(df_analysis["team"].dropna().unique().tolist())
    if not teams:
        teams = team_names
    figs = build_figures(df_analysis, df_class, weather_df, tracklimits_df, teams)
    graphs = [
        element
        for name, fig in figs.items()
        for element in (html.H3(name), dcc.Graph(figure=fig))
    ]
    serialized = {name: fig.to_dict() for name, fig in figs.items()}
    options = [{"label": t, "value": t} for t in team_names]
    return html.Div(graphs), serialized, options, teams

@app.callback(
    Output("download-report", "data"),
    Input("export-btn", "n_clicks"),
    State("fig-store", "data"),
    prevent_initial_call=True,
)

def on_export(n_clicks, data):
    if not data:
        raise PreventUpdate
    figs = {name: go.Figure(fig) if isinstance(fig, dict) else fig for name, fig in data.items()}
    path = "session_report.html"
    export_report(figs, path)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    os.remove(path)
    return dict(content=content, filename="session_report.html")

def run_app():
    # Open dashboard in the default browser. Update URL if the port changes.
    webbrowser.open_new("http://127.0.0.1:8050/")
    app.run(debug=False, threaded=False, use_reloader=False)

if __name__ == "__main__":
    run_app()