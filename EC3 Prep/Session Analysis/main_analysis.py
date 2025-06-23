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


from session_io import load_session_data
from data_process import unify_timestamps
from KPI_builder import (
    compute_top_speeds,
    lap_time_histogram,
    pace_delta,
    position_trace,
    sector_comparison,
    gap_matrix,
    climate_impact,
    track_limits_incidents,
    top_speed_locations,
    stint_boxplots,
    team_ranking,
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

    if not df_class.empty:
        class_driver = next(
            (c for c in ["driver_name", "driver_shortname", "driver_number"] if c in df_class.columns),
            None,
        )
        if class_driver:
            df_class = df_class.copy()
            df_class["driver"] = df_class[class_driver]

    return df_analysis, df_class, weather_df, tracklimits_df

def build_figures(df_analysis, df_class, weather_df, tracklimits_df):
    """Compute KPI figures using Plotly."""
    figs = {}
    try:
        ts = compute_top_speeds(df_analysis)
        figs["Top Speeds"] = px.bar(
            ts,
            x="driver",
            y="max_top_speed",
            color="team",
            color_discrete_map={
                "Campos Racing": "#custom1",
                "Griffin Core": "#custom2",
            },
        )
    except Exception:
        pass
    try:
        laps = lap_time_histogram(df_analysis)
        fig_lt = px.scatter(laps, x="lap", y="lap_time", color="driver", title="Lap Times")
        fig_lt.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
        figs["Lap Times"] = fig_lt
    except Exception:
        pass
    try:
        delta = pace_delta(df_analysis, df_analysis["driver"].iloc[0])
        figs["Pace Delta"] = px.line(delta, x="lap", y="delta", color="driver", title="Pace Delta")
    except Exception:
        pass
    try:
        if not df_class.empty:
            pos = position_trace(df_class)
            figs["Position Trace"] = px.line(pos, title="Position Trace")
    except Exception:
        pass
    try:
        sec_df = sector_comparison(df_analysis)
        sector_cols = [c for c in sec_df.columns if c.startswith("sector") and not c.endswith("_rank")]
        for col in sector_cols:
            df_sorted = sec_df.sort_values(col)
            figs[f"{col.capitalize()} Times"] = px.bar(df_sorted, x="driver", y=col, title=f"{col.capitalize()} Times")
    except Exception:
        pass
    try:
        gap = gap_matrix(df_analysis)
        figs["Gap Matrix"] = px.imshow(gap, text_auto=True, title="Gap Matrix")
    except Exception:
        pass
    try:
        if not weather_df.empty:
            clim = climate_impact(df_analysis, weather_df)
            figs["Climate Impact"] = px.scatter(clim["data"], x="temperature", y="lap_time", title="Climate Impact")
    except Exception:
        pass
    try:
        if not tracklimits_df.empty:
            inc = track_limits_incidents(tracklimits_df)
            figs["Track Limits"] = px.bar(inc, x="driver", y="incident", title="Track Limits Incidents")
    except Exception:
        pass
    try:
        loc = top_speed_locations(df_analysis)
        figs["Top Speed Locations"] = px.scatter(loc, x="track_pos", y="top_speed", color="driver", title="Top Speed Locations")
    except Exception:
        pass
    try:
        stint = stint_boxplots(df_analysis)
        figs["Stint Boxplot"] = px.box(stint, x="stint", y="lap_time", color="driver", title="Lap Time by Stint")
    except Exception:
        pass
    try:
        team = team_ranking(df_analysis)
        if not team.empty:
            figs["Team Ranking"] = px.bar(
                team,
                x="team",
                y="mean_top_speed",
                title="Team Average Top Speed",
            )
    except Exception:
        pass
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
    Input("load-btn", "n_clicks"),
    State("folder-input", "value"),
)
def on_load(n_clicks, folder):
    if not n_clicks or not folder or not os.path.isdir(folder):
        raise PreventUpdate
    df_analysis, df_class, weather_df, tracklimits_df = load_data(folder)
    figs = build_figures(df_analysis, df_class, weather_df, tracklimits_df)
    tabs = dcc.Tabs([
        dcc.Tab(dcc.Graph(figure=fig), label=name)
        for name, fig in figs.items()
    ])
    serialized = {name: fig.to_dict() for name, fig in figs.items()}
    return tabs, serialized

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
    app.run(debug=False)

if __name__ == "__main__":
    run_app()