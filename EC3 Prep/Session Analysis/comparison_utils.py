import pandas as pd
import plotly.express as px
from KPI_builder import compute_top_speeds
from data_process import parse_time_to_seconds
import plotly.graph_objects as go 

def build_comparison_figures(
    dfA: pd.DataFrame,
    dfB: pd.DataFrame,
    *,
    nameA: str = "Q1",
    nameB: str = "Q2",
    team_colors: dict[str, str] | None = None,   # ← nuevo
) -> dict[str, go.Figure]:

    # paleta por defecto si no se pasa ninguna
    if team_colors is None:
        uniq = pd.concat([dfA["number"], dfB["number"]]).unique()
        team_colors = {n: px.colors.qualitative.Plotly[i % 10]
                       for i, n in enumerate(uniq)}

    # ---------- Fastest-lap Δ ----------
    bestA = dfA.groupby("number")["lap_time"].min()
    bestB = dfB.groupby("number")["lap_time"].min()
    fast  = pd.concat([bestA, bestB], axis=1, keys=[nameA, nameB]).dropna()
    fast["delta"] = fast[nameB] - fast[nameA]

    fig_fast = px.bar(
        fast.reset_index(), x="number", y="delta",
        title=f"Fastest Lap Δ ({nameB} – {nameA})",
        color="number", color_discrete_map=team_colors,
    )

    # ---------- Top-speed Δ ------------
    topsA = compute_top_speeds(dfA).set_index("number")["max_top_speed"]
    topsB = compute_top_speeds(dfB).set_index("number")["max_top_speed"]
    tops  = pd.concat([topsA, topsB], axis=1, keys=[nameA, nameB]).dropna()
    tops["delta"] = tops[nameB] - tops[nameA]

    fig_speed = px.bar(
        tops.reset_index(), x="number", y="delta",
        title=f"Top Speed Δ ({nameB} – {nameA})",
        color="number", color_discrete_map=team_colors,
    )

    # ---------- Distribución de laps ----
    for df in (dfA, dfB):
        if not pd.api.types.is_numeric_dtype(df["lap_time"]):
            df["lap_time"] = df["lap_time"].apply(parse_time_to_seconds)

    dist = (
        pd.concat([
            dfA.assign(session=nameA)[["number", "lap_time", "session"]],
            dfB.assign(session=nameB)[["number", "lap_time", "session"]],
        ])
    )

    fig_violin = px.violin(
        dist, x="number", y="lap_time", color="session",
        box=True, points=False,
        title="Lap-time distribution",
    )

    return {
        "Fastest Lap Δ":      fig_fast,
        "Top Speed Δ":        fig_speed,
        "Lap-time distribution": fig_violin,
    }
