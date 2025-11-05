import plotly.graph_objects as go
import dash
from dash import dcc, html
from threading import Thread
import re, numpy as np
import os
import pandas as pd
from plotly.offline import plot
from scipy.signal import savgol_filter
from datetime import datetime
import plotly.graph_objects as go
from gui_v2 import load_track_channels
from pathlib import Path
from typing import Sequence, Union, List
import json

# ── Definición centralizada de KPIs ───────────────────────────────────────────
KPI_DEFINITIONS = [
    (
        "Wheel Load Max [Kg]",
        "Kg",
        "wheel_load_max",
        "wheel_load_std",
        1,
    ),
]

def kpi_point_with_var(title, unit, mean_values_list, var_values_list,
                        setup_names, kpi_labels=None):
    """Return a KPI scatter plot with variance bars."""
    if kpi_labels is None:
        kpi_labels = ['FL', 'FR', 'RL', 'RR']

    fig = go.Figure()
    for mean_vals, var_vals, name in zip(mean_values_list,
                                         var_values_list,
                                         setup_names):
        if np.isscalar(var_vals):
            var_vals = np.full_like(mean_vals, float(var_vals), dtype=float)
        fig.add_trace(go.Scatter(
            x=kpi_labels,
            y=mean_vals,
            mode='markers',
            name=name,
            marker=dict(size=10, symbol='circle'),
            error_y=dict(type='data', array=var_vals,
                         visible=True, thickness=1.5, width=3),
        ))

    fig.update_layout(
        title=title,
        yaxis_title=unit,
        xaxis_title="Wheel",
        yaxis=dict(zeroline=False, gridcolor='lightgrey'),
        legend_title="Setup",
        
        height=300,
        margin=dict(t=40, b=40, l=40, r=10),
    )
    return fig

# ── Estilos reutilizables para dashboards y reportes ──────────────────────────
GRID_STYLE = {"display": "grid","gridTemplateColumns": "repeat(auto-fit, minmax(768px, 1fr))","gap": "20px",}
CARD_STYLE = {"width": "100%","padding": "6px 8px",}
GRAPH_CFG = {"displayModeBar": False}
STANDARD_HEIGHT = "480px"
FIG_HEIGHT = 480

# ── Estilos CSS para Dash ─────────────────────────────────────────────────────
GRID_CSS  = "display:grid;grid-template-columns:repeat(auto-fit,minmax(768px,1fr));gap:20px"
CARD_CSS  = "width:100%;padding:6px 8px"             # mismo que CARD_STYLE
BODY_CSS  = "font-family:sans-serif;margin:0"
FIG_HEIGHT = 480                                      # Plotly (px)
STYLE_HEIGHT = f"{FIG_HEIGHT}px"                      # CSS

app = dash.Dash(__name__)
app.layout = html.Div([html.H3("Resultados no cargados")])
server = app.server

def _place_legend_outside(fig, side="right"):
    # side = 'right' o 'bottom'
    if side == "right":
        fig.update_layout(
            legend=dict(x=1.02, y=1, yanchor="top", xanchor="left",
                        orientation="v",  # vertical
                        font=dict(size=10)),
            margin=dict(r=90)  # deja un margen para la leyenda
        )
    else:  # bottom horizontal
        fig.update_layout(
            legend=dict(x=0.0, y=-0.25, xanchor="left",
                        orientation="h",
                        font=dict(size=10)),
            margin=dict(b=60)
        )

def smooth_signal(data, window=51, polyorder=3):
    """
    Aplica un filtro Savitzky-Golay a una señal o matriz de señales.
    Si data tiene forma (4, N), suaviza cada rueda.
    """
    if len(data.shape) == 1:
        return savgol_filter(data, window_length=window, polyorder=polyorder)
    else:
        return np.array([savgol_filter(channel, window_length=window, polyorder=polyorder) for channel in data])

def launch_dash(sol, post, setup_name="Setup"):
    def plot_data():
        graphs = []
        distance = np.cumsum(post['vx']) * np.gradient(sol.t)

        spring_travel = (post['z_wheel']) * 1000
        travel = smooth_signal(post['damper_travel'])*1000
        wheel_f = post['f_wheel']          # shape (4, N)
        grip_mask = post['grip_limited_lateral_mask']  # (N,)
        wheel_ld = post['wheel_load']       # (4,N)
        wheel_names = ["FL", "FR", "RL", "RR"]

        #v_damper = smooth_signal(post['v_damper'])
        heave_mm  = post['h']*1000 #(sol.y[0]) * 1000    # pasa a mm
        heave_ax_f = smooth_signal(post['dyn_hF'] * 1000)   # mm
        heave_ax_r = smooth_signal(post['dyn_hR']  * 1000)   # mm
        rh = smooth_signal(post['travel_rel']*1000)   # mm
        heave_filtered = smooth_signal(heave_mm)
        pitch_filtered = smooth_signal(np.degrees(sol.y[2]))
        roll_filtered = smooth_signal(np.degrees(sol.y[4]))
        spring_force_filtered = (post['f_spring'])
        damper_force_filtered = (post['f_damper'])
        arb_force_filtered = (post['f_arb'])
        f_bump_filtered  = (post['f_arb'])

        # Travel absoluto por rueda
        fig = go.Figure([
            go.Scatter(x=distance, y=spring_travel[0], name="Potencia  FL"),
            go.Scatter(x=distance, y=spring_travel[1], name="Potencia FR"),
            go.Scatter(x=distance, y=spring_travel[2], name="Potencia RL"),
            go.Scatter(x=distance, y=spring_travel[3], name="Potencia RR"),
        ])
        fig.update_layout(title="Spring Travel [mm]", xaxis_title="Distance [m]", yaxis_title="Travel [mm]")
        graphs.append(html.Div(
            dcc.Graph(figure=fig, config=GRAPH_CFG, style={"height": STANDARD_HEIGHT}),
            style=CARD_STYLE))
        
                # Travel absoluto por rueda
        fig = go.Figure([
            go.Scatter(x=distance, y=travel[0], name="damper Travel FL"),
            go.Scatter(x=distance, y=travel[1], name="damper Travel FR"),
            go.Scatter(x=distance, y=travel[2], name="damper Travel RL"),
            go.Scatter(x=distance, y=travel[3], name="damper Travel RR"),
        ])
        fig.update_layout(title="Damper Travel [mm]", xaxis_title="Distance [m]", yaxis_title="Travel [mm]")
        graphs.append(html.Div(
            dcc.Graph(figure=fig, config=GRAPH_CFG, style={"height": STANDARD_HEIGHT}),
            style=CARD_STYLE))
        
        # travels (filtrado)
        fig = go.Figure([
            go.Scatter(x=distance, y=rh[0], name="Ride Height FL [mm]"),
            go.Scatter(x=distance, y=rh[1], name="Ride Height FR [mm]"),
            go.Scatter(x=distance, y=rh[2], name="Ride Height RL [mm]"),
            go.Scatter(x=distance, y=rh[3], name="Ride Height RR [mm]"),
        ])
        fig.update_layout(title="RH [mm]", xaxis_title="Distance [m]", yaxis_title="RH [mm]")
        graphs.append(html.Div(
            dcc.Graph(figure=fig, config=GRAPH_CFG, style={"height": STANDARD_HEIGHT}),
            style=CARD_STYLE))
        fig = go.Figure([
            go.Scatter(x=distance, y=heave_ax_f, name="h Front [mm]"),
            go.Scatter(x=distance, y=heave_ax_r, name="h Rear [mm]")
        ])
        fig.update_layout(title="Heave per Axle (ΔRH) [mm]",
                          xaxis_title="Distance [m]",
                          yaxis_title="Heave [mm]")
        graphs.append(html.Div(
            dcc.Graph(figure=fig, config=GRAPH_CFG, style={"height": STANDARD_HEIGHT}),
            style=CARD_STYLE))

        # Pitch (filtrado)
        fig = go.Figure([
            go.Scatter(x=distance, y=heave_filtered, name="Heave [°]")
        ])
        fig.update_layout(title="Pitch Motion (Filtered)", xaxis_title="Distance [m]", yaxis_title="Pitch [°]")
        graphs.append(html.Div(
            dcc.Graph(figure=fig, config=GRAPH_CFG, style={"height": STANDARD_HEIGHT}),
            style=CARD_STYLE))
        
        # Roll (filtrado)
        fig = go.Figure([
            go.Scatter(x=distance, y=roll_filtered, name="Roll [°]")
        ])
        fig.update_layout(title="Roll Motion (Filtered)", xaxis_title="Distance [m]", yaxis_title="Roll [°]")
        graphs.append(html.Div(
            dcc.Graph(figure=fig, config=GRAPH_CFG, style={"height": STANDARD_HEIGHT}),
            style=CARD_STYLE))
        
        # === Zona de grip limitado lateral ===
        grip_limited_trace = None
        if 'grip_limited_lateral_mask' in post:
            mask = post['grip_limited_lateral_mask']
            avg_tire_load = np.mean(wheel_ld, axis=0)
            y_gl = np.where(mask, avg_tire_load, np.nan)
            grip_limited_trace = go.Scatter(
                x=distance, y=y_gl, name="Grip-Limited Zone", mode='lines', line=dict(color='black', width=3, dash='dot'),
                showlegend=True
            )

        # === Aerodynamic Downforce per Axle [N] ===
        ae_front = post.get('Fz_aero_front', np.zeros_like(post['vx']))
        ae_rear  = post.get('Fz_aero_rear',  np.zeros_like(post['vx']))

        fig = go.Figure([
            go.Scatter(x=distance, y=-ae_front, name="Downforce Front"),
            go.Scatter(x=distance, y=-ae_rear,  name="Downforce Rear"),
        ])
        fig.update_layout(
            title="Aerodynamic Downforce per Axle [N]", xaxis_title="Distance [m]", yaxis_title="Force [N]")
        graphs.append(html.Div(
            dcc.Graph(figure=fig, config=GRAPH_CFG, style={"height": STANDARD_HEIGHT}),
            style=CARD_STYLE))

        # Tire load
        fig_load = go.Figure()

        # Curvas de carga por rueda
        for i, name in enumerate(wheel_names):
            fig_load.add_trace(
                go.Scatter(x=distance, y=wheel_ld[i], name=name))

        # Marcadores grip-limited para cada rueda
        for i, name in enumerate(wheel_names):
            fig_load.add_trace(
                go.Scatter(
                    x=distance[grip_mask], y=wheel_ld[i, grip_mask],
                    mode='markers', name=f"{name} Grip-Limited",
                    marker=dict(color='black', symbol='line-ns-open'))
            )

        fig_load.update_layout(
            title="Tire Load per Wheel [N]", xaxis_title="Distance [m]", yaxis_title="Load [N]")
        graphs.append(html.Div(
            dcc.Graph(figure=fig_load, config=GRAPH_CFG, style={"height": STANDARD_HEIGHT}),
            style=CARD_STYLE))
        
        # === Bumpstop Forces por rueda ===
        fig = go.Figure([
            go.Scatter(x=distance, y=post['f_tire'][0], name=" FL"),
            go.Scatter(x=distance, y=post['f_tire'][1], name=" FR"),
            go.Scatter(x=distance, y=post['f_tire'][2], name=" RL"),
            go.Scatter(x=distance, y=post['f_tire'][3], name=" RR"),
        ])
        fig.update_layout(title="Force per Wheel [N]", xaxis_title="Distance [m]", yaxis_title="Force [N]")
        graphs.append(html.Div(
            dcc.Graph(figure=fig, config=GRAPH_CFG, style={"height": STANDARD_HEIGHT}),
            style=CARD_STYLE))

        fig = go.Figure([
            go.Scatter(x=distance, y=spring_force_filtered[0], name="Spring FL"),
            go.Scatter(x=distance, y=spring_force_filtered[1], name="Spring FR"),
            go.Scatter(x=distance, y=spring_force_filtered[2], name="Spring RL"),
            go.Scatter(x=distance, y=spring_force_filtered[3], name="Spring RR"),
        ])
        fig.update_layout(
            title="Spring Force per Wheel [N]",
            xaxis_title="Distance [m]",
            yaxis_title="Force [N]"
        )
        graphs.append(html.Div(
            dcc.Graph(figure=fig, config=GRAPH_CFG, style={"height": STANDARD_HEIGHT}),
            style=CARD_STYLE))

        fig = go.Figure([
            go.Scatter(x=distance, y=f_bump_filtered[0], name="dyn FL"),
            go.Scatter(x=distance, y=f_bump_filtered[1], name="dyn FR"),
            go.Scatter(x=distance, y=f_bump_filtered[2], name="dyn RL"),
            go.Scatter(x=distance, y=f_bump_filtered[3], name="dyn RR"),
        ])
        fig.update_layout(
            title="Damper Force per Wheel [N]",
            xaxis_title="Distance [m]",
            yaxis_title="Force [N]"
        )
        graphs.append(html.Div(
            dcc.Graph(figure=fig, config=GRAPH_CFG, style={"height": STANDARD_HEIGHT}),
            style=CARD_STYLE))

        app.layout = html.Div([
            html.H1(f"Resultados 7-Post Rig: {setup_name}"),
            html.Div(graphs, style=GRID_STYLE)
        ])

    plot_data()
    app.run(port=8050, debug=False)

# === Radar KPIs (5 métricas) con rangos fijos ==================================
RADAR_METRICS = [
    ("FRH RMS [mm]",        "frh_rms",        False,  1000.0),
    ("RRH RMS [mm]",        "rrh_rms",        False,  1000.0),
    ("Pitch RMS [°]",       "pitch_rms",      False,     1.0),
    ("Front Load RMS [N]",  "front_load_rms", False,    1.0),
    ("Rear Load RMS [N]",   "rear_load_rms",  False,    1.0),
]

# rangos "fijos" en UNIDADES ya escaladas por 'scale' (mm, °, N)
RADAR_FIXED_RANGES = {
    "frh_rms":        (0.0, 20.0),   # mm  
    "rrh_rms":        (0.0, 20.0),   # mm
    "pitch_rms":      (0.0, 0.35),   # °   
    "front_load_rms": (0.0, 5000), # N   
    "rear_load_rms":  (0.0, 18000), # N
}

def build_kpi_radar(
    kpi_data,
    labels,
    metrics=RADAR_METRICS,
    normalize="fixed",           # "dataset" | "fixed" | False
    fixed_ranges=RADAR_FIXED_RANGES
):

    theta = [m[0] for m in metrics]

    # matriz raw[M métricas, K setups] en unidades finales (mm,°,N)
    raw = np.array([
        [float(k.get(key, np.nan)) * scale for k in kpi_data]
        for (_lbl, key, _inv, scale) in metrics
    ], dtype=float)

    if normalize == "dataset":
        mins = np.nanmin(raw, axis=1)
        maxs = np.nanmax(raw, axis=1)
        span = np.where((maxs - mins) == 0.0, 1.0, (maxs - mins))
        vals = (raw - mins[:, None]) / span[:, None]
        title_suffix = " (normalised to dataset, higher is better)"
        radial_range = [0, 1]

    elif normalize == "fixed":
        # rangos base por métrica
        mins = np.array([fixed_ranges[m[1]][0] for m in metrics], dtype=float)
        maxs = np.array([fixed_ranges[m[1]][1] for m in metrics], dtype=float)

        # auto-expandir si hay datos fuera de rango (padding 5%)
        data_min = np.nanmin(raw, axis=1)
        data_max = np.nanmax(raw, axis=1)
        mins = np.minimum(mins, data_min * 0.95)
        maxs = np.maximum(maxs, data_max * 1.05)

        span = np.where((maxs - mins) == 0.0, 1.0, (maxs - mins))
        vals = (raw - mins[:, None]) / span[:, None]
        vals = np.clip(vals, 0.0, 1.0)

        title_suffix = ""
        radial_range = [0, 1]

    else:
        # sin normalizar (forma menos interpretable por mezclar unidades)
        vals = raw.copy()
        title_suffix = " (raw values)"
        radial_range = [float(np.nanmin(vals)), float(np.nanmax(vals))]

    # invierte donde “menos es mejor”
    for i, (_lbl, _key, invert, _s) in enumerate(metrics):
        if invert:
            vals[i] = 1.0 - vals[i] if normalize else -vals[i]

    vals = np.nan_to_num(vals, nan=0.0)

    fig = go.Figure()
    th = theta + [theta[0]]
    for j, name in enumerate(labels):
        r = vals[:, j].tolist()
        r.append(r[0])  # cerrar polígono

        custom = np.append(raw[:, j], raw[0, j])

        fig.add_trace(go.Scatterpolar(
            r=r, theta=th, fill='toself', name=name,
            customdata=custom,
            text=[name]*len(th),
            hovertemplate="<b>%{text}</b><br>%{theta}: %{customdata:.2f}<extra></extra>",
        ))

    FIG_HEIGHT = globals().get("FIG_HEIGHT", 420)
    fig.update_layout(
        title="KPIs – Radar" + title_suffix,
        polar=dict(radialaxis=dict(range=radial_range, showline=True,
                                   ticks='outside', tickfont=dict(size=9))),
        legend=dict(orientation="h", x=0, y=-0.20),
        margin=dict(t=40, b=80, l=40, r=40),
        height=FIG_HEIGHT
    )
    return fig

def build_kpi_radar_table(kpi_data, labels, metrics=RADAR_METRICS):
    """Tabla de valores brutos (mm, °, N) para no perder magnitud."""
    import numpy as np
    import plotly.graph_objects as go
    headers = ["KPI"] + list(labels)
    rows = []
    for (label, key, _inv, scale) in metrics:
        row = [label] + [float(k.get(key, float("nan")))*scale for k in kpi_data]
        rows.append(row)

    # transponer a columnas para go.Table
    cols = list(map(list, zip(*rows)))
    fig = go.Figure(data=[go.Table(
        header=dict(values=headers, fill_color='paleturquoise', align='left'),
        cells=dict(values=cols, fill_color='lavender', align='left')
    )])
    fig.update_layout(title="KPIs – Valores brutos", height=globals().get("FIG_HEIGHT", 480))
    return fig

def launch_dash_kpis(kpi_data, setup_names):
    from dash import Dash, dcc, html
    import plotly.graph_objects as go
    import numpy as np

    # ──────────────────────────────────────────────────────────────────────────────
    # Determinar la etiqueta (“label”) que mostrará cada punto:
    if len(setup_names) > 1:
        # Caso A: hay varios setups diferentes → mostramos el nombre del setup
        labels = setup_names
    else:
        # Caso B: solo un setup, pero varios tracks → mostramos el nombre de cada track
        labels = [k['track_name'] for k in kpi_data]
    # ──────────────────────────────────────────────────────────────────────────────

    app_kpi = Dash(__name__, title="KPIs", suppress_callback_exceptions=True)
    kpi_labels = ['FL', 'FR', 'RL', 'RR']
    kpi_labels_axes = ['Front', 'Rear']
    
    # --- DEFINICIÓN CENTRALIZADA DE LOS KPIs --- 
    kpi_definitions = KPI_DEFINITIONS

    # --- ARRANCAR LAYOUT CON UN TÍTULO PRINCIPAL ---
    layout = []

    # ── KPIs por rueda (puntos con barra de desviación) ──
    for title, unit, key_mean, key_std, factor in kpi_definitions:
        try:
            factor_use = 1 if title.startswith("Wheel Load") else factor
            mean_values_list = [k[key_mean] * factor_use for k in kpi_data]
            try:
                std_values_list = [k[key_std] * factor_use for k in kpi_data]
            except KeyError:
                std_values_list = [np.std(vals) * np.ones_like(vals)
                                   for vals in mean_values_list]
            fig = kpi_point_with_var(title, unit,
                                     mean_values_list, std_values_list,
                                     setup_names, kpi_labels)
            _place_legend_outside(fig, "bottom")
            layout.append(html.Div(
                dcc.Graph(figure=fig, config=GRAPH_CFG,
                          style={"height": STANDARD_HEIGHT}),
                style=CARD_STYLE))
               
        except KeyError:
            continue

    try:
        fig_radar = build_kpi_radar(kpi_data, labels, normalize="fixed")
        layout.append(html.Div(
            dcc.Graph(figure=fig_radar, config=GRAPH_CFG,
                    style={"height": STANDARD_HEIGHT}),
            style=CARD_STYLE))
        # tabla de valores brutos
        fig_radar_tbl = build_kpi_radar_table(kpi_data, labels)
        layout.append(html.Div(
            dcc.Graph(figure=fig_radar_tbl, config=GRAPH_CFG,
                    style={"height": STANDARD_HEIGHT}),
            style=CARD_STYLE))
    except Exception as e:
        print("[RADAR] no generado:", e)


    # --- KPIs PERSONALIZADOS: Road Noise, Pitch vs Distance, Pitch RMS, Ride Height RMS, Scatter, etc. ---
    # ── 2) Accumulated Road‐Noise Normalised by Lap Time ─────────────────────────
    try:
        noise_by_track = {}
        for k in kpi_data:
            tname = k['track_name']
            if tname not in noise_by_track:
                noise_by_track[tname] = (
                    k['tracknoise_accu_front'],
                    k['tracknoise_accu_rear']
                )

        tracks_unique = list(noise_by_track.keys())
        front_vals = [noise_by_track[t][0] for t in tracks_unique]
        rear_vals  = [noise_by_track[t][1] for t in tracks_unique]

        fig_accu = go.Figure()
        fig_accu.add_trace(go.Bar(
            name="Front Axle",
            x=tracks_unique,
            y=front_vals,
            marker_color='royalblue'
        ))
        fig_accu.add_trace(go.Bar(
            name="Rear  Axle",
            x=tracks_unique,
            y=rear_vals,
            marker_color='tomato'
        ))

        fig_accu.update_layout(
            title="Accumulated Road Track-noise Normalised by Lap Time",
            xaxis_title="Track",
            yaxis_title="Normalised Accu. Track-noise [mm/s]",
            barmode="group"
        )
        
        fig_accu.update_layout(height= FIG_HEIGHT, margin=dict(t=40, b=40, l=40, r=10))    
        layout.append(html.Div(
            dcc.Graph(figure=fig_accu, config=GRAPH_CFG, style={"height": STANDARD_HEIGHT}),
            style=CARD_STYLE))

    except KeyError:
        print("[DEBUG] No se encontró 'tracknoise_accu_*' en algún KPI")

    # ── 4) Tabla de Pitch RMS por Setup ──────────────────────────────────────────
    try:
        pitch_rms_vals = [float(k['pitch_rms']) for k in kpi_data]
        fig_pitch_table = go.Figure(data=[go.Table(
            header=dict(values=["Setup", "Pitch RMS [°]"],
                        fill_color='paleturquoise', align='left'),
            cells=dict(values=[setup_names, pitch_rms_vals],
                       fill_color='lavender', align='left'))
        ])
        fig_pitch_table.update_layout(title="Pitch RMS por Setup (Resumen Numérico)",
                                     height= FIG_HEIGHT,
                                     margin=dict(t=40, b=40, l=40, r=10))
        _place_legend_outside(fig_pitch_table, "bottom")
        layout.append(html.Div(
            dcc.Graph(figure=fig_pitch_table, config=GRAPH_CFG, style={"height": STANDARD_HEIGHT}),
            style=CARD_STYLE))
    except KeyError:
        pass


    # ── 6) FRH RMS vs Contact Patch Load RMS (dispersograma etiquetado) ────────
    try:
        frh_rms_vals = [k['frh_rms'] * 1000 for k in kpi_data]
        load_rms_vals_front = [k['front_load_rms'] for k in kpi_data]
        fig_scatter_frh = go.Figure(data=[
            go.Scatter(
                x=frh_rms_vals,
                y=load_rms_vals_front,
                mode='markers+text',
                text=labels,             
                textposition='top center'
            )
        ])
        fig_scatter_frh.update_layout(
            title="FRH RMS vs Contact Patch Load RMS",
            xaxis_title="Front Ride Height RMS [mm]",
            yaxis_title="Contact Patch Load RMS [N]",
            height= FIG_HEIGHT, margin=dict(t=40, b=40, l=40, r=10)
        )
        _place_legend_outside(fig, "bottom")   
        layout.append(html.Div(
            dcc.Graph(figure=fig_scatter_frh, config=GRAPH_CFG, style={"height": STANDARD_HEIGHT}),
            style=CARD_STYLE))
    except KeyError:
        pass

    # ── 7) RRH RMS vs Contact Patch Load RMS (dispersograma etiquetado) ────────
    try:
        rrh_rms_vals = [k['rrh_rms'] * 1000 for k in kpi_data]
        load_rms_vals_rear = [k['rear_load_rms'] for k in kpi_data]
        fig_scatter_rrh = go.Figure(data=[
            go.Scatter(
                x=rrh_rms_vals,
                y=load_rms_vals_rear,
                mode='markers+text',
                text=labels,             # <--- etiqueta inteligente
                textposition='top center'
            )
        ])
        fig_scatter_rrh.update_layout(
            title="RRH RMS vs Contact Patch Load RMS",
            xaxis_title="Rear Ride Height RMS [mm]",
            yaxis_title="Contact Patch Load RMS [N]",
            height= FIG_HEIGHT, margin=dict(t=40, b=40, l=40, r=10)
        )
        _place_legend_outside(fig_scatter_rrh, "bottom")
        layout.append(html.Div(
            dcc.Graph(figure=fig_scatter_rrh, config=GRAPH_CFG, style={"height": STANDARD_HEIGHT}),
            style=CARD_STYLE))
    except KeyError:
        pass

    # ── 5) Ride‑Height RMS (Front / Rear) en GLS ──────────────────────────────
    try:
        frh_vals      = np.array([k['frh_rms'] for k in kpi_data]) * 1e3  # mm
        rrh_vals      = np.array([k['rrh_rms'] for k in kpi_data]) * 1e3
        frh_std_vals  = np.array([k.get('frh_rms_std', 0) for k in kpi_data]) * 1e3
        rrh_std_vals  = np.array([k.get('rrh_rms_std', 0) for k in kpi_data]) * 1e3

        fig_rh_scatter = go.Figure()
        # pequeña “separación” en x para que Front y Rear no se pisen
        x_base = np.arange(len(setup_names))
        for i, name in enumerate(setup_names):
            fig_rh_scatter.add_trace(go.Scatter(
                x=[x_base[i]-0.1, x_base[i]+0.1],        # Front  | Rear
                y=[frh_vals[i], rrh_vals[i]],
                mode='markers',
                name=name,
                marker=dict(size=12),
                error_y=dict(type='data',
                            array=[frh_std_vals[i], rrh_std_vals[i]],
                            visible=True, width=3, thickness=1.2)
            ))
        fig_rh_scatter.update_layout(
            title="Ride‑Height RMS en GLS [mm]",
            xaxis=dict(tickvals=x_base,
                    ticktext=setup_names,
                    title="Setup"),
            yaxis_title="RMS Ride‑Height [mm]",
            height= FIG_HEIGHT, margin=dict(t=40, b=40, l=40, r=10)
        )
        _place_legend_outside(fig_rh_scatter, "bottom")
        layout.append(html.Div(
            dcc.Graph(figure=fig_rh_scatter, config=GRAPH_CFG, style={"height": STANDARD_HEIGHT}),
            style=CARD_STYLE))
    except KeyError:
        pass


    # ── 8) y 9) FRH / RRH   GLS vs NGLS  (scatter) ───────────────────────────
    def add_gls_vs_ngls(axis, key_rms, key_ngls, title):
        try:
            vals_gls   = np.array([k[key_rms]       for k in kpi_data]) * 1e3
            vals_ngls  = np.array([k[key_ngls]      for k in kpi_data]) * 1e3
            std_gls    = np.array([k[key_rms + '_std']      for k in kpi_data]) * 1e3
            std_ngls   = np.array([k[key_ngls + '_std']     for k in kpi_data]) * 1e3

            fig_cmp = go.Figure()
            x_base  = np.arange(len(setup_names))
            fig_cmp.add_trace(go.Scatter(
                x=x_base-0.1, y=vals_gls,
                mode='markers', name="Grip‑Limited",
                error_y=dict(type='data', array=std_gls,
                            visible=True, width=3)))
            fig_cmp.add_trace(go.Scatter(
                x=x_base+0.1, y=vals_ngls,
                mode='markers', name="No Grip‑Limited",
                error_y=dict(type='data', array=std_ngls,
                            visible=True, width=3)))
            fig_cmp.update_layout(
                title=title,
                xaxis=dict(tickvals=x_base, ticktext=setup_names, title="Setup"),
                yaxis_title="RMS [mm]",
                height= FIG_HEIGHT, margin=dict(t=40, b=40, l=40, r=10)
            )
            _place_legend_outside(fig_cmp, "bottom")
            layout.append(html.Div(
                dcc.Graph(figure=fig_cmp, config=GRAPH_CFG, style={"height": STANDARD_HEIGHT}),
                style=CARD_STYLE))
        except KeyError:
            pass

    add_gls_vs_ngls('front', 'frh_rms', 'frh_rms_nongrip',
                    "Front Ride‑Height RMS: GLS vs NGLS [mm]")
    add_gls_vs_ngls('rear',  'rrh_rms', 'rrh_rms_nongrip',
                    "Rear Ride‑Height RMS: GLS vs NGLS [mm]")


    # ── 14) y 15)  Load RMS  Braking vs Traction  (scatter) ──────────────────
    def braking_vs_traction(key_brake, key_trac, title, ytitle):
        try:
            brake_vals = np.array([k[key_brake] for k in kpi_data])
            trac_vals  = np.array([k[key_trac]  for k in kpi_data])
            std_brake  = np.array([k[key_brake + '_std'] for k in kpi_data])
            std_trac   = np.array([k[key_trac + '_std']  for k in kpi_data])

            fig_bt = go.Figure()
            x_base = np.arange(len(setup_names))
            fig_bt.add_trace(go.Scatter(
                x=x_base-0.1, y=brake_vals, name="Braking",
                mode='markers',
                error_y=dict(type='data', array=std_brake,
                            visible=True, width=3)))
            fig_bt.add_trace(go.Scatter(
                x=x_base+0.1, y=trac_vals,  name="Traction",
                mode='markers',
                error_y=dict(type='data', array=std_trac,
                            visible=True, width=3)))
            fig_bt.update_layout(
                title=title,
                xaxis=dict(tickvals=x_base, ticktext=setup_names, title="Setup"),
                yaxis_title=ytitle,
                height= FIG_HEIGHT, margin=dict(t=40, b=40, l=40, r=10)
            )
            _place_legend_outside(fig_bt, "bottom")
            layout.append(html.Div(
                dcc.Graph(figure=fig_bt, config=GRAPH_CFG, style={"height": STANDARD_HEIGHT}),
                style=CARD_STYLE))
        except KeyError:
            pass

    braking_vs_traction('front_load_rms_brake',    'front_load_rms_traction',
                        "Front Load RMS: Braking vs Traction [N]",
                        "Contact Patch Load RMS [N]")
    braking_vs_traction('rear_load_rms_brake',     'rear_load_rms_traction',
                        "Rear  Load RMS: Braking vs Traction [N]",
                        "Contact Patch Load RMS [N]")


    # ── 16) y 17)  Load RMS Front vs Rear  en Frenada / Tracción ─────────────
    def front_vs_rear(xlabels, front_vals, rear_vals, title):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=xlabels, y=front_vals, name="Front", mode='markers'))
        fig.add_trace(go.Scatter(
            x=xlabels, y=rear_vals,  name="Rear",  mode='markers'))
        fig.update_layout(
            title=title,
            xaxis_title="Setup",
            yaxis_title="CPL RMS [N]",
            height= FIG_HEIGHT, margin=dict(t=40, b=40, l=40, r=10)
        )
        _place_legend_outside(fig, "bottom")
        layout.append(html.Div(
            dcc.Graph(figure=fig, config=GRAPH_CFG, style={"height": STANDARD_HEIGHT}),
            style=CARD_STYLE))

    try:
        fv_brake = np.array([k['front_load_rms_brake'] for k in kpi_data])
        rv_brake = np.array([k['rear_load_rms_brake']  for k in kpi_data])
        front_vs_rear(setup_names, fv_brake, rv_brake,
                    "Contact Patch Load RMS en Frenada [N]")
    except KeyError:
        pass

    try:
        fv_trac = np.array([k['front_load_rms_traction'] for k in kpi_data])
        rv_trac = np.array([k['rear_load_rms_traction']  for k in kpi_data])
        front_vs_rear(setup_names, fv_trac, rv_trac,
                    "Contact Patch Load RMS en Tracción [N]")
    except KeyError:
        pass

    # ── 10) PSD Heave por eje (Front vs Rear, mm²/Hz) ──────────────────────────
    try:
        fig_psd_axes = go.Figure()
        for k, name in zip(kpi_data, setup_names):
            if ('f_psd_front' in k and 'psd_heave_front' in k and
                'f_psd_rear'  in k and 'psd_heave_rear'  in k):

                # PSD eje delantero (convertir m²/Hz → mm²/Hz)
                fig_psd_axes.add_trace(go.Scatter(
                    x=k['f_psd_front'],
                    y=np.array(k['psd_heave_front']) * 1e6,
                    mode='lines',
                    name=f"{name} – Front"
                ))
                # PSD eje trasero
                fig_psd_axes.add_trace(go.Scatter(
                    x=k['f_psd_rear'],
                    y=np.array(k['psd_heave_rear']) * 1e6,
                    mode='lines',
                    name=f"{name} – Rear"
                ))

        fig_psd_axes.update_layout(
            title="PSD of Heave Motion by Axle (Front vs Rear)",
            xaxis_title="Frequency (Hz)",
            yaxis_title="PSD Heave (mm²/Hz)",
            yaxis_type="log",
            height= FIG_HEIGHT, margin=dict(t=40, b=40, l=40, r=10)
        )
        _place_legend_outside(fig_psd_axes, "bottom")
        layout.append(html.Div(
            dcc.Graph(figure=fig_psd_axes, config=GRAPH_CFG, style={"height": STANDARD_HEIGHT}),
            style=CARD_STYLE))

    except Exception as e:
        print(f"[WARNING] Error al generar el PSD de heave por eje: {e}")

    # ── 12) PSD Pitch por eje (Front vs Rear, mm²/Hz) ──────────────────────────
    try:
        fig_psd_pitch_axes = go.Figure()
        for k, name in zip(kpi_data, setup_names):
            if ('f_psd_pitch_front' in k and 'psd_pitch_front' in k and
                'f_psd_pitch_rear'  in k and 'psd_pitch_rear'  in k):

                # PSD pitch→vertical eje delantero (m²/Hz → mm²/Hz)
                fig_psd_pitch_axes.add_trace(go.Scatter(
                    x=k['f_psd_pitch_front'],
                    y=np.array(k['psd_pitch_front']) * 1e6,
                    mode='lines',
                    name=f"{name} – Pitch Front"
                ))
                # PSD pitch→vertical eje trasero
                fig_psd_pitch_axes.add_trace(go.Scatter(
                    x=k['f_psd_pitch_rear'],
                    y=np.array(k['psd_pitch_rear']) * 1e6,
                    mode='lines',
                    name=f"{name} – Pitch Rear"
                ))

        fig_psd_pitch_axes.update_layout(
            title="PSD of Pitch‐Induced Vertical by Axle (Front vs Rear)",
            xaxis_title="Frequency (Hz)",
            yaxis_title="PSD Pitch→Vertical (mm²/Hz)",
            yaxis_type="log",
             height= FIG_HEIGHT, margin=dict(t=40, b=40, l=40, r=10)
        )
        _place_legend_outside(fig_psd_pitch_axes, "bottom")
        layout.append(html.Div(
            dcc.Graph(figure=fig_psd_pitch_axes, config=GRAPH_CFG, style={"height": STANDARD_HEIGHT}),
            style=CARD_STYLE))

    except Exception as e:
        print(f"[WARNING] Error al generar el PSD de pitch por eje: {e}")

    # 14a) Magnitud (dB) con smoothing
    fig_psd_load_mag = go.Figure()
    for k, name in zip(kpi_data, setup_names):
        if ('f_psd_load' in k and 'psd_load_mag_front' in k and 'psd_load_mag_rear' in k):
            f_load = np.array(k['f_psd_load'])
            # Suavizado de la magnitud
            mag_f_raw = np.array(k['psd_load_mag_front'])
            mag_r_raw = np.array(k['psd_load_mag_rear'])
            mag_f = smooth_signal(mag_f_raw, window=51, polyorder=3)
            mag_r = smooth_signal(mag_r_raw, window=51, polyorder=3)

            fig_psd_load_mag.add_trace(go.Scatter(
                x=f_load,
                y=mag_f,
                mode='lines',
                name=f"{name} – Mag Front [dB] (suavizado)",
            ))
            fig_psd_load_mag.add_trace(go.Scatter(
                x=f_load,
                y=mag_r,
                mode='lines',
                name=f"{name} – Mag Rear  [dB] (suavizado)",
            ))

    fig_psd_load_mag.update_layout(
        title="PSD de Carga – Magnitud (Front vs Rear, Suavizado)",
        xaxis=dict(title="Frecuencia [Hz]", type="log"),
        yaxis=dict(title="Magnitud [dB]"),
        height= FIG_HEIGHT,
        margin=dict(t=40, b=40, l=40, r=10)
    )
    _place_legend_outside(fig_psd_load_mag, "bottom")
    layout.append(html.Div(
        dcc.Graph(figure=fig_psd_load_mag, config=GRAPH_CFG, style={"height": STANDARD_HEIGHT}),
        style=CARD_STYLE))

    # ──────────────────────────────────────────────────────────────────────────
    app_kpi.layout = html.Div([
        html.H1("Comparativa de KPIs entre Setups"),
        html.Div(layout, style=GRID_STYLE)
    ])
    app_kpi.run(port=8051, debug=False)

def get_results_figures(sol, post, save_dir=None):
    import plotly.graph_objs as go
    import numpy as np
    import os
    figures = []
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    distance = np.cumsum(post['vx']) * np.gradient(sol.t)
    # Travel
    fig_travel = go.Figure()
    fig_travel.add_trace(go.Scatter(x=distance, y=np.mean(post['travel'][0:2], axis=0)*1000, name="Front"))
    fig_travel.add_trace(go.Scatter(x=distance, y=np.mean(post['travel'][2:4], axis=0)*1000, name="Rear"))
    fig_travel.update_layout(title="Suspension Travel [mm]", xaxis_title="Time [s]", yaxis_title="Travel [mm]")
    figures.append(fig_travel)
    if save_dir:
        fig_travel.write_html(os.path.join(save_dir, 'travel.html'))

    # Heave
    fig_heave = go.Figure()
    fig_heave.add_trace(go.Scatter(x=distance, y=sol.y[0]*1000, name="Heave"))
    fig_heave.update_layout(title="Heave [mm]", xaxis_title="Time [s]", yaxis_title="Heave [mm]")
    figures.append(fig_heave)
    if save_dir:
        fig_heave.write_html(os.path.join(save_dir, 'heave.html'))

    # Pitch
    fig_pitch = go.Figure()
    fig_pitch.add_trace(go.Scatter(x=distance, y=np.degrees(sol.y[2]), name="Pitch"))
    fig_pitch.update_layout(title="Pitch [°]", xaxis_title="Time [s]", yaxis_title="Pitch [°]")
    figures.append(fig_pitch)
    if save_dir:
        fig_roll.write_html(os.path.join(save_dir, 'roll.html'))

    # Roll
    fig_roll = go.Figure()
    fig_roll.add_trace(go.Scatter(x=distance, y=np.degrees(sol.y[4]), name="Roll"))
    fig_roll.update_layout(title="Roll [°]", xaxis_title="Time [s]", yaxis_title="Roll [°]")
    figures.append(fig_roll)
    if save_dir:
        fig_roll.write_html(os.path.join(save_dir, 'roll.html'))

    # Wheel Load
    fig_load = go.Figure()
    for i, label in enumerate(["FL", "FR", "RL", "RR"]):
        fig_load.add_trace(go.Scatter(x=distance, y=post['wheel_load'][i], name=label))
    fig_load.update_layout(
        title="Wheel Load per Wheel [N]",xaxis_title="Time [s]",yaxis_title="Load [N]")
    figures.append(fig_load)
    if save_dir:
        fig_load.write_html(os.path.join(save_dir, 'wheel_load.html'))

    if save_dir:
        for idx, fig in enumerate(figures, 1):
            fig.write_html(os.path.join(save_dir, f'result_{idx}.html'))
    return figures

def get_kpi_figures(data, save_dir: str | None = None):
    import os, re, math, numpy as np, plotly.graph_objects as go

    # Detectar si data son tuplas (sol, post, …) o dicts (post)
    if data and isinstance(data[0], tuple):
        kpi_data    = [post for (_, post, _, _) in data]
        setup_names = [os.path.basename(p).replace(".json", "")
                       for (_, _, p, _) in data]
    else:                                 # lista de dicts 'post'
        kpi_data    = data
        setup_names = [p.get("setup_name", f"Setup {i+1}")
                       for i, p in enumerate(kpi_data)]

    # ── REORDENAR ambos arrays en función del sufijo _088, _104, … ──────────
    def _extract_ratio(name: str) -> float:
        """
        'kspringF_088' → 0.88   |  'baseSetup' → 1.00
        """
        m = re.search(r"_([0-9]{2,3})$", name)
        return int(m.group(1))/100 if m else 1.0

    order = np.argsort([_extract_ratio(n) for n in setup_names])
    setup_names = [setup_names[i] for i in order]
    kpi_data    = [kpi_data[i]    for i in order]


    # === NUEVO BLOQUE: rellenar KPIs ausentes o cero ===
    for p in kpi_data:
        # FRH RMS y desviación típica
        if ('frh_rms' not in p) or (not p['frh_rms']):
            if 'RH_front' in p:
                arr = np.asarray(p['RH_front']).flatten()
                p['frh_rms'] = float(np.sqrt(np.mean(arr**2)))
                p['frh_rms_std'] = float(np.std(arr))
        # RRH RMS
        if ('rrh_rms' not in p) or (not p['rrh_rms']):
            if 'RH_rear' in p:
                arr = np.asarray(p['RH_rear']).flatten()
                p['rrh_rms'] = float(np.sqrt(np.mean(arr**2)))
                p['rrh_rms_std'] = float(np.std(arr))
        # FRH/ RRH no‑grip (usar el valor general si no hay máscara)
        if ('frh_rms_nongrip' not in p) or (not p['frh_rms_nongrip']):
            if 'RH_front' in p:
                arr = np.asarray(p['RH_front']).flatten()
                p['frh_rms_nongrip'] = float(np.sqrt(np.mean(arr**2)))
                p['frh_rms_nongrip_std'] = float(np.std(arr))
        if ('rrh_rms_nongrip' not in p) or (not p['rrh_rms_nongrip']):
            if 'RH_rear' in p:
                arr = np.asarray(p['RH_rear']).flatten()
                p['rrh_rms_nongrip'] = float(np.sqrt(np.mean(arr**2)))
                p['rrh_rms_nongrip_std'] = float(np.std(arr))
        # Carga RMS por eje (Front / Rear)
        if ('front_load_rms' not in p) or (not p['front_load_rms']):
            if 'wheel_load' in p:
                wl = np.asarray(p['wheel_load'])
                fl = np.mean(wl[0:2], axis=0).flatten()
                p['front_load_rms'] = float(np.sqrt(np.mean(fl**2)))
                p['front_load_rms_std'] = float(np.std(fl))
        if ('rear_load_rms' not in p) or (not p['rear_load_rms']):
            if 'wheel_load' in p:
                wl = np.asarray(p['wheel_load'])
                rl = np.mean(wl[2:4], axis=0).flatten()
                p['rear_load_rms'] = float(np.sqrt(np.mean(rl**2)))
                p['rear_load_rms_std'] = float(np.std(rl))
        # Si no hay valores de frenada o tracción, usar la RMS general
        if ('front_load_rms_brake' not in p) or (not p['front_load_rms_brake']):
            p['front_load_rms_brake'] = p.get('front_load_rms', 0.0)
            p['front_load_rms_brake_std'] = p.get('front_load_rms_std', 0.0)
        if ('rear_load_rms_brake' not in p) or (not p['rear_load_rms_brake']):
            p['rear_load_rms_brake'] = p.get('rear_load_rms', 0.0)
            p['rear_load_rms_brake_std'] = p.get('rear_load_rms_std', 0.0)
        if ('front_load_rms_traction' not in p) or (not p['front_load_rms_traction']):
            p['front_load_rms_traction'] = p.get('front_load_rms', 0.0)
            p['front_load_rms_traction_std'] = p.get('front_load_rms_std', 0.0)
        if ('rear_load_rms_traction' not in p) or (not p['rear_load_rms_traction']):
            p['rear_load_rms_traction'] = p.get('rear_load_rms', 0.0)
            p['rear_load_rms_traction_std'] = p.get('rear_load_rms_std', 0.0)

    # Etiquetas para los puntos: si hay más de un setup usamos su nombre;
    # si sólo hay un setup pero varias pistas usamos el nombre de la pista.
    if len(setup_names) > 1:
        labels = setup_names
    else:
        labels = [k.get("track_name", f"Track {i+1}") for i, k in enumerate(kpi_data)]

    # Lista donde guardaremos todas las figuras
    figs = []

    # 2) Wheel‑load Max (gráfica de puntos con barras)
    for title, unit, key_mean, key_std, factor in KPI_DEFINITIONS:
        # Construir listas de valores para cada setup
        mean_vals_list = []
        var_vals_list  = []
        valid_names    = []
        for name, post in zip(setup_names, kpi_data):
            if key_mean not in post:
                # Si falta la clave, omite ese setup para este KPI
                continue
            mean_arr = post[key_mean] * factor
            if key_std in post:
                var_arr = post[key_std] * factor
            else:
                # si no hay desviación estándar, usa la desviación de los valores
                var_arr = np.std(mean_arr) * np.ones_like(mean_arr)
            mean_vals_list.append(mean_arr)
            var_vals_list.append(var_arr)
            valid_names.append(name)
        if mean_vals_list:
            fig = kpi_point_with_var(title, unit,
                                     mean_vals_list, var_vals_list,
                                     valid_names, ['FL','FR','RL','RR'])
            fig.update_layout(height=FIG_HEIGHT, margin=dict(t=40,b=40,l=40,r=10))
            figs.append(fig)

    # 3) Ruido de pista acumulado normalizado
    try:
        noise_by_track = {}
        for k in kpi_data:
            tname = k.get('track_name','Unknown')
            if tname not in noise_by_track:
                # (front, rear)
                noise_by_track[tname] = (k['tracknoise_accu_front'],
                                         k['tracknoise_accu_rear'])
        if noise_by_track:
            tracks = list(noise_by_track.keys())
            front_vals = [noise_by_track[t][0] for t in tracks]
            rear_vals  = [noise_by_track[t][1] for t in tracks]
            fig_noise = go.Figure()
            fig_noise.add_trace(go.Bar(name="Front Axle", x=tracks, y=front_vals, marker_color='royalblue'))
            fig_noise.add_trace(go.Bar(name="Rear  Axle", x=tracks, y=rear_vals,  marker_color='tomato'))
            fig_noise.update_layout(
                title="Accumulated Road Track-noise Normalised by Lap Time",
                xaxis_title="Track",
                yaxis_title="Normalised Accu. Track-noise [mm/s]",
                barmode="group",
                height=FIG_HEIGHT,
                margin=dict(t=40,b=40,l=40,r=10)
            )
            figs.append(fig_noise)
    except KeyError:
        pass

    # 4) Tabla de Pitch RMS
    try:
        pitch_vals = [float(k['pitch_rms']) for k in kpi_data]
        fig_pitch = go.Figure(data=[go.Table(
            header=dict(values=["Setup","Pitch RMS [°]"],
                        fill_color='paleturquoise', align='left'),
            cells=dict(values=[setup_names,pitch_vals],
                       fill_color='lavender', align='left')
        )])
        fig_pitch.update_layout(
            title="Pitch RMS por Setup (Resumen Numérico)",
            height=FIG_HEIGHT,
            margin=dict(t=40,b=40,l=40,r=10)
        )
        figs.append(fig_pitch)
    except KeyError:
        pass

    # 5) FRH RMS vs Contact Patch Load RMS
    try:
        frh_vals_mm = [k['frh_rms'] * 1000 for k in kpi_data]
        load_front = [k['front_load_rms'] for k in kpi_data]
        fig_frh = go.Figure(data=[go.Scatter(
            x=frh_vals_mm, y=load_front,
            mode='markers+text',
            text=labels, textposition='top center'
        )])
        fig_frh.update_layout(
            title="FRH RMS vs Contact Patch Load RMS",
            xaxis_title="Front Ride Height RMS [mm]",
            yaxis_title="Contact Patch Load RMS [N]",
            height=FIG_HEIGHT,
            margin=dict(t=40,b=40,l=40,r=10)
        )
        figs.append(fig_frh)
    except KeyError:
        pass

    # 6) RRH RMS vs Contact Patch Load RMS
    try:
        rrh_vals_mm = [k['rrh_rms'] * 1000 for k in kpi_data]
        load_rear = [k['rear_load_rms'] for k in kpi_data]
        fig_rrh = go.Figure(data=[go.Scatter(
            x=rrh_vals_mm, y=load_rear,
            mode='markers+text',
            text=labels, textposition='top center'
        )])
        fig_rrh.update_layout(
            title="RRH RMS vs Contact Patch Load RMS",
            xaxis_title="Rear Ride Height RMS [mm]",
            yaxis_title="Contact Patch Load RMS [N]",
            height=FIG_HEIGHT,
            margin=dict(t=40,b=40,l=40,r=10)
        )
        figs.append(fig_rrh)
    except KeyError:
        pass

    # 7) Ride‑Height RMS en GLS (front/rear)
    try:
        frh_rms_vals = np.array([k['frh_rms'] for k in kpi_data]) * 1000  # [mm]
        rrh_rms_vals = np.array([k['rrh_rms'] for k in kpi_data]) * 1000
        frh_std_vals = np.array([k.get('frh_rms_std',0) for k in kpi_data]) * 1000
        rrh_std_vals = np.array([k.get('rrh_rms_std',0) for k in kpi_data]) * 1000
        fig_rh = go.Figure()
        x_base = np.arange(len(setup_names))
        for i, name in enumerate(setup_names):
            fig_rh.add_trace(go.Scatter(
                x=[x_base[i]-0.1, x_base[i]+0.1],
                y=[frh_rms_vals[i], rrh_rms_vals[i]],
                mode='markers', name=name,
                marker=dict(size=12),
                error_y=dict(type='data',
                             array=[frh_std_vals[i], rrh_std_vals[i]],
                             visible=True, width=3, thickness=1.2)
            ))
        fig_rh.update_layout(
            title="Ride‑Height RMS en GLS [mm]",
            xaxis=dict(tickvals=x_base, ticktext=setup_names, title="Setup"),
            yaxis_title="RMS Ride‑Height [mm]",
            height=FIG_HEIGHT,
            margin=dict(t=40,b=40,l=40,r=10)
        )
        figs.append(fig_rh)
    except KeyError:
        pass

    # 8) Comparativa GLS vs NGLS de FRH y RRH
    def add_gls_ngls(key_rms, key_ngls, title):
        try:
            vals_gls  = np.array([k[key_rms]  for k in kpi_data]) * 1000
            vals_ngls = np.array([k[key_ngls] for k in kpi_data]) * 1000
            std_gls   = np.array([k.get(key_rms  + '_std', 0) for k in kpi_data]) * 1000
            std_ngls  = np.array([k.get(key_ngls + '_std', 0) for k in kpi_data]) * 1000
            fig_cmp = go.Figure()
            x = np.arange(len(setup_names))
            fig_cmp.add_trace(go.Scatter(
                x=x-0.1, y=vals_gls, name="Grip‑Limited",
                mode='markers',
                error_y=dict(type='data', array=std_gls, visible=True, width=3)
            ))
            fig_cmp.add_trace(go.Scatter(
                x=x+0.1, y=vals_ngls, name="No Grip‑Limited",
                mode='markers',
                error_y=dict(type='data', array=std_ngls, visible=True, width=3)
            ))
            fig_cmp.update_layout(
                title=title,
                xaxis=dict(tickvals=x, ticktext=setup_names, title="Setup"),
                yaxis_title="RMS [mm]",
                height=FIG_HEIGHT,
                margin=dict(t=40,b=40,l=40,r=10)
            )
            figs.append(fig_cmp)
        except KeyError:
            return

    add_gls_ngls('frh_rms','frh_rms_nongrip',"Front Ride‑Height RMS: GLS vs NGLS [mm]")
    add_gls_ngls('rrh_rms','rrh_rms_nongrip',"Rear Ride‑Height RMS: GLS vs NGLS [mm]")

    # 9) Load RMS – Braking vs Traction (front y rear)
    def add_brake_trac(k_brake, k_trac, title, ytitle):
        try:
            v_brake = np.array([k[k_brake] for k in kpi_data])
            v_trac  = np.array([k[k_trac]  for k in kpi_data])
            std_brake = np.array([k.get(k_brake + '_std', 0) for k in kpi_data])
            std_trac  = np.array([k.get(k_trac  + '_std', 0) for k in kpi_data])
            fig_bt = go.Figure()
            x = np.arange(len(setup_names))
            fig_bt.add_trace(go.Scatter(
                x=x-0.1, y=v_brake, name="Braking",
                mode='markers',
                error_y=dict(type='data', array=std_brake, visible=True, width=3)
            ))
            fig_bt.add_trace(go.Scatter(
                x=x+0.1, y=v_trac,  name="Traction",
                mode='markers',
                error_y=dict(type='data', array=std_trac, visible=True, width=3)
            ))
            fig_bt.update_layout(
                title=title,
                xaxis=dict(tickvals=x, ticktext=setup_names, title="Setup"),
                yaxis_title=ytitle,
                height=FIG_HEIGHT,
                margin=dict(t=40,b=40,l=40,r=10)
            )
            figs.append(fig_bt)
        except KeyError:
            return

    add_brake_trac('front_load_rms_brake','front_load_rms_traction',
                   "Front Load RMS: Braking vs Traction [N]",
                   "Contact Patch Load RMS [N]")
    add_brake_trac('rear_load_rms_brake','rear_load_rms_traction',
                   "Rear  Load RMS: Braking vs Traction [N]",
                   "Contact Patch Load RMS [N]")

    # 10) Contact Patch Load RMS en Frenada / en Tracción (Front vs Rear)
    def add_cpl_front_vs_rear(front_vals, rear_vals,
                            front_std,  rear_std,      # ← nuevos
                            title):
        fig = go.Figure()

        # Front
        fig.add_trace(go.Scatter(
            x=setup_names,
            y=front_vals,
            mode='markers',
            name="Front",
            error_y=dict(
                type='data',
                array=front_std,            # ← usa front_std
                visible=True,
                width=3
            )
        ))

        # Rear
        fig.add_trace(go.Scatter(
            x=setup_names,
            y=rear_vals,
            mode='markers',
            name="Rear",
            error_y=dict(
                type='data',
                array=rear_std,             # ← usa rear_std
                visible=True,
                width=3
            )
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Setup",
            yaxis_title="CPL RMS [N]",
            height=FIG_HEIGHT,
            margin=dict(t=40, b=40, l=40, r=10)
        )
        figs.append(fig)


    # ── Frenada ────────────────────────────────────────────────────────────────
    try:
        fv_brake      = [k['front_load_rms_brake']       for k in kpi_data]
        rv_brake      = [k['rear_load_rms_brake']        for k in kpi_data]
        std_fv_brake  = [k['front_load_rms_brake_std']   for k in kpi_data]
        std_rv_brake  = [k['rear_load_rms_brake_std']    for k in kpi_data]

        add_cpl_front_vs_rear(fv_brake, rv_brake,
                            std_fv_brake, std_rv_brake,
                            "Contact Patch Load RMS en Frenada [N]")
    except KeyError:
        pass

    # ── Tracción ───────────────────────────────────────────────────────────────
    try:
        fv_trac      = [k['front_load_rms_traction']     for k in kpi_data]
        rv_trac      = [k['rear_load_rms_traction']      for k in kpi_data]
        std_fv_trac  = [k['front_load_rms_traction_std'] for k in kpi_data]
        std_rv_trac  = [k['rear_load_rms_traction_std']  for k in kpi_data]

        add_cpl_front_vs_rear(fv_trac, rv_trac,
                            std_fv_trac, std_rv_trac,
                            "Contact Patch Load RMS en Tracción [N]")
    except KeyError:
        pass

    # 14) Guardar HTML individuales opcionalmente
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for i, fig in enumerate(figs, 1):
            fname = os.path.join(save_dir, f"kpi_{i:02d}.html")
            fig.write_html(fname, include_plotlyjs="cdn")

    return figs

def run_kpi_comparison_in_thread(sim_results):
    from threading import Thread
    kpi_data = [post for _, post, _, _ in sim_results]
    setup_names = [os.path.basename(p).replace(".json", "") for _, _, p, _ in sim_results]
    thread = Thread(target=launch_dash_kpis, args=(kpi_data, setup_names))
    thread.start()
    
def save_kpi_report(
        data: Sequence[Union[dict, tuple]],
        out_dir: str = "report_kpis",
        mode: str = "single",
        grid_cols: int = 2,
        title: str = "KPIs – 7-Post Rig"
) -> List[go.Figure]:

    from visualizer_dash import get_kpi_figures, CARD_CSS, BODY_CSS

    figs = get_kpi_figures(data)
    out  = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if mode == "separate":
        for i, fig in enumerate(figs, 1):
            fig.write_html(out / f"kpi_{i:02d}.html",
                           include_plotlyjs="cdn", full_html=True)
    else:
        # ── cuadrícula N columnas según grid_cols ───────────────────────────
        grid_css = (f"display:grid;"
                    f"grid-template-columns:repeat({grid_cols},minmax(0,1fr));"
                    f"gap:20px")

        cards = [
            f'<div class="card">{plot(fig, include_plotlyjs=False, output_type="div")}</div>'
            for fig in figs
        ]

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
 body{{{BODY_CSS}}}
 .grid{{{grid_css}}}          /* ← usa la variable correcta  */
 .card{{{CARD_CSS}}}
</style></head><body>
<h1 style="margin:16px 20px 0">{title}</h1>
<div class="grid">
{''.join(cards)}
</div></body></html>"""

        (out / "kpis_report.html").write_text(html, encoding="utf-8")

    return figs

def export_full_report(setups, export_path="export_full_report.html"):
    """
    • Genera un HTML con:
        1. Portada + sección de KPIs (producida por save_kpi_report)
        2. Cualquier otra sección adicional que quieras añadir (opcional)
    • Guarda un CSV extendido por cada setup.
    """
    out_dir = os.path.dirname(export_path) or "."
    out_dir = Path(out_dir)

    # --------------------------------------------------------------------- #
    # 1)  Dashboard de KPIs vía save_kpi_report
    # --------------------------------------------------------------------- #
    posts = [post for (_, post, _, _) in setups]

    # genera kpis_report.html en la misma carpeta
    save_kpi_report(
        data=posts,
        out_dir=out_dir,
        mode="single",           # un único dashboard
        grid_cols=2,
        title="KPIs – 7-Post Rig"
    )

    # leemos solo el <body> de ese mini-dashboard para incrustarlo
    kpi_html_path = out_dir / "kpis_report.html"
    with open(kpi_html_path, "r", encoding="utf-8") as fh:
        kpi_html = fh.read()

    # extraer lo que hay entre <body> … </body>
    kpi_body = kpi_html.split("<body>")[1].split("</body>")[0]

    # --------------------------------------------------------------------- #
    # 2)  Montar el documento final
    # --------------------------------------------------------------------- #
    with open(export_path, "w", encoding="utf-8") as f:
        f.write("""<!DOCTYPE html><html><head>
<meta charset="utf-8">
<title>Reporte de Simulación – 7-Post Rig</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<link rel="stylesheet"
      href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap">
<style>body{font-family:Inter,sans-serif;margin:0;padding:0 20px}</style>
</head><body>
<h1>Reporte de Simulación – 7-Post Rig</h1>
""")
        # 2.1  KPIs producidos con save_kpi_report
        f.write(kpi_body)

        # 2.2  (Opcional) aquí podrías añadir más secciones…
        # f.write("<h2>Otra sección</h2> …")

        f.write("</body></html>")

    print(f"[INFO] HTML combinado exportado en {export_path}")

    # --------------------------------------------------------------------- #
    # 3)  CSV extendido por setup (igual que antes)
    # --------------------------------------------------------------------- #
    out_dir = os.path.dirname(export_path) or "."
    for sol, post, setup_path, track_path in setups:
        setup_name = os.path.basename(setup_path).replace(".json","")

        # === 1) Inputs ==============================================================
        track = load_track_channels(track_path)  # t[s], vx[m/s], ax[m/s2], ay[m/s2], ...
        t_in   = np.asarray(track['t'])
        vx_in  = np.asarray(track['vx'])
        ax_in  = np.asarray(track['ax'])
        ay_in  = np.asarray(track['ay'])
        thr_in = np.asarray(track['rpedal'])
        brk_in = np.asarray(track['brake'])
        ztrk_in = np.vstack(track['z_tracks'])   # (4,N) en m

        # === 2) Re-muestreo / garantía de longitud ==================================
        t = sol.t
        vx  = np.interp(t, t_in, vx_in) * 3.6
        ax  = np.interp(t, t_in, ax_in) / 9.81
        ay  = np.interp(t, t_in, ay_in) / 9.81
        thr = np.interp(t, t_in, thr_in)
        brk = np.interp(t, t_in, brk_in)

        # Distancia integrada (trapecios)
        dt = np.gradient(t)
        dist = np.cumsum(vx * dt)

        # === 3) Estados del solver ==================================================
        heave      = sol.y[0]                 # m
        heave_front = post['heave_front']*1000
        heave_rear  = post['heave_rear']*1000
        heave_rate = sol.y[1]
        pitch      = sol.y[2]                 # rad
        pitch_rate = sol.y[3]
        roll       = sol.y[4]                 # rad
        roll_rate  = sol.y[5]

        # === 4) Canales de salida del postprocesado =================================
        # (Todos tienen forma (4,N) salvo los listados como escalares/1D)
        trv     = post['travel']            # (4,N) m (neg = compresión)
        dmp_trv = post['damper_travel']     # (4,N) m
        Fspr    = post['f_spring']          # (4,N) N
        Fdmp    = post['f_damper']          # (4,N) N
        Fbump   = post['f_bump']            # (4,N) N
        Farb    = post['f_arb']             # (4,N) N
        Wload   = post['wheel_load']        # (4,N) N  (asegúrate: si venía en "kg", multiplica por 9.81)

        FzAeroF = post.get('Fz_aero_front', np.zeros_like(t))
        FzAeroR = post.get('Fz_aero_rear',  np.zeros_like(t))
        
        # heave por eje ya viene en mm en post:

        RH_front = post["RH_front"]*1000 #dyn_hF
        RH_rear  = post["RH_rear"]*1000  #dyn_hR

        # Máscaras grip
        gl_lat   = post.get('grip_limited_lateral_mask', np.zeros_like(t, dtype=bool))
        gl_brake = post.get('grip_brake_mask',            np.zeros_like(t, dtype=bool))
        gl_trac  = post.get('grip_traction_mask',         np.zeros_like(t, dtype=bool))

        # === 5) Construcción DataFrame ==============================================
        data = {
            # ----- Tiempo / distancia / entradas -----
            "t_s": t,
            "dist_m": dist,
            "CarSpeed": vx,
            "Ax": ax,
            "Ay": ay,
            "throttle": thr,
            "brake": brk,

            # ----- Estados chasis -----
            "heave_mm": heave*1000,
            "heave_rate_mps": heave_rate,
            "Front_heave": heave_front,
            "Rear_heave": heave_rear,
            "pitch_rad": pitch, "pitch_deg": np.degrees(pitch),
            "pitch_rate_rps": pitch_rate,
            "roll_rad": roll, "roll_deg": np.degrees(roll),
            "roll_rate_rps": roll_rate,

            # ----- Travels (relativos; compresión negativa) -----
            "travel_FL_mm": trv[0]*1000, "travel_FR_mm": trv[1]*1000,
            "travel_RL_mm": trv[2]*1000, "travel_RR_mm": trv[3]*1000,
            "FL_Damper": dmp_trv[0]*1000, "FR_Damper": dmp_trv[1]*1000,
            "RL_Damper": dmp_trv[2]*1000, "RR_Damper": dmp_trv[3]*1000,

            # ----- Fuerzas suspensión -----
            "Fspring_FL_N": Fspr[0], "Fspring_FR_N": Fspr[1],
            "Fspring_RL_N": Fspr[2], "Fspring_RR_N": Fspr[3],
            "Fbump_FL_N": Fbump[0], "Fbump_FR_N": Fbump[1],
            "Fbump_RL_N": Fbump[2], "Fbump_RR_N": Fbump[3],
            "Fdamper_FL_N": Fdmp[0], "Fdamper_FR_N": Fdmp[1],
            "Fdamper_RL_N": Fdmp[2], "Fdamper_RR_N": Fdmp[3],
            "Farb_FL_N": Farb[0], "Farb_FR_N": Farb[1],
            "Farb_RL_N": Farb[2], "Farb_RR_N": Farb[3],

            # ----- Fuerza neumático & wheel load -----
            "FL_Load": Wload[0], "FR_Load": Wload[1],
            "RL_Load": Wload[2], "RR_Load": Wload[3],

            # ----- Aero -----
            "F_AeroLoad": -FzAeroF/9.81,
            "R_AeroLoad":  -FzAeroR/9.81,
            "F_RH": RH_front,
            "R_RH": RH_rear,

            # ----- Máscaras grip -----
            "GripLat_mask":   gl_lat.astype(int),
            "GripBrake_mask": gl_brake.astype(int),
            "GripTrac_mask":  gl_trac.astype(int),
        }

        df_all = pd.DataFrame(data)

        # === 6) Guardar =============================================================
        csv_path = os.path.join(out_dir, f"{setup_name}_results_full.csv")
        df_all.to_csv(csv_path, index=False)
        print(f"[INFO] CSV extendido guardado en {csv_path}")
    # -------------------------------------------------------------------------------

def run_in_thread(sol, post, setup_name="Setup"):
    thread = Thread(target=launch_dash, args=(sol, post, setup_name))
    thread.start()