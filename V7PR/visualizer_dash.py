import plotly.graph_objects as go
import dash
from dash import dcc, html
from threading import Thread
import numpy as np
import os
import pandas as pd
from plotly.offline import plot
from scipy.signal import savgol_filter
from datetime import datetime
import plotly.graph_objects as go
from gui_v2 import load_track_channels

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
GRID_STYLE = {"display": "grid","gridTemplateColumns": "repeat(auto-fit, minmax(480px, 1fr))","gap": "20px",}
CARD_STYLE = {"width": "100%","padding": "6px 8px",}
GRAPH_CFG = {"displayModeBar": False}
STANDARD_HEIGHT = "320px"

app = dash.Dash(__name__)
app.layout = html.Div([html.H3("Resultados no cargados")])
server = app.server

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

        spring_travel = smooth_signal(post['travel_rel']) * 1000
        travel = post['z_s']*1000
        wheel_f = post['f_wheel']          # shape (4, N)
        grip_mask = post['grip_limited_lateral_mask']  # (N,)
        wheel_ld = post['wheel_load']       # (4,N)
        wheel_names = ["FL", "FR", "RL", "RR"]

        #v_damper = smooth_signal(post['v_damper'])
        travel_filtered = smooth_signal(travel)
        h0_static = float(sol.y[0][0])               # primera muestra = equilibrio
        heave_mm  = (sol.y[0] - h0_static) * 1000    # pasa a mm
        heave_ax_f = smooth_signal(post['heave_front'] * 1000)   # mm
        heave_ax_r = smooth_signal(post['heave_rear']  * 1000)   # mm
        rh = smooth_signal(post['RH']*1000)   # mm
        heave_filtered = smooth_signal(heave_mm)
        pitch_filtered = smooth_signal(np.degrees(sol.y[2]))
        roll_filtered = smooth_signal(np.degrees(sol.y[4]))
        wheel_f_filtered = smooth_signal(wheel_f)
        spring_force_filtered = smooth_signal(post['f_spring'])
        damper_force_filtered = smooth_signal(post['f_damper'])
        arb_force_filtered = smooth_signal(post['f_arb'])

        # Travel absoluto por rueda
        fig = go.Figure([
            go.Scatter(x=distance, y=spring_travel[0], name="Spring Travel FL"),
            go.Scatter(x=distance, y=spring_travel[1], name="Spring Travel FR"),
            go.Scatter(x=distance, y=spring_travel[2], name="Spring Travel RL"),
            go.Scatter(x=distance, y=spring_travel[3], name="Spring Travel RR"),
        ])
        fig.update_layout(title="Travel [mm]", xaxis_title="Distance [m]", yaxis_title="Travel [mm]")
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
            go.Scatter(x=distance, y=heave_ax_f, name="Heave Front [mm]"),
            go.Scatter(x=distance, y=heave_ax_r, name="Heave Rear [mm]")
        ])
        fig.update_layout(title="Heave per Axle (ΔRH) [mm]",
                          xaxis_title="Distance [m]",
                          yaxis_title="Heave [mm]")
        graphs.append(html.Div(
            dcc.Graph(figure=fig, config=GRAPH_CFG, style={"height": STANDARD_HEIGHT}),
            style=CARD_STYLE))

        # Pitch (filtrado)
        fig = go.Figure([
            go.Scatter(x=distance, y=pitch_filtered, name="Pitch [°]")
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
            go.Scatter(x=distance, y=post['f_bump'][0], name="Bumpstop FL"),
            go.Scatter(x=distance, y=post['f_bump'][1], name="Bumpstop FR"),
            go.Scatter(x=distance, y=post['f_bump'][2], name="Bumpstop RL"),
            go.Scatter(x=distance, y=post['f_bump'][3], name="Bumpstop RR"),
        ])
        fig.update_layout(title="Bumpstop Force per Wheel [N]", xaxis_title="Distance [m]", yaxis_title="Force [N]")
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
            go.Scatter(x=distance, y=damper_force_filtered[0], name="Damper FL"),
            go.Scatter(x=distance, y=damper_force_filtered[1], name="Damper FR"),
            go.Scatter(x=distance, y=damper_force_filtered[2], name="Damper RL"),
            go.Scatter(x=distance, y=damper_force_filtered[3], name="Damper RR"),
        ])
        fig.update_layout(
            title="Damper Force per Wheel [N]",
            xaxis_title="Distance [m]",
            yaxis_title="Force [N]"
        )
        graphs.append(html.Div(
            dcc.Graph(figure=fig, config=GRAPH_CFG, style={"height": STANDARD_HEIGHT}),
            style=CARD_STYLE))

        fig = go.Figure([
            go.Scatter(x=distance, y=arb_force_filtered[0], name="ARB FL"),
            go.Scatter(x=distance, y=arb_force_filtered[1], name="ARB FR"),
            go.Scatter(x=distance, y=arb_force_filtered[2], name="ARB RL"),
            go.Scatter(x=distance, y=arb_force_filtered[3], name="ARB RR"),
        ])
        fig.update_layout(
            title="ARB Force per Wheel [N]",
            xaxis_title="Distance [m]",
            yaxis_title="Force [N]"
        )
        graphs.append(html.Div(
            dcc.Graph(figure=fig, config=GRAPH_CFG, style={"height": STANDARD_HEIGHT}),
            style=CARD_STYLE))
        if "arb_torque_front" in post:
            fig_arb = go.Figure([
                go.Scatter(x=distance, y=post["arb_torque_front"], name="Front ARB"),
                go.Scatter(x=distance, y=post["arb_torque_rear"],  name="Rear ARB")
            ])
            fig_arb.update_layout(title="Anti-roll Bar Torque [Nm]", xaxis_title="Distance [m]", yaxis_title="Torque [Nm]")
            graphs.append(html.Div(
                dcc.Graph(figure=fig_arb, config=GRAPH_CFG, style={"height": STANDARD_HEIGHT}),
                style=CARD_STYLE))

        
        app.layout = html.Div([
            html.H1(f"Resultados 7-Post Rig: {setup_name}"),
            html.Div(graphs, style=GRID_STYLE)
        ])

    plot_data()
    app.run(port=8050, debug=False)

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
            layout.append(html.Div(
                dcc.Graph(figure=fig, config=GRAPH_CFG,
                          style={"height": STANDARD_HEIGHT}),
                style=CARD_STYLE))
        except KeyError:
            continue

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
        
        fig_accu.update_layout(height=300, margin=dict(t=40, b=40, l=40, r=10))
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
                                     height=300,
                                     margin=dict(t=40, b=40, l=40, r=10))
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
                text=labels,             # <--- etiqueta inteligente
                textposition='top center'
            )
        ])
        fig_scatter_frh.update_layout(
            title="FRH RMS vs Contact Patch Load RMS",
            xaxis_title="Front Ride Height RMS [mm]",
            yaxis_title="Contact Patch Load RMS [N]",
            height=300, margin=dict(t=40, b=40, l=40, r=10)
        )
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
             height=300, margin=dict(t=40, b=40, l=40, r=10)
        )
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
             height=300, margin=dict(t=40, b=40, l=40, r=10)
        )
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
                 height=300, margin=dict(t=40, b=40, l=40, r=10)
            )
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
                 height=300, margin=dict(t=40, b=40, l=40, r=10)
            )
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
             height=300, margin=dict(t=40, b=40, l=40, r=10)
        )
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
             height=300, margin=dict(t=40, b=40, l=40, r=10)
        )
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
             height=300, margin=dict(t=40, b=40, l=40, r=10)
        )
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
        legend=dict(x=0.01, y=0.99),
        
        height=300,
        margin=dict(t=40, b=40, l=40, r=10)
    )
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

# visualizer_dash.py

def build_kpi_figs_from_data(kpi_data, setup_names, save_dir=None):
    """
    Devuelve una lista de figuras Plotly exactamente iguales a las que
    muestra launch_dash_kpis, pero sin arrancar Dash.
    """
    import numpy as np
    import plotly.graph_objects as go
    figs = []

    # decide qué etiqueta mostrar en cada punto
    if len(setup_names) > 1:
        labels = setup_names
    else:
        labels = [k['track_name'] for k in kpi_data]

    # 1) Wheel Load Max por rueda
    for title, unit, key_mean, key_std, factor in KPI_DEFINITIONS:
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
                                     setup_names, ['FL','FR','RL','RR'])
            fig.update_layout(width=450, height=300,
                              margin=dict(t=40,b=40,l=40,r=10))
            figs.append(fig)
        except KeyError:
            pass

    # 2) Accumulated road noise
    try:
        noise_by_track = {}
        for k in kpi_data:
            tname = k['track_name']
            if tname not in noise_by_track:
                noise_by_track[tname] = (
                    k['tracknoise_accu_front'],
                    k['tracknoise_accu_rear'])
        tracks = list(noise_by_track.keys())
        front_vals = [noise_by_track[t][0] for t in tracks]
        rear_vals  = [noise_by_track[t][1] for t in tracks]
        fig_accu = go.Figure()
        fig_accu.add_trace(go.Bar(name="Front Axle", x=tracks, y=front_vals, marker_color='royalblue'))
        fig_accu.add_trace(go.Bar(name="Rear  Axle", x=tracks, y=rear_vals,  marker_color='tomato'))
        fig_accu.update_layout(
            title="Accumulated Road Track-noise Normalised by Lap Time",
            xaxis_title="Track", yaxis_title="Normalised Accu. Track-noise [mm/s]",
            barmode="group",
            width=450, height=300,
            margin=dict(t=40,b=40,l=40,r=10))
        figs.append(fig_accu)
    except KeyError:
        pass

    # 3) Tabla Pitch RMS
    try:
        pitch_vals = [float(k['pitch_rms']) for k in kpi_data]
        fig_pitch_table = go.Figure(data=[go.Table(
            header=dict(values=["Setup","Pitch RMS [°]"],
                        fill_color='paleturquoise', align='left'),
            cells=dict(values=[setup_names,pitch_vals],
                       fill_color='lavender', align='left')
        )])
        fig_pitch_table.update_layout(
            title="Pitch RMS por Setup (Resumen Numérico)",
            width=450, height=300,
            margin=dict(t=40,b=40,l=40,r=10))
        figs.append(fig_pitch_table)
    except KeyError:
        pass

    # 4) FRH RMS vs Contact Patch Load RMS
    try:
        frh_vals  = [k['frh_rms'] * 1000 for k in kpi_data]
        front_load= [k['front_load_rms'] for k in kpi_data]
        fig_frh = go.Figure(data=[go.Scatter(
            x=frh_vals, y=front_load,
            mode='markers+text',
            text=labels, textposition='top center'
        )])
        fig_frh.update_layout(
            title="FRH RMS vs Contact Patch Load RMS",
            xaxis_title="Front Ride Height RMS [mm]",
            yaxis_title="Contact Patch Load RMS [N]",
            width=450, height=300,
            margin=dict(t=40,b=40,l=40,r=10))
        figs.append(fig_frh)
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
             height=300, margin=dict(t=40, b=40, l=40, r=10)
        )
        figs.append(fig_scatter_rrh)
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
             height=300, margin=dict(t=40, b=40, l=40, r=10)
        )
        figs.append(fig_rh_scatter)
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
                 height=300, margin=dict(t=40, b=40, l=40, r=10)
            )
            figs.append(fig_cmp)
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
                 height=300, margin=dict(t=40, b=40, l=40, r=10)
            )
            figs.append(fig_bt)
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
             height=300, margin=dict(t=40, b=40, l=40, r=10)
        )
        figs.append(fig)

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

    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        for i, fig in enumerate(figs, 1):
            fig.write_html(os.path.join(save_dir, f'kpi_{i:02d}.html'),
                           include_plotlyjs='cdn')
    return figs


def get_kpi_figures(setups, save_dir: str | None = None):
    import os, numpy as np, plotly.graph_objects as go

    # convertir tupla de resultados en listas
    kpi_data = [post for (_, post, _, _) in setups]
    setup_names = []
    for _, post, setup_path, _ in setups:
        name = post.get("setup_name") or os.path.splitext(os.path.basename(setup_path))[0]
        setup_names.append(name)

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
            fig.update_layout(width=450, height=300, margin=dict(t=40,b=40,l=40,r=10))
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
                width=450, height=300,
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
            width=450, height=300,
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
            width=450, height=300,
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
            width=450, height=300,
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
            width=450, height=300,
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
                width=450, height=300,
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
                width=450, height=300,
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
    def add_cpl_front_vs_rear(front_vals, rear_vals, title):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=setup_names, y=front_vals, mode='markers', name="Front"
        ))
        fig.add_trace(go.Scatter(
            x=setup_names, y=rear_vals,  mode='markers', name="Rear"
        ))
        fig.update_layout(
            title=title,
            xaxis_title="Setup",
            yaxis_title="CPL RMS [N]",
            width=450, height=300,
            margin=dict(t=40,b=40,l=40,r=10)
        )
        figs.append(fig)

    try:
        fv_brake = [k['front_load_rms_brake'] for k in kpi_data]
        rv_brake = [k['rear_load_rms_brake']  for k in kpi_data]
        add_cpl_front_vs_rear(fv_brake, rv_brake, "Contact Patch Load RMS en Frenada [N]")
    except KeyError:
        pass

    try:
        fv_trac = [k['front_load_rms_traction'] for k in kpi_data]
        rv_trac = [k['rear_load_rms_traction']  for k in kpi_data]
        add_cpl_front_vs_rear(fv_trac, rv_trac, "Contact Patch Load RMS en Tracción [N]")
    except KeyError:
        pass

    # 11) PSD de heave (front vs rear) – convertimos a mm²/Hz en log
    try:
        fig_psd_heave = go.Figure()
        for post, name in zip(kpi_data, setup_names):
            if ('f_psd_front' in post and 'psd_heave_front' in post and
                'f_psd_rear'  in post and 'psd_heave_rear'  in post):
                f_front = post['f_psd_front']
                psd_f   = np.array(post['psd_heave_front']) * 1e6  # m²/Hz → mm²/Hz
                f_rear  = post['f_psd_rear']
                psd_r   = np.array(post['psd_heave_rear'])  * 1e6
                fig_psd_heave.add_trace(go.Scatter(
                    x=f_front, y=psd_f, mode='lines', name=f"{name} – Front"
                ))
                fig_psd_heave.add_trace(go.Scatter(
                    x=f_rear,  y=psd_r, mode='lines', name=f"{name} – Rear"
                ))
        if fig_psd_heave.data:
            fig_psd_heave.update_layout(
                title="PSD of Heave Motion by Axle (Front vs Rear)",
                xaxis_title="Frequency (Hz)",
                yaxis_title="PSD Heave (mm²/Hz)",
                yaxis_type="log",
                width=450, height=300,
                margin=dict(t=40,b=40,l=40,r=10)
            )
            figs.append(fig_psd_heave)
    except Exception:
        pass

    # 12) PSD de Pitch‑induced vertical (front vs rear)
    try:
        fig_psd_pitch = go.Figure()
        for post, name in zip(kpi_data, setup_names):
            if ('f_psd_pitch_front' in post and 'psd_pitch_front' in post and
                'f_psd_pitch_rear'  in post and 'psd_pitch_rear'  in post):
                f_front = post['f_psd_pitch_front']
                psd_f   = np.array(post['psd_pitch_front']) * 1e6
                f_rear  = post['f_psd_pitch_rear']
                psd_r   = np.array(post['psd_pitch_rear'])  * 1e6
                fig_psd_pitch.add_trace(go.Scatter(
                    x=f_front, y=psd_f, mode='lines', name=f"{name} – Pitch Front"
                ))
                fig_psd_pitch.add_trace(go.Scatter(
                    x=f_rear,  y=psd_r, mode='lines', name=f"{name} – Pitch Rear"
                ))
        if fig_psd_pitch.data:
            fig_psd_pitch.update_layout(
                title="PSD of Pitch‑Induced Vertical by Axle (Front vs Rear)",
                xaxis_title="Frequency (Hz)",
                yaxis_title="PSD Pitch→Vertical (mm²/Hz)",
                yaxis_type="log",
                width=450, height=300,
                margin=dict(t=40,b=40,l=40,r=10)
            )
            figs.append(fig_psd_pitch)
    except Exception:
        pass

    # 13) PSD de carga – Magnitud, suavizado con Savitzky‑Golay
    try:
        fig_psd_mag = go.Figure()
        for post, name in zip(kpi_data, setup_names):
            if ('f_psd_load' in post and 'psd_load_mag_front' in post and
                'psd_load_mag_rear' in post):
                f_load  = np.array(post['f_psd_load'])
                mag_f   = np.array(post['psd_load_mag_front'])
                mag_r   = np.array(post['psd_load_mag_rear'])
                # Suavizado ligero con Savitzky‑Golay
                mag_f_smooth = smooth_signal(mag_f, window=51, polyorder=3)
                mag_r_smooth = smooth_signal(mag_r, window=51, polyorder=3)
                fig_psd_mag.add_trace(go.Scatter(
                    x=f_load, y=mag_f_smooth, mode='lines', name=f"{name} – Mag Front [dB] (suavizado)"
                ))
                fig_psd_mag.add_trace(go.Scatter(
                    x=f_load, y=mag_r_smooth, mode='lines', name=f"{name} – Mag Rear  [dB] (suavizado)"
                ))
        if fig_psd_mag.data:
            fig_psd_mag.update_layout(
                title="PSD de Carga – Magnitud (Front vs Rear, Suavizado)",
                xaxis=dict(title="Frecuencia [Hz]", type="log"),
                yaxis=dict(title="Magnitud [dB]"),
                legend=dict(x=0.01, y=0.99),
                width=450, height=300,
                margin=dict(t=40,b=40,l=40,r=10)
            )
            figs.append(fig_psd_mag)
    except Exception:
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
    
def export_full_report(setups, export_path="export_full_report.html"):
    html_sections = []

    html_sections.append("<h1>Reporte de Simulación - Post Rig</h1>")
 
    # KPIs comparativos (siempre incluir, incluso con un solo setup)
    html_sections.append('<h2>📊 KPIs Comparativos</h2>')
    html_sections.append('<div style="display:flex;flex-wrap:wrap;gap:12px">')
    kpi_figs = get_kpi_figures(setups)
    for fig in kpi_figs:
        html_sections.append(plot(fig, include_plotlyjs=False, output_type='div'))
    html_sections.append('</div>')

    # Exporta a HTML
    with open(export_path, "w", encoding="utf-8") as f:
        f.write("<html><head>")
        f.write('<meta charset="utf-8">')
        f.write('<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>')
        f.write("<title>Reporte 4-Post Rig</title></head><body>")
        for section in html_sections:
            f.write(section)
        f.write("</body></html>")

    # --- EXPORTAR CSV POR SETUP ---
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
        # solve_ivp puede devolver sol.t idéntico a t_in si se integró en esos nodos;
        # por robustez interpolamos todos los inputs al tiempo de la solución.
        t = sol.t
        vx  = np.interp(t, t_in, vx_in) * 3.6
        ax  = np.interp(t, t_in, ax_in) / 9.81
        ay  = np.interp(t, t_in, ay_in) / 9.81
        thr = np.interp(t, t_in, thr_in)
        brk = np.interp(t, t_in, brk_in)
        ztrk = np.vstack([np.interp(t, t_in, ztrk_in[i]) for i in range(4)])

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

        # Unsprung (índices como en postprocess_7dof)
        zu_idx    = [8, 6, 10, 12]
        zudot_idx = [9, 7, 11, 13]
        zu     = np.stack([sol.y[i] for i in zu_idx])      # (4,N)
        zudot  = np.stack([sol.y[i] for i in zudot_idx])   # (4,N)

        # === 4) Canales de salida del postprocesado =================================
        # (Todos tienen forma (4,N) salvo los listados como escalares/1D)
        trv     = post['travel']            # (4,N) m (neg = compresión)
        dmp_trv = post['travel_rel']     # (4,N) m
        mext    = post['margen_ext']        # (4,N) m
        mcomp   = post['margen_comp']       # (4,N) m
        Fspr    = post['f_spring']          # (4,N) N
        Fdmp    = post['f_damper']          # (4,N) N
        Fbump   = post['f_bump']            # (4,N) N
        Farb    = post['f_arb']             # (4,N) N
        Ftire   = post['f_tire']            # (4,N) N
        Wload   = post['wheel_load']        # (4,N) N  (asegúrate: si venía en "kg", multiplica por 9.81)
        # Si tu versión de post devolviera wheel_load en "kg", descomenta:
        #Wload = Wload * 9.81

        FzAeroF = post.get('Fz_aero_front', np.zeros_like(t))
        FzAeroR = post.get('Fz_aero_rear',  np.zeros_like(t))
        
        # heave por eje ya viene en mm en post:
        RH_F_mm = np.mean(post['RH'][0:2], axis=0) * 1000  # Front axle
        RH_R_mm = np.mean(post['RH'][2:4], axis=0) * 1000  # Rear  axle

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
            "ztrk_FL_mm": ztrk[0]*1000, "ztrk_FR_mm": ztrk[1]*1000,
            "ztrk_RL_mm": ztrk[2]*1000, "ztrk_RR_mm": ztrk[3]*1000,

            # ----- Estados chasis -----
            "heave_mm": heave*1000,
            "heave_rate_mps": heave_rate,
            "heave_front_mm": heave_front,
            "heave_rear_mm": heave_rear,
            "pitch_rad": pitch, "pitch_deg": np.degrees(pitch),
            "pitch_rate_rps": pitch_rate,
            "roll_rad": roll, "roll_deg": np.degrees(roll),
            "roll_rate_rps": roll_rate,

            # ----- Unsprung abs -----
            "zu_FL_m": zu[0], "zu_FR_m": zu[1], "zu_RL_m": zu[2], "zu_RR_m": zu[3],
            "zudot_FL_mps": zudot[0], "zudot_FR_mps": zudot[1], "zudot_RL_mps": zudot[2], "zudot_RR_mps": zudot[3],

            # ----- Travels (relativos; compresión negativa) -----
            "travel_FL_mm": trv[0]*1000, "travel_FR_mm": trv[1]*1000,
            "travel_RL_mm": trv[2]*1000, "travel_RR_mm": trv[3]*1000,
            "damper_FL_mm": dmp_trv[0]*1000, "damper_FR_mm": dmp_trv[1]*1000,
            "damper_RL_mm": dmp_trv[2]*1000, "damper_RR_mm": dmp_trv[3]*1000,
            "margenExt_FL_mm": mext[0]*1000, "margenExt_FR_mm": mext[1]*1000,
            "margenExt_RL_mm": mext[2]*1000, "margenExt_RR_mm": mext[3]*1000,
            "margenComp_FL_mm": mcomp[0]*1000, "margenComp_FR_mm": mcomp[1]*1000,
            "margenComp_RL_mm": mcomp[2]*1000, "margenComp_RR_mm": mcomp[3]*1000,

            # ----- Fuerzas suspensión -----
            "Fspring_FL_N": Fspr[0], "Fspring_FR_N": Fspr[1],
            "Fspring_RL_N": Fspr[2], "Fspring_RR_N": Fspr[3],
            "Fbump_FL_N": Fbump[0], "Fbump_FR_N": Fbump[1],
            "Fbump_RL_N": Fbump[2], "Fbump_RR_N": Fbump[3],
            "Fdamper_FL_N": Fdmp[0], "Fdamper_FR_N": Fdmp[1],
            "Fdamper_RL_N": Fdmp[2], "Fdamper_RR_N": Fdmp[3],
            "Farb_FL_N": Farb[0], "Farb_FR_N": Farb[1],
            "Farb_RL_N": Farb[2], "Farb_RR_N": Farb[3],
            "ARB_torque_front_Nm": post.get("arb_torque_front", np.zeros_like(t)),
            "ARB_torque_rear_Nm":  post.get("arb_torque_rear",  np.zeros_like(t)),

            # ----- Fuerza neumático & wheel load -----
            "Ftire_FL_N": Ftire[0], "Ftire_FR_N": Ftire[1],
            "Ftire_RL_N": Ftire[2], "Ftire_RR_N": Ftire[3],
            "WheelLoad_FL_N": Wload[0], "WheelLoad_FR_N": Wload[1],
            "WheelLoad_RL_N": Wload[2], "WheelLoad_RR_N": Wload[3],

            # ----- Aero -----
            "FzAeroFront_N": -FzAeroF,
            "FzAeroRear_N":  -FzAeroR,
            "RH_F_mm": RH_F_mm,
            "RH_R_mm": RH_R_mm,

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