# visualizer_dash.py
import plotly.graph_objects as go
import dash
from dash import dcc, html
from threading import Thread
import numpy as np
import os 
from plotly.offline import plot
from scipy.signal import savgol_filter
from datetime import datetime

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
        dt_vec   = np.gradient(sol.t)
        distance = np.cumsum(post['vx'] * dt_vec)
        travel = np.vstack([smooth_signal(travel_i) for travel_i in post['travel']])
        wheel_f = post['wheel_load_N'] 
        grip_mask = post['grip_mask_lat']  # (N,)
        wheel_ld = post['wheel_load_kg']       # (4,N)
        wheel_names = ["FL", "FR", "RL", "RR"]
        heave_filtered = smooth_signal(sol.y[0])
        pitch_filtered = smooth_signal(np.degrees(sol.y[2]))
        roll_filtered = smooth_signal(np.degrees(sol.y[4]))

        # === Damper Travel por rueda ===
        dt = np.gradient(sol.t)
        # damper_travel está en metros, lo pasamos a mm
        damper_travel_mm = post['damper_travel'] * 1000  # (4, N)
        graphs.append(dcc.Graph(figure=go.Figure([
            go.Scatter(x=distance, y=damper_travel_mm[0], name="Damper Travel FL"),
            go.Scatter(x=distance, y=damper_travel_mm[1], name="Damper Travel FR"),
            go.Scatter(x=distance, y=damper_travel_mm[2], name="Damper Travel RL"),
            go.Scatter(x=distance, y=damper_travel_mm[3], name="Damper Travel RR"),
        ]).update_layout(
            title="Damper Travel por rueda [mm]",
            xaxis_title="Distance [m]",
            yaxis_title="Travel [mm]"
        )))
        
        # === Heave (filtrado) ===
        graphs.append(dcc.Graph(
            figure=go.Figure([
                go.Scatter(
                    x=distance,
                    y=heave_filtered * 1000,  # [mm]
                    name="Heave"
                )
            ]).update_layout(
                title="Heave Motion (Filtered)",
                xaxis_title="Distance [m]",
                yaxis_title="Heave [mm]"
            )
        ))

        # === Pitch (filtrado) ===
        graphs.append(dcc.Graph(
            figure=go.Figure([
                go.Scatter(
                    x=distance,
                    y=pitch_filtered,  # [°]
                    name="Pitch"
                )
            ]).update_layout(
                title="Pitch Motion (Filtered)",
                xaxis_title="Distance [m]",
                yaxis_title="Pitch [°]"
            )
        ))

        # === Roll (filtrado) ===
        graphs.append(dcc.Graph(
            figure=go.Figure([
                go.Scatter(
                    x=distance,
                    y=roll_filtered,   # [°]
                    name="Roll"
                )
            ]).update_layout(
                title="Roll Motion (Filtered)",
                xaxis_title="Distance [m]",
                yaxis_title="Roll [°]"
            )
        ))

        # === Zona de grip limitado lateral ===
        grip_limited_trace = None
        if 'grip_mask_lat' in post:
            mask = post['grip_mask_lat']                # (N,)
            # señal promedio de carga por eje [N]
            avg_tire_load = np.mean(post['wheel_load_N'], axis=0)
            # solo mostramos donde hay grip limitado
            y_gl = np.where(mask, avg_tire_load, np.nan)
            grip_limited_trace = go.Scatter(
                x=distance,
                y=y_gl,
                name="Grip-Limited Zone",
                mode='lines',
                line=dict(color='black', width=3, dash='dot'),
                showlegend=True
            )
        if grip_limited_trace:
            graphs.append(dcc.Graph(figure=go.Figure([grip_limited_trace]).update_layout(
                title="Zona de Grip-Limited (Carga Promedio) [N]",
                xaxis_title="Distance [m]",
                yaxis_title="Average Tire Load [N]"
            )))

        # === Aerodynamic Downforce per Axle [N] ===
        ae_front = post.get('Fz_aero_front', np.zeros_like(post['vx']))
        ae_rear  = post.get('Fz_aero_rear',  np.zeros_like(post['vx']))
        fig_aero = go.Figure([
            go.Scatter(x=distance, y=-ae_front, name="Downforce Front"),
            go.Scatter(x=distance, y=-ae_rear,  name="Downforce Rear"),
        ])
        fig_aero.update_layout(
            title="Aerodynamic Downforce per Axle [N]",
            xaxis_title="Distance [m]",
            yaxis_title="Force [N]"
        )
        graphs.append(dcc.Graph(figure=fig_aero))

        # Tire load
        fig_load = go.Figure()

        # Curvas de carga por rueda
        for i, name in enumerate(wheel_names):
            fig_load.add_trace(
                go.Scatter(
                    x=distance,
                    y=wheel_ld[i],
                    name=name
                )
            )

        # Marcadores grip-limited para cada rueda
        for i, name in enumerate(wheel_names):
            fig_load.add_trace(go.Scatter(
                x=distance[grip_mask],
                y=wheel_ld[i, grip_mask],
                mode='markers',
                name=f"{name} Grip-Limited",
                marker=dict(color='black', symbol='line-ns-open')
            ))

        fig_load.update_layout(
            title="Tire Load per Wheel [N]",
            xaxis_title="Distance [m]",
            yaxis_title="Load [N]"
        )
        graphs.append(dcc.Graph(figure=fig_load))
        
        # === Bumpstop Forces por rueda ===
        graphs.append(dcc.Graph(
            figure=go.Figure([
                go.Scatter(x=distance, y=post['f_bump'][0], name="Bumpstop FL"),
                go.Scatter(x=distance, y=post['f_bump'][1], name="Bumpstop FR"),
                go.Scatter(x=distance, y=post['f_bump'][2], name="Bumpstop RL"),
                go.Scatter(x=distance, y=post['f_bump'][3], name="Bumpstop RR"),
            ]).update_layout(
                title="Bumpstop Force per Wheel [N]",
                xaxis_title="Distance [m]",
                yaxis_title="Force [N]"
            ),
            id='bumpstop-forces',
            style={'height': '300px'},
            config={'displayModeBar': False}
        ))

        # === Ride Height RMS en zonas grip-limited ===
        if 'grip_mask_lat' in post:
            mask = post['grip_mask_lat']
            zf = (travel[0] + travel[1]) / 2
            zr = (travel[2] + travel[3]) / 2
            rms_zf = np.sqrt(np.mean(zf[mask]**2)) * 1000
            rms_zr = np.sqrt(np.mean(zr[mask]**2)) * 1000
            fig_rms_heights = go.Figure([
                go.Bar(x=["Front", "Rear"], y=[rms_zf, rms_zr])
            ])
            fig_rms_heights.update_layout(
                title="Ride Height RMS en grip-limited [mm]",
                yaxis_title="RMS [mm]"
            )
            graphs.append(dcc.Graph(figure=fig_rms_heights))

        # === FRH vs Contact Patch Load (RMS) ===
        frh = (travel[0] + travel[1]) / 2
        frh_rms = np.sqrt(np.mean(frh**2)) * 1000
        load_f = (wheel_ld[0] + wheel_ld[1]) / 2
        load_f_rms = np.sqrt(np.mean(load_f**2))
        fig_frh_vs_load = go.Figure(data=go.Scatter(
            x=[frh_rms], y=[load_f_rms],
            mode='markers+text',
            text=["Front"],
            textposition='top center'
        ))
        fig_frh_vs_load.update_layout(
            title="FRH RMS vs Contact Patch Load RMS",
            xaxis_title="Front Ride Height RMS [mm]",
            yaxis_title="Contact Patch Load RMS [N]"
        )
        graphs.append(dcc.Graph(figure=fig_frh_vs_load))
        

        app.layout = html.Div([
            html.H1(f"Resultados 7-Post Rig: {setup_name}"),
            html.H2("Señales dinámicas"),
            *graphs
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

    # Mapeo de claves antiguas -> nuevas para mantener compatibilidad
    key_map = {
        'f_tire_grip_limited_max': 'f_tire_gl_max',
        'f_tire_grip_limited_min': 'f_tire_gl_min',
        'frh_rms': 'heave_f_rms_mm',
        'rrh_rms': 'heave_r_rms_mm',
        'front_load_rms': 'load_f_rms_N',
        'rear_load_rms': 'load_r_rms_N',
        'frh_rms_nongrip': 'heave_f_rms_ng_mm',
        'rrh_rms_nongrip': 'heave_r_rms_ng_mm',
        'front_load_rms_nongrip': 'load_f_rms_ng',
        # Campos PSD/FFT actualizados
        'f_psd_front': 'f_psd_heave_f',
        'psd_heave_front': 'psd_heave_f',
        'f_psd_rear': 'f_psd_heave_r',
        'psd_heave_rear': 'psd_heave_r',
        'f_psd': 'f_psd_heave',
        'pitch_rms': 'pitch_rms_deg',
        'f_psd_pitch_front': 'f_psd_pitch_f',
        'psd_pitch_front': 'psd_pitch_f_dB',
        'f_psd_pitch_rear': 'f_psd_pitch_r',
        'psd_pitch_rear': 'psd_pitch_r_dB',
        'psd_load_mag_front': 'psd_load_f_dB',
        'psd_load_mag_rear': 'psd_load_r_dB',
    }

    def get_val(d, key):
        if key in d:
            return d[key]
        mapped = key_map.get(key)
        if mapped and mapped in d:
            return d[mapped]
        raise KeyError(key)

    def kpi_bar(title, unit, values_list):
        fig = go.Figure()
        for values, name in zip(values_list, setup_names):
            fig.add_trace(go.Bar(name=name, x=kpi_labels, y=values))
        fig.update_layout(
            title=title,
            yaxis_title=unit,
            barmode='group'
        )
        return dcc.Graph(figure=fig)

    # --- DEFINICIÓN CENTRALIZADA DE LOS KPIs --- 
    kpi_definitions = [
        ("Tire Load Max (Grip-Limited) [N]", "N", "f_tire_grip_limited_max", 1),
        ("Tire Load Min (Grip-Limited) [N]", "N", "f_tire_grip_limited_min", 1),
    ]

    # --- ARRANCAR LAYOUT CON UN TÍTULO PRINCIPAL ---
    layout = [html.H1("Comparativa de KPIs entre Setups")]

    for title, unit, key, factor in kpi_definitions:
        try:
            values_list = [get_val(k, key) * factor for k in kpi_data]
            layout.append(kpi_bar(title, unit, values_list))
        except KeyError:
            continue

    # --- KPIs PERSONALIZADOS: Road Noise, Pitch vs Distance, Pitch RMS, Ride Height RMS, Scatter, etc. ---

    # ─── 1) Road‐Noise: wheel vertical‐speed RMS [mm/s] ──────────────────────────
    try:
        front_noise_vals, rear_noise_vals = [], []
        track_labels = [k['track_name'] for k in kpi_data]

        for k in kpi_data:
            try:
                lap_time = float(k.get('lap_time', 1.0))    # [s]
                zt = k.get('zt_hp', k['zt'])                # [FL, FR, RL, RR] en metros
                n_samples = len(zt[0])
                if n_samples < 2:
                    raise ValueError("zt demasiado corto")

                # Δt = lap_time / (n_samples - 1)
                dt = lap_time / (n_samples - 1)

                # velocidad vertical (mm/s) y RMS por rueda
                v_mm_s = [np.gradient(z, dt) * 1000 for z in zt]  # 4 listas
                rms_vals = [np.sqrt(np.mean(v**2)) for v in v_mm_s]

                # promedio por eje
                front_noise_vals.append(0.5 * (rms_vals[0] + rms_vals[1]))
                rear_noise_vals .append(0.5 * (rms_vals[2] + rms_vals[3]))
            except Exception as inner_e:
                print(f"[WARN] Road-noise: {inner_e}")
                front_noise_vals.append(np.nan)
                rear_noise_vals .append(np.nan)

        fig_road_noise = go.Figure()
        fig_road_noise.add_trace(go.Bar(name="Front Axle", x=track_labels, y=front_noise_vals))
        fig_road_noise.add_trace(go.Bar(name="Rear  Axle", x=track_labels, y=rear_noise_vals))

        fig_road_noise.update_layout(
            title="Qual-Derived Road Noise (Wheel-Speed RMS)",
            xaxis_title="Track",
            yaxis_title="RMS [mm/s]",
            barmode="group"
        )
        layout.append(dcc.Graph(figure=fig_road_noise))

    except Exception as e:
        print(f"[ERROR] Fallo en el gráfico de Road Noise: {e}")

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
        layout.append(dcc.Graph(figure=fig_accu))

    except KeyError:
        print("[DEBUG] No se encontró 'tracknoise_accu_*' en algún KPI")

    # ── 3) Pitch vs Distance [°] (serie temporal) ───────────────────────────────
    try:
        fig_pitch_vs_distance = go.Figure()
        for kpi, name in zip(kpi_data, setup_names):
            if 'pitch_deg' in kpi and 'distance' in kpi:
                fig_pitch_vs_distance.add_trace(go.Scatter(
                    x=kpi['distance'],
                    y=kpi['pitch_deg'],
                    mode='lines',
                    name=name
                ))
        fig_pitch_vs_distance.update_layout(
            title="Pitch vs Distance [°]",
            xaxis_title="Distance [m]",
            yaxis_title="Pitch [°]"
        )
        layout.append(dcc.Graph(figure=fig_pitch_vs_distance))
    except Exception as e:
        print(f"[DEBUG] Error en Pitch vs Distance: {e}")

    # ── 4) Tabla de Pitch RMS por Setup ──────────────────────────────────────────
    try:
        pitch_rms_vals = [float(get_val(k, 'pitch_rms')) for k in kpi_data]
        fig_pitch_table = go.Figure(data=[go.Table(
            header=dict(values=["Setup", "Pitch RMS [°]"],
                        fill_color='paleturquoise', align='left'),
            cells=dict(values=[setup_names, pitch_rms_vals],
                       fill_color='lavender', align='left'))
        ])
        fig_pitch_table.update_layout(title="Pitch RMS por Setup (Resumen Numérico)")
        layout.append(dcc.Graph(figure=fig_pitch_table))
    except KeyError:
        pass

    # ── 5) Ride Height RMS (FRH, RRH) en GLS [mm] (barras) ─────────────────────
    try:
        frh_vals = [get_val(k, 'frh_rms') * 1000 for k in kpi_data]
        rrh_vals = [get_val(k, 'rrh_rms') * 1000 for k in kpi_data]
        fig_rh_bar = go.Figure(data=[
            go.Bar(name=name, x=["Front", "Rear"], y=[float(frh), float(rrh)])
            for name, frh, rrh in zip(setup_names, frh_vals, rrh_vals)
        ])
        fig_rh_bar.update_layout(title="Ride Height RMS en GLS [mm]")
        layout.append(dcc.Graph(figure=fig_rh_bar))
    except KeyError:
        pass

    # ── 6) FRH RMS vs Contact Patch Load RMS (dispersograma etiquetado) ────────
    try:
        frh_rms_vals = [get_val(k, 'frh_rms') * 1000 for k in kpi_data]
        load_rms_vals_front = [get_val(k, 'front_load_rms') for k in kpi_data]
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
            yaxis_title="Contact Patch Load RMS [N]"
        )
        layout.append(dcc.Graph(figure=fig_scatter_frh))
    except KeyError:
        pass

    # ── 7) RRH RMS vs Contact Patch Load RMS (dispersograma etiquetado) ────────
    try:
        rrh_rms_vals = [get_val(k, 'rrh_rms') * 1000 for k in kpi_data]
        load_rms_vals_rear = [get_val(k, 'rear_load_rms') for k in kpi_data]
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
            yaxis_title="Contact Patch Load RMS [N]"
        )
        layout.append(dcc.Graph(figure=fig_scatter_rrh))
    except KeyError:
        pass

    # ── 8) FRH: GLS vs NGLS (barras) ────────────────────────────────────────────
    try:
        frh_grip_vals = [get_val(k, 'frh_rms') * 1000 for k in kpi_data]
        frh_nongrip_vals = [get_val(k, 'frh_rms_nongrip') * 1000 for k in kpi_data]
        fig_frh_compare = go.Figure()
        fig_frh_compare.add_trace(go.Bar(
            name="Grip-Limited", x=setup_names, y=frh_grip_vals))
        fig_frh_compare.add_trace(go.Bar(
            name="No Grip-Limited", x=setup_names, y=frh_nongrip_vals))
        fig_frh_compare.update_layout(
            title="Front Ride Height RMS: GLS vs NGLS [mm]",
            xaxis_title="Setup",
            yaxis_title="FRH RMS [mm]",
            barmode="group"
        )
        layout.append(dcc.Graph(figure=fig_frh_compare))
    except KeyError:
        pass

    # ── 9) RRH: GLS vs NGLS (barras) ────────────────────────────────────────────
    try:
        rrh_grip_vals = [get_val(k, 'rrh_rms') * 1000 for k in kpi_data]
        rrh_nongrip_vals = [get_val(k, 'rrh_rms_nongrip') * 1000 for k in kpi_data]
        fig_rrh_compare = go.Figure()
        fig_rrh_compare.add_trace(go.Bar(
            name="Grip-Limited", x=setup_names, y=rrh_grip_vals))
        fig_rrh_compare.add_trace(go.Bar(
            name="No Grip-Limited", x=setup_names, y=rrh_nongrip_vals))
        fig_rrh_compare.update_layout(
            title="Rear Ride Height RMS: GLS vs NGLS [mm]",
            xaxis_title="Setup",
            yaxis_title="RRH RMS [mm]",
            barmode="group"
        )
        layout.append(dcc.Graph(figure=fig_rrh_compare))
    except KeyError:
        print("[DEBUG] Faltan datos para el gráfico de RRH RMS")

    # ── 14) Front Load RMS: Braking vs Traction (barras) ────────────────────────
    try:
        brake_vals = [k['front_load_rms_brake'] for k in kpi_data]
        traction_vals = [k['front_load_rms_traction'] for k in kpi_data]
        fig_brake_vs_traction_f = go.Figure()
        fig_brake_vs_traction_f.add_trace(go.Bar(
            name="Braking", x=setup_names, y=brake_vals))
        fig_brake_vs_traction_f.add_trace(go.Bar(
            name="Traction", x=setup_names, y=traction_vals))
        fig_brake_vs_traction_f.update_layout(
            title="Front Load RMS: Braking vs Traction [N]",
            xaxis_title="Setup",
            yaxis_title="Contact Patch Load RMS [N]",
            barmode="group"
        )
        layout.append(dcc.Graph(figure=fig_brake_vs_traction_f))
    except KeyError:
        pass

    # ── 15) Rear Load RMS: Braking vs Traction (barras) ────────────────────────
    try:
        brake_vals = [k['rear_load_rms_brake'] for k in kpi_data]
        traction_vals = [k['rear_load_rms_traction'] for k in kpi_data]
        fig_brake_vs_traction_r = go.Figure()
        fig_brake_vs_traction_r.add_trace(go.Bar(
            name="Braking", x=setup_names, y=brake_vals))
        fig_brake_vs_traction_r.add_trace(go.Bar(
            name="Traction", x=setup_names, y=traction_vals))
        fig_brake_vs_traction_r.update_layout(
            title="Rear Load RMS: Braking vs Traction [N]",
            xaxis_title="Setup",
            yaxis_title="Contact Patch Load RMS [N]",
            barmode="group"
        )
        layout.append(dcc.Graph(figure=fig_brake_vs_traction_r))
    except KeyError:
        pass

    # ── 16) Contact Patch Load RMS en Frenada (barras Front vs Rear) ───────────
    try:
        brake_vals_front = [k['front_load_rms_brake'] for k in kpi_data]
        brake_vals_rear  = [k['rear_load_rms_brake'] for k in kpi_data]
        fig_brake = go.Figure()
        fig_brake.add_trace(go.Bar(
            name="Front", x=setup_names, y=brake_vals_front))
        fig_brake.add_trace(go.Bar(
            name="Rear", x=setup_names, y=brake_vals_rear))
        fig_brake.update_layout(
            title="Contact Patch Load RMS en Frenada [N]",
            xaxis_title="Setup",
            yaxis_title="CPL RMS [N]",
            barmode="group"
        )
        layout.append(dcc.Graph(figure=fig_brake))
    except KeyError:
        pass

    # ── 17) Contact Patch Load RMS en Tracción (barras Front vs Rear) ─────────
    try:
        traction_vals_front = [k['front_load_rms_traction'] for k in kpi_data]
        traction_vals_rear  = [k['rear_load_rms_traction'] for k in kpi_data]
        fig_traction = go.Figure()
        fig_traction.add_trace(go.Bar(
            name="Front", x=setup_names, y=traction_vals_front))
        fig_traction.add_trace(go.Bar(
            name="Rear", x=setup_names, y=traction_vals_rear))
        fig_traction.update_layout(
            title="Contact Patch Load RMS en Tracción [N]",
            xaxis_title="Setup",
            yaxis_title="CPL RMS [N]",
            barmode="group"
        )
        layout.append(dcc.Graph(figure=fig_traction))
    except KeyError:
        pass

    # ── 10) PSD Heave por eje (Front vs Rear, mm²/Hz) ──────────────────────────
    try:
        fig_psd_axes = go.Figure()
        for k, name in zip(kpi_data, setup_names):
            try:
                f_front = get_val(k, 'f_psd_front')
                psd_front = np.array(get_val(k, 'psd_heave_front')) * 1e6
                f_rear = get_val(k, 'f_psd_rear')
                psd_rear = np.array(get_val(k, 'psd_heave_rear')) * 1e6
            except KeyError:
                continue

            # PSD eje delantero (convertir m²/Hz → mm²/Hz)
            fig_psd_axes.add_trace(go.Scatter(
                x=f_front,
                y=psd_front,
                mode='lines',
                name=f"{name} – Front"
            ))
            # PSD eje trasero
            fig_psd_axes.add_trace(go.Scatter(
                x=f_rear,
                y=psd_rear,
                mode='lines',
                name=f"{name} – Rear"
            ))

        fig_psd_axes.update_layout(
            title="PSD of Heave Motion by Axle (Front vs Rear)",
            xaxis_title="Frequency (Hz)",
            yaxis_title="PSD Heave (mm²/Hz)",
            yaxis_type="log"
        )
        layout.append(dcc.Graph(figure=fig_psd_axes))

    except Exception as e:
        print(f"[WARNING] Error al generar el PSD de heave por eje: {e}")

    # ── 11) PSD Heave por setup (solo chasis, mm²/Hz) ───────────────────────────
    try:
        fig_psd = go.Figure()
        for k, name in zip(kpi_data, setup_names):
            try:
                f_psd = get_val(k, 'f_psd')
                psd_h = np.array(get_val(k, 'psd_heave')) * 1e6
            except KeyError:
                continue
            fig_psd.add_trace(go.Scatter(
                x=f_psd,
                y=psd_h,
                mode='lines',
                name=name
            ))
        fig_psd.update_layout(
            title="Power Spectrum Density of Heave Motion",
            xaxis_title="Frequency (Hz)",
            yaxis_title="PSD Heave (mm²/Hz)",
            yaxis_type="log"
        )
        layout.append(dcc.Graph(figure=fig_psd))
    except Exception as e:
        print(f"[WARNING] Error al generar el PSD de heave: {e}")

    # ── 12) PSD Pitch por eje (Front vs Rear, mm²/Hz) ──────────────────────────
    try:
        fig_psd_pitch_axes = go.Figure()
        for k, name in zip(kpi_data, setup_names):
            try:
                f_pf = get_val(k, 'f_psd_pitch_front')
                psd_pf = np.array(get_val(k, 'psd_pitch_front')) * 1e6
                f_pr = get_val(k, 'f_psd_pitch_rear')
                psd_pr = np.array(get_val(k, 'psd_pitch_rear')) * 1e6
            except KeyError:
                continue

            # PSD pitch→vertical eje delantero (m²/Hz → mm²/Hz)
            fig_psd_pitch_axes.add_trace(go.Scatter(
                x=f_pf,
                y=psd_pf,
                mode='lines',
                name=f"{name} – Pitch Front"
            ))
            # PSD pitch→vertical eje trasero
            fig_psd_pitch_axes.add_trace(go.Scatter(
                x=f_pr,
                y=psd_pr,
                mode='lines',
                name=f"{name} – Pitch Rear"
            ))

        fig_psd_pitch_axes.update_layout(
            title="PSD of Pitch‐Induced Vertical by Axle (Front vs Rear)",
            xaxis_title="Frequency (Hz)",
            yaxis_title="PSD Pitch→Vertical (mm²/Hz)",
            yaxis_type="log"
        )
        layout.append(dcc.Graph(figure=fig_psd_pitch_axes))

    except Exception as e:
        print(f"[WARNING] Error al generar el PSD de pitch por eje: {e}")

    # ── 13) PSD Pitch por setup (rad²/Hz) ───────────────────────────────────────
    try:
        fig_psd_pitch = go.Figure()
        for k, name in zip(kpi_data, setup_names):
            if 'f_psd_pitch' in k and 'psd_pitch' in k:
                fig_psd_pitch.add_trace(go.Scatter(
                    x=k['f_psd_pitch'],
                    y=np.array(k['psd_pitch']),  # en rad²/Hz
                    mode='lines',
                    name=name
                ))
        fig_psd_pitch.update_layout(
            title="Power Spectrum Density of Pitch Motion",
            xaxis_title="Frequency (Hz)",
            yaxis_title="PSD Pitch (rad²/Hz)",
            yaxis_type="log"
        )
        layout.append(dcc.Graph(figure=fig_psd_pitch))
    except Exception as e:
        print(f"[WARNING] Error al generar el PSD de pitch: {e}")

    # 14a) Magnitud (dB) con smoothing
    fig_psd_load_mag = go.Figure()
    for k, name in zip(kpi_data, setup_names):
        try:
            f_load = np.array(get_val(k, 'f_psd_load'))
            mag_f_raw = np.array(get_val(k, 'psd_load_mag_front'))
            mag_r_raw = np.array(get_val(k, 'psd_load_mag_rear'))
        except KeyError:
            continue
        # Suavizado de la magnitud
        mag_f = smooth_signal(mag_f_raw, window=51, polyorder=3)
        mag_r = smooth_signal(mag_r_raw, window=51, polyorder=3)

        fig_psd_load_mag.add_trace(go.Scatter(
            x=f_load,
            y=mag_f,
            mode='lines',
            name=f"{name} – Mag Front [dB] (suavizado)"
        ))
        fig_psd_load_mag.add_trace(go.Scatter(
            x=f_load,
            y=mag_r,
            mode='lines',
            name=f"{name} – Mag Rear  [dB] (suavizado)"
        ))

    fig_psd_load_mag.update_layout(
        title="PSD de Carga – Magnitud (Front vs Rear, Suavizado)",
        xaxis=dict(title="Frecuencia [Hz]", type="log"),
        yaxis=dict(title="Magnitud [dB]"),
        legend=dict(x=0.01, y=0.99)
    )
    layout.append(dcc.Graph(figure=fig_psd_load_mag))

    # 14b) Fase (grados) con smoothing
    fig_psd_load_phase = go.Figure()
    for k, name in zip(kpi_data, setup_names):
        try:
            f_load = np.array(get_val(k, 'f_psd_load'))
            phase_f_raw = np.array(get_val(k, 'psd_load_phase_front'))
            phase_r_raw = np.array(get_val(k, 'psd_load_phase_rear'))
        except KeyError:
            continue
        # Suavizado de la fase
        phase_f = smooth_signal(phase_f_raw, window=51, polyorder=3)
        phase_r = smooth_signal(phase_r_raw, window=51, polyorder=3)

        fig_psd_load_phase.add_trace(go.Scatter(
            x=f_load,
            y=phase_f,
            mode='lines',
            name=f"{name} – Phase Front [°] (suavizado)"
        ))
        fig_psd_load_phase.add_trace(go.Scatter(
            x=f_load,
            y=phase_r,
            mode='lines',
            name=f"{name} – Phase Rear  [°] (suavizado)"
        ))

    fig_psd_load_phase.update_layout(
        title="PSD de Carga – Fase (Front vs Rear, Suavizado)",
        xaxis=dict(title="Frecuencia [Hz]", type="log"),
        yaxis=dict(title="Fase [°]"),
        legend=dict(x=0.01, y=0.99)
    )
    layout.append(dcc.Graph(figure=fig_psd_load_phase))

    # 14c) PSD de Fuerza del Damper – Magnitud (dB) FL y RL con suavizado
    fig_psd_damper_mag = go.Figure()
    for k, name in zip(kpi_data, setup_names):
        try:
            f_damp = np.array(get_val(k, 'f_psd_damper'))
            mag_FL = smooth_signal(np.array(get_val(k, 'psd_damper_mag_FL')), window=51, polyorder=3)
            mag_RL = smooth_signal(np.array(get_val(k, 'psd_damper_mag_RL')), window=51, polyorder=3)
        except KeyError:
            continue
        fig_psd_damper_mag.add_trace(go.Scatter(
            x=f_damp,
            y=mag_FL,
            mode='lines',
            name=f"{name} – Damper FL [dB] (suavizado)"
        ))
        fig_psd_damper_mag.add_trace(go.Scatter(
            x=f_damp,
            y=mag_RL,
            mode='lines',
            name=f"{name} – Damper RL [dB] (suavizado)"
        ))
    fig_psd_damper_mag.update_layout(
        title="PSD de Fuerza del Damper – Magnitud (FL vs RL, Suavizado)",
        xaxis=dict(title="Frecuencia [Hz]", type="log"),
        yaxis=dict(title="Magnitud [dB]"),
        legend=dict(x=0.01, y=0.99)
    )
    layout.append(dcc.Graph(figure=fig_psd_damper_mag))


    # ────────────────────────────────────────────────────────────────────────────
    app_kpi.layout = html.Div(layout)
    app_kpi.run(port=8051, debug=False)

def get_results_figures(sol, post):
    import plotly.graph_objs as go
    import numpy as np
    figures = []
    distance = np.cumsum(post['vx']) * np.gradient(sol.t)
    # Travel
    fig_travel = go.Figure()
    fig_travel.add_trace(go.Scatter(x=distance, y=np.mean(post['travel_rel'][0:2], axis=0)*1000, name="Front"))
    fig_travel.add_trace(go.Scatter(x=distance, y=np.mean(post['travel_rel'][2:4], axis=0)*1000, name="Rear"))
    fig_travel.update_layout(title="Suspension Travel [mm]", xaxis_title="Time [s]", yaxis_title="Travel [mm]")
    figures.append(fig_travel)

    # Heave
    fig_heave = go.Figure()
    fig_heave.add_trace(go.Scatter(x=distance, y=sol.y[0]*1000, name="Heave"))
    fig_heave.update_layout(title="Heave [mm]", xaxis_title="Time [s]", yaxis_title="Heave [mm]")
    figures.append(fig_heave)

    # Pitch
    fig_pitch = go.Figure()
    fig_pitch.add_trace(go.Scatter(x=distance, y=np.degrees(sol.y[2]), name="Pitch"))
    fig_pitch.update_layout(title="Pitch [°]", xaxis_title="Time [s]", yaxis_title="Pitch [°]")
    figures.append(fig_pitch)

    # Roll
    fig_roll = go.Figure()
    fig_roll.add_trace(go.Scatter(x=distance, y=np.degrees(sol.y[4]), name="Roll"))
    fig_roll.update_layout(title="Roll [°]", xaxis_title="Time [s]", yaxis_title="Roll [°]")
    figures.append(fig_roll)

    # Tire Load
    fig_tire = go.Figure()
    for i, label in enumerate(["FL", "FR", "RL", "RR"]):
        fig_tire.add_trace(go.Scatter(x=distance, y=post['f_tire'][i], name=label))
    fig_tire.update_layout(title="Tire Load per Wheel [N]", xaxis_title="Time [s]", yaxis_title="Load [N]")
    figures.append(fig_tire)

    return figures

def get_kpi_figures(setups):
    import os
    import numpy as np
    import plotly.graph_objects as go

    # ── Nombres de setups y etiquetas fijas ──
    setup_names = [os.path.splitext(os.path.basename(p))[0] for _, _, p, _ in setups]
    kpi_labels  = ['FL', 'FR', 'RL', 'RR']

    figures = []

    # Definimos el nuevo tamaño (20 % más que 1122×529)
    NEW_WIDTH  = int(1122 * 1.5)  # ≈ 1346
    NEW_HEIGHT = int(529  * 1.5)  # ≈ 634

    # === 1) Barras de KPIs centrales (Grip-Limited) ===
    kpi_definitions = [
        ("Tire Load Max (Grip-Limited) [N]", "N", "f_tire_grip_limited_max", 1),
        ("Tire Load Min (Grip-Limited) [N]", "N", "f_tire_grip_limited_min", 1),
    ]
    for title, unit, key, factor in kpi_definitions:
        try:
            # Cada post[key] debe ser un array de 4 valores [FL, FR, RL, RR]
            values_list = [post[key] * factor for _, post, _, _ in setups]
            fig = go.Figure()
            for values, name in zip(values_list, setup_names):
                fig.add_trace(go.Bar(name=name, x=kpi_labels, y=values))
            fig.update_layout(
                title=title,
                yaxis_title=unit,
                barmode='group',
                width=NEW_WIDTH,
                height=NEW_HEIGHT
            )
            figures.append(fig)
        except KeyError:
            continue

    # === 2) Road-Noise acumulado (wheel-speed RMS) por track ===
    try:
        noise_by_track = {}
        for _, post, _, _ in setups:
            tname = post['track_name']
            if tname in noise_by_track:
                continue
            front_accu = float(post.get('tracknoise_accu_front', 0))
            rear_accu  = float(post.get('tracknoise_accu_rear',  0))
            noise_by_track[tname] = (front_accu, rear_accu)

        tracks_unique = list(noise_by_track.keys())
        front_vals = [noise_by_track[t][0] for t in tracks_unique]
        rear_vals  = [noise_by_track[t][1] for t in tracks_unique]

        fig_accu = go.Figure()
        fig_accu.add_trace(go.Bar(name="Front Axle", x=tracks_unique, y=front_vals))
        fig_accu.add_trace(go.Bar(name="Rear  Axle", x=tracks_unique, y=rear_vals))
        fig_accu.update_layout(
            title="Accumulated Road Track-Noise Normalized by Lap Time",
            xaxis_title="Track",
            yaxis_title="Normalized Accu. Track-Noise [mm/s]",
            barmode="group",
            width=NEW_WIDTH,
            height=NEW_HEIGHT
        )
        figures.append(fig_accu)
    except (KeyError, TypeError):
        pass

    # === 3) Pitch vs Distance ===
    try:
        fig_pitch_vs_distance = go.Figure()
        for _, post, _, _ in setups:
            if 'pitch_deg' in post and 'distance' in post:
                fig_pitch_vs_distance.add_trace(go.Scatter(
                    x=post['distance'],
                    y=post['pitch_deg'],
                    mode='lines',
                    name=""
                ))
        for idx, trace in enumerate(fig_pitch_vs_distance.data):
            trace.name = setup_names[idx]
        fig_pitch_vs_distance.update_layout(
            title="Pitch vs Distance [°]",
            xaxis_title="Distance [m]",
            yaxis_title="Pitch [°]",
            width=NEW_WIDTH,
            height=NEW_HEIGHT
        )
        figures.append(fig_pitch_vs_distance)
    except Exception:
        pass

    # === 4) Pitch RMS (tabla numérica) ===
    try:
        pitch_rms_vals = [float(post.get('pitch_rms', 0)) for _, post, _, _ in setups]
        fig_pitch_table = go.Figure(data=[go.Table(
            header=dict(
                values=["Setup", "Pitch RMS [°]"],
                fill_color='paleturquoise',
                align='left'
            ),
            cells=dict(
                values=[setup_names, pitch_rms_vals],
                fill_color='lavender',
                align='left'
            )
        )])
        fig_pitch_table.update_layout(
            title="Pitch RMS por Setup (Resumen Numérico)",
            width=NEW_WIDTH,
            height=NEW_HEIGHT
        )
        figures.append(fig_pitch_table)
    except KeyError:
        pass

    # === 5) FRH vs RRH RMS (por setup) ===
    try:
        frh_vals = [float(post['frh_rms']) * 1000 for _, post, _, _ in setups]
        rrh_vals = [float(post['rrh_rms']) * 1000 for _, post, _, _ in setups]
        fig_frh_rrh = go.Figure()
        for name, frh, rrh in zip(setup_names, frh_vals, rrh_vals):
            fig_frh_rrh.add_trace(go.Bar(
                name=name,
                x=["Front", "Rear"],
                y=[frh, rrh]
            ))
        fig_frh_rrh.update_layout(
            title="Ride Height RMS en GLS [mm]",
            barmode='group',
            xaxis_title="Axle",
            width=NEW_WIDTH,
            height=NEW_HEIGHT
        )
        figures.append(fig_frh_rrh)
    except KeyError:
        pass

    # === 6) Scatter: FRH RMS vs Contact Patch Load RMS (Front) ===
    try:
        frh_rms_vals   = [float(post['frh_rms']) * 1000 for _, post, _, _ in setups]
        load_rms_front = [float(post['front_load_rms'])    for _, post, _, _ in setups]
        fig_h_vs_f = go.Figure()
        for name, frh, load_f in zip(setup_names, frh_rms_vals, load_rms_front):
            fig_h_vs_f.add_trace(go.Scatter(
                x=[frh],
                y=[load_f],
                mode='markers+text',
                text=[name],
                textposition='top center'
            ))
        fig_h_vs_f.update_layout(
            title="Front Ride Height RMS vs Contact Patch Load RMS",
            xaxis_title="Front Ride Height RMS [mm]",
            yaxis_title="Contact Patch Load RMS [N]",
            width=NEW_WIDTH,
            height=NEW_HEIGHT
        )
        figures.append(fig_h_vs_f)
    except KeyError:
        pass

    # === 7) Scatter: RRH RMS vs Contact Patch Load RMS (Rear) ===
    try:
        rrh_rms_vals  = [float(post['rrh_rms'])   * 1000 for _, post, _, _ in setups]
        load_rms_rear = [float(post['rear_load_rms']) for _, post, _, _ in setups]
        fig_rrh_vs_f = go.Figure()
        for name, rrh, load_r in zip(setup_names, rrh_rms_vals, load_rms_rear):
            fig_rrh_vs_f.add_trace(go.Scatter(
                x=[rrh],
                y=[load_r],
                mode='markers+text',
                text=[name],
                textposition='top center'
            ))
        fig_rrh_vs_f.update_layout(
            title="Rear Ride Height RMS vs Contact Patch Load RMS",
            xaxis_title="Rear Ride Height RMS [mm]",
            yaxis_title="Contact Patch Load RMS [N]",
            width=NEW_WIDTH,
            height=NEW_HEIGHT
        )
        figures.append(fig_rrh_vs_f)
    except KeyError:
        pass

    # === 8) Ride Height RMS (barra Front/Rear) ===
    try:
        fig_rh = go.Figure()
        for _, post, _, _ in setups:
            frh_val = float(post.get('frh_rms', 0)) * 1000
            rrh_val = float(post.get('rrh_rms', 0)) * 1000
            fig_rh.add_trace(go.Bar(
                x=["Front", "Rear"],
                y=[frh_val, rrh_val],
                name=""
            ))
        for idx, trace in enumerate(fig_rh.data):
            trace.name = setup_names[idx]
        fig_rh.update_layout(
            title="Ride Height RMS [mm]",
            barmode='group',
            xaxis_title="Axle",
            width=NEW_WIDTH,
            height=NEW_HEIGHT
        )
        figures.append(fig_rh)
    except Exception:
        pass

    # === 9) Brake vs Tracción – Resumen (Front/Rear) ===
    try:
        brake_vals_front = [float(post['front_load_rms_brake']) for _, post, _, _ in setups]
        brake_vals_rear  = [float(post['rear_load_rms_brake'])  for _, post, _, _ in setups]
        fig_brake = go.Figure()
        fig_brake.add_trace(go.Bar(name="Front", x=setup_names, y=brake_vals_front))
        fig_brake.add_trace(go.Bar(name="Rear",  x=setup_names, y=brake_vals_rear))
        fig_brake.update_layout(
            title="Contact Patch Load RMS en Frenada [N]",
            xaxis_title="Setup",
            yaxis_title="CPL RMS [N]",
            barmode="group",
            width=NEW_WIDTH,
            height=NEW_HEIGHT
        )
        figures.append(fig_brake)
    except KeyError:
        pass

    try:
        traction_vals_front = [float(post['front_load_rms_traction']) for _, post, _, _ in setups]
        traction_vals_rear  = [float(post['rear_load_rms_traction'])  for _, post, _, _ in setups]
        fig_traction = go.Figure()
        fig_traction.add_trace(go.Bar(name="Front", x=setup_names, y=traction_vals_front))
        fig_traction.add_trace(go.Bar(name="Rear",  x=setup_names, y=traction_vals_rear))
        fig_traction.update_layout(
            title="Contact Patch Load RMS en Tracción [N]",
            xaxis_title="Setup",
            yaxis_title="CPL RMS [N]",
            barmode="group",
            width=NEW_WIDTH,
            height=NEW_HEIGHT
        )
        figures.append(fig_traction)
    except KeyError:
        pass

    # === 10) Heave PSD [mm²/Hz] (Global) ===
    try:
        fig_psd_heave = go.Figure()
        for _, post, _, _ in setups:
            if 'f_psd' in post and 'psd_heave' in post:
                fig_psd_heave.add_trace(go.Scatter(
                    x=post['f_psd'],
                    y=np.array(post['psd_heave']) * 1e6,  # m²/Hz → mm²/Hz
                    mode='lines',
                    name=""
                ))
        for idx, trace in enumerate(fig_psd_heave.data):
            trace.name = setup_names[idx]
        fig_psd_heave.update_layout(
            title="Power Spectrum Density of Heave Motion",
            xaxis_title="Frequency [Hz]",
            yaxis_title="PSD Heave (mm²/Hz)",
            yaxis_type="log",
            width=NEW_WIDTH,
            height=NEW_HEIGHT
        )
        figures.append(fig_psd_heave)
    except Exception:
        pass

    # === 11) Pitch PSD [rad²/Hz] (Global) ===
    try:
        fig_psd_pitch = go.Figure()
        for _, post, _, _ in setups:
            if 'f_psd_pitch' in post and 'psd_pitch' in post:
                fig_psd_pitch.add_trace(go.Scatter(
                    x=post['f_psd_pitch'],
                    y=np.array(post['psd_pitch']),
                    mode='lines',
                    name=""
                ))
        for idx, trace in enumerate(fig_psd_pitch.data):
            trace.name = setup_names[idx]
        fig_psd_pitch.update_layout(
            title="Power Spectrum Density of Pitch Motion",
            xaxis_title="Frequency [Hz]",
            yaxis_title="PSD Pitch (rad²/Hz)",
            yaxis_type="log",
            width=NEW_WIDTH,
            height=NEW_HEIGHT
        )
        figures.append(fig_psd_pitch)
    except Exception:
        pass

    # === 12) Heave PSD por eje (Front vs Rear) en mm²/Hz ===
    try:
        fig_psd_heave_axes = go.Figure()
        for _, post, _, _ in setups:
            if ('f_psd_front' in post and 'psd_heave_front' in post and
                'f_psd_rear'  in post and 'psd_heave_rear'  in post):

                # Heave Front [mm²/Hz]
                fig_psd_heave_axes.add_trace(go.Scatter(
                    x=post['f_psd_front'],
                    y=np.array(post['psd_heave_front']) * 1e6,
                    mode='lines',
                    name=f"{os.path.splitext(os.path.basename(post['track_name']))[0]} – Front"
                ))
                # Heave Rear [mm²/Hz]
                fig_psd_heave_axes.add_trace(go.Scatter(
                    x=post['f_psd_rear'],
                    y=np.array(post['psd_heave_rear']) * 1e6,
                    mode='lines',
                    name=f"{os.path.splitext(os.path.basename(post['track_name']))[0]} – Rear"
                ))

        fig_psd_heave_axes.update_layout(
            title="PSD of Heave Motion by Axle (Front vs Rear)",
            xaxis_title="Frequency [Hz]",
            yaxis_title="PSD Heave (mm²/Hz)",
            yaxis_type="log",
            width=NEW_WIDTH,
            height=NEW_HEIGHT
        )
        figures.append(fig_psd_heave_axes)
    except Exception:
        pass

    # === 12+1) Pitch PSD por eje (Front vs Rear) en mm²/Hz ===
    try:
        fig_psd_pitch_axes = go.Figure()
        for _, post, _, _ in setups:
            if ('f_psd_pitch_front' in post and 'psd_pitch_front' in post and
                'f_psd_pitch_rear'  in post and 'psd_pitch_rear'  in post):

                # Pitch→Vertical Front [mm²/Hz]
                fig_psd_pitch_axes.add_trace(go.Scatter(
                    x=post['f_psd_pitch_front'],
                    y=np.array(post['psd_pitch_front']) * 1e6,
                    mode='lines',
                    name=f"{os.path.splitext(os.path.basename(post['track_name']))[0]} – Front"
                ))
                # Pitch→Vertical Rear [mm²/Hz]
                fig_psd_pitch_axes.add_trace(go.Scatter(
                    x=post['f_psd_pitch_rear'],
                    y=np.array(post['psd_pitch_rear']) * 1e6,
                    mode='lines',
                    name=f"{os.path.splitext(os.path.basename(post['track_name']))[0]} – Rear"
                ))

        fig_psd_pitch_axes.update_layout(
            title="PSD of Pitch‐Induced Vertical by Axle (Front vs Rear)",
            xaxis_title="Frequency [Hz]",
            yaxis_title="PSD Pitch→Vertical (mm²/Hz)",
            yaxis_type="log",
            width=NEW_WIDTH,
            height=NEW_HEIGHT
        )
        figures.append(fig_psd_pitch_axes)
    except Exception:
        pass

    return figures

def run_kpi_comparison_in_thread(sim_results):
    from threading import Thread
    kpi_data = [post for _, post, _, _ in sim_results]
    setup_names = [os.path.basename(p).replace(".json", "") for _, _, p, _ in sim_results]
    thread = Thread(target=launch_dash_kpis, args=(kpi_data, setup_names))
    thread.start()
    
def export_full_report(setups, export_path="export_full_report.html"):
    html_sections = []

    html_sections.append("<h1>Reporte de Simulación - 4-Post Rig</h1>")
    html_sections.append('<ul>')
    html_sections.append('<li><a href="#kpis">📊 Ver Comparativa de KPIs</a></li>')
    for idx, (_, _, setup_path, _) in enumerate(setups):
        setup_name = os.path.basename(setup_path).replace(".json", "")
        anchor = f"setup_{idx}"
        html_sections.append(f'<li><a href="#{anchor}">📁 {setup_name}</a></li>')
    html_sections.append('</ul>')

    # Resultados por setup
    for idx, (sol, post, setup_path, _) in enumerate(setups):
        setup_name = os.path.basename(setup_path).replace(".json", "")
        anchor = f"setup_{idx}"
        html_sections.append(f'<div id="{anchor}"><h2>Resultados - {setup_name}</h2>')
        figures = get_results_figures(sol, post)
        for i, fig in enumerate(figures):
            html_sections.append(plot(fig, include_plotlyjs=(idx == 0 and i == 0), output_type='div'))
        html_sections.append('</div><hr>')

    # KPIs comparativos (siempre incluir, incluso con un solo setup)
    html_sections.append('<div id="kpis"><h2>📊 KPIs Comparativos</h2>')
    kpi_figs = get_kpi_figures(setups)
    for fig in kpi_figs:
        html_sections.append(plot(fig, include_plotlyjs=False, output_type='div'))
    html_sections.append('</div><hr>')

    # Exporta a HTML
    with open(export_path, "w", encoding="utf-8") as f:
        f.write("<html><head>")
        f.write('<meta charset="utf-8">')
        f.write('<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>')
        f.write("<title>Reporte 4-Post Rig</title></head><body>")
        for section in html_sections:
            f.write(section)
        f.write("</body></html>")

def run_in_thread(sol, post, setup_name="Setup"):
    thread = Thread(target=launch_dash, args=(sol, post, setup_name))
    thread.start()