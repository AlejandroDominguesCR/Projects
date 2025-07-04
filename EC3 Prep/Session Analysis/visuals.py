from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QFormLayout, QStackedWidget,
    QFileDialog, QApplication, QLineEdit, QCheckBox
)
from PyQt6.QtCore import Qt
import plotly.express as px
import pandas as pd
import os
from plotly.subplots import make_subplots 
import webbrowser
from dash import dash_table  
import plotly.graph_objects as go
import random
from data_process import load_session_data, unify_timestamps, convert_time_column
from KPI_builder import (
    compute_top_speeds, track_limit_rate, team_ranking,
    ideal_lap_gap, best_sector_times, lap_time_history,
    slipstream_stats, sector_slipstream_stats,
    pit_stop_summary, lap_time_consistency,
    build_driver_tables,
    build_fastest_lap_table,
)

from main_analysis import export_report, get_team_colors

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Análisis de Sesión de Carreras")
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.init_intro()
        self.init_graphs()
        self.stack.setCurrentWidget(self.intro_page)

    def init_intro(self):
        # Página inicial con inputs y selector de carpeta
        self.intro_page = QWidget()
        layout = QVBoxLayout(self.intro_page)
        form = QFormLayout()
        self.year_input = QLineEdit()
        self.category_input = QLineEdit()
        self.event_input = QLineEdit()
        self.session_input = QLineEdit()
        form.addRow("Año:", self.year_input)
        form.addRow("Categoría:", self.category_input)
        form.addRow("Evento:", self.event_input)
        form.addRow("Sesión (FP, R1, R2, Q, C):", self.session_input)
        layout.addLayout(form)
        self.sector_toggle = QCheckBox("Incluir sectores en tabla")
        layout.addWidget(self.sector_toggle, alignment=Qt.AlignmentFlag.AlignLeft)
        self.select_btn = QPushButton("Seleccionar carpeta CSV")
        self.select_btn.clicked.connect(self.on_select)
        layout.addWidget(self.select_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        self.intro_page.setLayout(layout)
        self.stack.addWidget(self.intro_page)

    def init_graphs(self):
        # Página de confirmación de generación de KPIs
        self.graphs_page = QWidget()
        layout = QVBoxLayout(self.graphs_page)
        self.info_label = QLabel("Reporte HTML generado en la carpeta seleccionada.")
        self.back_btn = QPushButton("Volver")
        self.back_btn.clicked.connect(lambda: self.stack.setCurrentWidget(self.intro_page))
        layout.addWidget(self.info_label, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.back_btn, alignment=Qt.AlignmentFlag.AlignRight)
        self.graphs_page.setLayout(layout)
        self.stack.addWidget(self.graphs_page)

    def on_select(self):
        # Selección de carpeta y carga de datos
        folder = QFileDialog.getExistingDirectory(self, "Selecciona carpeta de sesión")
        if not folder:
            return
        data = load_session_data(folder)
        # Preprocesar timestamps
        for key, df in data.items():
            data[key] = unify_timestamps(df, 'time')
        # Seleccionar dataframes clave
        analysis_key = next((k for k in data if 'analysis' in k.lower()), None)
        class_key = next((k for k in data if 'classification' in k.lower()), None)
        weather_key = next((k for k in data if 'weather' in k.lower()), None)
        tracklimits_key = next((k for k in data if 'tracklimits' in k.lower()), None)
        df_analysis = data.get(analysis_key, list(data.values())[0])
        df_class = data.get(class_key, pd.DataFrame())
        weather_df = data.get(weather_key, pd.DataFrame())
        tracklimits_df = data.get(tracklimits_key, pd.DataFrame())
        # Normalizar columna 'driver'
        driver_col = next((c for c in ['driver_name', 'driver_shortname', 'driver_number'] if c in df_analysis.columns), None)
        if driver_col is None:
            raise KeyError('No se encontró columna de piloto en Analysis')
        df_analysis = df_analysis.copy()
        df_analysis['driver'] = df_analysis[driver_col]
        df_analysis = convert_time_column(df_analysis, 'lap_time')
        driver_tables = build_driver_tables(df_analysis)
        fast_table = build_fastest_lap_table(df_analysis, df_class)
        # --- 1) Generar un mapa de colores por piloto ---
        ts = compute_top_speeds(df_analysis)
        team_colors = get_team_colors()
        for team in ts['team'].unique():
            team_colors.setdefault(team, f'#{random.randint(0,0xFFFFFF):06x}')

        # Diccionario de figuras para el reporte
        figs: dict[str, go.Figure] = {}

        ss = slipstream_stats(df_analysis)
        ss_sector = sector_slipstream_stats(df_analysis)

        if not ss.empty:
            # Lap-time mínimo
            ss_lap = ss.sort_values("min_lap_time_with_slip")
            order_lap = ss_lap["driver"].tolist()
            fig_slip_lap = px.bar(
                ss_lap,
                x="driver",
                y=["min_lap_time_no_slip", "min_lap_time_with_slip"],
                barmode="group",
                title="Lap-time mínimo – Con vs Sin rebufo",
            )
            fig_slip_lap.update_layout(
                xaxis={"categoryorder": "array", "categoryarray": order_lap}
            )

            # Top-speed máximo
            ss_spd = ss.sort_values("max_top_speed_with_slip", ascending=False)
            order_spd = ss_spd["driver"].tolist()
            fig_slip_speed = px.bar(
                ss_spd,
                x="driver",
                y=["max_top_speed_no_slip", "max_top_speed_with_slip"],
                barmode="group",
                title="Top Speed máxima – Con vs Sin rebufo",
            )
            fig_slip_speed.update_layout(
                xaxis={"categoryorder": "array", "categoryarray": order_spd}
            )

            figs["Slipstream Lap-time (min)"]  = fig_slip_lap
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

        df_tmp = df_tmp.groupby("driver", group_keys=False).apply(_flag_slip)

 
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

        # Definir KPIs
        kpis = {
            'top_speeds': lambda: compute_top_speeds(df_analysis),
            'track_rate': (
                lambda: track_limit_rate(tracklimits_df, df_analysis)
                if not tracklimits_df.empty else None
            ),
            'team_ranking': lambda: team_ranking(df_analysis),
            'ideal_lap_gap': lambda: ideal_lap_gap(df_analysis),
            'sector_comparison': lambda: best_sector_times(df_analysis),
            'lap_history':   lambda: lap_time_history(df_analysis),
            'pit_summary':   lambda: pit_stop_summary(df_analysis),
            'lap_consistency': lambda: lap_time_consistency(df_analysis),
        }
        # Generar archivos para cada KPI
        for name, func in kpis.items():
            try:
                df_out = func()
            except Exception:
                continue
            if df_out is None or (isinstance(df_out, pd.DataFrame) and df_out.empty):
                continue
            # Crear figura según KPI
            try:
                if name == 'top_speeds':
                    df_ts = df_out.sort_values('max_top_speed', ascending=False)
                    fig = px.bar(
                        df_ts,
                        x='driver', y='max_top_speed',
                        color='team', color_discrete_map=team_colors,
                        title='Top Speeds (ordenado desc)'
                    )
                    fig.update_layout(
                        xaxis={'categoryorder':'array','categoryarray':df_ts['driver'].tolist()}
                    )
                    # ── aquí el bloque dinámico ──
                    ymin, ymax = df_ts['max_top_speed'].min(), df_ts['max_top_speed'].max()
                    delta = ymax - ymin
                    fig.update_layout(
                        yaxis=dict(
                            range=[ymin - 0.05 * delta, ymax + 0.10 * delta],
                            title="Velocidad Máxima"
                        )
                    )
                    figs[name] = fig
                elif name == 'track_rate':
                    fig = px.bar(df_out, x='driver', y='rate', title='Track Limits per Lap')
                    figs[name] = fig
                elif name == 'team_ranking':
                    fig = px.bar(
                        df_out,
                        x='team', y='mean_top_speed',
                        color='team', color_discrete_map=team_colors,
                        title='Team Ranking (mean top speed)'
                    )
                    # ── bloque dinámico ──
                    vals = df_out['mean_top_speed']
                    ymin, ymax = vals.min(), vals.max()
                    delta = ymax - ymin
                    fig.update_layout(
                        yaxis=dict(
                            range=[ymin - 0.05 * delta, ymax + 0.10 * delta],
                            title="Velocidad Media de Equipo"
                        )
                    )
                    figs[name] = fig
                elif name == 'ideal_lap_gap':
                    ig = df_out.sort_values('best_lap', ascending=True).reset_index(drop=True)
                    drivers = ig['driver'].tolist()
                    cols = [team_colors[t] for t in ig['team']]
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=drivers, y=ig['ideal_time'], marker_color=cols, name='Vuelta Ideal'))
                    fig.add_trace(go.Scatter(x=drivers, y=ig['best_lap'], mode='markers',
                                            marker=dict(size=10, color=cols, symbol='circle'),
                                            name='Mejor Vuelta Real'))
                    for i in range(len(drivers)):
                        fig.add_shape(
                            type='line',
                            x0=drivers[i],
                            x1=drivers[i],
                            y0=ig.loc[i, 'ideal_time'],
                            y1=ig.loc[i, 'best_lap'],
                            line=dict(color=cols[i], dash='dash'),
                        )
                    fig.update_layout(
                        title="Gap a Vuelta Ideal vs Mejor Vuelta Real",
                        xaxis={'categoryorder':'array','categoryarray':drivers},
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                    )
                    # ── bloque dinámico ──
                    ymin = min(ig['ideal_time'].min(), ig['best_lap'].min())
                    ymax = max(ig['ideal_time'].max(), ig['best_lap'].max())
                    delta = ymax - ymin
                    fig.update_layout(
                        yaxis=dict(
                            range=[ymin - 0.05 * delta, ymax + 0.10 * delta],
                            title="Tiempo (s)"
                        )
                    )
                    figs[name] = fig
                elif name == 'lap_history':
                    df_hist = df_out.sort_values(['lap','driver'])
                    fig = px.line(
                        df_hist,
                        x='lap', y='lap_time',
                        color='team', line_group='driver',
                        color_discrete_map=team_colors,
                        title="Histórico de tiempos por vuelta"
                    )
                    vals = df_hist['lap_time']
                    ymin, ymax = vals.min(), vals.max()
                    delta = ymax - ymin
                    fig.update_layout(
                        yaxis=dict(
                            range=[ymin - 0.05 * delta, ymax + 0.10 * delta],
                            title="Tiempo (s)"
                        )
                    )
                    figs[name] = fig
                elif name == 'sector_comparison':
                    for sec in ['sector1','sector2','sector3']:
                        df_temp = df_out[['driver','team', sec]].copy()
                        best = df_temp[sec].min()
                        df_temp['diff'] = df_temp[sec] - best
                        # Fuerza orden ascendente en la diferencia (0.0 → más rápido)
                        df_temp = df_temp.sort_values('diff', ascending=True).reset_index(drop=True)

                        fig = px.bar(
                            df_temp,
                            x='driver',
                            y='diff',
                            color='team',
                            color_discrete_map=team_colors,
                            title=f"Diferencia en {sec.upper()} vs mejor"
                        )
                        fig.update_layout(
                            xaxis={'categoryorder':'array','categoryarray':df_temp['driver'].tolist()},
                            yaxis_title="Diferencia (s)"
                        )

                        ymin, ymax = df_temp['diff'].min(), df_temp['diff'].max()
                        delta = ymax - ymin
                        fig.update_layout(
                            yaxis=dict(
                                range=[ymin - 0.05 * delta, ymax + 0.10 * delta],
                                title="Diferencia (s)"
                            )
                        )
                        figs[f"{sec.upper()} Diff"] = fig

                elif name == 'pit_summary':
                    df_sorted = df_out.sort_values('best_pit_time', ascending=True)
                    drivers = df_sorted['driver'].tolist()
                    if 'team' in df_sorted.columns:
                        cols = [team_colors.get(t, '#333333') for t in df_sorted['team']]
                    else:
                        cols = ['#333333'] * len(df_sorted)
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=drivers, y=df_sorted['best_pit_time'],
                        marker_color=cols, name='Mejor Parada'
                    ))
                    fig.add_trace(go.Bar(
                        x=drivers, y=df_sorted['mean_pit_time'],
                        marker_color=cols, opacity=0.6, name='Media Paradas'
                    ))
                    fig.update_layout(
                        barmode='group',
                        title='Pit Stop Summary',
                        xaxis={'categoryorder':'array','categoryarray':drivers},
                        yaxis_title='Tiempo (s)'
                    )
                    figs['Pit Stop Summary'] = fig
                elif name == 'lap_consistency':
                    df_sorted = df_out.sort_values('lap_time_std', ascending=True)
                    fig = px.bar(
                        df_sorted,
                        x='driver', y='lap_time_std',
                        color='team', color_discrete_map=team_colors,
                        title='Lap Time Consistency'
                    )
                    ymin, ymax = df_sorted['lap_time_std'].min(), df_sorted['lap_time_std'].max()
                    delta = ymax - ymin
                    fig.update_layout(
                        yaxis=dict(
                            range=[ymin - 0.05 * delta, ymax + 0.10 * delta],
                            title='Desviación (s)'
                        )
                    )
                    figs['Lap Time Consistency'] = fig
                else:
                    continue
            except Exception:
                continue
        export_report(
            figs,
            os.path.join(folder, 'session_report.html'),
            driver_tables=driver_tables,
            fast_table=fast_table,
        )
        webbrowser.open(f"file://{os.path.join(folder, 'session_report.html')}")
        self.stack.setCurrentWidget(self.graphs_page)

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = MainWindow(); window.show(); sys.exit(app.exec())
