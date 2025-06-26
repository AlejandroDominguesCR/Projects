from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QFormLayout, QStackedWidget,
    QFileDialog, QApplication, QLineEdit
)
from PyQt6.QtCore import Qt
import plotly.express as px
import pandas as pd
import os
import webbrowser
import plotly.graph_objects as go
import random
from data_process import load_session_data, unify_timestamps, convert_time_column
from KPI_builder import (
    compute_top_speeds, track_limit_rate, team_ranking,
    ideal_lap_gap, best_sector_times, lap_time_history
)

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
        self.select_btn = QPushButton("Seleccionar carpeta CSV")
        self.select_btn.clicked.connect(self.on_select)
        layout.addWidget(self.select_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        self.intro_page.setLayout(layout)
        self.stack.addWidget(self.intro_page)

    def init_graphs(self):
        # Página de confirmación de generación de KPIs
        self.graphs_page = QWidget()
        layout = QVBoxLayout(self.graphs_page)
        self.info_label = QLabel("KPIs generados en HTML en la carpeta seleccionada.")
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
        # --- 1) Generar un mapa de colores por piloto ---
        ts = compute_top_speeds(df_analysis)
        team_colors = {}
        for team in ts['team'].unique():
            if team == 'Campos Racing':
                team_colors[team] = '#FF5733'
            elif team == 'Griffin Core':
                team_colors[team] = '#33C1FF'
            else:
                team_colors[team] = f'#{random.randint(0,0xFFFFFF):06x}'
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
                elif name == 'track_rate':
                    fig = px.bar(df_out, x='driver', y='rate', title='Track Limits per Lap')
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
                    html = fig.to_html(include_plotlyjs='cdn')
                    path = os.path.join(folder, "lap_history.html")
                    with open(path,'w') as f: f.write(html)
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
                else:
                    continue
                # Guardar HTML
                html = fig.to_html(include_plotlyjs='cdn')
                path = os.path.join(folder, f"{name}.html")
                with open(path, 'w') as f:
                    f.write(html)
            except Exception:
                continue
        # Abrir carpeta en navegador y mostrar confirmación
        webbrowser.open(f'file://{folder}')
        self.stack.setCurrentWidget(self.graphs_page)

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = MainWindow(); window.show(); sys.exit(app.exec())
