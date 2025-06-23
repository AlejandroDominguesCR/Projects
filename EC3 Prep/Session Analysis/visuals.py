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
import random
from session_io import load_session_data
from data_process import unify_timestamps, convert_time_column
from KPI_builder import (
    compute_top_speeds, lap_time_histogram, pace_delta,
    position_trace, sector_comparison, gap_matrix,
    climate_impact, track_limits_incidents,
    top_speed_locations, stint_boxplots,
    team_ranking, lap_time_consistency, 
    ideal_lap_gap, track_limit_rate,
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
        # Analysis
        driver_col = next((c for c in ['driver_name', 'driver_shortname', 'driver_number'] if c in df_analysis.columns), None)
        if driver_col is None:
            raise KeyError('No se encontró columna de piloto en Analysis')
        df_analysis = df_analysis.copy()
        df_analysis['driver'] = df_analysis[driver_col]
        df_analysis = convert_time_column(df_analysis, 'lap_time')
        # Classification
        if not df_class.empty:
            class_driver = next((c for c in ['driver_name', 'driver_shortname', 'driver_number'] if c in df_class.columns), None)
            if class_driver:
                df_class = df_class.copy()
                df_class['driver'] = df_class[class_driver]
        # Definir KPIs
        kpis = {
            'lap_times': lambda: lap_time_histogram(df_analysis),
            'lap_histogram': lambda: lap_time_histogram(df_analysis, df_analysis['driver'].iloc[0]),
            'pace_delta': lambda: pace_delta(df_analysis, df_analysis['driver'].iloc[0]),
            'position_trace': lambda: position_trace(df_class) if not df_class.empty else None,
            'sector_comp': lambda: sector_comparison(df_analysis),
            'gap_matrix': lambda: gap_matrix(df_analysis),
            'ideal_lap_gap': lambda: ideal_lap_gap(df_analysis),
            'climate_impact': lambda: climate_impact(df_analysis, weather_df) if not weather_df.empty else None,
            'track_incidents': lambda: track_limits_incidents(tracklimits_df) if not tracklimits_df.empty else None,
            'track_rate': lambda: track_limit_rate(tracklimits_df, df_analysis) if not tracklimits_df.empty else None,
            'top_speed_loc': lambda: top_speed_locations(df_analysis),
            'stint_box': lambda: stint_boxplots(df_analysis),
            'team_ranking': lambda: team_ranking(df_analysis),
            'lap_consistency': lambda: lap_time_consistency(df_analysis),
        }
        # Generar archivos para cada KPI
        for name, func in kpis.items():
            try:
                out = func()
            except Exception:
                continue
            if out is None:
                continue
            # Si es dict con data, extraer df
            if isinstance(out, dict) and 'data' in out:
                df_out = out['data']
            else:
                df_out = out
            # Guardar output
            try:
                if isinstance(df_out, pd.DataFrame):
                    if name == 'lap_times':
                        fig = px.scatter(df_out, x='lap', y='lap_time', color='driver', title='Lap Times')
                        fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
                    elif name == 'track_rate':
                        fig = px.bar(df_out, x='driver', y='rate', title='Track Limits per Lap')
                    else:
                        fig = px.bar(df_out, x=df_out.columns[0], y=df_out.columns[1], title=name)
                    html = fig.to_html(include_plotlyjs='cdn')
                    path = os.path.join(folder, f"{name}.html")
                    with open(path, 'w') as f:
                        f.write(html)
                else:
                    with open(os.path.join(folder, f"{name}.txt"), 'w') as f:
                        f.write(str(out))
            except Exception:
                pass
        # Abrir carpeta destino en navegador
        webbrowser.open(f'file://{folder}')
        # Mostrar página de confirmación
        self.stack.setCurrentWidget(self.graphs_page)