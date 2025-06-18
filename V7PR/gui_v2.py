import sys
import numpy as np
import pandas as pd
import os
import visualizer_dash
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QScrollArea,
    QLabel, QFileDialog, QTextEdit, QHBoxLayout, QSpinBox, QComboBox, QGroupBox, QTabWidget, QGridLayout, QProgressBar, QLineEdit, QFormLayout, QMessageBox, QListWidget, QAbstractItemView, QTableWidgetItem
)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from visualizer_dash import export_full_report
from PyQt5.QtCore import Qt, QMimeData, QTimer, QUrl
from PyQt5.QtGui import QPixmap, QPalette, QColor, QFont, QIcon, QDoubleValidator, QIntValidator
from concurrent.futures import ProcessPoolExecutor, as_completed
from model import run_vehicle_model_simple, postprocess_7dof

def set_dark_theme(app):
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(45, 45, 45))
    palette.setColor(QPalette.AlternateBase, QColor(60, 60, 60))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(45, 45, 45))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Highlight, QColor(38, 79, 120))
    palette.setColor(QPalette.HighlightedText, Qt.white)
    app.setPalette(palette)

class DragDropLabel(QLabel):
    def __init__(self, text, file_callback):
        super().__init__(text)
        self.setAcceptDrops(True)
        self.file_callback = file_callback
        self.setStyleSheet("border: 2px dashed #888; padding: 20px; color: #aaa;")
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
            self.setStyleSheet("border: 2px solid #4e9a06; background: #222; color: #fff;")
        else:
            event.ignore()
    def dragLeaveEvent(self, event):
        self.setStyleSheet("border: 2px dashed #888; padding: 20px; color: #aaa;")
    def dropEvent(self, event):
        self.setStyleSheet("border: 2px dashed #888; padding: 20px; color: #aaa;")
        for url in event.mimeData().urls():
            self.file_callback(url.toLocalFile())

def parse_json_setup(json_data):
    import numpy as np
    # Extrae masas y parámetros globales
    m_car = json_data["config"]["chassis"]["carRunningMass"]["mCar"]
    wbal_f = json_data["config"]["chassis"]["carRunningMass"]["rWeightBalF"]
    mHubF = json_data["config"]["chassis"]["mHubF"]
    mHubR = json_data["config"]["chassis"]["mHubR"]
    ICar = json_data["config"]["chassis"]["ICar"]
    wheelbase = abs(json_data["config"]["chassis"]["rRideR"][0] - json_data["config"]["chassis"]["rRideF"][0])
    track_f = 2*json_data["config"]["suspension"]["front"]["external"]["pickUpPts"]["rUserTCP"][1]
    track_r = 2*json_data["config"]["suspension"]["rear"]["external"]["pickUpPts"]["rUserTCP"][1]

    # Extrae rigidez de barra estabilizadora (antiroll bar)
    kARB_F = json_data["config"]["suspension"]["front"]["internal"]["antiRollBar"]["kAntiRollBar"]
    kARB_R = json_data["config"]["suspension"]["rear"]["internal"]["antiRollBar"]["kAntiRollBar"]
    
    # Masas por esquina 
    ms_f = ((m_car * wbal_f)/ 2) - mHubF
    ms_r = ((m_car * (1 - wbal_f) )/ 2) - mHubR
    mu_f = mHubF 
    mu_r = mHubR 

    # Front
    spring_f = json_data["config"]["suspension"]["front"]["internal"]["spring"]
    bump_f = json_data["config"]["suspension"]["front"]["internal"]["bumpStop"]
    damper_f = json_data["config"]["suspension"]["front"]["internal"]["damper"]

    # Rear
    spring_r = json_data["config"]["suspension"]["rear"]["internal"]["spring"]
    bump_r = json_data["config"]["suspension"]["rear"]["internal"]["bumpStop"]
    damper_r = json_data["config"]["suspension"]["rear"]["internal"]["damper"]

    # Neumático (ejemplo)
    kt_f = 373100 #276500
    kt_r = 282100 #397900, 282100, 269000

    hRideF = json_data["config"]["chassis"].get("hRideFSetup")
    hRideR = json_data["config"]["chassis"].get("hRideRSetup")
    zCoG = json_data["config"]["chassis"].get("zCoG")

    # Esquinas: FL, FR, RL, RR
    params = []
    
    stroke_FL = 0.029#0.01965
    stroke_FR = 0.029#0.01965
    stroke_RL = 0.059#0.0305
    stroke_RR = 0.059#0.0305

    for i, (ms, mu, spring, bump, damper, kt, stroke) in enumerate([
        (ms_f, mu_f, spring_f, bump_f, damper_f, kt_f, stroke_FL),  # FL
        (ms_f, mu_f, spring_f, bump_f, damper_f, kt_f, stroke_FR),  # FR
        (ms_r, mu_r, spring_r, bump_r, damper_r, kt_r, stroke_RL),  # RL
        (ms_r, mu_r, spring_r, bump_r, damper_r, kt_r, stroke_RR),  # RR
    ]):
        # --- Corrección damper: separar compresión y extensión ---
        v = np.array(damper["vDamperBasis"])
        F = np.array(damper["FDamperLU"])
        damper_f_ext = F[v > 0]
        damper_v_ext = v[v > 0]
        damper_f_comp = F[v < 0]
        damper_v_comp = v[v < 0]

        # Si no hay valores negativos/positivos, usar arrays vacíos
        if damper_f_ext.size == 0:
            damper_f_ext = np.array([0.0])
            damper_v_ext = np.array([0.0])
        if damper_f_comp.size == 0:
            damper_f_comp = np.array([0.0])
            damper_v_comp = np.array([0.0])

        # --- Añadir curva muelle+bumpstop para el recorrido máximo ---
        spring_travel = np.linspace(0, stroke, 100)
        kSpring = spring["kSpring"]
        FSpringPreload = spring.get("FSpringPreload", 0)
        spring_force = kSpring * spring_travel + FSpringPreload
        bump_x = np.array(bump["xData"])
        bump_f = np.array(bump["FData"])
        bump_gap = bump["xFreeGap"]
        bump_force = np.zeros_like(spring_travel)
        bump_indices = spring_travel > bump_gap
        bump_force[bump_indices] = np.interp(spring_travel[bump_indices] - bump_gap, bump_x, bump_f, left=0, right=bump_f[-1])
        total_force = spring_force + bump_force
        p = {
            "ms": ms,
            "mu": mu,
            "kSpring": kSpring,
            "spring_x": spring_travel,
            "spring_f": spring_force,
            "bump_f_interp": bump_force,
            "spring_bump_total_f": total_force,
            "bump_x": bump_x,
            "bump_f": bump_f,
            "bump_gap": bump_gap,
            "damper_v": v,
            "damper_f_ext": damper_f_ext,
            "damper_v_ext": damper_v_ext,
            "damper_f_comp": damper_f_comp,
            "damper_v_comp": damper_v_comp,
            "kt": kt,
            "stroke": stroke,
        }
        params.append(p)

    # Diccionario de parámetros globales
    global_setup = {
        'mCar': m_car,
        'mHubF': mHubF,
        'mHubR': mHubR,
        'rWeightBalF': wbal_f,
        'ICar': ICar,
        'wheelbase': wheelbase,
        'track_f': track_f,
        'track_r': track_r,
        'kVerticalSuspensionComplianceF': json_data["config"]["chassis"].get("kVerticalSuspensionComplianceF"),
        'kVerticalSuspensionComplianceR': json_data["config"]["chassis"].get("kVerticalSuspensionComplianceR"),
        'kARB_F': kARB_F,
        'kARB_R': kARB_R,
        'hRideF': hRideF,
        'hRideR': hRideR,
        'zCoG': zCoG
    }

    global_setup.update({
    "gap_bumpstop_FL": params[0]["bump_gap"],
    "gap_bumpstop_FR": params[1]["bump_gap"],
    "gap_bumpstop_RL": params[2]["bump_gap"],
    "gap_bumpstop_RR": params[3]["bump_gap"]
    })

    chassis = json_data['config']['chassis']
    tyres = json_data['config']['tyres']

    aero = {
        'zCoG': chassis.get('zCoG', 0.0),
        'kUndertrayFront': chassis.get('kUndertrayFront', 0),
        'kUndertrayMid': chassis.get('kUndertrayMid', 0),
        'kUndertrayRear': chassis.get('kUndertrayRear', 0),
        'rUndertrayFront': chassis.get('rUndertrayFront', [0, 0, 0]),
        'rUndertrayMid': chassis.get('rUndertrayMid', [0, 0, 0]),
        'rUndertrayRear': chassis.get('rUndertrayRear', [0, 0, 0])
    }

    tires = {
        'front': {
            'scaling': tyres['front']['SCALING_COEFFICIENTS'],
            'longitudinal': tyres['front']['LONGITUDINAL_COEFFICIENTS'],
            'lateral': tyres['front']['LATERAL_COEFFICIENTS'],
            'dimension': tyres['front']['DIMENSION'],
            'vertical': tyres['front']['VERTICAL']
        },
        'rear': {
            'scaling': tyres['rear']['SCALING_COEFFICIENTS'],
            'longitudinal': tyres['rear']['LONGITUDINAL_COEFFICIENTS'],
            'lateral': tyres['rear']['LATERAL_COEFFICIENTS'],
            'dimension': tyres['rear']['DIMENSION'],
            'vertical': tyres['rear']['VERTICAL']
        }
    }
    # --- AERODINÁMICA ---
    global_setup.update({
        "aero_full": aero  # información extendida del undertray
    })

    aero_poly = json_data['config']['aero']
    def parse_poly(poly_list):
        return {entry['expression']: entry['coefficient'] for entry in poly_list}

    aero_polynomials = {
        'CLiftBodyF': parse_poly(aero_poly.get('PolynomialCLiftBodyFDefinition', [])),
        'CLiftBodyR': parse_poly(aero_poly.get('PolynomialCLiftBodyRDefinition', [])),
        'CDragBody': parse_poly(aero_poly.get('PolynomialCDragBodyDefinition', [])),
        'DRS_CLiftF': parse_poly(aero_poly.get('DRS', {}).get('CLiftBodyFDRSPolynomial', [])),
        'DRS_CLiftR': parse_poly(aero_poly.get('DRS', {}).get('CLiftBodyRDRSPolynomial', [])),
        'DRS_CDrag': parse_poly(aero_poly.get('DRS', {}).get('CDragBodyDRSPolynomial', [])),
    }

    global_setup['aero_polynomials'] = aero_polynomials
    global_setup['aero_flapAngles'] = aero_poly.get('flapAngles', {})
    global_setup['aero_offsets'] = aero_poly.get('coefficientOffsets', {})
    global_setup['aero_DRS'] = aero_poly.get('DRS', {})
    global_setup['tires'] = tires
    return params, global_setup

def prepare_simple_params(params, global_setup):

    # Prepara un diccionario plano con los parámetros necesarios para vehicle_model_simple
    # params: lista de 4 esquinas (FL, FR, RL, RR)
    # global_setup: diccionario global

    from scipy.interpolate import interp1d
    # Muelle: usar directamente el valor de kSpring del JSON para cada esquina
    kFL = params[0]['kSpring']
    kFR = params[1]['kSpring']
    kRL = params[2]['kSpring']
    kRR = params[3]['kSpring']


    # Damper y bumpstop interpoladores
    # --- Corrección: usar compresión y extensión según el signo de la velocidad ---
    def damper_interp_factory(v_ext, f_ext, v_comp, f_comp):
        from scipy.interpolate import interp1d
        # Crear interpoladores lineales con extrapolación
        interp_ext = interp1d(v_ext, f_ext, kind='linear', fill_value='extrapolate', bounds_error=False)
        interp_comp = interp1d(v_comp, f_comp, kind='linear', fill_value='extrapolate', bounds_error=False)
        def damper_func(v):
            v = np.asarray(v)
            out = np.zeros_like(v)
            mask_ext = v > 0
            mask_comp = v < 0
            if np.any(mask_ext):
                out[mask_ext] = interp_ext(v[mask_ext])
            if np.any(mask_comp):
                out[mask_comp] = interp_comp(v[mask_comp])
            out[v == 0] = 0.0
            return out
        return damper_func
    damper_front = damper_interp_factory(params[0]['damper_v_ext'], params[0]['damper_f_ext'], params[0]['damper_v_comp'], params[0]['damper_f_comp'])
    damper_rear = damper_interp_factory(params[2]['damper_v_ext'], params[2]['damper_f_ext'], params[2]['damper_v_comp'], params[2]['damper_f_comp'])
    bumpstop_front_interp = interp1d(params[0]['bump_x'], params[0]['bump_f'], kind='linear', fill_value='extrapolate', bounds_error=False)
    bumpstop_rear_interp = interp1d(params[2]['bump_x'], params[2]['bump_f'], kind='linear', fill_value='extrapolate', bounds_error=False)

    bumpstop_front = lambda x: np.where(x < 0, 0, bumpstop_front_interp(x))
    bumpstop_rear = lambda x: np.where(x < 0, 0, bumpstop_rear_interp(x))

    # Rigidez de instalación (compliance vertical)
    # Rigidez de instalación total por eje, repartirla entre las dos esquinas
    kaxle_f = global_setup.get('kVerticalSuspensionComplianceF', 0.0)
    kaxle_r = global_setup.get('kVerticalSuspensionComplianceR', 0.0)
    # Cada esquina “ve” la mitad de la rigidez del eje
    kinstf = kaxle_f / 2.0
    kinstr = kaxle_r / 2.0


    # Rigidez de barra estabilizadora (anti roll bar)
    k_arb_f = global_setup.get('kARB_F', 0)
    k_arb_r = global_setup.get('kARB_R', 0)

    # --- Cálculo de topes físicos (top-out y bumpstop) para cada esquina ---
    z_topout_FL = float(min(params[0]['spring_x'][0], params[0]['bump_x'][0]))
    z_bottomout_FL = float(max(params[0]['spring_x'][-1], params[0]['bump_x'][-1]))

    z_topout_FR = float(min(params[1]['spring_x'][0], params[1]['bump_x'][0]))
    z_bottomout_FR = float(max(params[1]['spring_x'][-1], params[1]['bump_x'][-1]))

    z_topout_RL = float(min(params[2]['spring_x'][0], params[2]['bump_x'][0]))
    z_bottomout_RL = float(max(params[2]['spring_x'][-1], params[2]['bump_x'][-1]))

    z_topout_RR = float(min(params[3]['spring_x'][0], params[3]['bump_x'][0]))
    z_bottomout_RR = float(max(params[3]['spring_x'][-1], params[3]['bump_x'][-1]))

    # --- Masa suspendida total: suma de las 4 esquinas ---
    ms_total = sum([params[i]['ms'] for i in range(4)])

    stroke_FL = params[0].get("stroke")
    stroke_FR = params[1].get("stroke")
    stroke_RL = params[2].get("stroke")
    stroke_RR = params[3].get("stroke")

    # --- Aerodinámica: interpoladores para ClA y CdA ---
    aero_v = global_setup.get('aero_v', np.array([0]))
    aero_ClA = global_setup.get('aero_ClA', np.array([0]))
    aero_CdA = global_setup.get('aero_CdA', np.array([0]))

    tires = global_setup.get('tires', {})
    tire_front = tires.get('front', {})
    tire_rear = tires.get('rear', {})
    
    ClA_interp = interp1d(aero_v, aero_ClA, kind='linear', fill_value='extrapolate', bounds_error=False)
    CdA_interp = interp1d(aero_v, aero_CdA, kind='linear', fill_value='extrapolate', bounds_error=False)
 
    return {
        'ms': ms_total,
        'Ixx': global_setup['ICar'][0],
        'Iyy': global_setup['ICar'][4],
        'lf': global_setup['wheelbase'] * (1 - global_setup['rWeightBalF']),
        'lr': global_setup['wheelbase'] * global_setup['rWeightBalF'],
        'tf': global_setup['track_f'],
        'tr': global_setup['track_r'],
        'mHubF': global_setup['mHubF'],
        'mHubR': global_setup['mHubR'],
        'hRideF': global_setup['hRideF'],
        'hRideR': global_setup['hRideR'],
        'rWeightBalF': global_setup['rWeightBalF'],
        'kFL': kFL,
        'kFR': kFR,
        'kRL': kRL,
        'kRR': kRR,
        'kinstf': kinstf,  
        'kinstr': kinstr,
        'ktf': params[0]['kt'],
        'ktr': params[2]['kt'],
        'z_FL_free': params[0]['bump_gap'],
        'z_FR_free': params[1]['bump_gap'],
        'z_RL_free': params[2]['bump_gap'],
        'z_RR_free': params[3]['bump_gap'],
        'bumpstop_front': bumpstop_front,
        'bumpstop_rear': bumpstop_rear,
        'damper_front': damper_front,
        'damper_rear': damper_rear,
        'spring_x_FL': params[0]['spring_x'],
        'spring_x_FR': params[1]['spring_x'],
        'spring_x_RL': params[2]['spring_x'],
        'spring_x_RR': params[3]['spring_x'],
        'bump_x_FL': params[0]['bump_x'],
        'bump_x_FR': params[1]['bump_x'],
        'bump_x_RL': params[2]['bump_x'],
        'bump_x_RR': params[3]['bump_x'],
        'k_arb_f': k_arb_f,
        'k_arb_r': k_arb_r,
        'stroke_FL': stroke_FL,
        'stroke_FR': stroke_FR,
        'stroke_RL': stroke_RL,
        'stroke_RR': stroke_RR,
        'z_topout_FL': z_topout_FL,
        'z_bottomout_FL': z_bottomout_FL,
        'z_topout_FR': z_topout_FR,
        'z_bottomout_FR': z_bottomout_FR,
        'z_topout_RL': z_topout_RL,
        'z_bottomout_RL': z_bottomout_RL,
        'z_topout_RR': z_topout_RR,
        'z_bottomout_RR': z_bottomout_RR,
        'aero_ClA': ClA_interp,
        'aero_CdA': CdA_interp,
        'tire_front': tire_front,
        'tire_rear': tire_rear,
        'aero_polynomials': global_setup['aero_polynomials'],
        'gap_bumpstop_FL': global_setup['gap_bumpstop_FL'],
        'gap_bumpstop_FR': global_setup['gap_bumpstop_FR'],
        'gap_bumpstop_RL': global_setup['gap_bumpstop_RL'],
        'gap_bumpstop_RR': global_setup['gap_bumpstop_RR'],
        'x_FL_static': params[0]['spring_bump_total_f'][0] / kFL,
        'x_FR_static': params[1]['spring_bump_total_f'][0] / kFR,
        'x_RL_static': params[2]['spring_bump_total_f'][0] / kRL,
        'x_RR_static': params[3]['spring_bump_total_f'][0] / kRR,
    }

def simulate_combo(setup_path, track_path, kt_overrides=None):
    import json
    from gui_v2 import parse_json_setup, prepare_simple_params, load_track_channels
    from model import run_vehicle_model_simple, postprocess_7dof
    from pathlib import Path

    # 1) Carga el JSON original
    with open(setup_path, 'r') as f:
        setup_data_raw = json.load(f)

    # 2) Extrae params y global_setup
    params, global_setup = parse_json_setup(setup_data_raw)
    #    'params' es lista de 4 dicts: FL, FR, RL, RR con clave "kt" incl.
    simple_params = prepare_simple_params(params, global_setup)

    # 3) Si el usuario pasó kt_overrides, lo inyectamos:
    if kt_overrides is not None:
        # Ejemplo de kt_overrides = {'front': 400000, 'rear': 300000}
        if kt_overrides.get('front') is not None:
            simple_params['ktf'] = kt_overrides['front']
        if kt_overrides.get('rear') is not None:
            simple_params['ktr'] = kt_overrides['rear']

    # 4) Carga el track y corre la simulación
    track_data = load_track_channels(track_path)
    t_vec   = track_data['t']
    z_tracks= track_data['z_tracks']
    vx      = track_data['vx']
    ax      = track_data['ax']
    ay      = track_data['ay']
    rpedal  = track_data['rpedal']
    pbrake  = track_data['brake']

    sol = run_vehicle_model_simple(t_vec, z_tracks, vx, ax, ay, rpedal, pbrake, simple_params)
    simple_params['track_name'] = Path(track_path).stem
    post = postprocess_7dof(sol, simple_params, z_tracks, t_vec, rpedal, pbrake, vx)

    return sol, post, setup_path, track_path

def load_track_channels(track_path):
    import pandas as pd

    df = pd.read_csv(track_path)
    columns_required = ['Times', 'Car_Speed', 'Ax', 'Ay', 'rPedal', 'pBrake', 'Zp_FL', 'Zp_FR', 'Zp_RL', 'Zp_RR']
    missing = [col for col in columns_required if col not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el track: {', '.join(missing)}")

    data = {
        't': df['Times'].values,
        'vx': df['Car_Speed'].values,
        'ax': df['Ax'].values,
        'ay': df['Ay'].values,
        'rpedal': df['rPedal'].values,
        'brake': df['pBrake'].values,
        'z_tracks': [
            df['Zp_FL'].values,
            df['Zp_FR'].values,
            df['Zp_RL'].values,
            df['Zp_RR'].values
        ]
    }

    # Conversión vienen en mm
    data['z_tracks'] = [z / 1000.0 for z in data['z_tracks']]

    return data

class SevenPostRigGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("7-Post Rig Simulator")
        self.setWindowIcon(QIcon("icon.png"))
        self.resize(1100, 800)
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)
        self.tabs.setStyleSheet("""
            QTabBar::tab {
                background: #bbb;
                color: #222;
                padding: 10px 26px;
                min-width: 96px;
                min-height: 26px;
                font-size: 13px;
                font-weight: bold;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #e0e0e0;
                color: #1a237e;
            }
            QTabBar::tab:hover {
                background: #d5d5d5;
                color: #1565c0;
            }
            QTabWidget::pane {
                border-top: 3px solid #1565c0;
                top: -1em;
            }
        """)
        main_layout.addWidget(self.tabs)

        # --- Configuración ---
        self.config_tab = QWidget()
        config_layout = QVBoxLayout()
        group_setup = QGroupBox("Carga de Setups y Tracks")
        group_setup.setStyleSheet("""
            QGroupBox {
                color: #1565c0;
                font-weight: bold;
                font-size: 15px;
                background: transparent;
                border-radius: 8px;
                margin-top: 1.5em;
            }
            QGroupBox:title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 10px;
                background: transparent;
                color: #1565c0;
            }
        """)
        # Widget contenedor con fondo claro para los DragDropLabel
        dragdrop_container = QWidget()
        dragdrop_container.setStyleSheet("background: #e0e0e0; border-radius: 8px;")
        group_layout = QGridLayout(dragdrop_container)
        # NUEVO: Listas para setups y tracks
        self.setup_list = QListWidget()
        self.setup_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setup_list.setStyleSheet("background: #fff; color: #1565c0; font-weight: bold;")
        self.track_list = QListWidget()
        self.track_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.track_list.setStyleSheet("background: #fff; color: #1565c0; font-weight: bold;")
        group_layout.addWidget(QLabel("Setups (JSON):"), 0, 0)
        group_layout.addWidget(self.setup_list, 1, 0)
        group_layout.addWidget(QLabel("Tracks (CSV):"), 0, 1)
        group_layout.addWidget(self.track_list, 1, 1)
        # Botones para cargar archivos (más visibles)
        btn_load_setups = QPushButton("Añadir Setups")
        btn_load_setups.setStyleSheet("""
            QPushButton {
                background: #1565c0;
                color: #fff;
                font-size: 15px;
                font-weight: bold;
                border-radius: 8px;
                padding: 10px 24px;
                margin: 8px 0;
            }
            QPushButton:hover {
                background: #1976d2;
                color: #fff;
            }
        """)
        btn_load_setups.clicked.connect(self.add_setups)
        btn_load_tracks = QPushButton("Añadir Tracks")
        btn_load_tracks.setStyleSheet("""
            QPushButton {
                background: #388e3c;
                color: #fff;
                font-size: 15px;
                font-weight: bold;
                border-radius: 8px;
                padding: 10px 24px;
                margin: 8px 0;
            }
            QPushButton:hover {
                background: #43a047;
                color: #fff;
            }
        """)
        btn_load_tracks.clicked.connect(self.add_tracks)
        group_layout.addWidget(btn_load_setups, 2, 0)
        group_layout.addWidget(btn_load_tracks, 2, 1)
        # Área de listas más pequeña
        self.setup_list.setMaximumHeight(80)
        self.track_list.setMaximumHeight(80)
        group_setup.setLayout(QVBoxLayout())
        group_setup.layout().addWidget(dragdrop_container)
        config_layout.addWidget(group_setup)

        # --- NUEVO: Formulario de parámetros ---
        self.param_group = QGroupBox("Parámetros del Setup")
        self.param_form = QFormLayout()
        self.param_edits = {}  # clave: (rueda, nombre), valor: QLineEdit
        self.param_group.setLayout(self.param_form)
        config_layout.addWidget(self.param_group)
        self.param_group.setVisible(False)

        # Botón para guardar cambios
        self.save_params_btn = QPushButton("Guardar Cambios")
        self.save_params_btn.setStyleSheet(
            "QPushButton { background: #2679b8; color: white; border-radius: 8px; padding: 8px; }"
            "QPushButton:hover { background: #3a9ad9; }"
        )
        self.save_params_btn.clicked.connect(self.save_params_tab)
        config_layout.addWidget(self.save_params_btn)
        self.save_params_btn.setVisible(False)

        self.config_tab.setLayout(config_layout)
        self.tabs.addTab(self.config_tab, "Configuración")

        # NUEVA PESTAÑA: Visualización y edición de parámetros del setup
        self.params_tab = QWidget()
        params_layout = QVBoxLayout()
        # Selector de setup a editar
        self.edit_setup_selector = QComboBox()
        self.edit_setup_selector.setStyleSheet("background: #e0e0e0; color: #1565c0; font-weight: bold;")
        self.edit_setup_selector.currentIndexChanged.connect(self.on_edit_setup_changed)
        params_layout.addWidget(self.edit_setup_selector)
        self.params_table_group = QGroupBox("Parámetros del Setup Seleccionado")
        self.params_table_group.setStyleSheet("color: #1565c0; font-weight: bold; font-size: 15px;")
        self.params_form = QFormLayout()
        self.params_edits = {}  # (rueda, clave): QLineEdit
        self.params_table_group.setLayout(self.params_form)
        params_layout.addWidget(self.params_table_group)
        self.apply_params_btn = QPushButton("Aplicar Cambios")
        self.apply_params_btn.setStyleSheet(
            "QPushButton { background: #2679b8; color: white; border-radius: 8px; padding: 8px; font-weight: bold; }"
            "QPushButton:hover { background: #3a9ad9; }"
        )
        self.apply_params_btn.clicked.connect(self.save_params_tab)
        params_layout.addWidget(self.apply_params_btn)
        self.params_tab.setLayout(params_layout)
        self.tabs.addTab(self.params_tab, "Editar Setup")

        # --- Simulación ---
        self.sim_tab = QWidget()
        sim_layout = QVBoxLayout()
        self.sim_button = QPushButton("Iniciar Simulación (Ctrl+R)")
        self.sim_button.setToolTip("Ejecuta la simulación con los archivos cargados")
        self.sim_button.setShortcut("Ctrl+R")
        self.sim_button.clicked.connect(self.run_simulation)
        self.sim_button.setStyleSheet("""
            QPushButton { background: #2679b8; color: white; border-radius: 8px; padding: 10px; }
            QPushButton:hover { background: #3a9ad9; }
        """)
        sim_layout.addWidget(self.sim_button)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        sim_layout.addWidget(self.progress)
        self.sim_status = QLabel("")
        sim_layout.addWidget(self.sim_status)
        self.sim_tab.setLayout(sim_layout)
        self.tabs.addTab(self.sim_tab, "Simulación")

        # --- Resultados ---
        self.results_tab = QWidget()
        results_layout = QVBoxLayout()

        # Selector
        self.result_selector = QComboBox()
        self.result_selector.currentIndexChanged.connect(self.show_results)
        results_layout.addWidget(self.result_selector)

        # Web view de Dash
        self.web_view = QWebEngineView()
        results_layout.addWidget(self.web_view)

        self.results_tab.setLayout(results_layout)
        self.tabs.addTab(self.results_tab, "Resultados")

        # --- KPIs ---
        self.kpi_tab = QWidget()  
        kpi_layout = QVBoxLayout()  
        self.kpi_web_view = QWebEngineView()
        self.kpi_web_view.load(QUrl("http://127.0.0.1:8051"))
        self.export_button = QPushButton("Exportar Reporte HTML")
        self.export_button.setStyleSheet("background: #1565c0; color: white; padding: 10px;")
        self.export_button.clicked.connect(self.export_report)
        kpi_layout.addWidget(self.export_button)
        kpi_layout.addWidget(self.kpi_web_view)

        self.kpi_tab.setLayout(kpi_layout)
        self.tabs.addTab(self.kpi_tab, "KPIs")

        # Feedback visual
        self.feedback_label = QLabel("")
        self.feedback_label.setStyleSheet("font-weight: bold;")
        main_layout.addWidget(self.feedback_label)

    def load_setup(self, filepath):
        # Carga y muestra los parámetros en el formulario
        import json
        # 1) Abrir y guardar JSON completo
        with open(filepath, 'r') as f:
            json_data = json.load(f)

        # Guardamos el JSON original y la ruta para luego sobreescribir
        self.raw_setup_json = json_data
        self.current_editing_setup = filepath
        self.kt_overrides = {'front': None, 'rear': None}

        # 2) Obtener sólo los parámetros “planares” de las 4 esquinas
        #    parse_json_setup devuelve (params, global_setup) donde params es lista de 4 dicts
        if isinstance(json_data, dict):
            parsed = parse_json_setup(json_data)
            setup_data = parsed[0]  # params
        elif isinstance(json_data, tuple):
            setup_data = json_data[0]
        else:
            setup_data = json_data  # ya sería lista de 4 dicts

        # 3) Construir el formulario de edición igual que antes
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        self.params_edits = {}
        wheel_labels = ['FL', 'FR', 'RL', 'RR']

        for i, corner in enumerate(setup_data):
            wheel_group = QGroupBox(f"Rueda {wheel_labels[i]}")
            wheel_group.setStyleSheet("""
                QGroupBox {
                    color: #1565c0;
                    font-weight: bold;
                    font-size: 14px;
                    border: 1px solid #b0bec5;
                    border-radius: 8px;
                    margin-top: 10px;
                    background: #f7fafd;
                }
                QGroupBox:title {
                    subcontrol-origin: margin;
                    subcontrol-position: top left;
                    padding: 0 8px;
                }
            """)
            form = QFormLayout()
            for key in sorted(corner.keys()):
                val = corner[key]
                edit = QLineEdit(str(val) if val is not None else "")
                edit.setStyleSheet("""
                    QLineEdit {
                        color: #222;
                        background: #fff;
                        border: 1px solid #90caf9;
                        border-radius: 4px;
                        padding: 4px 8px;
                    }
                    QLineEdit:focus {
                        border: 1.5px solid #1565c0;
                        background: #e3f2fd;
                    }
                """)
                if isinstance(val, (float, int)) or val is None:
                    edit.setValidator(QDoubleValidator())
                edit.setToolTip(f"Parámetro '{key}' de la rueda {wheel_labels[i]}")
                form.addRow(QLabel(key), edit)
                self.params_edits[(i, key)] = edit

            wheel_group.setLayout(form)
            main_layout.addWidget(wheel_group)

        scroll.setWidget(main_widget)

        # Limpiar layout anterior y añadir el nuevo scroll
        for idx in reversed(range(self.params_table_group.layout().count())):
            self.params_table_group.layout().itemAt(idx).widget().setParent(None)
        self.params_table_group.layout().addWidget(scroll)

        self.params_table_group.setVisible(True)
        self.apply_params_btn.setVisible(True)
        self.tabs.setCurrentIndex(self.tabs.indexOf(self.params_tab))

    def save_params_tab(self):
        import json

        # 1) Partimos del JSON completo que guardamos al cargar
        json_data = self.raw_setup_json

        # 2) Para cada parámetro editado, sobrescribimos el valor en la sección correspondiente
        for (i, key), edit in self.params_edits.items():
            text = edit.text()
            try:
                val = float(text) if '.' in text or 'e' in text.lower() else int(text)
            except ValueError:
                val = text

            # Determinar si es “front” (i < 2) o “rear” (i >= 2)
            side = 'front' if i < 2 else 'rear'
            # Atajos a las secciones en el JSON original:
            spring = json_data['config']['suspension'][side]['internal']['spring']
            bump   = json_data['config']['suspension'][side]['internal']['bumpStop']
            damper = json_data['config']['suspension'][side]['internal']['damper']

            # Sobrescribir en el lugar correcto, si existe la clave
            if key in spring:
                spring[key] = val
            elif key in bump:
                bump[key] = val
            elif key in damper:
                damper[key] = val
            else:
                # Si la clave no está en spring / bumpStop / damper, la ignoramos
                continue

        # 3) Escribir el JSON actualizado en el mismo archivo (sobrescribimos)
        with open(self.current_editing_setup, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)

        self.feedback("✅ Cambios guardados en el archivo original.", "success")

        # 4) Volver a recargar el formulario para reflejar los cambios
        self.load_setup_params_tab(self.current_editing_setup)

    def load_track(self, filepath):
        self.track_data = filepath
        self.track_label.setText(f"Track cargado: {filepath}")
        self.feedback("✅ Track cargado correctamente.", "success")

    def add_setups(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Selecciona archivos de Setup (JSON)", "", "JSON Files (*.json)")
        for f in files:
            if f not in [self.setup_list.item(i).text() for i in range(self.setup_list.count())]:
                self.setup_list.addItem(f)
        self.feedback(f"{len(files)} setup(s) añadidos.", "success")
        # Actualiza el selector de edición
        self.edit_setup_selector.clear()
        for i in range(self.setup_list.count()):
            self.edit_setup_selector.addItem(self.setup_list.item(i).text())
        # Si solo se añade uno, lo mostramos en la pestaña de edición
        if len(files) == 1:
            self.load_setup_params_tab(files[0])
        elif self.setup_list.count() > 0:
            self.load_setup_params_tab(self.setup_list.item(0).text())

    def add_tracks(self):
        import pandas as pd
        files, _ = QFileDialog.getOpenFileNames(self, "Selecciona archivos de Track (CSV)", "", "CSV Files (*.csv)")
        for f in files:
            if f not in [self.track_list.item(i).text() for i in range(self.track_list.count())]:
                try:
                    df = pd.read_csv(f)
                    required = ["Times", "Car_Speed", "Ax", "Ay", "rPedal", "pBrake", "Zp_FL", "Zp_FR", "Zp_RL", "Zp_RR"]
                    missing = [col for col in required if col not in df.columns]
                    if missing:
                        self.feedback(f"⚠️ Track {os.path.basename(f)} está incompleto. Faltan: {', '.join(missing)}", "warning")
                    else:
                        self.track_list.addItem(f)
                        self.feedback(f"✅ Track {os.path.basename(f)} añadido correctamente.", "success")
                except Exception as e:
                    self.feedback(f"❌ Error leyendo {os.path.basename(f)}: {str(e)}", "error")

    def on_edit_setup_changed(self, idx):
        if idx < 0 or idx >= self.edit_setup_selector.count():
            return
        filepath = self.edit_setup_selector.itemText(idx)
        if filepath:
            self.load_setup_params_tab(filepath)

    def load_setup_params_tab(self, filepath):
        import json
        from PyQt5.QtWidgets import QWidget, QScrollArea, QVBoxLayout, QGroupBox, QFormLayout, QLabel, QLineEdit

        with open(filepath, 'r') as f:
            setup_data_raw = json.load(f)
            self.raw_setup_json = setup_data_raw
            self.current_editing_setup = filepath

        if isinstance(setup_data_raw, dict):
            corner_list, _ = parse_json_setup(setup_data_raw)
        elif isinstance(setup_data_raw, tuple):
            corner_list = setup_data_raw[0]
        else:
            corner_list = setup_data_raw

        front = corner_list[0]
        rear  = corner_list[2]
        axle_data = [("Front", front), ("Rear", rear)]

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)

        self.kt_overrides = {'front': None, 'rear': None}
        self.params_edits = {}

        for axle_name, corner in axle_data:
            side = 'front' if axle_name == "Front" else 'rear'
            group = QGroupBox(f"Eje {axle_name}")
            form = QFormLayout()

            # 1) kt (editable, en memoria)
            val_kt = corner.get("kt", "")
            edit_kt = QLineEdit(str(val_kt) if val_kt is not None else "")
            edit_kt.setValidator(QDoubleValidator())
            edit_kt.setToolTip(f"Rigidez de neumático (kt) eje {axle_name}")
            form.addRow(QLabel("kt"), edit_kt)
            self.params_edits[(side, "kt")] = edit_kt

            # 2) Los demás valores escalares que queramos (ms, mu, kSpring, FSpringPreload, bump_gap, stroke)
            valores_escalares = {
                "ms",              # SOLO LECTURA: masa suspendida
                "mu",              # SOLO LECTURA: masa no suspendida
                "kSpring",         # editable
                "FSpringPreload",  # editable si existiera (de lo contrario corner["FSpringPreload"] = 0)
                "bump_gap",        # editable
                "stroke",          # editable
            }

            for key in sorted(corner.keys()):
                if key == "kt":
                    continue
                if key in valores_escalares:
                    valor = corner.get(key, "")
                    edit = QLineEdit(str(valor) if valor is not None else "")
                    edit.setValidator(QDoubleValidator())
                    edit.setToolTip(f"Parámetro '{key}' eje {axle_name}")
                    # Si key es “ms” o “mu”, ponemos solo lectura:
                    if key in {"ms", "mu"}:
                        edit.setReadOnly(True)
                        edit.setStyleSheet("background: #444; color: #ddd;")
                    form.addRow(QLabel(key), edit)
                    self.params_edits[(side, key)] = edit

            group.setLayout(form)
            main_layout.addWidget(group)

        scroll.setWidget(main_widget)

        for i in reversed(range(self.params_table_group.layout().count())):
            self.params_table_group.layout().itemAt(i).widget().setParent(None)
        self.params_table_group.layout().addWidget(scroll)

        self.params_table_group.setVisible(True)
        self.apply_params_btn.setVisible(True)
        self.tabs.setCurrentIndex(self.tabs.indexOf(self.params_tab))
    
    def save_params_tab(self):
        import json
        from PyQt5.QtWidgets import QMessageBox

        if not hasattr(self, 'raw_setup_json') or not hasattr(self, 'current_editing_setup'):
            QMessageBox.warning(self, "Error", "No hay ningún setup cargado para guardar.")
            return

        for (side, key), edit in self.params_edits.items():
            text = edit.text()
            try:
                val = float(text) if '.' in text or 'e' in text.lower() else int(text)
            except ValueError:
                val = text

            if key == "kt":
                # guardamos solo en memoria
                self.kt_overrides[side] = val

            elif key in {"ms", "mu"}:
                # Los dejamos solo lectura, no modificamos el JSON aquí.
                # (Podrías mostrar un warning si alguien intenta cambiar “ms”/“mu”.)
                continue

            else:
                # sobreecribir el JSON original para los parámetros escalares
                spring = self.raw_setup_json['config']['suspension'][side]['internal']['spring']
                bump   = self.raw_setup_json['config']['suspension'][side]['internal']['bumpStop']
                damper = self.raw_setup_json['config']['suspension'][side]['internal']['damper']

                if key == "kSpring":
                    spring["kSpring"] = val
                elif key == "FSpringPreload":
                    spring["FSpringPreload"] = val
                elif key == "bump_gap":
                    bump["xFreeGap"] = val
                elif key == "stroke":
                    # En tu parser original, ‘stroke’ se asigna manualmente; no hay clave directa en JSON.
                    # Si tu JSON no tiene ‘stroke’, aquí no lo toques o aclara su ruta real.
                    pass
                else:
                    # Si faltara algún otro campo, ignorarlo
                    continue

        # Guardamos el JSON sin tocar “kt”
        with open(self.current_editing_setup, 'w', encoding='utf-8') as f:
            json.dump(self.raw_setup_json, f, indent=2)

        self.feedback("✅ Cambios guardados en el JSON (excluyendo kt).", "success")
        print(f"[DEBUG] kt actualizados (en memoria): {self.kt_overrides}")

        # recargar la UI
        self.load_setup_params_tab(self.current_editing_setup)

    def run_simulation(self):
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import traceback

        if self.setup_list.count() == 0 or self.track_list.count() == 0:
            self.feedback("❌ Debes añadir al menos un setup y un track.", "error")
            return

        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.sim_status.setText("Simulando...")
        self.sim_results = []
        self.result_selector.clear()

        combo_list = []
        for i in range(self.setup_list.count()):
            setup_path = self.setup_list.item(i).text()
            for j in range(self.track_list.count()):
                track_path = self.track_list.item(j).text()
                combo_list.append((setup_path, track_path))

        self.progress.setMaximum(len(combo_list))

        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(simulate_combo, setup, track, self.kt_overrides)
                for setup, track in combo_list
            ]

            for i, future in enumerate(as_completed(futures)):
                try:
                    sol, post, setup_path, track_path = future.result()

                    # === VALIDACIÓN DE TRACK ===
                    import pandas as pd
                    df = pd.read_csv(track_path)
                    if 'Zp_FL' in df.columns:
                        z_tracks = [
                            df['Zp_FL'].values,
                            df['Zp_FR'].values,
                            df['Zp_RL'].values,
                            df['Zp_RR'].values
                        ]
                    elif 'z_FL (m)' in df.columns:
                        z_tracks = [
                            df['z_FL (m)'].values,
                            df['z_FR (m)'].values,
                            df['z_RL (m)'].values,
                            df['z_RR (m)'].values
                        ]
                    else:
                        z_tracks = []

                    for idx, zt in enumerate(z_tracks):
                        dz = np.abs(np.diff(zt))
                        if np.any(dz > 0.05):
                            msg = (
                                f"⚠️ Track {['FL','FR','RL','RR'][idx]} "
                                f"tiene saltos > 0.05 m (máx: {dz.max():.3f} m)"
                            )
                            self.feedback(msg, "warning")

                    # === GUARDAR RESULTADOS ===
                    self.sim_results.append((sol, post, setup_path, track_path))
                    name = (
                        f"Setup: {os.path.basename(setup_path)} | "
                        f"Track: {os.path.basename(track_path)}"
                    )
                    self.result_selector.addItem(name)

                    # === LOG DE RESULTADO ===
                    print(f"\n[DEBUG] Simulación: {name}")
                    print(f"  Tiempo final: {sol.t[-1]:.3f} s, pasos: {len(sol.t)}")
                    print(f"  Travel FL final: {post['travel'][0][-1]*1000:.2f} mm")
                    print(f"  Travel RL final: {post['travel'][2][-1]*1000:.2f} mm")
                    print(f"  Fuerza muelle FL (inicio): {post['f_spring'][0][:5]}")
                    print(f"  Fuerza neumático FL (inicio): {post['f_tire'][0][:5]}")
                    print(f"  Fuerza amortiguador FL (inicio): {post['f_damper'][0][:5]}")
                    print("[DEBUG] Topes físicos por esquina (topout / bottomout):")
                    for wheel in ["FL", "FR", "RL", "RR"]:
                        zt = post[f'z_topout_{wheel}']   # valor escalar [m]
                        zb = post[f'z_bottomout_{wheel}']
                        print(f"  {wheel}: {zt*1000:.2f} mm / {zb*1000:.2f} mm")
                    # ───────────────────────────────────────────────────────────

                    for eje, idxs in zip(['Front', 'Rear'], [(0,1),(2,3)]):
                        travel = np.mean([post['travel'][i]*1000 for i in idxs], axis=0)
                        f_spring = np.mean([post['f_spring'][i] for i in idxs], axis=0)
                        f_damper = np.mean([post['f_damper'][i] for i in idxs], axis=0)
                        f_tire = np.mean([post['f_tire'][i] for i in idxs], axis=0)
                        print(
                            f"  {eje} travel max: {np.max(travel):.2f} mm, "
                            f"min: {np.min(travel):.2f} mm"
                        )
                        print(f"  {eje} spring max: {np.max(f_spring):.2f} N")
                        print(f"  {eje} damper max: {np.max(f_damper):.2f} N")
                        print(f"  {eje} tire max: {np.max(f_tire):.2f} N")

                except Exception as e:
                    tb = traceback.format_exc()
                    print(f"[ERROR] Simulación fallida:\n{tb}")
                    self.feedback(
                        f"❌ Error en simulación:\n{type(e).__name__}: {str(e)}",
                        "error"
                    )

                self.progress.setValue(i + 1)

        self.sim_status.setText("Simulación completada.")
        self.feedback("✅ Simulaciones finalizadas.", "success")

        if self.sim_results:
            self.show_results()
            visualizer_dash.run_kpi_comparison_in_thread(self.sim_results)
            self.kpi_web_view.load(QUrl("http://127.0.0.1:8051"))
            self.tabs.setCurrentIndex(self.tabs.indexOf(self.results_tab))

    def show_results(self, idx=None):
        if not hasattr(self, 'sim_results') or self.result_selector.count() == 0:
            return
        idx = self.result_selector.currentIndex() if idx is None else idx
        if idx < 0 or idx >= len(self.sim_results):
            return
        sol, post, setup_path, track_path = self.sim_results[idx]
        setup_name = os.path.basename(setup_path).replace(".json", "")
        visualizer_dash.run_in_thread(sol, post, setup_name=setup_name)
        self.web_view.load(QUrl("http://127.0.0.1:8050"))

    def export_report(self):
        if not hasattr(self, 'sim_results') or not self.sim_results:
            self.feedback("⚠️ No hay resultados para exportar.", "warning")
            return
        from visualizer_dash import export_full_report
        export_path, _ = QFileDialog.getSaveFileName(self, "Guardar Reporte HTML", "", "HTML Files (*.html)")
        if export_path:
            try:
                export_full_report(self.sim_results, export_path)
                self.feedback(f"✅ Reporte exportado en {export_path}", "success")
            except Exception as e:
                self.feedback(f"❌ Error al exportar: {str(e)}", "error")
        
    def feedback(self, msg, level):
        color = {"success": "#4e9a06", "warning": "#f6c700", "error": "#d7263d"}.get(level, "#aaa")
        self.feedback_label.setText(msg)
        self.feedback_label.setStyleSheet(f"color: {color}; font-weight: bold;")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    set_dark_theme(app)
    window = SevenPostRigGUI()
    window.show()
    sys.exit(app.exec_())
