# This is a Plotly version of the original Matplotlib-based Grafico_de_prueba.py
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import json
import os
import sys


CHASSIS_COLOR_MAP = {
    '006': '#FF0000',  # rojo
    '018': "#FFCC00",  # amarillo claro
    '080': '#000000',  #negro
    '057': "#FF00C8",  # lila claro
    '099': '#7FD27F',  # verde claro
    '159': '#7FA6FF',  # azul claro
}

FALLBACK_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c',
                   '#d62728', '#9467bd', '#8c564b']

def pick_color(file_name: str, idx: int) -> str:
    """
    Devuelve el color fijo si el nombre de archivo contiene un nº de chasis
    conocido; en caso contrario usa la paleta de respaldo.
    """
    base = os.path.basename(file_name)
    # Busca la sub-cadena '_XXX' dentro del nombre (Campos_006, Campos_159, …)
    for chassis, col in CHASSIS_COLOR_MAP.items():
        if f'_{chassis}' in base:
            return col
    # Si no coincide, color de respaldo (cíclico)
    return FALLBACK_COLORS[idx % len(FALLBACK_COLORS)]

def resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        return os.path.join(os.path.dirname(sys.executable), relative_path)
    return os.path.join(os.path.dirname(__file__), relative_path)

def cargar_datos_json(ruta_archivo):
    with open(ruta_archivo, "r") as json_file:
        datos = json.load(json_file)
    return datos

def es_numero(valor):
    try:
        float(valor)
        return True
    except ValueError:
        return False

def get_threshold(config, percent_key, abs_key, reference, mode):
    """Return threshold based on config and mode (percent or absolute)."""
    if mode == 'absolute' and abs_key in config:
        return config[abs_key]
    elif percent_key in config:
        return config[percent_key] * reference
    else:
        return None


def extraer_stats_12pts(data, variable_y):
    """Compute mean and std for the 9 phase labels using WinTAX logic."""
    config_path = resource_path('config_conditions.json')
    with open(config_path, 'r') as f:
        config_all = json.load(f)
    mode = config_all.get('mode', 'percentage')
    config = config_all['absolute_conditions'] if mode == 'absolute' else config_all['percentage_conditions']

    def _num_list(values, scale=1.0):
        return [float(v) * scale if es_numero(v) else np.nan for v in values]

    if 'CarSpeed' in data['Datos']:
        vel = _num_list(data['Datos']['CarSpeed'])
        ax_vals = _num_list(data['Datos']['Ax'])
        brake = _num_list(data['Datos']['Brake_Press'])
        throttle = _num_list(data['Datos'].get('Throttle', [np.nan]*len(vel)))
    else:
        vel = _num_list(data['Datos']['vCar'], 3.6)
        ax_vals = _num_list(data['Datos']['gLong'])
        brake = _num_list(data['Datos']['Brake_Total'])
        throttle = _num_list(data['Datos'].get('rThrottlePedal', [np.nan]*len(vel)))

    y = _num_list(data['Datos'][variable_y])

    min_decel = min([v for v in ax_vals if not np.isnan(v) and v < 0], default=None)
    max_brake = max([v for v in brake if not np.isnan(v)], default=None)
    max_vel = max([v for v in vel if not np.isnan(v)], default=None)

    stats = {}

    # Helper to compute mean/std
    def _mean_std(indices, label, pos):
        valores = [y[i] for i in indices if i < len(y) and not np.isnan(y[i])]
        if valores:
            stats[label] = (float(np.mean(valores)), float(np.std(valores)))
        else:
            stats[label] = (None, None)

    # Early Entry LS
    speed_low = get_threshold(config["early_entry_ls"], "speed_lower_percent", "speed_lower", max_vel, mode)
    speed_high = get_threshold(config["early_entry_ls"], "speed_upper_percent", "speed_upper", max_vel, mode)
    brake_low = get_threshold(config["early_entry_ls"], "brake_lower_percent", "brake_lower", max_brake, mode)
    brake_high = get_threshold(config["early_entry_ls"], "brake_upper_percent", "brake_upper", max_brake, mode)
    ax_low = get_threshold(config["early_entry_ls"], "ax_lower_percent", "ax_lower", abs(min_decel) if min_decel is not None else None, mode)
    ax_high = get_threshold(config["early_entry_ls"], "ax_upper_percent", "ax_upper", abs(min_decel) if min_decel is not None else None, mode)
    if None not in (speed_low, speed_high, brake_low, brake_high, ax_low, ax_high):
        idx_ls = [i for i in range(len(vel)) if not np.isnan(vel[i]) and speed_low <= vel[i] <= speed_high and not np.isnan(brake[i]) and brake_low <= brake[i] <= brake_high and not np.isnan(ax_vals[i]) and -ax_high <= ax_vals[i] <= -ax_low]
    else:
        idx_ls = []
    _mean_std(idx_ls, 'early_ls', None)

    # Mid Corner LS
    throttle_thresh = get_threshold(config["mid_corner_ls"], "throttle_threshold_percent", "throttle_threshold", max(throttle) if throttle else 1, mode)
    try:
        from app_dash import get_default_conditions
        defaults = get_default_conditions()
        default_throttle_thresh = defaults["percentage_conditions"]["mid_corner_ls"]["throttle_threshold"]
    except Exception:
        default_throttle_thresh = 0.05
    if throttle_thresh is None:
        throttle_thresh = default_throttle_thresh
    idx_mid_ls = [i for i in range(len(throttle)) if throttle[i] is not None and not np.isnan(throttle[i]) and abs(throttle[i]) < throttle_thresh]
    _mean_std(idx_mid_ls, 'mid_ls', None)

    # Exit LS
    throttle_diff = np.diff(throttle)
    idx_exit_ls = [i + 1 for i, d in enumerate(throttle_diff) if not np.isnan(d) and d > 0.05]
    _mean_std(idx_exit_ls, 'exit_ls', None)

    # Early Entry MS
    speed_low = get_threshold(config["early_entry_ms"], "speed_lower_percent", "speed_lower", max_vel, mode)
    speed_high = get_threshold(config["early_entry_ms"], "speed_upper_percent", "speed_upper", max_vel, mode)
    brake_low = get_threshold(config["early_entry_ms"], "brake_lower_percent", "brake_lower", max_brake, mode)
    brake_high = get_threshold(config["early_entry_ms"], "brake_upper_percent", "brake_upper", max_brake, mode)
    idx_early_ms = [i for i in range(len(vel)) if not np.isnan(vel[i]) and speed_low <= vel[i] <= speed_high and not np.isnan(brake[i]) and brake_low <= brake[i] <= brake_high]
    _mean_std(idx_early_ms, 'early_ms', None)

    # Mid Corner MS
    throttle_lower = get_threshold(config["mid_corner_ms"], "throttle_lower_percent", "throttle_lower", max(throttle) if throttle else 1, mode)
    throttle_upper = get_threshold(config["mid_corner_ms"], "throttle_upper_percent", "throttle_upper", max(throttle) if throttle else 1, mode)
    idx_mid_ms = [i for i in range(len(throttle)) if throttle[i] is not None and not np.isnan(throttle[i]) and throttle_lower <= throttle[i] <= throttle_upper]
    _mean_std(idx_mid_ms, 'mid_ms', None)

    # Exit MS
    throttle_diff = np.diff(throttle)
    idx_exit_ms = [i + 1 for i, d in enumerate(throttle_diff) if not np.isnan(d) and d > 0.05]
    _mean_std(idx_exit_ms, 'exit_ms', None)

    # Early Entry HS
    speed_low = get_threshold(config["early_entry_hs"], "speed_lower_percent", "speed_lower", max_vel, mode)
    speed_high = get_threshold(config["early_entry_hs"], "speed_upper_percent", "speed_upper", max_vel, mode)
    brake_upper = get_threshold(config["early_entry_hs"], "brake_upper_percent", "brake_upper", max_brake, mode)
    throttle_low = get_threshold(config["early_entry_hs"], "throttle_lower_percent", "throttle_lower", max(throttle) if throttle else 1, mode)
    throttle_high = get_threshold(config["early_entry_hs"], "throttle_upper_percent", "throttle_upper", max(throttle) if throttle else 1, mode)
    idx_early_hs = [i for i in range(len(vel)) if not np.isnan(vel[i]) and speed_low <= vel[i] <= speed_high and not np.isnan(brake[i]) and brake[i] <= brake_upper and not np.isnan(throttle[i]) and throttle_low <= throttle[i] <= throttle_high]
    _mean_std(idx_early_hs, 'early_hs', None)

    # Mid Corner HS
    throttle_lower = get_threshold(config["mid_corner_hs"], "throttle_lower_percent", "throttle_lower", max(throttle) if throttle else 1, mode)
    throttle_upper = get_threshold(config["mid_corner_hs"], "throttle_upper_percent", "throttle_upper", max(throttle) if throttle else 1, mode)
    idx_mid_hs = [i for i in range(len(throttle)) if throttle[i] is not None and not np.isnan(throttle[i]) and throttle_lower <= throttle[i] <= throttle_upper]
    _mean_std(idx_mid_hs, 'mid_hs', None)

    # Exit HS
    throttle_diff_hs = np.diff(throttle)
    idx_exit_hs = [i + 1 for i, d in enumerate(throttle_diff_hs) if not np.isnan(d) and d < -0.05]
    _mean_std(idx_exit_hs, 'exit_hs', None)

    return stats

def main_multiple_archivos_plotly(archivos_json, variable_y):
    import json
    # Load config
    config_path = resource_path('config_conditions.json')
    with open(config_path, 'r') as f:
        config_all = json.load(f)
    mode = config_all.get('mode', 'percentage')
    if mode == 'absolute':
        config = config_all['absolute_conditions']
    else:
        config = config_all['percentage_conditions']
    # Fallback for missing 'straight_line_points'
    if "straight_line_points" not in config:
        config["straight_line_points"] = {"v60": 0.6, "v75": 0.75, "vEOS": 1.0, "tol": 0.02}

    fig = go.Figure()
    if not archivos_json or len(archivos_json) > 6:
        print("Debe proporcionar entre 1 y 6 archivos.")
        return go.Figure()
    colores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
            '#9467bd', '#8c564b']
    posiciones = {
        'v60': 5, 'v75': 12, 'vEOS': 20,
        'early_ls': 32, 'mid_ls': 40, 'exit_ls': 48,
        'early_ms': 60, 'mid_ms': 68, 'exit_ms': 76,
        'early_hs': 90, 'mid_hs': 98, 'exit_hs': 106,
    }
    for idx, archivo in enumerate(archivos_json):
        data = cargar_datos_json(archivo)
        nombre = os.path.basename(archivo)
        color = pick_color(nombre, idx)
        puntos_x = []
        puntos_y = []
        puntos_std = []
        labels = []
        # --- Straight line points: v60, v75, vEOS ---
        if "CarSpeed" in data['Datos']:
            var_x = "CarSpeed"
            var_ay = "Ay"
            x = [float(val) if es_numero(val) else np.nan for val in data['Datos'][var_x]]
            y = [float(val) if es_numero(val) else np.nan for val in data['Datos'][variable_y]]
            ay = [float(val) if es_numero(val) else np.nan for val in data['Datos'][var_ay]]
        elif "vCar" in data['Datos']:
            var_x = "vCar"
            var_ay = "gExtAccY"
            x = [float(val) * 3.6 if es_numero(val) else np.nan for val in data['Datos'][var_x]]
            y = [float(val) if es_numero(val) else np.nan for val in data['Datos'][variable_y]]
            ay = [float(val) if es_numero(val) else np.nan for val in data['Datos'][var_ay]]
        else:
            continue
        max_x = max(x)
        tol = config["straight_line_points"].get("tol", 0.02)
        tol_absolute = config["straight_line_points"].get("tol_absolute", 2)
        for key, pos, frac in zip(['v60', 'v75', 'vEOS'], [posiciones['v60'], posiciones['v75'], posiciones['vEOS']], [config["straight_line_points"]["v60"], config["straight_line_points"]["v75"], config["straight_line_points"]["vEOS"]]):
            if mode == 'absolute':
                objetivo = config["straight_line_points"][key]
                tolerance = tol_absolute 
            else:
                objetivo = max_x * frac
                tolerance = objetivo * tol
            indices = [i for i, val in enumerate(x) if abs(val - objetivo) <= tolerance and abs(ay[i]) < 0.05]
                # --- DEBUG PRINTS ---
            print(f"[DEBUG] {key} | mode: {mode}")
            print(f"  objetivo: {objetivo}, tolerance: {tolerance}")
            print(f"  Number of indices found: {len(indices)}")
            if indices:
                print(f"  Sample x values: {[x[i] for i in indices[:5]]}")
                print(f"  Sample ay values: {[ay[i] for i in indices[:5]]}")
                print(f"  Sample y values: {[y[i] for i in indices[:5]]}")
            else:
                print("  No indices found for this point.")
            # --- END DEBUG PRINTS ---
            valores_y = [y[i] for i in indices if i < len(y) and not np.isnan(y[i])]
            if valores_y:
                puntos_x.append(pos)
                puntos_y.append(np.mean(valores_y))
                puntos_std.append(np.std(valores_y))
                labels.append(key)
            else:
                puntos_x.append(pos)
                puntos_y.append(None)
                puntos_std.append(None)
                labels.append(key)

        # --- LS points: early_ls, mid_ls, exit_ls ---
        # early_ls
        pos_early_ls = 32
        pos_mid_ls = 40
        pos_exit_ls = 48

        # Early Entry LS logic (simplified, ported from Matplotlib)
        if "CarSpeed" in data['Datos']:
            var_vel = "CarSpeed"
            var_ax = "Ax"
            var_brake = "Brake_Press"
        else:
            var_vel = "vCar"
            var_ax = "gLong"
            var_brake = "Brake_Total"

        vel = [float(val) * 3.6 if var_vel == "vCar" and es_numero(val) else float(val) if es_numero(val) else np.nan for val in data['Datos'][var_vel]]
        ax_vals = [float(val) if es_numero(val) else np.nan for val in data['Datos'][var_ax]]
        brake = [float(val) if es_numero(val) else np.nan for val in data['Datos'][var_brake]]

        # Early LS: find indices with strong braking, low speed, and significant brake
        min_decel = min([v for v in ax_vals if not np.isnan(v) and v < 0], default=None)
        max_brake = max([v for v in brake if not np.isnan(v)], default=None)
        max_vel = max([v for v in vel if not np.isnan(v)], default=(None))
        # Use new config keys for early_entry_ls
        speed_low = get_threshold(config["early_entry_ls"], "speed_lower_percent", "speed_lower", max_vel, mode)
        speed_high = get_threshold(config["early_entry_ls"], "speed_upper_percent", "speed_upper", max_vel, mode)
        brake_low = get_threshold(config["early_entry_ls"], "brake_lower_percent", "brake_lower", max_brake, mode)
        brake_high = get_threshold(config["early_entry_ls"], "brake_upper_percent", "brake_upper", max_brake, mode)
        ax_low = get_threshold(config["early_entry_ls"], "ax_lower_percent", "ax_lower", abs(min_decel) if min_decel is not None else None, mode)
        ax_high = get_threshold(config["early_entry_ls"], "ax_upper_percent", "ax_upper", abs(min_decel) if min_decel is not None else None, mode)
        # Debug prints for early_entry_ls thresholds and values
        print(f"[DEBUG] early_entry_ls thresholds for {nombre}:")
        print(f"  speed_low: {speed_low}, speed_high: {speed_high}")
        print(f"  brake_low: {brake_low}, brake_high: {brake_high}")
        print(f"  ax_low: {ax_low}, ax_high: {ax_high}")
        # Check for None thresholds before filtering
        if None not in (speed_low, speed_high, brake_low, brake_high, ax_low, ax_high):
            indices_early_ls = [
                i for i in range(len(vel))
                if not np.isnan(vel[i]) and speed_low <= vel[i] <= speed_high
                and not np.isnan(brake[i]) and brake_low <= brake[i] <= brake_high
                and not np.isnan(ax_vals[i]) and -ax_high <= ax_vals[i] <= -ax_low
            ]
            print(f"[DEBUG] early_entry_ls: Found {len(indices_early_ls)} indices.")
            if indices_early_ls:
                print(f"[DEBUG] Sample values at first index:")
                i = indices_early_ls[0]
                print(f"  vel: {vel[i]}, brake: {brake[i]}, ax: {ax_vals[i]}")
            valores_y_early_ls = [y[i] for i in indices_early_ls if i < len(y) and not np.isnan(y[i])]
            if valores_y_early_ls:
                puntos_x.append(pos_early_ls)
                puntos_y.append(np.mean(valores_y_early_ls))
                puntos_std.append(np.std(valores_y_early_ls))
                labels.append('early_ls')
            else:
                puntos_x.append(pos_early_ls)
                puntos_y.append(None)
                puntos_std.append(None)
                labels.append('early_ls')
        else:
            print(f"[DEBUG] early_entry_ls: One or more thresholds are None.")
            puntos_x.append(pos_early_ls)
            puntos_y.append(None)
            puntos_std.append(None)
            labels.append('early_ls')
                
        # Mid Corner LS: throttle near zero
        if "CarSpeed" in data['Datos']:
            var_throttle = "Throttle"
        else:
            var_throttle = "rThrottlePedal"
        throttle = [float(val) if es_numero(val) else np.nan for val in data['Datos'].get(var_throttle, [np.nan]*len(y))]
        throttle_thresh = get_threshold(config["mid_corner_ls"], "throttle_threshold_percent", "throttle_threshold", max(throttle) if throttle else 1, mode)

        try:
            from app_dash import get_default_conditions
            defaults = get_default_conditions()
            # Use percentage_conditions/mid_corner_ls/throttle_threshold as default
            default_throttle_thresh = defaults["percentage_conditions"]["mid_corner_ls"]["throttle_threshold"]
        except Exception:
            default_throttle_thresh = 0.05  # fallback default

        if throttle_thresh is None:
            throttle_thresh = default_throttle_thresh

        indices_mid_ls = [
            i for i in range(len(throttle))
            if throttle[i] is not None and not np.isnan(throttle[i]) and abs(throttle[i]) < throttle_thresh
        ]
        valores_y_mid_ls = [y[i] for i in indices_mid_ls if i < len(y) and not np.isnan(y[i])]
        if valores_y_mid_ls:
            puntos_x.append(pos_mid_ls)
            puntos_y.append(np.mean(valores_y_mid_ls))
            puntos_std.append(np.std(valores_y_mid_ls))
            labels.append('mid_ls')
        else:
            puntos_x.append(pos_mid_ls)
            puntos_y.append(None)
            puntos_std.append(None)
            labels.append('mid_ls')
        # Exit LS: throttle increasing
        throttle_diff = np.diff(throttle)
        indices_exit_ls = [i+1 for i, d in enumerate(throttle_diff) if not np.isnan(d) and d > 0.05]
        valores_y_exit_ls = [y[i] for i in indices_exit_ls if i < len(y) and not np.isnan(y[i])]
        if valores_y_exit_ls:
            puntos_x.append(pos_exit_ls)
            puntos_y.append(np.mean(valores_y_exit_ls))
            puntos_std.append(np.std(valores_y_exit_ls))
            labels.append('exit_ls')
        else:
            puntos_x.append(pos_exit_ls)
            puntos_y.append(None)
            puntos_std.append(None)
            labels.append('exit_ls')
        # --- MS points: early_ms, mid_ms, exit_ms ---
        pos_early_ms = posiciones['early_ms']
        pos_mid_ms = posiciones['mid_ms']
        pos_exit_ms = posiciones['exit_ms']
        # Early Entry MS (example: speed between 50-70% max, moderate brake)
        speed_low = get_threshold(config["early_entry_ms"], "speed_lower_percent", "speed_lower", max_vel, mode)
        speed_high = get_threshold(config["early_entry_ms"], "speed_upper_percent", "speed_upper", max_vel, mode)
        brake_low = get_threshold(config["early_entry_ms"], "brake_lower_percent", "brake_lower", max_brake, mode)
        brake_high = get_threshold(config["early_entry_ms"], "brake_upper_percent", "brake_upper", max_brake, mode)
        indices_early_ms = [i for i in range(len(vel)) if not np.isnan(vel[i]) and speed_low <= vel[i] <= speed_high and not np.isnan(brake[i]) and brake_low <= brake[i] <= brake_high]
        valores_y_early_ms = [y[i] for i in indices_early_ms if i < len(y) and not np.isnan(y[i])]
        if valores_y_early_ms:
            puntos_x.append(pos_early_ms)
            puntos_y.append(np.mean(valores_y_early_ms))
            puntos_std.append(np.std(valores_y_early_ms))
            labels.append('early_ms')
        else:
            puntos_x.append(pos_early_ms)
            puntos_y.append(None)
            puntos_std.append(None)
            labels.append('early_ms')
        # Mid Corner MS (example: throttle between 2-20%)
        throttle_lower = get_threshold(config["mid_corner_ms"], "throttle_lower_percent", "throttle_lower", max(throttle) if throttle else 1, mode)
        throttle_upper = get_threshold(config["mid_corner_ms"], "throttle_upper_percent", "throttle_upper", max(throttle) if throttle else 1, mode)
        indices_mid_ms = [i for i in range(len(throttle)) if throttle[i] is not None and not np.isnan(throttle[i]) and throttle_lower <= throttle[i] <= throttle_upper]
        valores_y_mid_ms = [y[i] for i in indices_mid_ms if i < len(y) and not np.isnan(y[i])]
        if valores_y_mid_ms:
            puntos_x.append(pos_mid_ms)
            puntos_y.append(np.mean(valores_y_mid_ms))
            puntos_std.append(np.std(valores_y_mid_ms))
            labels.append('mid_ms')
        else:
            puntos_x.append(pos_mid_ms)
            puntos_y.append(None)
            puntos_std.append(None)
            labels.append('mid_ms')
        # Exit MS: throttle increasing, similar to exit_ls
        throttle_diff = np.diff(throttle)
        indices_exit_ms = [i+1 for i, d in enumerate(throttle_diff) if not np.isnan(d) and d > 0.05]
        valores_y_exit_ms = [y[i] for i in indices_exit_ms if i < len(y) and not np.isnan(y[i])]
        if valores_y_exit_ms:
            puntos_x.append(pos_exit_ms)
            puntos_y.append(np.mean(valores_y_exit_ms))
            puntos_std.append(np.std(valores_y_exit_ms))
            labels.append('exit_ms')
        else:
            puntos_x.append(pos_exit_ms)
            puntos_y.append(None)
            puntos_std.append(None)
            labels.append('exit_ms')
        # --- HS points: early_hs, mid_hs, exit_hs ---
        pos_early_hs = posiciones['early_hs']
        pos_mid_hs = posiciones['mid_hs']
        pos_exit_hs = posiciones['exit_hs']
        # Early Entry HS (example: speed 60-80% max, low brake, high throttle)
        speed_low_hs = get_threshold(config["early_entry_hs"], "speed_lower_percent", "speed_lower", max_vel, mode)
        speed_high_hs = get_threshold(config["early_entry_hs"], "speed_upper_percent", "speed_upper", max_vel, mode)
        brake_upper_hs = get_threshold(config["early_entry_hs"], "brake_upper_percent", "brake_upper", max_brake, mode)
        throttle_low_hs = get_threshold(config["early_entry_hs"], "throttle_lower_percent", "throttle_lower", max(throttle) if throttle else 1, mode)
        throttle_high_hs = get_threshold(config["early_entry_hs"], "throttle_upper_percent", "throttle_upper", max(throttle) if throttle else 1, mode)
        indices_early_hs = [i for i in range(len(vel)) if not np.isnan(vel[i]) and speed_low_hs <= vel[i] <= speed_high_hs and not np.isnan(brake[i]) and brake[i] <= brake_upper_hs and not np.isnan(throttle[i]) and throttle_low_hs <= throttle[i] <= throttle_high_hs]
        valores_y_early_hs = [y[i] for i in indices_early_hs if i < len(y) and not np.isnan(y[i])]
        if valores_y_early_hs:
            puntos_x.append(pos_early_hs)
            puntos_y.append(np.mean(valores_y_early_hs))
            puntos_std.append(np.std(valores_y_early_hs))
            labels.append('early_hs')
        else:
            puntos_x.append(pos_early_hs)
            puntos_y.append(None)
            puntos_std.append(None)
            labels.append('early_hs')
        # Mid Corner HS (example: throttle > 50%)
        throttle_lower_hs = get_threshold(config["mid_corner_hs"], "throttle_lower_percent", "throttle_lower", max(throttle) if throttle else 1, mode)
        throttle_upper_hs = get_threshold(config["mid_corner_hs"], "throttle_upper_percent", "throttle_upper", max(throttle) if throttle else 1, mode)
        indices_mid_hs = [i for i in range(len(throttle)) if throttle[i] is not None and not np.isnan(throttle[i]) and throttle_lower_hs <= throttle[i] <= throttle_upper_hs]
        valores_y_mid_hs = [y[i] for i in indices_mid_hs if i < len(y) and not np.isnan(y[i])]
        if valores_y_mid_hs:
            puntos_x.append(pos_mid_hs)
            puntos_y.append(np.mean(valores_y_mid_hs))
            puntos_std.append(np.std(valores_y_mid_hs))
            labels.append('mid_hs')
        else:
            puntos_x.append(pos_mid_hs)
            puntos_y.append(None)
            puntos_std.append(None)
            labels.append('mid_hs')
        # Exit HS: throttle decreasing, similar to exit_ls
        throttle_diff_hs = np.diff(throttle)
        indices_exit_hs = [i+1 for i, d in enumerate(throttle_diff_hs) if not np.isnan(d) and d < -0.05]
        valores_y_exit_hs = [y[i] for i in indices_exit_hs if i < len(y) and not np.isnan(y[i])]
        if valores_y_exit_hs:
            puntos_x.append(pos_exit_hs)
            puntos_y.append(np.mean(valores_y_exit_hs))
            puntos_std.append(np.std(valores_y_exit_hs))
            labels.append('exit_hs')
        else:
            puntos_x.append(pos_exit_hs)
            puntos_y.append(None)
            puntos_std.append(None)
            labels.append('exit_hs')
        # Plot all points
        # Group points by phase for connecting lines
        group_indices = [
            [0, 1, 2],      # Straight line: v60, v75, vEOS
            [3, 4, 5],      # LS: early_ls, mid_ls, exit_ls
            [6, 7, 8],      # MS: early_ms, mid_ms, exit_ms
            [9, 10, 11],    # HS: early_hs, mid_hs, exit_hs
        ]
        for group in group_indices:
            xs = [puntos_x[i] for i in group if i < len(puntos_x)]
            ys = [puntos_y[i] for i in group if i < len(puntos_y)]
            # Connect if at least 2 valid points in the group
            valid = [(x, y) for x, y in zip(xs, ys) if y is not None]
            if len(valid) >= 2:
                xs_valid, ys_valid = zip(*valid)
                fig.add_trace(go.Scatter(
                    x=xs_valid,
                    y=ys_valid,
                    mode='lines',
                    line=dict(color=color, width=2, dash='solid'),
                    showlegend=False,
                    hoverinfo='skip',
                    marker=dict(opacity=0)
                ))
        # Add the markers and labels for all points
        fig.add_trace(go.Scatter(
            x=puntos_x,
            y=puntos_y,
            mode='markers+text',
            name=nombre,
            marker=dict(color=color, size=12, line=dict(width=2, color='black')),
            text=labels,
            textposition='top center',
            error_y=dict(
                type='data',
                array=puntos_std,
                visible=True
            )
        ))
    fig.update_layout(
        title=f"12-Point Plot for {variable_y}",
        xaxis=dict(
            tickvals=[posiciones[k] for k in ['v60','v75','vEOS','early_ls','mid_ls','exit_ls','early_ms','mid_ms','exit_ms','early_hs','mid_hs','exit_hs']],
            ticktext=['0.60 Vmax', '0.75 Vmax', 'vEOS', 'Early Entry LS', 'Mid Corner LS', 'Exit LS', 'Early Entry MS', 'Mid Corner MS', 'Exit MS', 'Early Entry HS', 'Mid Corner HS', 'Exit HS'],
            title="Phase"
        ),
        yaxis_title=variable_y,
        hovermode='closest'
    )
    return fig

def main_comparacion_plotly(archivos_json_wintax, archivos_json_canopy, variable_y_wintax, variable_y_canopy):
    import json
    # Load config
    config_path = resource_path('config_conditions.json')
    with open(config_path, 'r') as f:
        config_all = json.load(f)
    mode = config_all.get('mode', 'percentage')
    if mode == 'absolute':
        config = config_all['absolute_conditions']
    else:
        config = config_all['percentage_conditions']
    if "straight_line_points" not in config:
        config["straight_line_points"] = {"v60": 0.6, "v75": 0.75, "vEOS": 1.0, "tol": 0.02}
    fig = go.Figure()
    colores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    posiciones = {'v60': 5, 'v75': 12, 'vEOS': 20, 'early_ls': 32, 'mid_ls': 40, 'exit_ls': 48, 'early_ms': 60, 'mid_ms': 68, 'exit_ms': 76, 'early_hs': 90, 'mid_hs': 98, 'exit_hs': 106}
    # Plot WinTAX files
    for idx, archivo in enumerate(archivos_json_wintax):
        data = cargar_datos_json(archivo)
        nombre = os.path.basename(archivo) + ' (WinTAX)'
        color = pick_color(nombre, idx)
        puntos_x = []
        puntos_y = []
        puntos_std = []
        labels = []
        if "CarSpeed" in data['Datos']:
            var_x = "CarSpeed"
            var_ay = "Ay"
            x = [float(val) if es_numero(val) else np.nan for val in data['Datos'][var_x]]
            y = [float(val) if es_numero(val) else np.nan for val in data['Datos'][variable_y_wintax]]
            ay = [float(val) if es_numero(val) else np.nan for val in data['Datos'][var_ay]]
            max_x = max(x)
            tol = config["straight_line_points"].get("tol", 0.02)
            tol_absolute = config["straight_line_points"].get("tol_absolute", 2)
            for key, pos, frac in zip(['v60', 'v75', 'vEOS'], [posiciones['v60'], posiciones['v75'], posiciones['vEOS']], [config["straight_line_points"]["v60"], config["straight_line_points"]["v75"], config["straight_line_points"]["vEOS"] ]):
                if mode == 'absolute':
                    objetivo = config["straight_line_points"][key]
                    tolerance = tol_absolute
                else:
                    objetivo = max_x * frac
                    tolerance = objetivo * tol
                indices = [i for i, val in enumerate(x) if abs(val - objetivo) <= tolerance and abs(ay[i]) < 0.05]
                valores_y = [y[i] for i in indices if i < len(y) and not np.isnan(y[i])]
                if valores_y:
                    puntos_x.append(pos)
                    puntos_y.append(np.mean(valores_y))
                    puntos_std.append(np.std(valores_y))
                    labels.append(key)
                else:
                    puntos_x.append(pos)
                    puntos_y.append(None)
                    puntos_std.append(None)
                    labels.append(key)
            # --- LS points: early_ls, mid_ls, exit_ls ---
            pos_early_ls = posiciones['early_ls']
            pos_mid_ls = posiciones['mid_ls']
            pos_exit_ls = posiciones['exit_ls']
            # Use flexible variable selection and new threshold logic for early_ls
            if "CarSpeed" in data['Datos']:
                var_vel = "CarSpeed"
                var_ax = "Ax"
                var_brake = "Brake_Press"
                vel = [float(val) if es_numero(val) else np.nan for val in data['Datos'][var_vel]]
            else:
                var_vel = "vCar"
                var_ax = "gLong"
                var_brake = "Brake_Total"
                vel = [float(val) * 3.6 if es_numero(val) else np.nan for val in data['Datos'][var_vel]]
            ax_vals = [float(val) if es_numero(val) else np.nan for val in data['Datos'][var_ax]]
            brake = [float(val) if es_numero(val) else np.nan for val in data['Datos'][var_brake]]
            min_decel = min([v for v in ax_vals if not np.isnan(v) and v < 0], default=None)
            max_brake = max([v for v in brake if not np.isnan(v)], default=None)
            max_vel = max([v for v in vel if not np.isnan(v)], default=None)
            # Use new config keys for early_entry_ls
            speed_low = get_threshold(config["early_entry_ls"], "speed_lower_percent", "speed_lower", max_vel, mode)
            speed_high = get_threshold(config["early_entry_ls"], "speed_upper_percent", "speed_upper", max_vel, mode)
            brake_low = get_threshold(config["early_entry_ls"], "brake_lower_percent", "brake_lower", max_brake, mode)
            brake_high = get_threshold(config["early_entry_ls"], "brake_upper_percent", "brake_upper", max_brake, mode)
            ax_low = get_threshold(config["early_entry_ls"], "ax_lower_percent", "ax_lower", abs(min_decel) if min_decel is not None else None, mode)
            ax_high = get_threshold(config["early_entry_ls"], "ax_upper_percent", "ax_upper", abs(min_decel) if min_decel is not None else None, mode)
            # Debug prints for early_entry_ls thresholds and values
            print(f"[DEBUG] early_entry_ls thresholds for {nombre}:")
            print(f"  speed_low: {speed_low}, speed_high: {speed_high}")
            print(f"  brake_low: {brake_low}, brake_high: {brake_high}")
            print(f"  ax_low: {ax_low}, ax_high: {ax_high}")
            # Check for None thresholds before filtering
            if None not in (speed_low, speed_high, brake_low, brake_high, ax_low, ax_high):
                indices_early_ls = [
                    i for i in range(len(vel))
                    if not np.isnan(vel[i]) and speed_low <= vel[i] <= speed_high
                    and not np.isnan(brake[i]) and brake_low <= brake[i] <= brake_high
                    and not np.isnan(ax_vals[i]) and -ax_high <= ax_vals[i] <= -ax_low
                ]
                print(f"[DEBUG] early_entry_ls: Found {len(indices_early_ls)} indices.")
                if indices_early_ls:
                    print(f"[DEBUG] Sample values at first index:")
                    i = indices_early_ls[0]
                    print(f"  vel: {vel[i]}, brake: {brake[i]}, ax: {ax_vals[i]}")
                valores_y_early_ls = [y[i] for i in indices_early_ls if i < len(y) and not np.isnan(y[i])]
                if valores_y_early_ls:
                    puntos_x.append(pos_early_ls)
                    puntos_y.append(np.mean(valores_y_early_ls))
                    puntos_std.append(np.std(valores_y_early_ls))
                    labels.append('early_ls')
                else:
                    puntos_x.append(pos_early_ls)
                    puntos_y.append(None)
                    puntos_std.append(None)
                    labels.append('early_ls')
            else:
                print(f"[DEBUG] early_entry_ls: One or more thresholds are None.")
                puntos_x.append(pos_early_ls)
                puntos_y.append(None)
                puntos_std.append(None)
                labels.append('early_ls')
            var_throttle = "Throttle"
            throttle = [float(val) if es_numero(val) else np.nan for val in data['Datos'].get(var_throttle, [np.nan]*len(y))]
            throttle_thresh = get_threshold(config["mid_corner_ls"], "throttle_threshold_percent", "throttle_threshold", max(throttle) if throttle else 1, mode)

            try:
                from app_dash import get_default_conditions
                defaults = get_default_conditions()
                # Use percentage_conditions/mid_corner_ls/throttle_threshold as default
                default_throttle_thresh = defaults["percentage_conditions"]["mid_corner_ls"]["throttle_threshold"]
            except Exception:
                default_throttle_thresh = 0.05  # fallback default

            if throttle_thresh is None:
                throttle_thresh = default_throttle_thresh

            indices_mid_ls = [
                i for i in range(len(throttle))
                if throttle[i] is not None and not np.isnan(throttle[i]) and abs(throttle[i]) < throttle_thresh
            ]
            valores_y_mid_ls = [y[i] for i in indices_mid_ls if i < len(y) and not np.isnan(y[i])]
            if valores_y_mid_ls:
                puntos_x.append(pos_mid_ls)
                puntos_y.append(np.mean(valores_y_mid_ls))
                puntos_std.append(np.std(valores_y_mid_ls))
                labels.append('mid_ls')
            else:
                puntos_x.append(pos_mid_ls)
                puntos_y.append(None)
                puntos_std.append(None)
                labels.append('mid_ls')
            throttle_diff = np.diff(throttle)
            indices_exit_ls = [i+1 for i, d in enumerate(throttle_diff) if not np.isnan(d) and d > 0.05]
            valores_y_exit_ls = [y[i] for i in indices_exit_ls if i < len(y) and not np.isnan(y[i])]
            if valores_y_exit_ls:
                puntos_x.append(pos_exit_ls)
                puntos_y.append(np.mean(valores_y_exit_ls))
                puntos_std.append(np.std(valores_y_exit_ls))
                labels.append('exit_ls')
            else:
                puntos_x.append(pos_exit_ls)
                puntos_y.append(None)
                puntos_std.append(None)
                labels.append('exit_ls')
            # --- MS points: early_ms, mid_ms, exit_ms ---
            pos_early_ms = posiciones['early_ms']
            pos_mid_ms = posiciones['mid_ms']
            pos_exit_ms = posiciones['exit_ms']
            # Early Entry MS (example: speed between 50-70% max, moderate brake)
            speed_low = get_threshold(config["early_entry_ms"], "speed_lower_percent", "speed_lower", max_vel, mode)
            speed_high = get_threshold(config["early_entry_ms"], "speed_upper_percent", "speed_upper", max_vel, mode)
            brake_low = get_threshold(config["early_entry_ms"], "brake_lower_percent", "brake_lower", max_brake, mode)
            brake_high = get_threshold(config["early_entry_ms"], "brake_upper_percent", "brake_upper", max_brake, mode)
            indices_early_ms = [i for i in range(len(vel)) if not np.isnan(vel[i]) and speed_low <= vel[i] <= speed_high and not np.isnan(brake[i]) and brake_low <= brake[i] <= brake_high]
            valores_y_early_ms = [y[i] for i in indices_early_ms if i < len(y) and not np.isnan(y[i])]
            if valores_y_early_ms:
                puntos_x.append(pos_early_ms)
                puntos_y.append(np.mean(valores_y_early_ms))
                puntos_std.append(np.std(valores_y_early_ms))
                labels.append('early_ms')
            else:
                puntos_x.append(pos_early_ms)
                puntos_y.append(None)
                puntos_std.append(None)
                labels.append('early_ms')
            # Mid Corner MS (example: throttle between 2-20%)
            throttle_lower = get_threshold(config["mid_corner_ms"], "throttle_lower_percent", "throttle_lower", max(throttle) if throttle else 1, mode)
            throttle_upper = get_threshold(config["mid_corner_ms"], "throttle_upper_percent", "throttle_upper", max(throttle) if throttle else 1, mode)
            indices_mid_ms = [i for i in range(len(throttle)) if throttle[i] is not None and not np.isnan(throttle[i]) and throttle_lower <= throttle[i] <= throttle_upper]
            valores_y_mid_ms = [y[i] for i in indices_mid_ms if i < len(y) and not np.isnan(y[i])]
            if valores_y_mid_ms:
                puntos_x.append(pos_mid_ms)
                puntos_y.append(np.mean(valores_y_mid_ms))
                puntos_std.append(np.std(valores_y_mid_ms))
                labels.append('mid_ms')
            else:
                puntos_x.append(pos_mid_ms)
                puntos_y.append(None)
                puntos_std.append(None)
                labels.append('mid_ms')
            # Exit MS (example: throttle increasing, similar to exit_ls)
            throttle_diff = np.diff(throttle)
            indices_exit_ms = [i+1 for i, d in enumerate(throttle_diff) if not np.isnan(d) and d > 0.05]
            valores_y_exit_ms = [y[i] for i in indices_exit_ms if i < len(y) and not np.isnan(y[i])]
            if valores_y_exit_ms:
                puntos_x.append(pos_exit_ms)
                puntos_y.append(np.mean(valores_y_exit_ms))
                puntos_std.append(np.std(valores_y_exit_ms))
                labels.append('exit_ms')
            else:
                puntos_x.append(pos_exit_ms)
                puntos_y.append(None)
                puntos_std.append(None)
                labels.append('exit_ms')
            # --- HS points: early_hs, mid_hs, exit_hs ---
            pos_early_hs = posiciones['early_hs']
            pos_mid_hs = posiciones['mid_hs']
            pos_exit_hs = posiciones['exit_hs']
            # Early Entry HS (example: speed 60-80% max, low brake, high throttle)
            speed_low_hs = get_threshold(config["early_entry_hs"], "speed_lower_percent", "speed_lower", max_vel, mode)
            speed_high_hs = get_threshold(config["early_entry_hs"], "speed_upper_percent", "speed_upper", max_vel, mode)
            brake_upper_hs = get_threshold(config["early_entry_hs"], "brake_upper_percent", "brake_upper", max_brake, mode)
            throttle_low_hs = get_threshold(config["early_entry_hs"], "throttle_lower_percent", "throttle_lower", max(throttle) if throttle else 1, mode)
            throttle_high_hs = get_threshold(config["early_entry_hs"], "throttle_upper_percent", "throttle_upper", max(throttle) if throttle else 1, mode)
            indices_early_hs = [i for i in range(len(vel)) if not np.isnan(vel[i]) and speed_low_hs <= vel[i] <= speed_high_hs and not np.isnan(brake[i]) and brake[i] <= brake_upper_hs and not np.isnan(throttle[i]) and throttle_low_hs <= throttle[i] <= throttle_high_hs]
            valores_y_early_hs = [y[i] for i in indices_early_hs if i < len(y) and not np.isnan(y[i])]
            if valores_y_early_hs:
                puntos_x.append(pos_early_hs)
                puntos_y.append(np.mean(valores_y_early_hs))
                puntos_std.append(np.std(valores_y_early_hs))
                labels.append('early_hs')
            else:
                puntos_x.append(pos_early_hs)
                puntos_y.append(None)
                puntos_std.append(None)
                labels.append('early_hs')
            # Mid Corner HS (example: throttle > 50%)
            throttle_lower_hs = get_threshold(config["mid_corner_hs"], "throttle_lower_percent", "throttle_lower", max(throttle) if throttle else 1, mode)
            throttle_upper_hs = get_threshold(config["mid_corner_hs"], "throttle_upper_percent", "throttle_upper", max(throttle) if throttle else 1, mode)
            indices_mid_hs = [i for i in range(len(throttle)) if throttle[i] is not None and not np.isnan(throttle[i]) and throttle_lower_hs <= throttle[i] <= throttle_upper_hs]
            valores_y_mid_hs = [y[i] for i in indices_mid_hs if i < len(y) and not np.isnan(y[i])]
            if valores_y_mid_hs:
                puntos_x.append(pos_mid_hs)
                puntos_y.append(np.mean(valores_y_mid_hs))
                puntos_std.append(np.std(valores_y_mid_hs))
                labels.append('mid_hs')
            else:
                puntos_x.append(pos_mid_hs)
                puntos_y.append(None)
                puntos_std.append(None)
                labels.append('mid_hs')
            # Exit HS (example: throttle decreasing, similar to exit_ls)
            throttle_diff_hs = np.diff(throttle)
            indices_exit_hs = [i+1 for i, d in enumerate(throttle_diff_hs) if not np.isnan(d) and d < -0.05]
            valores_y_exit_hs = [y[i] for i in indices_exit_hs if i < len(y) and not np.isnan(y[i])]
            if valores_y_exit_hs:
                puntos_x.append(pos_exit_hs)
                puntos_y.append(np.mean(valores_y_exit_hs))
                puntos_std.append(np.std(valores_y_exit_hs))
                labels.append('exit_hs')
            else:
                puntos_x.append(pos_exit_hs)
                puntos_y.append(None)
                puntos_std.append(None)
                labels.append('exit_hs')
        # Plot all points
        # Group points by phase for connecting lines
        group_indices = [
            [0, 1, 2],      # Straight line: v60, v75, vEOS
            [3, 4, 5],      # LS: early_ls, mid_ls, exit_ls
            [6, 7, 8],      # MS: early_ms, mid_ms, exit_ms
            [9, 10, 11],    # HS: early_hs, mid_hs, exit_hs
        ]
        for group in group_indices:
            xs = [puntos_x[i] for i in group if i < len(puntos_x)]
            ys = [puntos_y[i] for i in group if i < len(puntos_y)]
            # Connect if at least 2 valid points in the group
            valid = [(x, y) for x, y in zip(xs, ys) if y is not None]
            if len(valid) >= 2:
                xs_valid, ys_valid = zip(*valid)
                fig.add_trace(go.Scatter(
                    x=xs_valid,
                    y=ys_valid,
                    mode='lines',
                    line=dict(color=color, width=2, dash='solid'),
                    showlegend=False,
                    hoverinfo='skip',
                    marker=dict(opacity=0)
                ))
        # Add the markers and labels for all points
        fig.add_trace(go.Scatter(
            x=puntos_x,
            y=puntos_y,
            mode='markers+text',
            name=nombre,
            marker=dict(color=color, size=12, line=dict(width=2, color='black')),
            text=labels,
            textposition='top center',
            error_y=dict(
                type='data',
                array=puntos_std,
                visible=True
            )
        ))
    # Plot Canopy files
    for idx, archivo in enumerate(archivos_json_canopy):
        data = cargar_datos_json(archivo)
        color = colores[(idx + len(archivos_json_wintax)) % len(colores)]
        nombre = os.path.basename(archivo) + ' (Canopy)'
        puntos_x = []
        puntos_y = []
        puntos_std = []
        labels = []
        if "vCar" in data['Datos']:
            var_x = "vCar"
            var_ay = "gExtAccY"
            x = [float(val) * 3.6 if es_numero(val) else np.nan for val in data['Datos'][var_x]]
            y = [float(val) if es_numero(val) else np.nan for val in data['Datos'][variable_y_canopy]]
            ay = [float(val) if es_numero(val) else np.nan for val in data['Datos'][var_ay]]
            max_x = max(x)
            tol = config["straight_line_points"].get("tol", 0.02)
            tol_absolute = config["straight_line_points"].get("tol_absolute", 2)
            for key, pos, frac in zip(['v60', 'v75', 'vEOS'], [posiciones['v60'], posiciones['v75'], posiciones['vEOS']], [config["straight_line_points"]["v60"], config["straight_line_points"]["v75"], config["straight_line_points"]["vEOS"]]):
                if mode == 'absolute':
                    objetivo = config["straight_line_points"][key]
                    tolerance = tol_absolute
                else:
                    objetivo = max_x * frac
                    tolerance = objetivo * tol
                indices = [i for i, val in enumerate(x) if abs(val - objetivo) <= tolerance and abs(ay[i]) < 0.05]
                valores_y = [y[i] for i in indices if i < len(y) and not np.isnan(y[i])]
                if valores_y:
                    puntos_x.append(pos)
                    puntos_y.append(np.mean(valores_y))
                    puntos_std.append(np.std(valores_y))
                    labels.append(key)
                else:
                    puntos_x.append(pos)
                    puntos_y.append(None)
                    puntos_std.append(None)
                    labels.append(key)
            # --- LS points: early_ls, mid_ls, exit_ls ---
            pos_early_ls = posiciones['early_ls']
            pos_mid_ls = posiciones['mid_ls']
            pos_exit_ls = posiciones['exit_ls']
            var_vel = "vCar"
            var_ax = "gLong"
            var_brake = "Brake_Total"
            vel = [float(val) * 3.6 if var_vel == "vCar" and es_numero(val) else float(val) if es_numero(val) else np.nan for val in data['Datos'][var_vel]]
            ax_vals = [float(val) if es_numero(val) else np.nan for val in data['Datos'][var_ax]]
            brake = [float(val) if es_numero(val) else np.nan for val in data['Datos'][var_brake]]
            # Use same threshold logic as WinTAX for Canopy
            min_decel = min([v for v in ax_vals if not np.isnan(v) and v < 0], default=None)
            max_brake = max([v for v in brake if not np.isnan(v)], default=None)
            max_vel = max([v for v in vel if not np.isnan(v)], default=None)
            speed_low = get_threshold(config["early_entry_ls"], "speed_lower_percent", "speed_lower", max_vel, mode)
            speed_high = get_threshold(config["early_entry_ls"], "speed_upper_percent", "speed_upper", max_vel, mode)
            brake_low = get_threshold(config["early_entry_ls"], "brake_lower_percent", "brake_lower", max_brake, mode)
            brake_high = get_threshold(config["early_entry_ls"], "brake_upper_percent", "brake_upper", max_brake, mode)
            ax_low = get_threshold(config["early_entry_ls"], "ax_lower_percent", "ax_lower", abs(min_decel) if min_decel is not None else None, mode)
            ax_high = get_threshold(config["early_entry_ls"], "ax_upper_percent", "ax_upper", abs(min_decel) if min_decel is not None else None, mode)
            # Debug prints for early_entry_ls thresholds and values
            print(f"[DEBUG] early_entry_ls thresholds for {nombre}:")
            print(f"  speed_low: {speed_low}, speed_high: {speed_high}")
            print(f"  brake_low: {brake_low}, brake_high: {brake_high}")
            print(f"  ax_low: {ax_low}, ax_high: {ax_high}")
            # Check for None thresholds before filtering
            if None not in (speed_low, speed_high, brake_low, brake_high, ax_low, ax_high):
                indices_early_ls = [
                    i for i in range(len(vel))
                    if not np.isnan(vel[i]) and speed_low <= vel[i] <= speed_high
                    and not np.isnan(brake[i]) and brake_low <= brake[i] <= brake_high
                    and not np.isnan(ax_vals[i]) and -ax_high <= ax_vals[i] <= -ax_low
                ]
                print(f"[DEBUG] early_entry_ls: Found {len(indices_early_ls)} indices.")
                if indices_early_ls:
                    print(f"[DEBUG] Sample values at first index:")
                    i = indices_early_ls[0]
                    print(f"  vel: {vel[i]}, brake: {brake[i]}, ax: {ax_vals[i]}")
                valores_y_early_ls = [y[i] for i in indices_early_ls if i < len(y) and not np.isnan(y[i])]
                if valores_y_early_ls:
                    puntos_x.append(pos_early_ls)
                    puntos_y.append(np.mean(valores_y_early_ls))
                    puntos_std.append(np.std(valores_y_early_ls))
                    labels.append('early_ls')
                else:
                    puntos_x.append(pos_early_ls)
                    puntos_y.append(None)
                    puntos_std.append(None)
                    labels.append('early_ls')
            else:
                print(f"[DEBUG] early_entry_ls: One or more thresholds are None.")
                puntos_x.append(pos_early_ls)
                puntos_y.append(None)
                puntos_std.append(None)
                labels.append('early_ls')
            var_throttle = "rThrottlePedal"
            throttle = [float(val) if es_numero(val) else np.nan for val in data['Datos'].get(var_throttle, [np.nan]*len(y))]
            throttle_thresh = get_threshold(config["mid_corner_ls"], "throttle_threshold_percent", "throttle_threshold", max(throttle) if throttle else 1, mode)

            try:
                from app_dash import get_default_conditions
                defaults = get_default_conditions()
                # Use percentage_conditions/mid_corner_ls/throttle_threshold as default
                default_throttle_thresh = defaults["percentage_conditions"]["mid_corner_ls"]["throttle_threshold"]
            except Exception:
                default_throttle_thresh = 0.05  # fallback default

            if throttle_thresh is None:
                throttle_thresh = default_throttle_thresh

            indices_mid_ls = [
                i for i in range(len(throttle))
                if throttle[i] is not None and not np.isnan(throttle[i]) and abs(throttle[i]) < throttle_thresh
            ]
            valores_y_mid_ls = [y[i] for i in indices_mid_ls if i < len(y) and not np.isnan(y[i])]
            if valores_y_mid_ls:
                puntos_x.append(pos_mid_ls)
                puntos_y.append(np.mean(valores_y_mid_ls))
                puntos_std.append(np.std(valores_y_mid_ls))
                labels.append('mid_ls')
            else:
                puntos_x.append(pos_mid_ls)
                puntos_y.append(None)
                puntos_std.append(None)
                labels.append('mid_ls')
            throttle_diff = np.diff(throttle)
            indices_exit_ls = [i+1 for i, d in enumerate(throttle_diff) if not np.isnan(d) and d > 0.05]
            valores_y_exit_ls = [y[i] for i in indices_exit_ls if i < len(y) and not np.isnan(y[i])]
            if valores_y_exit_ls:
                puntos_x.append(pos_exit_ls)
                puntos_y.append(np.mean(valores_y_exit_ls))
                puntos_std.append(np.std(valores_y_exit_ls))
                labels.append('exit_ls')
            else:
                puntos_x.append(pos_exit_ls)
                puntos_y.append(None)
                puntos_std.append(None)
                labels.append('exit_ls')
            # --- MS points: early_ms, mid_ms, exit_ms ---
            pos_early_ms = posiciones['early_ms']
            pos_mid_ms = posiciones['mid_ms']
            pos_exit_ms = posiciones['exit_ms']
            # Early Entry MS (example: speed between 50-70% max, moderate brake)
            speed_low = get_threshold(config["early_entry_ms"], "speed_lower_percent", "speed_lower", max_vel, mode)
            speed_high = get_threshold(config["early_entry_ms"], "speed_upper_percent", "speed_upper", max_vel, mode)
            brake_low = get_threshold(config["early_entry_ms"], "brake_lower_percent", "brake_lower", max_brake, mode)
            brake_high = get_threshold(config["early_entry_ms"], "brake_upper_percent", "brake_upper", max_brake, mode)
            indices_early_ms = [i for i in range(len(vel)) if not np.isnan(vel[i]) and speed_low <= vel[i] <= speed_high and not np.isnan(brake[i]) and brake_low <= brake[i] <= brake_high]
            valores_y_early_ms = [y[i] for i in indices_early_ms if i < len(y) and not np.isnan(y[i])]
            if valores_y_early_ms:
                puntos_x.append(pos_early_ms)
                puntos_y.append(np.mean(valores_y_early_ms))
                puntos_std.append(np.std(valores_y_early_ms))
                labels.append('early_ms')
            else:
                puntos_x.append(pos_early_ms)
                puntos_y.append(None)
                puntos_std.append(None)
                labels.append('early_ms')
            # Mid Corner MS (example: throttle between 2-20%)
            throttle_lower = get_threshold(config["mid_corner_ms"], "throttle_lower_percent", "throttle_lower", max(throttle) if throttle else 1, mode)
            throttle_upper = get_threshold(config["mid_corner_ms"], "throttle_upper_percent", "throttle_upper", max(throttle) if throttle else 1, mode)
            indices_mid_ms = [i for i in range(len(throttle)) if throttle[i] is not None and not np.isnan(throttle[i]) and throttle_lower <= throttle[i] <= throttle_upper]
            valores_y_mid_ms = [y[i] for i in indices_mid_ms if i < len(y) and not np.isnan(y[i])]
            if valores_y_mid_ms:
                puntos_x.append(pos_mid_ms)
                puntos_y.append(np.mean(valores_y_mid_ms))
                puntos_std.append(np.std(valores_y_mid_ms))
                labels.append('mid_ms')
            else:
                puntos_x.append(pos_mid_ms)
                puntos_y.append(None)
                puntos_std.append(None)
                labels.append('mid_ms')
            # Exit MS (example: throttle increasing, similar to exit_ls)
            throttle_diff = np.diff(throttle)
            indices_exit_ms = [i+1 for i, d in enumerate(throttle_diff) if not np.isnan(d) and d > 0.05]
            valores_y_exit_ms = [y[i] for i in indices_exit_ms if i < len(y) and not np.isnan(y[i])]
            if valores_y_exit_ms:
                puntos_x.append(pos_exit_ms)
                puntos_y.append(np.mean(valores_y_exit_ms))
                puntos_std.append(np.std(valores_y_exit_ms))
                labels.append('exit_ms')
            else:
                puntos_x.append(pos_exit_ms)
                puntos_y.append(None)
                puntos_std.append(None)
                labels.append('exit_ms')
            # --- HS points: early_hs, mid_hs, exit_hs ---
            pos_early_hs = posiciones['early_hs']
            pos_mid_hs = posiciones['mid_hs']
            pos_exit_hs = posiciones['exit_hs']
            # Early Entry HS (example: speed 60-80% max, low brake, high throttle)
            speed_low_hs = get_threshold(config["early_entry_hs"], "speed_lower_percent", "speed_lower", max_vel, mode)
            speed_high_hs = get_threshold(config["early_entry_hs"], "speed_upper_percent", "speed_upper", max_vel, mode)
            brake_upper_hs = get_threshold(config["early_entry_hs"], "brake_upper_percent", "brake_upper", max_brake, mode)
            throttle_low_hs = get_threshold(config["early_entry_hs"], "throttle_lower_percent", "throttle_lower", max(throttle) if throttle else 1, mode)
            throttle_high_hs = get_threshold(config["early_entry_hs"], "throttle_upper_percent", "throttle_upper", max(throttle) if throttle else 1, mode)
            indices_early_hs = [i for i in range(len(vel)) if not np.isnan(vel[i]) and speed_low_hs <= vel[i] <= speed_high_hs and not np.isnan(brake[i]) and brake[i] <= brake_upper_hs and not np.isnan(throttle[i]) and throttle_low_hs <= throttle[i] <= throttle_high_hs]
            valores_y_early_hs = [y[i] for i in indices_early_hs if i < len(y) and not np.isnan(y[i])]
            if valores_y_early_hs:
                puntos_x.append(pos_early_hs)
                puntos_y.append(np.mean(valores_y_early_hs))
                puntos_std.append(np.std(valores_y_early_hs))
                labels.append('early_hs')
            else:
                puntos_x.append(pos_early_hs)
                puntos_y.append(None)
                puntos_std.append(None)
                labels.append('early_hs')
            # Mid Corner HS (example: throttle > 50%)
            throttle_lower_hs = get_threshold(config["mid_corner_hs"], "throttle_lower_percent", "throttle_lower", max(throttle) if throttle else 1, mode)
            throttle_upper_hs = get_threshold(config["mid_corner_hs"], "throttle_upper_percent", "throttle_upper", max(throttle) if throttle else 1, mode)
            indices_mid_hs = [i for i in range(len(throttle)) if throttle[i] is not None and not np.isnan(throttle[i]) and throttle_lower_hs <= throttle[i] <= throttle_upper_hs]
            valores_y_mid_hs = [y[i] for i in indices_mid_hs if i < len(y) and not np.isnan(y[i])]
            if valores_y_mid_hs:
                puntos_x.append(pos_mid_hs)
                puntos_y.append(np.mean(valores_y_mid_hs))
                puntos_std.append(np.std(valores_y_mid_hs))
                labels.append('mid_hs')
            else:
                puntos_x.append(pos_mid_hs)
                puntos_y.append(None)
                puntos_std.append(None)
                labels.append('mid_hs')
            # Exit HS (example: throttle decreasing, similar to exit_ls)
            throttle_diff_hs = np.diff(throttle)
            indices_exit_hs = [i+1 for i, d in enumerate(throttle_diff_hs) if not np.isnan(d) and d < -0.05]
            valores_y_exit_hs = [y[i] for i in indices_exit_hs if i < len(y) and not np.isnan(y[i])]
            if valores_y_exit_hs:
                puntos_x.append(pos_exit_hs)
                puntos_y.append(np.mean(valores_y_exit_hs))
                puntos_std.append(np.std(valores_y_exit_hs))
                labels.append('exit_hs')
            else:
                puntos_x.append(pos_exit_hs)
                puntos_y.append(None)
                puntos_std.append(None)
                labels.append('exit_hs')
        # Plot all points
        # Group points by phase for connecting lines
        group_indices = [
            [0, 1, 2],      # Straight line: v60, v75, vEOS
            [3, 4, 5],      # LS: early_ls, mid_ls, exit_ls
            [6, 7, 8],      # MS: early_ms, mid_ms, exit_ms
            [9, 10, 11],    # HS: early_hs, mid_hs, exit_hs
        ]
        for group in group_indices:
            xs = [puntos_x[i] for i in group if i < len(puntos_x)]
            ys = [puntos_y[i] for i in group if i < len(puntos_y)]
            # Connect if at least 2 valid points in the group
            valid = [(x, y) for x, y in zip(xs, ys) if y is not None]
            if len(valid) >= 2:
                xs_valid, ys_valid = zip(*valid)
                fig.add_trace(go.Scatter(
                    x=xs_valid,
                    y=ys_valid,
                    mode='lines',
                    line=dict(color=color, width=2, dash='solid'),
                    showlegend=False,
                    hoverinfo='skip',
                    marker=dict(opacity=0)
                ))
        # Add the markers and labels for all points
        fig.add_trace(go.Scatter(
            x=puntos_x,
            y=puntos_y,
            mode='markers+text',
            name=nombre,
            marker=dict(color=color, size=12, line=dict(width=2, color='black')),
            text=labels,
            textposition='top center',
            error_y=dict(
                type='data',
                array=puntos_std,
                visible=True
            )
        ))
    fig.update_layout(
        title=f"12-Point Plot Comparison",
        xaxis=dict(
            tickvals=[posiciones['v60'], posiciones['v75'], posiciones['vEOS'], posiciones['early_ls'], posiciones['mid_ls'], posiciones['exit_ls'], posiciones['early_ms'], posiciones['mid_ms'], posiciones['exit_ms'], posiciones['early_hs'], posiciones['mid_hs'], posiciones['exit_hs']],
            ticktext=['0.60 Vmax', '0.75 Vmax', 'vEOS', 'Early Entry LS', 'Mid Corner LS', 'Exit LS', 'Early Entry MS', 'Mid Corner MS', 'Exit MS', 'Early Entry HS', 'Mid Corner HS', 'Exit HS'],
            title="Phase"
        ),
        yaxis_title="Variable seleccionada",
        hovermode='closest'
    )
    return fig

def matrix_detailed_analysis_plotly(archivos_json, variable_y):
    """Return a 3×3 matrix plot comparing setups without straight-line points."""

    filas = ['Low speed', 'Medium speed', 'High speed']
    columnas = ['Entry', 'Mid', 'Exit']
    label_map = {
        (0, 0): 'early_ls', (0, 1): 'mid_ls', (0, 2): 'exit_ls',
        (1, 0): 'early_ms', (1, 1): 'mid_ms', (1, 2): 'exit_ms',
        (2, 0): 'early_hs', (2, 1): 'mid_hs', (2, 2): 'exit_hs',
    }

    fig = make_subplots(
        rows=3,
        cols=3,
        shared_xaxes=False,
        shared_yaxes=False,
        horizontal_spacing=0.04,
        vertical_spacing=0.07,
        subplot_titles=[f"{r} – {c}" for r in filas for c in columnas],
    )

    for idx, archivo in enumerate(archivos_json):
        data = cargar_datos_json(archivo)
        nombre = os.path.basename(archivo).replace('_procesado.json', '')
        color = pick_color(nombre, idx)
        stats = extraer_stats_12pts(data, variable_y)

        for (r, c), lbl in label_map.items():
            mean, std = stats.get(lbl, (None, None))
            if mean is None:
                continue
            fig.add_trace(
                go.Scatter(
                    x=[idx],
                    y=[mean],
                    error_y=dict(type='data', array=[std], visible=True),
                    mode='markers',
                    marker=dict(color=color, size=14, line=dict(width=1)),
                    name=nombre if r == 0 and c == 0 else None,
                    showlegend=(r == 0 and c == 0),
                ),
                row=r + 1,
                col=c + 1,
            )

    fig.update_xaxes(showticklabels=False)
    fig.update_layout(
        title=f"Detailed Analysis – {variable_y}",
        height=900,
        width=1400,
        hovermode='closest',
        legend_title="Setup / Archivo",
    )
    return fig



def plot_dual_axis(wintax_jsons, canopy_jsons, y1, y2=None, y1_canopy=None, y2_canopy=None):
    fig = go.Figure()

    # WINTAX - Eje izquierdo
    if y1:
        for file in wintax_jsons:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            x = data['Datos'].get('Time') or data['Datos'].get('LapDistance')
            y = data['Datos'].get(y1)
            if x and y:
                fig.add_trace(go.Scatter(
                    x=x, y=y, name=f"{y1} (W)", yaxis='y1', mode='lines'
                ))

    # CANOPY - Eje izquierdo (equivalente a Y1)
    if y1_canopy:
        for file in canopy_jsons:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            x = data['Datos'].get('Time') or data['Datos'].get('LapDistance')
            y = data['Datos'].get(y1_canopy)
            if x and y:
                fig.add_trace(go.Scatter(
                    x=x, y=y, name=f"{y1_canopy} (C)", yaxis='y1', mode='lines'
                ))

    # WINTAX - Eje derecho (Y2)
    if y2:
        for file in wintax_jsons:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            x = data['Datos'].get('Time') or data['Datos'].get('LapDistance')
            y = data['Datos'].get(y2)
            if x and y:
                fig.add_trace(go.Scatter(
                    x=x, y=y, name=f"{y2} (W)", yaxis='y2', mode='lines'
                ))

    # CANOPY - Eje derecho (equivalente a Y2)
    if y2_canopy:
        for file in canopy_jsons:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            x = data['Datos'].get('Time') or data['Datos'].get('LapDistance')
            y = data['Datos'].get(y2_canopy)
            if x and y:
                fig.add_trace(go.Scatter(
                    x=x, y=y, name=f"{y2_canopy} (C)", yaxis='y2', mode='lines'
                ))

    # Layout con dos ejes Y
    fig.update_layout(
        title="Dual Axis Comparison",
        xaxis=dict(title='Time or Lap Distance'),
        yaxis=dict(title='Y1 Axis', side='left'),
        yaxis2=dict(title='Y2 Axis', overlaying='y', side='right'),
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=40, r=40, t=40, b=40)
    )

    return fig

