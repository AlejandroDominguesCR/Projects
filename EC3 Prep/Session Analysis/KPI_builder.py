# src/metrics.py
import pandas as pd
import numpy as np
from scipy.stats import linregress

def detect_stints(laps: pd.DataFrame) -> pd.DataFrame:
    """Detecta secuencias de vueltas según reglas de stint (O, W, P)."""
    # TODO: implementar lógica específica del proveedor
    return laps

def compute_top_speeds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la velocidad máxima ('top_speed') por piloto.
    Además incorpora el equipo (si existe) y devuelve ordenado.
    Busca columnas 'driver_name', 'driver_shortname' o 'driver_number'.
    Y columnas de equipo: 'team', 'constructor' o 'team_name'.
    """
    # Detectar columna de piloto
    if 'driver_name' in df.columns:
        driver_col = 'driver_name'
    elif 'driver_shortname' in df.columns:
        driver_col = 'driver_shortname'
    elif 'driver_number' in df.columns:
        driver_col = 'driver_number'
    else:
        raise KeyError('No se encontró columna de piloto')
    # Detectar columna de equipo
    team_col = None
    for col in ['team', 'constructor', 'team_name']:
        if col in df.columns:
            team_col = col; break
    # Verificar columna de velocidad
    if 'top_speed' not in df.columns:
        raise KeyError('No se encontró columna top_speed')
    # Agrupar y calcular máxima velocidad y equipo asociado
    if team_col:
        grouped = df.groupby([driver_col, team_col])['top_speed'].max().reset_index()
        result = grouped.rename(columns={driver_col: 'driver', team_col: 'team', 'top_speed': 'max_top_speed'})
    else:
        grouped = df.groupby(driver_col)['top_speed'].max().reset_index()
        result = grouped.rename(columns={driver_col: 'driver', 'top_speed': 'max_top_speed'})
    # Ordenar de mayor a menor
    result = result.sort_values('max_top_speed', ascending=False).reset_index(drop=True)
    return result

def pace_comparison(df: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """Compara lap times contra piloto de referencia."""
    if 'lap_time' not in df.columns:
        raise KeyError('No se encontró columna lap_time')

    # Detectar columna de piloto (igual que compute_top_speeds)
    for col in ('driver_name', 'driver_shortname', 'driver_number'):
        if col in df.columns:
            driver_col = col
            break
    else:
        raise KeyError('No se encontró columna de piloto')

    # Tiempo medio de referencia usando la columna detectada
    ref = df[df[driver_col] == baseline]['lap_time'].mean()

    # Diferencia respecto a la referencia
    df['delta'] = df['lap_time'] - ref
    return df

def position_trace(df: pd.DataFrame) -> pd.DataFrame:
    """Evolución de posiciones vuelta a vuelta."""
    lap_col = 'lap'
    pos_col = 'pos'
    # Piloto: preferimos shortname o driver_name
    if 'driver_shortname' in df.columns:
        driver_col = 'driver_shortname'
    elif 'driver_name' in df.columns:
        driver_col = 'driver_name'
    elif 'driver_number' in df.columns:
        driver_col = 'driver_number'
    else:
        raise KeyError('No se encontró columna de piloto')
    return df.pivot(index=lap_col, columns=driver_col, values=pos_col)

def lap_time_histogram(df: pd.DataFrame, driver: str) -> pd.DataFrame:
    """Datos para histograma de lap times de un piloto."""
    series = df[df['driver'] == driver]['lap_time']
    return series.dropna()

def pace_delta(df: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Delta de lap_time vuelta a vuelta vs reference."""
    ref = df[df['driver'] == reference].set_index('lap')['lap_time']
    df2 = df.set_index('lap')
    df2['delta'] = df2['lap_time'] - ref
    return df2.reset_index()

def sector_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Media de sectores por piloto."""
    sectors = [c for c in df.columns if c.startswith('sector')]
    return df.groupby('driver')[sectors].mean().reset_index()

def gap_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Matriz de diferencias medias entre pilotos."""
    drivers = df['driver'].unique()
    mat = pd.DataFrame(index=drivers, columns=drivers, dtype=float)
    mean_times = df.groupby('driver')['lap_time'].mean()
    for i in drivers:
        for j in drivers:
            mat.loc[i,j] = mean_times[i] - mean_times[j]
    return mat

def climate_impact(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Join de lap_times con datos meteorológicos por timestamp y regresión."""
    merged = pd.merge_asof(df.sort_values('time'), weather.sort_values('time'), on='time')
    slope, intercept, r, p, stderr = linregress(merged['temperature'], merged['lap_time'])
    return {'slope': slope, 'r_value': r, 'data': merged}

def track_limits_incidents(df: pd.DataFrame) -> pd.DataFrame:
    """Cuenta de incidentes de track limits por piloto."""
    return df.groupby('driver')['incident'].sum().reset_index()

def top_speed_locations(df: pd.DataFrame) -> pd.DataFrame:
    """Lugar de cada top_speed por piloto (track position)."""
    idx = df.groupby('driver')['top_speed'].idxmax()
    return df.loc[idx, ['driver','track_pos','top_speed']]

def stint_boxplots(df: pd.DataFrame) -> pd.DataFrame:
    """Distribución de lap_time por stint y piloto."""
    return df[['driver','stint','lap_time']]

def team_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Ranking de equipos por suma de max_top_speed."""
    tops = compute_top_speeds(df)
    if 'team' in tops.columns:
        return tops.groupby('team')['max_top_speed'].sum().sort_values(ascending=False).reset_index()
    return pd.DataFrame()
