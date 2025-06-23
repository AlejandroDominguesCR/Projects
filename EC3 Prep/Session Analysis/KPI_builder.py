# src/metrics.py
import pandas as pd
import numpy as np
from scipy.stats import linregress
from data_process import parse_time_to_seconds

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
        grouped = (
            df.groupby([driver_col, team_col])['top_speed']
            .max()
            .reset_index()
        )
        result = grouped.rename(
            columns={driver_col: 'driver', team_col: 'team', 'top_speed': 'max_top_speed'}
        )
    else:
        grouped = df.groupby(driver_col)['top_speed'].max().reset_index()
        result = grouped.rename(
            columns={driver_col: 'driver', 'top_speed': 'max_top_speed'}
        )
        result['team'] = None
        result = result[['driver', 'team', 'max_top_speed']]

    result.sort_values('max_top_speed', ascending=False, inplace=True)
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

    laps = df['lap_time']
    if not np.issubdtype(laps.dtype, np.number):
        laps = laps.apply(parse_time_to_seconds)

    # Tiempo medio de referencia usando la columna detectada
    ref = laps[df[driver_col] == baseline].mean()

    # Diferencia respecto a la referencia
    df = df.copy()
    df['delta'] = laps - ref
    return df

def position_trace(df: pd.DataFrame) -> pd.DataFrame:
    """Evolución de posiciones vuelta a vuelta."""
    lap_col = "lap" if "lap" in df.columns else "lap_number"
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

    df_clean = (
        df.groupby([lap_col, driver_col], as_index=False)[pos_col]
        .last()
    )
    return df_clean.pivot(index=lap_col, columns=driver_col, values=pos_col)

def lap_time_histogram(df: pd.DataFrame, lap_start: int | None = None, lap_end: int | None = None) -> pd.DataFrame:
    """Return lap time data optionally limited to a lap range.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least ``lap``, ``lap_time`` and ``driver`` columns.
    lap_start : int, optional
        First lap of the range to keep. If ``None`` it defaults to the minimum lap
        in the dataframe.
    lap_end : int, optional
        Last lap of the range to keep. If ``None`` it defaults to the maximum lap
        in the dataframe.

    Returns
    -------
    pd.DataFrame
        DataFrame with the columns ``lap``, ``lap_time`` and ``driver`` filtered
        by the requested lap range.
    """
    lap_col = "lap" if "lap" in df.columns else "lap_number"

    required = {lap_col, "lap_time", "driver"}
    if not required.issubset(df.columns):
        missing = required.difference(df.columns)
        raise KeyError(f"Missing columns in dataframe: {', '.join(missing)}")

    out = df.loc[:, [lap_col, "lap_time", "driver"]].copy()
    if lap_start is not None or lap_end is not None:
        if lap_start is None:
            lap_start = int(out[lap_col].min())
        if lap_end is None:
            lap_end = int(out[lap_col].max())
        out = out[(out[lap_col] >= lap_start) & (out[lap_col] <= lap_end)]

    return out.dropna(subset=["lap_time"]).reset_index(drop=True)

def pace_delta(df: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Delta de lap_time vuelta a vuelta vs reference."""
    laps = df['lap_time']
    if not np.issubdtype(laps.dtype, np.number):
        laps = laps.apply(parse_time_to_seconds)

    ref = laps[df['driver'] == reference]
    ref = ref.set_axis(df.loc[df['driver'] == reference, 'lap'])

    df2 = df.set_index('lap').copy()
    df2['delta'] = laps.values - ref.reindex(df2.index).values
    return df2.reset_index()

def sector_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Ranking medio de tiempos por sector para cada piloto.

    La función calcula la media de cada sector para todos los pilotos y
    genera una columna adicional con el ranking por sector (1 = el más
    rápido). Las columnas de ranking se denominan ``<sector>_rank``.
    """

    sectors = [c for c in df.columns if c.startswith("sector") and not c.endswith("_rank")]
    if not sectors:
        return pd.DataFrame()

    result = df.groupby("driver")[sectors].mean().reset_index()

    for sec in sectors:
        rank_col = f"{sec}_rank"
        result[rank_col] = result[sec].rank(method="min", ascending=True).astype(int)

    return result

def gap_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Matriz de diferencias medias entre pilotos."""
    drivers = df['driver'].unique()
    mat = pd.DataFrame(index=drivers, columns=drivers, dtype=float)

    if not np.issubdtype(df['lap_time'].dtype, np.number):
        laps = df['lap_time'].apply(parse_time_to_seconds)
    else:
        laps = df['lap_time']

    mean_times = df.groupby('driver').apply(lambda d: laps.loc[d.index].mean())
    for i in drivers:
        for j in drivers:
            mat.loc[i,j] = mean_times[i] - mean_times[j]
    return mat

def climate_impact(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Join de lap_times con datos meteorológicos por timestamp y regresión."""
    merged = pd.merge_asof(df.sort_values('time'), weather.sort_values('time'), on='time')

    laps = merged['lap_time']
    if not np.issubdtype(laps.dtype, np.number):
        laps = laps.apply(parse_time_to_seconds)

    slope, intercept, r, p, stderr = linregress(merged['temperature'], laps)
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
    """Ranking de equipos por velocidad media."""
    tops = compute_top_speeds(df)
    if 'team' in tops.columns:
        result = (
            tops.groupby('team')['max_top_speed']
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        result = result.rename(columns={'max_top_speed': 'mean_top_speed'})
        return result
    return pd.DataFrame()
