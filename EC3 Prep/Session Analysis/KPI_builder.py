# src/metrics.py
import pandas as pd
import numpy as np
from scipy.stats import linregress
from data_process import parse_time_to_seconds
import warnings
import re

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

def track_limit_rate(track_df: pd.DataFrame, laps_df: pd.DataFrame) -> pd.DataFrame:
    """Calcula la tasa de infracciones de track limits por vuelta.

    Parameters
    ----------
    track_df : pd.DataFrame
        DataFrame con incidencias de track limits. Debe contener una columna
        de piloto y una columna ``incident`` con el recuento de infracciones.
    laps_df : pd.DataFrame
        DataFrame con las vueltas completadas. Debe incluir columna de piloto y
        ``lap`` o ``lap_number``.

    Returns
    -------
    pd.DataFrame
        DataFrame con ``driver``, ``incidents``, ``laps`` y ``rate``.
    """

    track_driver = next(
        (c for c in ['driver', 'driver_name', 'driver_shortname', 'driver_number'] if c in track_df.columns),
        None,
    )
    if track_driver is None:
        raise KeyError('No se encontró columna de piloto en track_df')

    lap_driver = next(
        (c for c in ['driver', 'driver_name', 'driver_shortname', 'driver_number'] if c in laps_df.columns),
        None,
    )
    if lap_driver is None:
        raise KeyError('No se encontró columna de piloto en laps_df')

    lap_col = 'lap' if 'lap' in laps_df.columns else 'lap_number'
    if lap_col not in laps_df.columns:
        raise KeyError('No se encontró columna de vueltas en laps_df')

    if 'incident' not in track_df.columns:
        if 'total_infractions' in track_df.columns:
            track_df = track_df.rename(columns={'total_infractions': 'incident'})
        else:
            raise KeyError('No se encontró columna de incidentes en track_df')

    incidents = track_df.groupby(track_driver)['incident'].sum()

    laps = (
        laps_df.drop_duplicates(subset=[lap_driver, lap_col])
        .groupby(lap_driver)[lap_col]
        .count()
    )

    result = (
        pd.DataFrame({'driver': incidents.index, 'incidents': incidents.values})
        .merge(
            pd.DataFrame({'driver': laps.index, 'laps': laps.values}),
            on='driver',
            how='outer'
        )
        .fillna(0)
    )

    result['rate'] = result['incidents'] / result['laps'].replace(0, np.nan)
    result['rate'] = result['rate'].fillna(0)
    return result

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

def ideal_lap_gap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada piloto calcula:
      - ideal_time = suma de sus mejores tiempos de sector
      - best_lap   = mejor tiempo de vuelta real (lap_time)
      - ideal_gap  = best_lap - ideal_time
    Devuelve driver, team (si existe), ideal_time, best_lap, ideal_gap.
    """
    if 'driver' not in df.columns or 'lap_time' not in df.columns:
        raise KeyError("Faltan columnas 'driver' o 'lap_time'")
    
    # Convertir lap_time a segundos si hace falta
    df_num = df[['driver'] + (['team'] if 'team' in df.columns else []) + ['lap_time']].copy()
    if not np.issubdtype(df_num['lap_time'].dtype, np.number):
        df_num['lap_time'] = df_num['lap_time'].apply(parse_time_to_seconds)

    # Mejor vuelta real
    best = df_num.groupby(['driver'] + (['team'] if 'team' in df_num.columns else []))['lap_time']\
                 .min().reset_index(name='best_lap')
    
    # Mejor sector por piloto
    bst = best_sector_times(df)
    if bst.empty:
        return pd.DataFrame()
    # Sumar mejores sectores para ideal_time
    bst['ideal_time'] = bst[['sector1','sector2','sector3']].sum(axis=1)

    # Unir best_lap e ideal_time
    merged = pd.merge(best, bst, on=['driver'] + (['team'] if 'team' in bst.columns else []))
    merged['ideal_gap'] = merged['best_lap'] - merged['ideal_time']
    return merged[['driver'] + (['team'] if 'team' in merged.columns else [])
                  + ['ideal_time','best_lap','ideal_gap']]

def sector_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la media de cada sector (sector1, sector2, sector3) por piloto.
    Sólo toma las columnas *_seconds si existen, para evitar duplicados.
    Devuelve un DataFrame con columnas: driver, [team], sector1, sector2, sector3.
    """
    # 1) Buscamos primero las columnas *_seconds
    secs_pattern = re.compile(r'(?i)^(?:s|sector)([123])_seconds$')
    sector_secs = [c for c in df.columns if secs_pattern.match(c)]
    
    # 2) Si no hay *_seconds, buscamos las sin sufijo
    if sector_secs:
        sector_cols = sector_secs
    else:
        nosecs_pattern = re.compile(r'(?i)^(?:s|sector)([123])$')
        sector_cols = [c for c in df.columns if nosecs_pattern.match(c)]
    
    # Si no encontramos ni unas ni otras, devolvemos vacío
    if not sector_cols:
        return pd.DataFrame()
    
    # Mapeo de cada columna original a sector1/2/3
    rename_map = {}
    for col in sector_cols:
        m = re.match(r'(?i)^(?:s|sector)([123])', col)
        if m:
            rename_map[col] = f"sector{m.group(1)}"
    
    # Columnas que vamos a usar: driver, [team], y sólo las de rename_map
    base = ['driver'] + (['team'] if 'team' in df.columns else [])
    use_cols = base + list(rename_map.keys())
    df_sec = df[use_cols].copy()
    
    # Convertir a segundos si hace falta (parse_time_to_seconds aplica NaN si no puede)
    for orig in rename_map.keys():
        if not pd.api.types.is_numeric_dtype(df_sec[orig]):
            df_sec[orig] = df_sec[orig].apply(parse_time_to_seconds)
    
    # Renombramos a sector1, sector2, sector3
    df_sec.rename(columns=rename_map, inplace=True)
    
    # Agrupamos y hacemos la media
    group_cols = base
    result = (
        df_sec
        .groupby(group_cols)[[f"sector{n}" for n in ['1','2','3']]]
        .mean()
        .reset_index()
    )
    return result

def best_sector_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Best sector times per driver with rankings.

    The function detects columns named either ``sectorN`` or ``sN`` (case
    insensitive). For each sector it computes the minimum time by driver and
    adds a ranking column ``<sector>_rank`` where ``1`` represents the fastest
    driver.
    """

    if "driver" not in df.columns:
        raise KeyError("'driver' column not found")

    sector_cols = []
    for col in df.columns:
        if col.lower().startswith("sector") and col[6:].isdigit():
            sector_cols.append(col)
        elif re.match(r"(?i)^s\d+$", col):
            sector_cols.append(col)

    if not sector_cols:
        return pd.DataFrame()

    df_num = df[["driver", *sector_cols]].copy()
    for col in sector_cols:
        if not np.issubdtype(df_num[col].dtypes, np.number):
            df_num[col] = df_num[col].apply(parse_time_to_seconds)

    best = df_num.groupby("driver")[sector_cols].min().reset_index()

    for col in sector_cols:
        best[f"{col}_rank"] = best[col].rank(method="min", ascending=True).astype(int)

    return best

def best_sector_times(df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada piloto devuelve su mejor (mínimo) tiempo en cada sector.
    Solo toma las columnas *_seconds si existen; 
    en caso contrario, busca columnas s1, s2, s3.
    Columnas resultantes: driver, [team], sector1, sector2, sector3.
    """
    # 1) Detección de columnas con sufijo *_seconds
    secs_pattern = re.compile(r'(?i)^(?:s|sector)([123])_seconds$')
    orig_secs = [c for c in df.columns if secs_pattern.match(c)]
    
    # 2) Si no hay *_seconds, buscamos s1, s2, s3
    if orig_secs:
        sector_cols = orig_secs
    else:
        nosuffix = re.compile(r'(?i)^(?:s|sector)([123])$')
        sector_cols = [c for c in df.columns if nosuffix.match(c)]
    
    if not sector_cols:
        return pd.DataFrame()
    
    # 3) Mapeo a nombres estandarizados sector1/2/3
    rename_map = {
        col: f"sector{re.search(r'([123])', col).group(1)}"
        for col in sector_cols
    }
    
    # 4) Selección de columnas
    base = ['driver'] + (['team'] if 'team' in df.columns else [])
    use_cols = base + sector_cols
    df_sec = df[use_cols].copy()
    
    # 5) A segundos si fuese texto
    for orig in sector_cols:
        if not pd.api.types.is_numeric_dtype(df_sec[orig]):
            df_sec[orig] = df_sec[orig].apply(parse_time_to_seconds)
    
    # 6) Renombrado unívoco
    df_sec.rename(columns=rename_map, inplace=True)
    
    # 7) Agrupamos y tomamos mínimo → el sector más rápido
    group_cols = base
    best = (
        df_sec
        .groupby(group_cols)[list(rename_map.values())]
        .min()
        .reset_index()
    )
    return best

def lap_time_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve un DataFrame con columna de vuelta y lap_time para cada piloto.
    Columnas: driver, team (si existe), lap, lap_time.
    """
    # Detectar nombre de columna de vuelta
    lap_col = 'lap_number' if 'lap_number' in df.columns else 'lap'
    if lap_col not in df.columns or 'lap_time' not in df.columns:
        return pd.DataFrame()

    df_hist = df[[lap_col, 'driver'] + (['team'] if 'team' in df.columns else []) + ['lap_time']]\
        .dropna(subset=[lap_col, 'lap_time'])\
        .copy()
    # Asegurarnos de que lap es int y lap_time float
    df_hist[lap_col] = df_hist[lap_col].astype(int)
    df_hist['lap_time'] = df_hist['lap_time'].astype(float)
    return df_hist


