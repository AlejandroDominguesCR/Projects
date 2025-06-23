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

def sector_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Ranking medio de tiempos por sector para cada piloto.

    La función detecta automáticamente columnas de sector llamadas
    ``sector1``, ``s1`` o ``s1_seconds`` (lo mismo para sectores 2 y 3),
    ignorando mayúsculas/minúsculas. Estas columnas se convierten a
    segundos mediante :func:`parse_time_to_seconds` antes de calcular la
    media de cada sector. Además se generan columnas ``<sector>_rank`` con
    la posición de cada piloto en dicho sector (1 = el más rápido).
    """

    # Detectar columnas de sector (sector1, s1, s1_seconds, ...)
    sector_map: dict[str, str] = {}
    for col in df.columns:
        m = re.match(r"(?i)^(?:sector|s)([123])(?:_seconds)?$", col)
        if m and int(m.group(1)) in (1, 2, 3) and m.group(1) not in sector_map:
            sector_map[m.group(1)] = col

    if not sector_map:
        return pd.DataFrame()

    # Crear DataFrame con driver y columnas de sector normalizadas
    use_cols = ["driver"] + [sector_map[k] for k in sorted(sector_map.keys())]
    df_sectors = df[use_cols].copy()

    for c in sector_map.values():
        if not np.issubdtype(df_sectors[c].dtype, np.number):
            df_sectors[c] = df_sectors[c].apply(parse_time_to_seconds)

    # Renombrar a formato canonical sector<n>
    rename_dict = {sector_map[k]: f"sector{int(k)}" for k in sector_map}
    df_sectors.rename(columns=rename_dict, inplace=True)

    sectors = list(rename_dict.values())
    result = df_sectors.groupby("driver")[sectors].mean().reset_index()

    # Añadir columnas de ranking
    for sec in sectors:
        rank_col = f"{sec}_rank"
        result[rank_col] = result[sec].rank(method="min", ascending=True).astype(int)

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

    """Calculate ideal lap from sector bests and compare to best lap time."""

    if "driver" not in df.columns or "lap_time" not in df.columns:
        raise KeyError("Missing 'driver' or 'lap_time' column")

    # detect sector columns with either "sector" or "s<number>" naming
    sector_map = {}
    for col in df.columns:
        if col.endswith("_rank"):
            continue
        m = re.match(r"(?i)^sector(\d+)", col)
        if not m:
            m = re.match(r"(?i)^s(\d+)", col)
        if m:
            sector_map[col] = f"sector{m.group(1)}"

    sectors = sorted(set(sector_map.values()), key=lambda x: int(re.search(r"\d+", x).group()))

    if not sectors:
        raise KeyError("No sector columns found (expected columns like 'sector1' or 's1')")

    df_num = df[["driver", "lap_time", *sector_map.keys()]].copy()
    df_num.rename(columns=sector_map, inplace=True)

    for col in ["lap_time", *sectors]:
        if not np.issubdtype(df_num[col].dtype, np.number):
            df_num[col] = df_num[col].apply(parse_time_to_seconds)

    best_sectors = df_num.groupby("driver")[sectors].min()
    ideal_lap = best_sectors.sum(axis=1)
    best_lap = df_num.groupby("driver")["lap_time"].min()

    result = (
        pd.DataFrame({
            "driver": best_lap.index,
            "ideal_lap": ideal_lap.values,
            "best_lap": best_lap.values,
        })
        .reset_index(drop=True)
    )
    result["ideal_gap"] = result["best_lap"] - result["ideal_lap"]
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
        if not np.issubdtype(df_num[col].dtype, np.number):
            df_num[col] = df_num[col].apply(parse_time_to_seconds)

    best = df_num.groupby("driver")[sector_cols].min().reset_index()

    for col in sector_cols:
        best[f"{col}_rank"] = best[col].rank(method="min", ascending=True).astype(int)

    return best

