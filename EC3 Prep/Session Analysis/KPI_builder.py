# src/metrics.py
import pandas as pd
import numpy as np
from scipy.stats import linregress
from data_process import parse_time_to_seconds
import warnings

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
    if "lap" in df.columns:
        lap_col = "lap"
    elif "lap_number" in df.columns:
        lap_col = "lap_number"
    else:
        raise KeyError("lap column not found in classification data")
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
    df = df.dropna(subset=["time"])
    weather = weather.dropna(subset=["time"])
    merged = pd.merge_asof(
        df.sort_values("time"),
        weather.sort_values("time"),
        on="time",
    )
    laps = merged['lap_time']
    if not np.issubdtype(laps.dtype, np.number):
        laps = laps.apply(parse_time_to_seconds)

    slope, intercept, r, p, stderr = linregress(merged['temperature'], laps)
    return {'slope': slope, 'r_value': r, 'data': merged}

def track_limits_incidents(df: pd.DataFrame) -> pd.DataFrame:
    """Cuenta de incidentes de track limits por piloto."""
    return df.groupby('driver')['incident'].sum().reset_index()

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

def top_speed_locations(df: pd.DataFrame) -> pd.DataFrame:
    """Lugar de cada ``top_speed`` por piloto (``track_pos``)."""

    if 'track_pos' not in df.columns:
        warnings.warn("Missing 'track_pos' column", UserWarning)
        return pd.DataFrame()

    idx = df.groupby('driver')['top_speed'].idxmax().dropna()
    result = df.loc[idx, ['driver', 'track_pos', 'top_speed']]
    return result.dropna()

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

def lap_time_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Compute lap time standard deviation by driver.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``lap_time`` and ``driver`` columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``driver``, ``lap_std`` and ``lap_mean``
        sorted by ``lap_std`` ascending.
    """

    if "lap_time" not in df.columns or "driver" not in df.columns:
        raise KeyError("Missing 'lap_time' or 'driver' column")

    laps = df["lap_time"]
    if not np.issubdtype(laps.dtype, np.number):
        laps = laps.apply(parse_time_to_seconds)

    grouped = df.assign(lap_time=laps).groupby("driver")
    result = grouped["lap_time"].agg(lap_std="std", lap_mean="mean").reset_index()
    result.sort_values("lap_std", inplace=True)
    return result

def ideal_lap_gap(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate ideal lap from sector bests and compare to best lap time."""

    if "driver" not in df.columns or "lap_time" not in df.columns:
        raise KeyError("Missing 'driver' or 'lap_time' column")

    sectors = [c for c in df.columns if c.startswith("sector") and not c.endswith("_rank")]
    if not sectors:
        raise KeyError("No sector columns found")

    df_num = df[["driver", "lap_time", *sectors]].copy()

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