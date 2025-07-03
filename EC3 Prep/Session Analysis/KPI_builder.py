# src/metrics.py
import pandas as pd
import numpy as np
from scipy.stats import linregress
from data_process import parse_time_to_seconds
import warnings
import re

def detect_stints(laps: pd.DataFrame, session: str) -> pd.DataFrame:
    """Detect stints based on lap time behaviour.

    The lap time column is converted to seconds before evaluating each lap.
    For race sessions the reference per driver is the mean lap time while for
    the rest of sessions the best lap is used.  Laps slower than ``1.5`` times
    the mean (race) are tagged as ``"SC"`` (Safety Car) whereas the rest receive
    ``"R"``.  In non race sessions laps slower than ``1.3`` times the best lap
    are tagged as ``"OUT"`` and the quicker ones as ``"PUSH"``.

    Stint boundaries are detected whenever the label changes or when the column
    ``CROSSING_FINISH_LINE_IN_PIT`` is not empty.  A new ``stint_id`` is
    generated incrementally for each driver.

    Parameters
    ----------
    laps : pd.DataFrame
        DataFrame with at least ``driver`` and ``lap_time`` columns.
    session : str
        Name of the session.  If it contains ``"race"`` the race rules are
        applied.

    Returns
    -------
    pd.DataFrame
        ``laps`` with two extra columns: ``stint_label`` and ``stint_id``.
    """

    if "driver" not in laps.columns or "lap_time" not in laps.columns:
        return laps

    df = laps.copy()

    if not pd.api.types.is_numeric_dtype(df["lap_time"]):
        df["_lap_time_sec"] = df["lap_time"].apply(parse_time_to_seconds)
    else:
        df["_lap_time_sec"] = df["lap_time"].astype(float)

    pit_col = next(
        (c for c in df.columns if c.lower() == "crossing_finish_line_in_pit"),
        None,
    )

    is_race = "race" in session.lower()

    if is_race:
        ref = df.groupby("driver")["_lap_time_sec"].transform("mean")
        df["stint_label"] = np.where(
            df["_lap_time_sec"] >= 1.5 * ref, "SC", "R"
        )
    else:
        ref = df.groupby("driver")["_lap_time_sec"].transform("min")
        df["stint_label"] = np.where(
            df["_lap_time_sec"] >= 1.3 * ref, "OUT", "PUSH"
        )

    lap_col = "lap" if "lap" in df.columns else (
        "lap_number" if "lap_number" in df.columns else None
    )
    sort_cols = ["driver"] + ([lap_col] if lap_col else [])
    df = df.sort_values(sort_cols).reset_index(drop=True)

    in_pit = (
        df[pit_col].fillna("").astype(str).str.strip() != ""
        if pit_col is not None else pd.Series(False, index=df.index)
    )

    prev_label = df.groupby("driver")["stint_label"].shift(1)
    new_stint = in_pit | (df["stint_label"] != prev_label) | (
        df["driver"] != df["driver"].shift(1)
    )

    df["stint_id"] = df.groupby("driver")[new_stint].cumsum().astype(int)
    df["stint_id"] += 1

    df = df.drop(columns=["_lap_time_sec"])
    return df

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
    """Return lap, driver, team and lap_time history.

    The function attempts to extract the lap column (either ``lap`` or
    ``lap_number``) together with ``driver`` and ``lap_time``.  If ``team``
    exists, it is included as well.  The returned DataFrame always contains
    the column ``lap`` (``lap_number`` is renamed accordingly).
    """

    lap_col = None
    for col in ["lap", "lap_number"]:
        if col in df.columns:
            lap_col = col
            break

    if lap_col is None or "driver" not in df.columns or "lap_time" not in df.columns:
        return pd.DataFrame()

    cols = [lap_col, "driver"]
    if "team" in df.columns:
        cols.append("team")
    cols.append("lap_time")

    df_hist = (
        df[cols]
        .dropna(subset=[lap_col, "lap_time"])
        .copy()
    )

    if lap_col != "lap":
        df_hist = df_hist.rename(columns={lap_col: "lap"})

    df_hist["lap"] = df_hist["lap"].astype(int)
    df_hist["lap_time"] = df_hist["lap_time"].astype(float)
    ordered_cols = ["lap", "driver"] + (["team"] if "team" in df_hist.columns else []) + ["lap_time"]
    return df_hist[ordered_cols]

def pit_stop_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Best and mean pit stop time per driver.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ``driver`` and ``pit_time``.  If ``team`` exists
        it is also included in the result.

    Returns
    -------
    pd.DataFrame
        Columns ``driver`` [, ``team``] ``best_pit_time`` and ``mean_pit_time``.
    """

    if "driver" not in df.columns or "pit_time" not in df.columns:
        raise KeyError("Faltan columnas 'driver' o 'pit_time'")

    cols = ["driver", "pit_time"] + (["team"] if "team" in df.columns else [])
    data = df[cols].copy()

    if not pd.api.types.is_numeric_dtype(data["pit_time"]):
        data["pit_time"] = data["pit_time"].apply(parse_time_to_seconds)

    group_cols = ["driver"] + (["team"] if "team" in data.columns else [])
    result = (
        data.groupby(group_cols)["pit_time"]
        .agg(["min", "mean"])
        .reset_index()
        .rename(columns={"min": "best_pit_time", "mean": "mean_pit_time"})
    )
    return result

def lap_time_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Lap time standard deviation per driver."""

    if "driver" not in df.columns or "lap_time" not in df.columns:
        raise KeyError("Faltan columnas 'driver' o 'lap_time'")

    data = df[["driver", "lap_time"]].copy()
    if not pd.api.types.is_numeric_dtype(data["lap_time"]):
        data["lap_time"] = data["lap_time"].apply(parse_time_to_seconds)

    result = (
        data.groupby("driver")["lap_time"]
        .std()
        .reset_index(name="lap_time_std")
    )
    return result


    import pandas as pd
from KPI_builder import best_sector_times  # para recuperar los mejores sectores

def extract_session_summary(df_analysis: pd.DataFrame,
                            tracklimits_df: pd.DataFrame,
                            output_path: str):
    """
    Extrae un resumen de métricas por piloto y lo vuelca en CSV/Excel.
    
    Parámetros
    ----------
    df_analysis : DataFrame
        DataFrame de la sesión con columnas 'driver', 'lap_time', etc.
    tracklimits_df : DataFrame
        DataFrame con las infracciones de track limits (columna 'driver' y 'incident').
    output_path : str
        Ruta de salida. Según la extensión genera CSV (.csv) o Excel (.xlsx).
    """
    drivers = df_analysis['driver'].unique()
    rows = {}
    # Pre-cálculo de mejores sectores
    bst = best_sector_times(df_analysis).set_index('driver')

    for drv in drivers:
        dfa = df_analysis[df_analysis['driver'] == drv]
        total_time      = dfa['lap_time'].sum()
        num_laps        = int(dfa['lap_time'].count())
        mean_lap        = dfa['lap_time'].mean()
        best_lap        = dfa['lap_time'].min()
        # Sectores
        if drv in bst.index:
            sec1 = bst.at[drv, 'sector1']
            sec2 = bst.at[drv, 'sector2']
            sec3 = bst.at[drv, 'sector3']
        else:
            sec1 = sec2 = sec3 = float('nan')
        # Track limits
        tl = tracklimits_df[tracklimits_df['driver'] == drv]
        incidents = int(tl['incident'].sum()) if 'incident' in tl.columns else 0
        rate      = incidents / num_laps if num_laps>0 else float('nan')

        rows[drv] = {
            'total_lap_time':   total_time,
            'num_laps':         num_laps,
            'mean_lap_time':    mean_lap,
            'best_lap_time':    best_lap,
            'best_sector1':     sec1,
            'best_sector2':     sec2,
            'best_sector3':     sec3,
            'num_incidents':    incidents,
            'track_limit_rate': rate,
        }

    # Construimos el DataFrame: índices = métricas, columnas = pilotos
    summary = pd.DataFrame(rows).T.T
    # Transponemos dos veces para que queden filas=métricas, columnas=pilotos
    # primero: df index=pilotos, cols=métricas; segundo: index=métricas, cols=pilotos

    # Guardado según extensión
    if output_path.lower().endswith('.xlsx'):
        summary.to_excel(output_path)
    else:
        summary.to_csv(output_path, index=True)

    return summary

def slipstream_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return mean lap time and top speed for laps with and without slipstream."""

    slip_col = next(
        (
            c
            for c in df.columns
            if "slip" in c.lower() or "draft" in c.lower() or "rebufo" in c.lower()
        ),
        None,
    )
    if slip_col is None or "lap_time" not in df.columns or "top_speed" not in df.columns:
        return pd.DataFrame()

    data = df[[slip_col, "lap_time", "top_speed"]].copy()

    if not pd.api.types.is_numeric_dtype(data["lap_time"]):
        data["lap_time"] = data["lap_time"].apply(parse_time_to_seconds)

    true_vals = {"1", "true", "yes", "y", "si", "sí"}
    data["slipstream"] = data[slip_col].astype(str).str.lower().isin(true_vals)

    result = (
        data.groupby("slipstream")
        .agg(
            mean_lap_time=("lap_time", "mean"),
            mean_top_speed=("top_speed", "mean"),
            laps=("lap_time", "count"),
        )
        .reset_index()
    )

    result["slipstream"] = result["slipstream"].map({True: "Con rebufo", False: "Sin rebufo"})
    return result