# src/metrics.py
import pandas as pd
import numpy as np
from scipy.stats import linregress
from data_process import parse_time_to_seconds
import warnings
import re
from collections import OrderedDict
from scipy.stats import median_abs_deviation

_INV_0_6745 = 1.4826              # 1 / 0.6745  → escala MAD → σ

def _mad(series: pd.Series) -> float:
    """Median Absolute Deviation escalada a σ (robusta)."""
    med = np.median(series)
    return np.median(np.abs(series - med)) * _INV_0_6745

def detect_stints(laps: pd.DataFrame, session: str) -> pd.DataFrame:
    """Detect stints based on lap time behaviour.

    The lap time column is converted to seconds before evaluating each lap.
    For race sessions the reference per number is the mean lap time while for
    the rest of sessions the best lap is used.  Laps slower than ``1.5`` times
    the mean (race) are tagged as ``"SC"`` (Safety Car) whereas the rest receive
    ``"R"``.  In non race sessions laps slower than ``1.3`` times the best lap
    are tagged as ``"OUT"`` and the quicker ones as ``"PUSH"``.

    Stint boundaries are detected whenever the label changes or when the column
    ``CROSSING_FINISH_LINE_IN_PIT`` is not empty.  A new ``stint_id`` is
    generated incrementally for each number.

    Parameters
    ----------
    laps : pd.DataFrame
        DataFrame with at least ``number`` and ``lap_time`` columns.
    session : str
        Name of the session.  If it contains ``"race"`` the race rules are
        applied.

    Returns
    -------
    pd.DataFrame
        ``laps`` with two extra columns: ``stint_label`` and ``stint_id``.
    """

    if "number" not in laps.columns or "lap_time" not in laps.columns:
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
        ref = df.groupby("number")["_lap_time_sec"].transform("mean")
        df["stint_label"] = np.where(
            df["_lap_time_sec"] >= 1.5 * ref, "SC", "R"
        )
    else:
        ref = df.groupby("number")["_lap_time_sec"].transform("min")
        df["stint_label"] = np.where(
            df["_lap_time_sec"] >= 1.3 * ref, "OUT", "PUSH"
        )

    lap_col = "lap" if "lap" in df.columns else (
        "lap_number" if "lap_number" in df.columns else None
    )
    sort_cols = ["number"] + ([lap_col] if lap_col else [])
    df = df.sort_values(sort_cols).reset_index(drop=True)

    in_pit = (
        df[pit_col].fillna("").astype(str).str.strip() != ""
        if pit_col is not None else pd.Series(False, index=df.index)
    )

    prev_label = df.groupby("number")["stint_label"].shift(1)
    new_stint = in_pit | (df["stint_label"] != prev_label) | (
        df["number"] != df["number"].shift(1)
    )

    df["stint_id"] = df.groupby("number")[new_stint].cumsum().astype(int)
    df["stint_id"] += 1

    df = df.drop(columns=["_lap_time_sec"])
    return df

def _detect_driver_col(df: pd.DataFrame) -> str:
    """Devuelve la mejor columna disponible para identificar UNÍVOCAMENTE al piloto."""
    for c in ["number", "driver_shortname", "driver_name", "driver_number"]:
        if c in df.columns:
            return c
    raise KeyError("No se encontró columna de piloto")

def compute_top_speeds(df: pd.DataFrame) -> pd.DataFrame:
    # 1) Columna única de piloto
    driver_col = _detect_driver_col(df)

    # 2) Columna de equipo (igual que antes)
    team_col = next((c for c in ["team", "constructor", "team_name"] if c in df.columns), None)

    if "top_speed" not in df.columns:
        raise KeyError("No se encontró columna top_speed")

    if team_col:
        grouped = (
            df.groupby([driver_col, team_col])["top_speed"]
              .max()
              .reset_index()
              .rename(columns={
                  driver_col: "number",
                  team_col:   "team",
                  "top_speed": "max_top_speed",
              })
        )
    else:
        grouped = (
            df.groupby(driver_col)["top_speed"]
              .max()
              .reset_index()
              .rename(columns={
                  driver_col: "number",
                  "top_speed": "max_top_speed",
              })
        )
        grouped["team"] = None
        grouped = grouped[["number", "team", "max_top_speed"]]

    return grouped.sort_values("max_top_speed", ascending=False)

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
        DataFrame con ``number``, ``incidents``, ``laps`` y ``rate``.
    """

    track_driver = next(
        (c for c in ['number', 'driver_name', 'driver_shortname', 'driver_number'] if c in track_df.columns),
        None,
    )
    if track_driver is None:
        raise KeyError('No se encontró columna de piloto en track_df')

    lap_driver = next(
        (c for c in ['number', 'driver_name', 'driver_shortname', 'driver_number'] if c in laps_df.columns),
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
        pd.DataFrame({'number': incidents.index, 'incidents': incidents.values})
        .merge(
            pd.DataFrame({'number': laps.index, 'laps': laps.values}),
            on='number',
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
    Devuelve number, team (si existe), ideal_time, best_lap, ideal_gap.
    """
    if 'number' not in df.columns or 'lap_time' not in df.columns:
        raise KeyError("Faltan columnas 'number' o 'lap_time'")
    
    # Convertir lap_time a segundos si hace falta
    df_num = df[['number'] + (['team'] if 'team' in df.columns else []) + ['lap_time']].copy()
    if not np.issubdtype(df_num['lap_time'].dtype, np.number):
        df_num['lap_time'] = df_num['lap_time'].apply(parse_time_to_seconds)

    # Mejor vuelta real
    best = df_num.groupby(['number'] + (['team'] if 'team' in df_num.columns else []))['lap_time']\
                 .min().reset_index(name='best_lap')
    
    # Mejor sector por piloto
    bst = best_sector_times(df)
    if bst.empty:
        return pd.DataFrame()
    # Sumar mejores sectores para ideal_time
    bst['ideal_time'] = bst[['sector1','sector2','sector3']].sum(axis=1)

    # Unir best_lap e ideal_time
    merged = pd.merge(best, bst, on=['number'] + (['team'] if 'team' in bst.columns else []))
    merged['ideal_gap'] = merged['best_lap'] - merged['ideal_time']
    return merged[['number'] + (['team'] if 'team' in merged.columns else [])
                  + ['ideal_time','best_lap','ideal_gap']]

def sector_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la media de cada sector (sector1, sector2, sector3) por piloto.
    Sólo toma las columnas *_seconds si existen, para evitar duplicados.
    Devuelve un DataFrame con columnas: number, [team], sector1, sector2, sector3.
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
    
    # Columnas que vamos a usar: number, [team], y sólo las de rename_map
    base = ['number'] + (['team'] if 'team' in df.columns else [])
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
    """Best sector times per number with rankings.

    The function detects columns named either ``sectorN`` or ``sN`` (case
    insensitive). For each sector it computes the minimum time by number and
    adds a ranking column ``<sector>_rank`` where ``1`` represents the fastest
    number.
    """

    if "number" not in df.columns:
        raise KeyError("'number' column not found")

    sector_cols = []
    for col in df.columns:
        if col.lower().startswith("sector") and col[6:].isdigit():
            sector_cols.append(col)
        elif re.match(r"(?i)^s\d+$", col):
            sector_cols.append(col)

    if not sector_cols:
        return pd.DataFrame()

    df_num = df[["number", *sector_cols]].copy()
    for col in sector_cols:
        if not np.issubdtype(df_num[col].dtypes, np.number):
            df_num[col] = df_num[col].apply(parse_time_to_seconds)

    best = df_num.groupby("number")[sector_cols].min().reset_index()

    for col in sector_cols:
        best[f"{col}_rank"] = best[col].rank(method="min", ascending=True).astype(int)

    return best

def best_sector_times(df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada piloto devuelve su mejor (mínimo) tiempo en cada sector.
    Solo toma las columnas *_seconds si existen; 
    en caso contrario, busca columnas s1, s2, s3.
    Columnas resultantes: number, [team], sector1, sector2, sector3.
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
    base = ['number'] + (['team'] if 'team' in df.columns else [])
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
    """Return lap, number, team and lap_time history.

    The function attempts to extract the lap column (either ``lap`` or
    ``lap_number``) together with ``number`` and ``lap_time``.  If ``team``
    exists, it is included as well.  The returned DataFrame always contains
    the column ``lap`` (``lap_number`` is renamed accordingly).
    """

    lap_col = None
    for col in ["lap", "lap_number"]:
        if col in df.columns:
            lap_col = col
            break

    if lap_col is None or "number" not in df.columns or "lap_time" not in df.columns:
        return pd.DataFrame()

    cols = [lap_col, "number"]
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
    ordered_cols = ["lap", "number"] + (["team"] if "team" in df_hist.columns else []) + ["lap_time"]
    return df_hist[ordered_cols]

def pit_stop_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Best and mean pit stop time per number.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ``number`` and ``pit_time``.  If ``team`` exists
        it is also included in the result.

    Returns
    -------
    pd.DataFrame
        Columns ``number`` [, ``team``] ``best_pit_time`` and ``mean_pit_time``.
    """

    if "number" not in df.columns or "pit_time" not in df.columns:
        raise KeyError("Faltan columnas 'number' o 'pit_time'")

    cols = ["number", "pit_time"] + (["team"] if "team" in df.columns else [])
    data = df[cols].copy()

    if not pd.api.types.is_numeric_dtype(data["pit_time"]):
        data["pit_time"] = data["pit_time"].apply(parse_time_to_seconds)

    group_cols = ["number"] + (["team"] if "team" in data.columns else [])
    result = (
        data.groupby(group_cols)["pit_time"]
        .agg(["min", "mean"])
        .reset_index()
        .rename(columns={"min": "best_pit_time", "mean": "mean_pit_time"})
    )
    return result

def lap_time_consistency(
    df: pd.DataFrame,
    *,
    threshold: float = 0.08,   # 10 % sobre la mejor vuelta
    trim: float = 0.10,        # 10 % a cada lado
    min_laps: int = 3,
) -> pd.DataFrame:
    """
    Consistencia por piloto usando desviación estándar recortada.

    1. Filtra vueltas > (1+threshold)×best_lap del piloto.
    2. Elimina trim·100 % superior e inferior.
    3. Calcula σ poblacional.  Devuelve NaN si quedan < min_laps.
    """

    req = {"number", "lap_time"}
    if req - set(df.columns):
        raise KeyError(f"Faltan columnas {req}")

    data = df[["number", "lap_time"]].copy()
    if not pd.api.types.is_numeric_dtype(data["lap_time"]):
        data["lap_time"] = data["lap_time"].apply(parse_time_to_seconds)

    # Mejor vuelta y filtro 10 %
    best = data.groupby("number")["lap_time"].transform("min")
    data = data[data["lap_time"] <= best * (1 + threshold)]

    def _trimmed_std(series: pd.Series) -> float:
        n = len(series)
        if n < min_laps:
            return np.nan
        k = int(np.floor(n * trim))
        trimmed = series.sort_values().iloc[k:n - k] if k else series
        return trimmed.std(ddof=0)          # σ poblacional

    result = (
        data.groupby("number")["lap_time"]
            .apply(_trimmed_std)
            .rename("lap_time_std")          # nombre que espera build_figures
            .reset_index()
    )
    return result

from KPI_builder import best_sector_times  

def extract_session_summary(df_analysis: pd.DataFrame,
                            tracklimits_df: pd.DataFrame,
                            output_path: str):
    """
    Extrae un resumen de métricas por piloto y lo vuelca en CSV/Excel.
    
    Parámetros
    ----------
    df_analysis : DataFrame
        DataFrame de la sesión con columnas 'number', 'lap_time', etc.
    tracklimits_df : DataFrame
        DataFrame con las infracciones de track limits (columna 'number' y 'incident').
    output_path : str
        Ruta de salida. Según la extensión genera CSV (.csv) o Excel (.xlsx).
    """
    drivers = df_analysis['number'].unique()
    rows = {}
    # Pre-cálculo de mejores sectores
    bst = best_sector_times(df_analysis).set_index('number')

    for drv in drivers:
        dfa = df_analysis[df_analysis['number'] == drv]
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
        tl = tracklimits_df[tracklimits_df['number'] == drv]
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

def slipstream_stats(
    df: pd.DataFrame,
    *,
    fast_threshold: float = 0.02,
    dt_min: float = 0.20,
    dt_max: float = 2.50,
    topspeed_delta: float = 6.0,
) -> pd.DataFrame:
    """
    KPIs de rebufo por piloto (number).

    Una vuelta cuenta como 'con rebufo' si:
      ▸ En su propio paso de meta el coche llega ≤ 2.5 s detrás de otro coche
        (y ambos iban en vuelta rápida).
      ▸ O bien la vuelta anterior cumplía ese criterio
        → propagamos el efecto de rebufo al inicio de la vuelta actual.

    Requiere columnas:
        number, number, lap_number, lap_time, top_speed, hour [, team]
    """
    required = {
        "number", "number", "lap_number",
        "lap_time", "top_speed", "hour"
    }
    if not required.issubset(df.columns):
        return pd.DataFrame()

    extra = ["team"] if "team" in df.columns else []
    gap_cols_present = [c for c in ("GapAhead_S1", "GapAhead_S2", "GapAhead_S3")
                        if c in df.columns]
    use_cols = list(required | set(extra) | set(gap_cols_present))
    data = df[use_cols].copy()

    # ─── Formatos ──────────────────────────────────────────────────────────
    if not pd.api.types.is_numeric_dtype(data["lap_time"]):
        data["lap_time"] = data["lap_time"].apply(parse_time_to_seconds)

    data["timestamp"] = pd.to_datetime(
        data["hour"], format="%H:%M:%S.%f", errors="coerce"
    )
    data = data.sort_values("timestamp").reset_index(drop=True)

    # ─── Bandera de vuelta rápida / Δt con el coche anterior ───────────────
    best_lap = data.groupby("number")["lap_time"].transform("min")
    data["competitive"] = data["lap_time"] <= (1 + fast_threshold) * best_lap
      
    median_speed = data["top_speed"].median()
    delta_t   = (data["timestamp"] - data["timestamp"].shift()).dt.total_seconds()
    prev_num  = data["number"].shift()
    speed_thr = (
        data.groupby("number")["top_speed"].transform("median") + topspeed_delta
    )

    # ① Rebufo al cruzar meta -------------------------------------------------
    data["slip_meta"] = (
        data["competitive"]
        & (prev_num != data["number"])
        & delta_t.between(dt_min, dt_max)
        & (data["top_speed"] >= speed_thr)
    )

    # ② Rebufo en sectores ----------------------------------------------------
    need_cols = {"GapAhead_S1", "GapAhead_S2", "GapAhead_S3"}
    if need_cols <= set(data.columns):      
        for lab in ("S1", "S2", "S3"):
            gcol = f"GapAhead_{lab}"
            data[f"slip_{lab}"] = (
                data["competitive"]
                & (data[gcol].between(dt_min, dt_max))
            )
    else:
        data[["slip_S1", "slip_S2", "slip_S3"]] = False

    # ③ Bandera global + propagación -----------------------------------------
    data["slip_base"] = (
        data["slip_meta"] | data["slip_S1"] | data["slip_S2"] | data["slip_S3"]
    )
    data = data.sort_values(["number", "lap_number"]).reset_index(drop=True)
    data["slip_prev"] = data.groupby("number")["slip_base"].shift(fill_value=False)
    data["slipstream"] = data["slip_base"] | data["slip_prev"]

    # ─── KPIs agregados ────────────────────────────────────────────────────
    stats = (
        data
        .groupby("number")
        .apply(lambda g: pd.Series({

            # extremos
            "min_lap_time_with_slip":  g.loc[g["slipstream"], "lap_time"].min(),
            "min_lap_time_no_slip":    g.loc[~g["slipstream"], "lap_time"].min(),
            "max_top_speed_with_slip": g.loc[g["slipstream"], "top_speed"].max(),
            "max_top_speed_no_slip":   g.loc[~g["slipstream"], "top_speed"].max(),

            
        }),include_groups=False)
        .reset_index()
    )

    # añade team si existe
    if "team" in data.columns:
        stats = stats.merge(
            data[["number", "team"]].drop_duplicates("number"),
            on="number", how="left"
        )

    return stats

def sector_slipstream_stats(
    df: pd.DataFrame,
    *,
    fast_threshold: float = 0.05,
    dt_min: float = 0.20,
    dt_max: float = 2.50,
    topspeed_delta: float = 6.0,
) -> pd.DataFrame:
    """Slipstream KPIs per number by sector.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis dataframe including ``T1``, ``T2`` and ``T3`` timestamps
        together with lap metrics (``lap_time``, ``top_speed`` and
        ``lap_number``).

    Returns
    -------
    pd.DataFrame
        Aggregated slipstream statistics for sector 1 and sector 2.  Column
        names are suffixed with ``_s1`` or ``_s2``.
    """

    req_base = {
        "number",
        "number",
        "lap_number",
        "lap_time",
        "top_speed",
        "T1",
        "T2",
        "T3",
    }

    if not req_base.issubset(df.columns):
        return pd.DataFrame()

    # locate sector time columns
    s1_col = next((c for c in df.columns if c.lower() in {"s1_seconds", "s1", "sector1"}), None)
    s2_col = next((c for c in df.columns if c.lower() in {"s2_seconds", "s2", "sector2"}), None)
    if s1_col is None or s2_col is None:
        return pd.DataFrame()

    extra = ["team"] if "team" in df.columns else []
    use_cols = list(req_base | {s1_col, s2_col} | set(extra))
    # ── selección mínima de columnas (añadimos dinámicamente GapAhead_S*) ──
    gap_cols_present = [c for c in ("GapAhead_S1", "GapAhead_S2", "GapAhead_S3")
                        if c in df.columns]
    use_cols = list(req_base | {s1_col, s2_col} | set(extra) | set(gap_cols_present))
    data = df[use_cols].copy()

    if not pd.api.types.is_numeric_dtype(data["lap_time"]):
        data["lap_time"] = data["lap_time"].apply(parse_time_to_seconds)

    for col in [s1_col, s2_col]:
        if not pd.api.types.is_numeric_dtype(data[col]):
            data[col] = data[col].apply(parse_time_to_seconds)

    for tcol in ["T1", "T2", "T3"]:
        data[tcol] = pd.to_datetime(data[tcol], format="%H:%M:%S.%f", errors="coerce")

    data = data.sort_values("T3").reset_index(drop=True)

    # ─── Vuelta competitiva (≤ 110 % de su mejor) ---------------------------
    best_lap = data.groupby("number")["lap_time"].transform("min")
    data["competitive"] = data["lap_time"] <= (1 + fast_threshold) * best_lap


    # umbral de velocidad: mediana del propio piloto + 4 km/h
    speed_thr = (
        data.groupby("number")["top_speed"].transform("median") + topspeed_delta
        )

    # ─── Sector 1 detection ────────────────────────────────────────────────
    s1 = data.sort_values("T1").reset_index()
    dt1 = (s1["T1"] - s1["T1"].shift()).dt.total_seconds()
    prev_drv = s1["number"].shift()
    prev_fast = s1["competitive"].shift()
    s1["slip_flag_s1"] = (
        s1["competitive"]
        & (prev_drv != s1["number"])
        & dt1.between(dt_min, dt_max)                 # mismo rango que meta
        & (s1["top_speed"] >= speed_thr)
    )
    s1 = s1.sort_values(["number", "lap_number"]).reset_index(drop=True)
    s1["slip_prev_s1"] = s1.groupby("number")["slip_flag_s1"].shift(fill_value=False)
    s1["slipstream_s1"] = s1["slip_flag_s1"] | s1["slip_prev_s1"]
    s1 = s1.sort_values("index")
    data["slipstream_s1"] = s1["slipstream_s1"].values

    # ─── Sector 2 detection ────────────────────────────────────────────────
    s2 = data.sort_values("T2").reset_index()
    dt2 = (s2["T2"] - s2["T2"].shift()).dt.total_seconds()
    prev_drv2 = s2["number"].shift()
    prev_fast2 = s2["competitive"].shift()
    s2["slip_flag_s2"] = (
        s2["competitive"]
        & (prev_drv2 != s2["number"])
        & dt2.between(dt_min, dt_max)
        & (s2["top_speed"] >= speed_thr)
    )
    s2 = s2.sort_values(["number", "lap_number"]).reset_index(drop=True)
    s2["slip_prev_s2"] = s2.groupby("number")["slip_flag_s2"].shift(fill_value=False)
    s2["slipstream_s2"] = s2["slip_flag_s2"] | s2["slip_prev_s2"]
    s2 = s2.sort_values("index")
    data["slipstream_s2"] = s2["slipstream_s2"].values

    # ─── KPIs per number ─────────────────────────────────────────────────--
    def _agg(g: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "avg_time_with_slip_s1": g.loc[g["slipstream_s1"], s1_col].mean(),
                "avg_time_no_slip_s1": g.loc[~g["slipstream_s1"], s1_col].mean(),
                "min_time_with_slip_s1": g.loc[g["slipstream_s1"], s1_col].min(),
                "min_time_no_slip_s1": g.loc[~g["slipstream_s1"], s1_col].min(),
                "slip_laps_s1": g["slipstream_s1"].sum(),
                "avg_time_with_slip_s2": g.loc[g["slipstream_s2"], s2_col].mean(),
                "avg_time_no_slip_s2": g.loc[~g["slipstream_s2"], s2_col].mean(),
                "min_time_with_slip_s2": g.loc[g["slipstream_s2"], s2_col].min(),
                "min_time_no_slip_s2": g.loc[~g["slipstream_s2"], s2_col].min(),
                "slip_laps_s2": g["slipstream_s2"].sum(),
                "total_laps": len(g),
            }
        )

    stats = data.groupby("number").apply(_agg).reset_index()
    stats["slip_pct_s1"] = stats["slip_laps_s1"] / stats["total_laps"] * 100
    stats["slip_pct_s2"] = stats["slip_laps_s2"] / stats["total_laps"] * 100

    if "team" in data.columns:
        stats = stats.merge(
            data[["number", "team"]].drop_duplicates("number"),
            on="number",
            how="left",
        )

    return stats

from collections import OrderedDict

def build_driver_tables(
    df: pd.DataFrame,
    teams: list[str] | None = None,
    *,
    filter_fast: bool = True,
    include_sectors: bool = True,
    include_sector_gaps: bool = False,
) -> "OrderedDict[str, pd.DataFrame]":
    """
    Devuelve OrderedDict {number: DataFrame} con todas las vueltas que

    GapStart se basa SIEMPRE en la vuelta anterior del mismo piloto:
      • con sectores → GapAhead_S3 de la vuelta anterior
      • sin sectores → GapAhead  de la vuelta anterior
    """

    # ── 1· limitar a equipos seleccionados ────────────────────────────
    df = df.copy()
    if teams is not None:
        df = df[df["team"].isin(teams)]
    if df.empty:
        return OrderedDict()

    # ── 2· normalizar tiempos ─────────────────────────────────────────
    if not pd.api.types.is_numeric_dtype(df["lap_time"]):
        df["lap_time"] = df["lap_time"].apply(parse_time_to_seconds)

    # ── 3· timestamp global + GapAhead (meta) ─────────────────────────
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.to_datetime(df["hour"], errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)
    if "GapAhead" not in df.columns:
        df["GapAhead"] = df["timestamp"].diff().dt.total_seconds().abs()

    # ── 4· localizar columnas de sector (si existen) ──────────────────
    sector_aliases = {
        "sector1": ["sector1", "s1", "sector1_seconds", "s1_seconds"],
        "sector2": ["sector2", "s2", "sector2_seconds", "s2_seconds"],
        "sector3": ["sector3", "s3", "sector3_seconds", "s3_seconds"],
    }
    sector_cols = {
        std: next((c for c in aliases if c in df.columns), None)
        for std, aliases in sector_aliases.items()
    }

    gap_cols: list[str] = []

    # ── 5· gaps por sector (si hay los 3 splits) ──────────────────────
    if all(sector_cols.values()):
        for col in sector_cols.values():
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].apply(parse_time_to_seconds)

        if {"GapAhead_S1", "GapAhead_S2", "GapAhead_S3"} - set(df.columns):
            df["T3"] = df["timestamp"]
            df["T2"] = df["T3"] - pd.to_timedelta(df[sector_cols["sector3"]], unit="s")
            df["T1"] = df["T2"] - pd.to_timedelta(df[sector_cols["sector2"]], unit="s")

            for lab, tcol in zip(("S1", "S2", "S3"), ("T1", "T2", "T3")):
                colname = f"GapAhead_{lab}"
                if colname not in df.columns:
                    df[colname] = (
                        df.loc[df[tcol].sort_values().index, tcol]
                          .diff().dt.total_seconds().abs()
                          .reindex(df.index)
                    )
                if include_sector_gaps and colname not in gap_cols:
                    gap_cols.append(colname)

        # GapStart basado en GapAhead_S3 de la vuelta anterior
        df["GapStart"] = df.groupby("number")["GapAhead_S3"].shift(1)
    else:
        # sin sectores → GapStart = GapAhead de la vuelta anterior
        df["GapStart"] = df.groupby("number")["GapAhead"].shift(1)

    if include_sector_gaps:
        for colname in ("GapAhead_S1", "GapAhead_S2", "GapAhead_S3"):
            if colname in df.columns and colname not in gap_cols:
                gap_cols.append(colname)


    # ── 3·5 FILTRO idéntico al de build_fastest_lap_table ─────────────
    if filter_fast:
        best = df.groupby("number")["lap_time"].transform("min")
        mask = (df["lap_time"] <= 1.20 * best) 
        df = df[mask]
        
    # ── 6· sectores a mostrar (si se piden) ───────────────────────────
    sector_map = {
        "sector1": "Sector1", "s1": "Sector1",
        "sector2": "Sector2", "s2": "Sector2",
        "sector3": "Sector3", "s3": "Sector3",
    }
    sec_cols: list[str] = []
    if include_sectors:
        for raw, std in sector_map.items():
            if raw in df.columns:
                if not pd.api.types.is_numeric_dtype(df[raw]):
                    df[std] = df[raw].apply(parse_time_to_seconds)
                else:
                    df[std] = df[raw]
                sec_cols.append(std)

    # 7· columnas finales y tablas por piloto
    base_cols = [
        "team",
        "lap_number", "lap_time",
        *sec_cols,
        "GapStart", *gap_cols,
        "top_speed",
    ]

    # ahora añadimos también los gap de sectores
    rename = {
        "team":         "Team",
        "lap_number":   "Vuelta",
        "lap_time":     "LapTime",
        "GapStart":     "GapStart",
        "GapAhead_S1":  "GapSector1",
        "GapAhead_S2":  "GapSector2",
        "GapAhead_S3":  "GapSector3",
        "top_speed":    "TopSpeed",
    }

    tables = OrderedDict()
    for drv, g in df.groupby("number"):
        tbl = g[base_cols].rename(columns=rename)
        tables[drv] = tbl.sort_values("LapTime").reset_index(drop=True)

    return tables

def build_fastest_lap_table(
    df: pd.DataFrame,
    df_class: pd.DataFrame | None = None,
    *,
    teams: list[str] | None = None,
    filter_fast: bool = True,
    include_sectors: bool = True,
    include_sector_gaps: bool = False,
) -> pd.DataFrame:
    """
    Igual que antes, pero ahora acepta `teams` para filtrar la tabla
    **tras** haber calculado todos los gaps en el DataFrame completo,
    de modo que quitar pilotos no altere los valores de gap.
    """

    # 0 · normalizar “number”
    if "number" not in df.columns:
        for src in ("driver_name","driver_shortname","driver_number"):
            if src in df.columns:
                df = df.rename(columns={src:"number"})
                break

    if df.empty or {"number","lap_time"} - set(df.columns):
        return pd.DataFrame()

    df = df.copy()

    # 1 · timestamp + GapAhead (meta)
    if "timestamp" not in df.columns and "hour" in df.columns:
        df["timestamp"] = pd.to_datetime(df["hour"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["GapAhead"] = df["timestamp"].diff().dt.total_seconds().abs()

    # 2 · splits y gaps sectoriales (si existen)
    sector_aliases = {
        "sector1": ["sector1","s1","sector1_seconds","s1_seconds"],
        "sector2": ["sector2","s2","sector2_seconds","s2_seconds"],
        "sector3": ["sector3","s3","sector3_seconds","s3_seconds"],
    }
    sector_cols = {std: next((c for c in aliases if c in df.columns), None)
                   for std,aliases in sector_aliases.items()}

    if all(sector_cols.values()):
        # convertir a segundos
        for c in sector_cols.values():
            df[c] = pd.to_numeric(df[c].apply(parse_time_to_seconds), errors="coerce")

        df["T3"] = df["timestamp"]
        df["T2"] = df["T3"] - pd.to_timedelta(df[sector_cols["sector3"]], unit="s")
        df["T1"] = df["T2"] - pd.to_timedelta(df[sector_cols["sector2"]], unit="s")

        df = df.assign(
            GapAhead_S1=lambda d: (
                d.loc[d["T1"].sort_values().index,"T1"]
                 .diff().dt.total_seconds().abs()
                 .reindex(d.index)
            ),
            GapAhead_S2=lambda d: (
                d.loc[d["T2"].sort_values().index,"T2"]
                 .diff().dt.total_seconds().abs()
                 .reindex(d.index)
            ),
            GapAhead_S3=lambda d: (
                d.loc[d["T3"].sort_values().index,"T3"]
                 .diff().dt.total_seconds().abs()
                 .reindex(d.index)
            ),
        )
        # GapStart en bruto
        df["GapStart"] = df.groupby("number")["GapAhead_S3"].shift(1)

        if include_sectors:
            df["Sector1"] = df[sector_cols["sector1"]]
            df["Sector2"] = df[sector_cols["sector2"]]
            df["Sector3"] = df[sector_cols["sector3"]]
    else:
        # solo GapAhead meta
        df["GapStart"] = df.groupby("number")["GapAhead"].shift(1)

    # 3 · filtro rápido SOLO por lap_time (no tocamos gaps)
    if filter_fast:
        best = df.groupby("number")["lap_time"].transform("min")
        df = df[df["lap_time"] <= 1.20 * best].copy()

    # 4 · asegurar lap_time en segundos
    if not pd.api.types.is_numeric_dtype(df["lap_time"]):
        df["lap_time"] = df["lap_time"].apply(parse_time_to_seconds)

    # 5 · extraer índice de la best lap
    idx = df.groupby("number")["lap_time"].idxmin()
    base = df.loc[idx].copy()

    # 6 · selección dinámica de columnas
    cols = ["number","team","lap_time","GapStart","top_speed"]
    if include_sectors:
        cols += [c for c in ("Sector1","Sector2","Sector3") if c in base.columns]
    if include_sector_gaps:
        cols += [c for c in ("GapAhead_S1","GapAhead_S2","GapAhead_S3") if c in base.columns]
    base = base[cols].rename(columns={"lap_time":"BestLap"})

    # 7 · ordenar según df_class o por BestLap
    if df_class is not None and {"position","number"} <= set(df_class.columns):
        order = (
            df_class.dropna(subset=["number","position"])
                    .sort_values("position")["number"]
                    .astype(str)
                    .tolist()
        )
        base["number"] = base["number"].astype(str)
        base = (
            base.set_index("number")
                .reindex(order)
                .dropna(subset=["BestLap"])
                .reset_index()
        )
    else:
        base = base.sort_values("BestLap").reset_index(drop=True)

    # 8 · **ahora** filtramos SOLO para mostrar los teams seleccionados
    if teams is not None:
        base = base[base["team"].isin(teams)].reset_index(drop=True)

    return base

def slipstream_gap_gain(
    df: pd.DataFrame,
    *,
    gap_col: str = "GapAhead",          # o "GapAhead_S3"
    fast_threshold: float = 0.10,       # ← ⚠ % sobre la best-lap aceptado
    bins: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.5, 5.0, float("inf")),
    ref_gap: float = 5.0,
    min_laps: int = 3,
) -> pd.DataFrame:
    """
    Ganancia media de lap-time según el *gap* al coche precedente,
    calculada **solo con vueltas rápidas**.

    • Se consideran rápidas las vueltas con
      lap_time ≤ (1+fast_threshold) × best_lap del mismo número.
    • Para cada número se toma como referencia la mediana de sus
      vueltas con gap ≥ ref_gap (≈ sin rebufo).
    • Δt = ref_lap − lap_time  →  positivo = tiempo ganado.
    """

    # ── 0 · comprobaciones mínimas ───────────────────────────────────
    if {"number", "lap_time", gap_col} - set(df.columns):
        return pd.DataFrame()

    data = df[["number", "lap_time", gap_col]].copy()

    if not pd.api.types.is_numeric_dtype(data["lap_time"]):
        data["lap_time"] = data["lap_time"].apply(parse_time_to_seconds)
    if not pd.api.types.is_numeric_dtype(data[gap_col]):
        data[gap_col] = pd.to_numeric(data[gap_col], errors="coerce")

    data = data.dropna(subset=["lap_time", gap_col])

    # ── 1 · filtrar sólo vueltas rápidas ─────────────────────────────
    best = data.groupby("number")["lap_time"].transform("min")
    data = data[data["lap_time"] <= best * (1 + fast_threshold)]

    # ── 2 · referencia “sin rebufo” y Δt ─────────────────────────────
    ref = (
        data.loc[data[gap_col] >= ref_gap]
            .groupby("number")["lap_time"]
            .median()
            .rename("ref_lap")
    )
    data = data.join(ref, on="number")
    data = data.dropna(subset=["ref_lap"])     # coches sin vueltas “sin rebufo”

    data["delta"] = data["lap_time"] - data["ref_lap"] 
    data["gap_bin"] = pd.cut(data[gap_col], bins=bins, right=False)

    # ── 3 · estadísticos por bin ─────────────────────────────────────
    result = (
        data.groupby("gap_bin")
            .agg(
                mean_gain=("delta", "mean"),
                std_gain=("delta", "std"),
                laps=("delta", "count"),
            )
            .query("laps >= @min_laps")
            .reset_index()
    )
    result["gap_range"] = result["gap_bin"].astype(str)
    return result[["gap_range", "laps", "mean_gain", "std_gain"]]

def slipstream_sector_gap_gain(
    df: pd.DataFrame,
    *,
    bins: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.5, 5.0, float("inf")),
    ref_gap: float = 5.0,
    fast_threshold: float = 0.10,   # ≤ 110 % de la best-lap
    min_laps: int = 3,
) -> pd.DataFrame:
    """
    Ganancia media de tiempo de sector vs gap del mismo sector.
    Devuelve: sector • gap_range • laps • mean_gain • std_gain
    """

    req = {
        "number", "lap_time",
        "Sector1", "Sector2", "Sector3",
        "GapAhead_S1", "GapAhead_S2", "GapAhead_S3",
    }
    if req - set(df.columns):
        return pd.DataFrame()

    data = df.copy()

    # ── asegurar todos los tiempos en segundos ──────────────────
    for col in ["lap_time", "Sector1", "Sector2", "Sector3"]:
        if not pd.api.types.is_numeric_dtype(data[col]):
            data[col] = data[col].apply(parse_time_to_seconds)

    # ── sólo vueltas rápidas ────────────────────────────────────
    best = data.groupby("number")["lap_time"].transform("min")
    data = data[data["lap_time"] <= best * (1 + fast_threshold)]

    # ── helper por sector ───────────────────────────────────────
    def _sector_gain(sec: str, gap: str) -> pd.DataFrame:
        # ── 1· referencia “sin rebufo” de cada piloto ───────────────────
        ref = (
            data.loc[data[gap] >= ref_gap]
                .groupby("number")[sec]
                .median()
                .rename("ref")
        )

        # ── 2· Δsector-time y bin de gap ────────────────────────────────
        d = data.join(ref, on="number").dropna(subset=["ref"])          # elimina pilotos sin ref
        d["delta"]   = d[sec] - d["ref"] 
        d["gap_bin"] = pd.cut(d[gap], bins=bins, right=False)

        # ── 3· estadísticos por bin ─────────────────────────────────────
        out = (
            d.groupby("gap_bin")["delta"]
              .agg(mean_gain="mean", std_gain="std", laps="count")
              .reset_index()
        )
        out = out[out["laps"] >= min_laps]                              # exige muestras mínimas

        # ── 4· orden numérico de los intervalos ─────────────────────────
        cat_order = [str(pd.Interval(bins[i], bins[i + 1], closed="left"))
                     for i in range(len(bins) - 1)]
        out["gap_range"] = pd.Categorical(
            out["gap_bin"].astype(str),
            categories=cat_order,
            ordered=True,
        )
        out = out.sort_values("gap_range")

        # ── 5· columnas finales ─────────────────────────────────────────
        out["sector"] = sec
        return out[["sector", "gap_range", "laps", "mean_gain", "std_gain"]]

    # ── unir resultados de S1/S2/S3 ─────────────────────────────
    return pd.concat([
        _sector_gain("Sector1", "GapAhead_S1"),
        _sector_gain("Sector2", "GapAhead_S2"),
        _sector_gain("Sector3", "GapAhead_S3"),
    ], ignore_index=True)