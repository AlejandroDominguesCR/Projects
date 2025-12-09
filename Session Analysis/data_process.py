from __future__ import annotations

import os
import re
import warnings
from typing import Dict, List

import pandas as pd
import xml.etree.ElementTree as ET

# --- Alias de columnas de timing que pueden cambiar entre eventos ------------
# Importante: estos nombres son los NORMALIZADOS por `_read_csv`
# (minúsculas, espacios -> "_").
# --- Alias de columnas de timing que pueden cambiar entre eventos ------------
# Importante: estos nombres son los NORMALIZADOS por `_read_csv`
# (minúsculas, espacios -> "_").
# =============================================================================
# CANDIDATOS DE NOMBRES DE COLUMNA (alias por tipo lógico)
# =============================================================================

COLUMN_CANDIDATES: dict[str, list[str]] = {
    # --- PILOTO -------------------------------------------------------------
    "driver": [
        "driver",           # genérico
        "driver_name",
        "driver short name",
        "driver_shortname",
        "driver_short",
        "driver_abbname",
        "driver_code",
        "screen",           # 'CR_2', 'TEC_76', etc. en entry list
        "screen_name",
        "screenname",
        "name",             # muy común en timing
        "surname and name",
        "surname&name",
        "surname_name",
        "pilot",
        "piloto",
    ],

    # --- NÚMERO DE COCHE / DORSAL ------------------------------------------
    "number": [
        "number",
        "no",
        "nr",
        "car_number",
        "car number",
        "car no",
        "carno",
        "race_number",
        "driver_number",
        "competition_number",
        "start_no",
        "dorsal",
        "num",              # por si acaso
    ],

    # --- POSICIÓN (CLASIFICACIÓN) ------------------------------------------
    "position": [
        "position",
        "pos",
        "pos.",
        "final_position",
        "overall_position",
        "classification_position",
        "class_position",
        "class_pos",
        "rank",
        "place",
        "grid_position",    # a veces se usa en qualy
        "grid_pos",
    ],

    # --- EQUIPO -------------------------------------------------------------
    "team": [
        "team",
        "TEAM",
        "team_name",
        "team name",
        "teamname",
        "team_shortname",
        "team_short",
        "team_code",
        "race_team",
        "race team",
        "racing_team",
        "racing team",
        "ecm_team_id",

        "entrant",
        "entrant_name",
        "entrant short name",
        "entrant_shortname",
        "entrant_short",

        "competitor",
        "competitor_name",
        "competitor short name",
        "competitor_shortname",

        "escuderia",
        "écurie",   # por si algún promotor va fino con el francés
        "ecurie",
        "equipe",
        "equipo",
    ],

    # --- NÚMERO DE VUELTA ---------------------------------------------------
    "lap_number": [
        "lap_number",
        "lapnumber",
        "lap no",
        "lap_no",
        "lap",
        "lap nr",
        "n_lap",
    ],

    # --- TIEMPO DE VUELTA ---------------------------------------------------
    "lap_time": [
        "lap_time",
        "laptime",
        "lap time",
        "bestlap",
        "best_lap",
        "best lap",
        "fastest_lap",
        "fastestlap",
        "fastesttime",
        "time_lap",
        "time lap",
        "time",      # OJO: solo si el CSV es claramente "lap table"
    ],

    # --- SECTORES (S1/S2/S3 en segundos) -----------------------------------
    "s1": [
        "s1",
        "s1_seconds",
        "sector1",
        "sector_1",
        "sector 1",
        "sector_1_time",
        "sector1time",
    ],
    "s2": [
        "s2",
        "s2_seconds",
        "sector2",
        "sector_2",
        "sector 2",
        "sector_2_time",
        "sector2time",
    ],
    "s3": [
        "s3",
        "s3_seconds",
        "sector3",
        "sector_3",
        "sector 3",
        "sector_3_time",
        "sector3time",
    ],

    # --- TOP SPEED ----------------------------------------------------------
    "top_speed": [
        "top_speed",
        "topspeed",
        "top speed",
        "max_top_speed",
        "max topspeed",
        "max speed",
        "vmax",
        "vmáx",
        # genérico tipo speedtrap
        "speedtrap",
        "speed trap",
        "speed_trap",
    ],
}


def find_column(
    df: pd.DataFrame,
    logical_name: str,
    *,
    required: bool = True,
) -> str | None:
    """
    Devuelve el nombre de la columna de `df` que mejor coincide con
    `logical_name` usando el diccionario COLUMN_CANDIDATES.

    `logical_name` es un nombre lógico ("position", "number", "driver"...).
    Si `required=True` y no se encuentra ninguna, lanza KeyError.
    """

    # Columnas ya normalizadas por _read_csv
    available = set(df.columns)

    # Lista de candidatos para ese nombre lógico
    candidates = COLUMN_CANDIDATES.get(logical_name, [])
    # Nos aseguramos de probar también el propio logical_name
    if logical_name not in candidates:
        candidates = [logical_name] + candidates

    for cand in candidates:
        cand_norm = cand.strip().lower()
        if cand_norm in available:
            return cand_norm

    if required:
        raise KeyError(
            f"No se encontró columna para '{logical_name}'. "
            f"Probados: {candidates}. Disponibles: {sorted(available)}"
        )
    return None

def standardise_analysis_core_columns(
    df: pd.DataFrame,
    *,
    require_team: bool = True,
    require_driver: bool = True,
) -> pd.DataFrame:
    """
    Normaliza nombres de columnas críticos para el análisis de vueltas.

    - Usa COLUMN_CANDIDATES para localizar 'team', 'driver', 'number'.
    - Renombra al nombre canónico (team, driver, number).
    - Comprueba que exista 'lap_number' o 'lap' y lo normaliza a 'lap_number'.

    Lanza KeyError con un mensaje explicativo si falta algún campo requerido.
    """
    df = df.copy()

    # --- driver --------------------------------------------------------
    dcol = find_column(df, "driver", required=require_driver)
    if dcol and dcol != "driver":
        df = df.rename(columns={dcol: "driver"})

    # --- team ----------------------------------------------------------
    tcol = find_column(df, "team", required=require_team)
    if tcol and tcol != "team":
        df = df.rename(columns={tcol: "team"})

    # --- number (si no existe ya 'number') -----------------------------
    try:
        ncol = find_column(df, "number", required=False)
    except KeyError:
        ncol = None

    if ncol and ncol != "number" and "number" not in df.columns:
        df = df.rename(columns={ncol: "number"})

    # --- lap_number ----------------------------------------------------
    if "lap_number" not in df.columns:
        if "lap" in df.columns:
            df = df.rename(columns={"lap": "lap_number"})
        else:
            raise KeyError(
                "No se encontró columna de vuelta ('lap_number' ni 'lap') "
                f"en df_analysis. Columnas disponibles: {sorted(df.columns)}"
            )

    return df

def get_position_col(df: pd.DataFrame, *, required: bool = True) -> str | None:
    """Devuelve la columna de posición (position / pos / ...)."""
    return find_column(df, "position", required=required)

def get_number_col(df: pd.DataFrame, *, required: bool = True) -> str | None:
    """Devuelve la columna de número (number / nr / no / driver_number / ...)."""
    return find_column(df, "number", required=required)

def get_driver_col(df: pd.DataFrame, *, required: bool = True) -> str | None:
    """Devuelve la columna de piloto (driver_name / driver / driver_shortname / ...)."""
    return find_column(df, "driver", required=required)

def _read_csv(path: str) -> pd.DataFrame:
    """Read a CSV file (auto-separator) and normalise columns."""

    df = pd.read_csv(
        path,
        sep=None,              
        engine="python",
        encoding="utf-8-sig",
    )

    # Drop unnamed columns that usually appear due to trailing separators
    df = df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")], errors="ignore")

    # Normalise column names: strip spaces, replace spaces by underscore and
    # lower case everything
    df.columns = [re.sub(r"\s+", "_", c.strip()).lower() for c in df.columns]
    return df

def _safe_int(x):
    try:
        if x is None:
            return None
        return int(str(x).strip())
    except (TypeError, ValueError):
        return None

def _safe_float(x):
    try:
        if x is None:
            return None
        s = str(x).strip().replace(",", ".")
        return float(s)
    except (TypeError, ValueError):
        return None

def load_session_data(folder_path: str) -> Dict[str, pd.DataFrame]:
    """
    Carga todos los CSV y, si existen, los XML de timing de la sesión.

    - CSV: se comporta como siempre -> name -> DataFrame (raw)
    - Si el CSV tiene formato Jerez, se genera además:
        name + "_analysis" -> DataFrame por vuelta (convert_jerez_timing)
    - XML:
        * Report (T01_Timing_1)     -> key "classification_xml"
        * SectorAnalysis (T01_Timing_2) -> key "analysis_xml"
    """

    data: Dict[str, pd.DataFrame] = {}

    # 1) CSVs
    for fname in os.listdir(folder_path):
        path = os.path.join(folder_path, fname)
        if not os.path.isfile(path):
            continue
        if not fname.lower().endswith(".csv"):
            continue
        path = os.path.join(folder_path, fname)
        key = os.path.splitext(fname)[0]
        data[key] = _read_csv(path)
    return data

def export_raw_session(folder_path: str, output_path: str) -> None:
    """Combine all session CSVs into a single Excel workbook."""

    data = load_session_data(folder_path)
    with pd.ExcelWriter(output_path) as writer:
        for name, df in data.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)

def parse_time_to_seconds(value: str | float | int) -> float:
    """Parse a ``mm:ss.xxx`` or ``hh:mm:ss.xxx`` time string to seconds."""

    if pd.isna(value):
        return float("nan")
    text = str(value).strip().replace(",", ".")

    match = re.match(r"^(\d+):(\d{2})\.(\d+)$", text)
    if match:
        minutes, seconds, millis = match.groups()
        return int(minutes) * 60 + float(f"{seconds}.{millis}")

    match = re.match(r"^(\d+):(\d{2}):(\d{2})\.(\d+)$", text)
    if match:
        hours, minutes, seconds, millis = match.groups()
        total = int(hours) * 3600
        total += int(minutes) * 60
        total += float(f"{seconds}.{millis}")
        return total

    try:
        return float(text)
    except ValueError:
        return float("nan")

def convert_jerez_timing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adaptador específico para los CSV de timing tipo Jerez
    (Lap / IP / Time Lap / Time Sector FL->IP / Time Sector -> Sector).

    Recibe el DataFrame tal y como sale de `_read_csv`
    (columnas en minúsculas y con '_' en vez de espacios) y devuelve
    un DataFrame "analysis" estándar por vuelta con columnas:
      - driver_name
      - driver_number
      - team              (si no hay info, se deja vacío)
      - lap_number
      - lap_time          (string m:ss.xxx)
      - s1, s2, s3        (float, en segundos)
      - hour              (hora del paso por meta, si existe 'time')
    """

    required = {"lap", "ip", "number", "name", "time_lap", "time_sector_fl->ip"}
    if not required.issubset(df.columns):
        # No tiene la pinta del timing de Jerez → no tocamos nada
        return df

    df = df.copy()
    rows: list[dict] = []

    for (car, lap), grp in df.groupby(["number", "lap"]):
        # Fila de FL (línea de meta) de esa vuelta
        grp_fl = grp[grp["ip"].astype(str).str.upper() == "FL"]
        if grp_fl.empty:
            continue

        # Por si hubiera más de una, tomamos la última en el tiempo
        fl = grp_fl.sort_values("time").iloc[-1] if "time" in grp_fl.columns else grp_fl.iloc[-1]

        lap_time_raw = str(fl["time_lap"]).strip()
        # Saltamos PIT IN / PIT EXIT / vacíos
        if not lap_time_raw or "PIT" in lap_time_raw.upper() or lap_time_raw == "------":
            continue

        lt_sec = parse_time_to_seconds(lap_time_raw)
        if not lt_sec or pd.isna(lt_sec) or lt_sec <= 0:
            continue

        # IP1 e IP2 de esa vuelta
        ip1 = grp[grp["ip"].astype(str).str.upper() == "IP1"]
        ip2 = grp[grp["ip"].astype(str).str.upper() == "IP2"]

        s1 = s2 = s3 = float("nan")

        # --- S1: FL → IP1 -------------------------------------------------
        if not ip1.empty:
            ip1_row = ip1.sort_values("time").iloc[-1] if "time" in ip1.columns else ip1.iloc[-1]
            s1 = parse_time_to_seconds(ip1_row["time_sector_fl->ip"])

        # --- S2: IP1 → IP2 -----------------------------------------------
        if not ip2.empty:
            ip2_row = ip2.sort_values("time").iloc[-1] if "time" in ip2.columns else ip2.iloc[-1]
            t_fl_ip2 = parse_time_to_seconds(ip2_row["time_sector_fl->ip"])
            if not pd.isna(t_fl_ip2) and not pd.isna(s1):
                # definición que usamos: S2 = (FL→IP2) - (FL→IP1)
                s2 = t_fl_ip2 - s1
            else:
                # fallback: usar Time Sector -> Sector en IP2 si falta algo
                if "time_sector_->_sector" in ip2_row.index:
                    s2 = parse_time_to_seconds(ip2_row["time_sector_->_sector"])

        # --- S3: IP2 → FL siguiente --------------------------------------
        if not pd.isna(lt_sec) and not pd.isna(s1) and not pd.isna(s2):
            s3 = lt_sec - s1 - s2

        rows.append(
            {
                "driver_name": fl["name"],
                "driver_number": car,
                "team": "",  # si luego quieres, lo rellenamos con df_class
                "lap_number": lap,
                "lap_time": lap_time_raw,
                "s1": s1,
                "s2": s2,
                "s3": s3,
                "hour": str(fl["time"]) if "time" in fl.index else "",
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["driver_number", "lap_number"]).reset_index(drop=True)
    return out

def convert_time_column(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Convert ``time_col`` of `df`` to seconds if present."""

    if time_col not in df.columns:
        return df

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        df = df.copy()
        df[time_col] = df[time_col].apply(parse_time_to_seconds)

    # ✅ Vueltas inválidas (0 o negativas) se marcan como NaN
    df.loc[df[time_col] <= 0, time_col] = float("nan")

    df = df.sort_values(time_col).reset_index(drop=True)
    return df

def unify_timestamps(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    """Convert ``time_col`` to ``datetime`` if it exists and sort by it."""

    if time_col not in df.columns:
        return df

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.sort_values(time_col).reset_index(drop=True)
    return df

def build_driver_matrix(
    df_analysis: pd.DataFrame,
    tracklimits_df: pd.DataFrame | None = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Return a nested dict ``{team: {driver: laps_df}}``.

    Se asegura de que `df_analysis` tenga al menos:
      - team
      - driver
      - lap_number

    `tracklimits_df` es opcional y, cuando se proporciona, se añade
    una columna ``track_limits`` con el nº de infracciones por vuelta.
    """

    # 1) Normalizar columnas clave (team, driver, number, lap_number)
    df = standardise_analysis_core_columns(
        df_analysis,
        require_team=True,
        require_driver=True,
    )

    # Orden por driver y vuelta para tener algo consistente
    df = df.sort_values(["driver", "lap_number"]).reset_index(drop=True)

    # 2) Construir matriz {team: {driver: df_vueltas}}
    driver_matrix: Dict[str, Dict[str, pd.DataFrame]] = {}
    for (team, driver), group in df.groupby(["team", "driver"]):
        driver_matrix.setdefault(team, {})[driver] = group.reset_index(drop=True)

    # 3) Track limits (si hay DataFrame)
    if tracklimits_df is not None and not tracklimits_df.empty:
        # Normalizamos nombres básicos también aquí
        tl = tracklimits_df.rename(columns=lambda c: c.strip().lower()).copy()

        # Asegurar 'team' y 'driver' usando candidatos
        t_team = find_column(tl, "team", required=False)
        if t_team and t_team != "team":
            tl = tl.rename(columns={t_team: "team"})

        t_driver = find_column(tl, "driver", required=False)
        if t_driver and t_driver != "driver":
            tl = tl.rename(columns={t_driver: "driver"})

        # Lap: aceptamos 'lap' o 'lap_number'
        lap_col = "lap" if "lap" in tl.columns else "lap_number"
        if lap_col not in tl.columns:
            raise KeyError(
                "tracklimits_df debe contener 'lap' o 'lap_number' para poder "
                "contar infracciones por vuelta."
            )

        tl_counts = (
            tl.groupby(["team", "driver", lap_col])
              .size()
              .reset_index(name="track_limits")
        )

        # Mapear a la matriz driver_matrix
        for _, row in tl_counts.iterrows():
            team = row["team"]
            driver = row["driver"]
            lap = row[lap_col]

            if team in driver_matrix and driver in driver_matrix[team]:
                df_driver = driver_matrix[team][driver]
                mask = df_driver["lap_number"] == lap

                if "track_limits" not in df_driver.columns:
                    df_driver["track_limits"] = 0

                df_driver.loc[mask, "track_limits"] = row["track_limits"]
                driver_matrix[team][driver] = df_driver

    return driver_matrix

def seconds_to_mmss(t: float | int | str) -> str:
    """Convierte segundos (float) → 'm:ss.mmm'.  NaN/None → ''.

    Acepta floats, ints o strings numéricos. Si el valor no es convertible
    o es NaN, devuelve cadena vacía para evitar errores en las tablas.
    """
    # Caso típico: ya viene formateado tipo "1:32.123" → lo dejamos como está
    if isinstance(t, str) and ":" in t:
        return t

    try:
        t = float(t)
    except (TypeError, ValueError):
        # Si no se puede convertir a número (None, '', etc.) devolvemos vacío
        return ""

    # Manejar explícitamente NaN
    if pd.isna(t):
        return ""

    m, s = divmod(t, 60)
    return f"{int(m):d}:{s:06.3f}"

def compute_gap_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute gap-related columns if not already present.

    Parameters
    ----------
    df:
        DataFrame that should contain at least a ``hour`` column.  The
        function adds ``timestamp`` (if missing), ``GapAhead`` and, when the
        three sector times are available, ``GapAhead_S1``, ``GapAhead_S2`` and
        ``GapAhead_S3``.

    Returns
    -------
    pd.DataFrame
        The DataFrame with the computed columns.
    """

    df = df.copy()

    # ── timestamp global + GapAhead (meta) ─────────────────────────────
    if "timestamp" not in df.columns:
        if "hour" in df.columns:
            df["timestamp"] = pd.to_datetime(df["hour"], errors="coerce")
        else:
            return df

    df = df.sort_values("timestamp").reset_index(drop=True)

    if "GapAhead" not in df.columns:
        df["GapAhead"] = df["timestamp"].diff().dt.total_seconds().abs()

    # ── gaps por sector (S1/S2/S3) si existen ─────────────────────────
    need_cols = {"GapAhead_S1", "GapAhead_S2", "GapAhead_S3"}
    if need_cols - set(df.columns):
        sector_aliases = {
            "sector1": ["sector1", "s1", "sector1_seconds", "s1_seconds"],
            "sector2": ["sector2", "s2", "sector2_seconds", "s2_seconds"],
            "sector3": ["sector3", "s3", "sector3_seconds", "s3_seconds"],
        }
        sector_cols = {
            std: next((c for c in aliases if c in df.columns), None)
            for std, aliases in sector_aliases.items()
        }

        if all(sector_cols.values()):
            for col in sector_cols.values():
                if not pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].apply(parse_time_to_seconds)

            df["T3"] = df["timestamp"]
            df["T2"] = df["T3"] - pd.to_timedelta(df[sector_cols["sector3"]], unit="s")
            df["T1"] = df["T2"] - pd.to_timedelta(df[sector_cols["sector2"]], unit="s")

            df["GapAhead_S1"] = (
                df.loc[df["T1"].sort_values().index, "T1"]
                  .diff().dt.total_seconds().abs()
                  .reindex(df.index)
            )
            df["GapAhead_S2"] = (
                df.loc[df["T2"].sort_values().index, "T2"]
                  .diff().dt.total_seconds().abs()
                  .reindex(df.index)
            )
            df["GapAhead_S3"] = (
                df.loc[df["T3"].sort_values().index, "T3"]
                  .diff().dt.total_seconds().abs()
                  .reindex(df.index)
            )

    return df