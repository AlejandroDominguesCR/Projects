import os
import warnings
import pandas as pd

# Mapeo de sinónimos a nombres estándar
COLUMN_MAP = {
    "lap_number": "lap",
    "time_utc_seconds": "time",
    "elapsed": "time",
    "tl_nr": "lap",
    "total_infractions": "incident",
}

def convert_lap_time(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte ``lap_time`` a segundos si la columna existe."""
    if "lap_time" in df.columns:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            td = pd.to_timedelta(df["lap_time"], errors="coerce")
        df["lap_time"] = td.dt.total_seconds()
    return df

def load_session_data(folder_path: str) -> dict:
    """
    Lee todos los CSVs de una carpeta de sesión usando ';' como separador,
    normaliza nombres de columnas y devuelve un dict de DataFrames.
    """
    data = {}
    for fname in os.listdir(folder_path):
        if fname.lower().endswith('.csv'):
            key = os.path.splitext(fname)[0]
            path = os.path.join(folder_path, fname)
            df = pd.read_csv(
                path,
                sep=';',
                engine='python',
                encoding='utf-8-sig',
            )
            # Normalizar nombres de columnas
            df.columns = (
                df.columns
                .str.strip()
                .str.lower()
                .str.replace(' ', '_', regex=False)
                .str.replace('-', '_', regex=False)
            )
            # Eliminar columnas vacías o unnamed
            df = df.loc[:, ~df.columns.str.startswith('unnamed')]

            # Renombrar columnas conocidas
            if 'total_infractions' in df.columns:
                df = df.rename(columns={'total_infractions': 'incident'})

            # Inferir stints si es posible
            df = _infer_stints(df)

            data[key] = df

    for raw, df in data.items():
        df.rename(columns=COLUMN_MAP, inplace=True)
        convert_lap_time(df)

    return data

def _infer_stints(df: pd.DataFrame) -> pd.DataFrame:
    """Infer stint number from pit stop information if missing."""
    if "stint" in df.columns:
        return df

    lap_col = "lap_number" if "lap_number" in df.columns else "lap"
    driver_col = None
    for col in ("driver", "driver_name", "driver_shortname", "driver_number"):
        if col in df.columns:
            driver_col = col
            break
    if driver_col is None or lap_col not in df.columns:
        df["stint"] = None
        return df

    pit_cols = [c for c in ("pit_time", "crossing_finish_line_in_pit") if c in df.columns]
    if not pit_cols:
        df["stint"] = None
        return df

    df = df.sort_values([driver_col, lap_col]).copy()
    df["stint"] = 1

    for driver, idx in df.groupby(driver_col).groups.items():
        stint = 1
        ordered = df.loc[idx].sort_values(lap_col).index.to_list()
        df.loc[ordered[0], "stint"] = stint
        for prev, curr in zip(ordered[:-1], ordered[1:]):
            prev_pit = False
            if "pit_time" in pit_cols:
                val = str(df.at[prev, "pit_time"]).strip()
                prev_pit |= val not in ("", "0", "0:00", "0:00:00.000")
            if not prev_pit and "crossing_finish_line_in_pit" in pit_cols:
                val = str(df.at[prev, "crossing_finish_line_in_pit"]).strip()
                prev_pit |= bool(val)
            if prev_pit:
                stint += 1
            df.at[curr, "stint"] = stint

    return df