## src/data_process.py
import os
import pandas as pd
import warnings
from session_io import load_session_data

def unify_timestamps(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """
    Convierte la columna de tiempo a datetime y ordena por ésta.
    Suprime warnings de pandas cuando no infiere el formato.
    Si no existe la columna, devuelve el DataFrame intacto.
    """
    col = time_col.lower()
    if col not in df.columns:
        return df
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        df[col] = pd.to_datetime(
            df[col],
            errors='coerce'
        )
    df = df.sort_values(col)
    return df.reset_index(drop=True)


def process_session(
    folder: str,
    output_root: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "processed"),
) -> None:
    """Procesa una sola carpeta de sesión y guarda los CSV normalizados.

    - Lee los CSVs de ``folder``.
    - Aplica ``load_session_data`` y ``unify_timestamps``.
    - Guarda los DataFrames en ``output_root/<nombre carpeta>``.
    """
    folder = os.path.abspath(folder)
    output_root = os.path.abspath(output_root)
    if not os.path.isdir(folder):
        raise FileNotFoundError(folder)
    session_name = os.path.basename(os.path.normpath(folder))
    out_folder = os.path.join(output_root, session_name)
    os.makedirs(out_folder, exist_ok=True)
    data = load_session_data(folder)
    for key, df in data.items():
        df = unify_timestamps(df, "time")
        out_path = os.path.join(out_folder, f"{key}.parquet")
        df.to_parquet(out_path)


def process_all_sessions(
    data_root: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Data"),
    output_root: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "processed"),
) -> None:
    """Procesa todas las carpetas de sesiones y guarda los CSV normalizados.

    - Itera sobre subdirectorios dentro de ``data_root``.
    - Para cada CSV se aplican ``load_session_data`` y ``unify_timestamps``.
    - Los DataFrames resultantes se guardan en ``output_root`` replicando la estructura.
    """
    data_root = os.path.abspath(data_root)
    output_root = os.path.abspath(output_root)
    os.makedirs(output_root, exist_ok=True)
    for name in os.listdir(data_root):
        folder = os.path.join(data_root, name)
        if not os.path.isdir(folder):
            continue
        process_session(folder, output_root)
