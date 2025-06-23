## src/data_process.py
import pandas as pd
import warnings

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