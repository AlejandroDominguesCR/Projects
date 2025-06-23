import os
import pandas as pd

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
            df = pd.read_csv(path, sep=';', engine='python')
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
            data[key] = df
    return data
