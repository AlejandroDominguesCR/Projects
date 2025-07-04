import csv
import json
import os
import pandas as pd
import numpy as np

datos_registrados_WinTAX = []
datos_registrados_Canopy = []

def procesar_archivos_wintax(archivos_wintax):
    print("DEBUG: Starting procesar_archivos_wintax for files:", archivos_wintax)  # Debug
    global datos_registrados_WinTAX  # Accede a la variable global
    carpeta_salida = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../procesados")
)
    os.makedirs(carpeta_salida, exist_ok=True)  # Crea la carpeta si no existe

    # Collect all data for variance calculation
    all_data = []
    for archivo in archivos_wintax:
        print(f"Procesando archivo WinTAX: {archivo}")
        datos_registrados = registrar_datos_wintax(archivo)
        datos_registrados_WinTAX.append(datos_registrados)
        all_data.append(datos_registrados["Datos"])

    # Compute variance for each variable at each index
    if len(all_data) > 1:
        variables = datos_registrados_WinTAX[0]["Variables"]
        varianzas = {}
        for var in variables:
            # Gather all values for this variable across files, per index
            values_matrix = []
            # 'all_data' already contains only the data dictionaries so we access
            # the variable directly
            min_len = min(len(d[var]) for d in all_data)
            for d in all_data:
                # Convert to float, pad/cut to min_len
                vals = [float(x) if str(x).replace('.', '', 1).replace('-', '', 1).isdigit() else np.nan for x in d[var][:min_len]]
                values_matrix.append(vals)
            # Transpose to get values at each index
            values_matrix = np.array(values_matrix)
            varianzas[var] = np.nanvar(values_matrix, axis=0).tolist()
    else:
        varianzas = {var: [0.0]*len(datos_registrados_WinTAX[0]["Datos"][var]) for var in datos_registrados_WinTAX[0]["Variables"]}

    # Save each file with variance
    for idx, archivo in enumerate(archivos_wintax):
        datos_registrados = datos_registrados_WinTAX[idx]
        datos_registrados["Varianza"] = {var: varianzas[var] for var in datos_registrados["Variables"]}
        nombre_base = os.path.basename(archivo)
        nombre_json = os.path.join(carpeta_salida, os.path.splitext(nombre_base)[0] + "_procesado.json")
        with open(nombre_json, "w") as json_file:
            json.dump(datos_registrados, json_file, indent=4)
        print(f"DEBUG: Datos guardados en: {nombre_json}")

def registrar_datos_wintax(ruta_archivo):
    """
    Lee un archivo CSV de WinTAX y registra sus datos en un diccionario.
    """
    datos_registrados = {
        "Variables": [],  # Lista de variables (encabezados)
        "Datos": {}       # Datos organizados por variable
    }

    with open(ruta_archivo, mode='r') as file:
        lector_csv = csv.reader(file)
        encabezados = next(lector_csv)  # Leer la primera fila como encabezados

        # Guardar los encabezados en la lista de variables
        datos_registrados["Variables"] = encabezados

        # Inicializar el diccionario de datos
        for encabezado in encabezados:
            datos_registrados["Datos"][encabezado] = []

        # Leer el resto de las filas y agregarlas al diccionario
        for fila in lector_csv:
            for idx, valor in enumerate(fila):
                datos_registrados["Datos"][encabezados[idx]].append(valor)

    return datos_registrados

def procesar_archivos_canopy(archivos_canopy):
    print("DEBUG: Starting procesar_archivos_canopy for files:", archivos_canopy)  # Debug
    global datos_registrados_Canopy  # Accede a la variable global
    carpeta_salida = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../procesados")
)
    os.makedirs(carpeta_salida, exist_ok=True)  # Crea la carpeta si no existe

    all_data = []
    for archivo in archivos_canopy:
        print(f"Procesando archivo Canopy: {archivo}")
        datos_registrados = registrar_datos_canopy(archivo)
        if datos_registrados:
            datos_registrados_Canopy.append(datos_registrados)
            all_data.append(datos_registrados["Datos"])
        else:
            print(f"DEBUG: No datos registrados para {archivo}")

    # Compute variance for each variable at each index
    if len(all_data) > 1 and all_data:
        variables = datos_registrados_Canopy[0]["Variables"]
        varianzas = {}
        for var in variables:
            min_len = min(len(d[var]) for d in all_data if var in d)
            values_matrix = []
            for d in all_data:
                vals = [float(x) if str(x).replace('.', '', 1).replace('-', '', 1).isdigit() else np.nan for x in d[var][:min_len]]
                values_matrix.append(vals)
            values_matrix = np.array(values_matrix)
            varianzas[var] = np.nanvar(values_matrix, axis=0).tolist()
    elif all_data:
        varianzas = {var: [0.0]*len(datos_registrados_Canopy[0]["Datos"][var]) for var in datos_registrados_Canopy[0]["Variables"]}
    else:
        varianzas = {}

    # Save each file with variance
    for idx, archivo in enumerate(archivos_canopy):
        if idx < len(datos_registrados_Canopy):
            datos_registrados = datos_registrados_Canopy[idx]
            datos_registrados["Varianza"] = {var: varianzas.get(var, []) for var in datos_registrados["Variables"]}
            nombre_base = os.path.basename(archivo)
            nombre_json = os.path.join(carpeta_salida, os.path.splitext(nombre_base)[0] + "_procesado.json")
            with open(nombre_json, "w") as json_file:
                json.dump(datos_registrados, json_file, indent=4)
            print(f"DEBUG: Datos guardados en: {nombre_json}")

def registrar_datos_canopy(ruta_archivo):
    """
    Procesa un archivo CSV proveniente de Canopy y extrae los datos.
    """
    try:
        valid_lines = []
        max_columns = 0
        with open(ruta_archivo, 'r', encoding='utf-8') as file:
            for line in file:
                columns = line.strip().split(',')
                if len(columns) >= 1:
                    valid_lines.append(columns)
                else:
                    columns.extend([np.nan] * (max_columns - len(columns)))
                    valid_lines.append(columns)

        valid_lines = [line + [np.nan] * (max_columns - len(line)) for line in valid_lines]
        df = pd.DataFrame(valid_lines)
        max_columns = df.apply(lambda row: len(row.dropna()), axis=1).max()

        if df.iloc[0, 0] == 'DynamicLap':
            print(f"Archivo detectado como Canopy: {ruta_archivo}")
            variable_names = df.iloc[1, :].tolist()
            units = df.iloc[2, :].tolist()
            data = df.iloc[3:, :].values
            canopy_df = pd.DataFrame(data, columns=variable_names)
            datos_registrados = {
                "Variables": variable_names,
                "Unidades": units,
                "Datos": canopy_df.dropna().to_dict(orient="list"),
            }
            return datos_registrados
        else:
            print(f"El archivo {ruta_archivo} no parece ser un archivo de Canopy válido.")
            return None

    except Exception as e:
        print(f"Error al procesar el archivo {ruta_archivo}: {e}")
        return None

def obtener_variables_separadas(datos_wintax, datos_canopy):
    """
    Obtiene listas separadas de variables a partir de los datos procesados de WinTAX y Canopy.
    """
    global datos_registrados_WinTAX, datos_registrados_Canopy  # Accede a las variables globales

    variables_wintax = set()
    variables_canopy = set()

    for dato in datos_registrados_WinTAX:
        variables_archivo = dato.get("Variables", [])
        variables_wintax.update(variables_archivo)  # Agrega las variables al conjunto de WinTAX

    for dato in datos_registrados_Canopy:
        variables_archivo = dato.get("Variables", [])
        variables_canopy.update(variables_archivo)  # Agrega las variables al conjunto de Canopy

    return list(variables_wintax), list(variables_canopy)

