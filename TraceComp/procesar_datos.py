import csv
import json
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

try:
    import variable_mapping as vm  # mismo directorio
except ImportError:
    vm = None  # desactiva mapeo si no se encuentra

datos_registrados_WinTAX = []
datos_registrados_Canopy = []

def procesar_archivos_wintax(
    archivos_wintax: List[str],
    header_map_overrides: Optional[Dict[str, Dict[str, str]]] = None,
    platform: str = "wintax",
    reset: bool = True,
):
    """
    Procesa una tanda de archivos WinTAX y guarda un JSON por archivo.

    Parámetros
    ----------
    archivos_wintax : list[str]
        Rutas absolutas de los CSV WinTAX a procesar.
    header_map_overrides : dict | None
        Overrides rol->header por archivo (clave = basename CSV).
    platform : str
        Etiqueta de origen; se usa en el JSON de salida ("wintax").
    reset : bool
        Si True (por defecto), limpia la lista global datos_registrados_WinTAX
        antes de procesar esta tanda, evitando contaminación de runs previos.
    """
    print("DEBUG: Starting procesar_archivos_wintax for files:", archivos_wintax)
    global datos_registrados_WinTAX

    # --- Reset global state (opcional) --------------------------------------
    if reset:
        datos_registrados_WinTAX = []  # o datos_registrados_WinTAX.clear()

    carpeta_salida = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../procesados")
    )
    os.makedirs(carpeta_salida, exist_ok=True)

    # Cargar alias_db una sola vez (si usas variable_mapping)
    try:
        import variable_mapping as vm
        alias_db = vm.load_alias_db()
    except ImportError:
        vm = None
        alias_db = None

    registros_locales = []
    all_data = []  # para varianza

    for archivo in archivos_wintax:
        print(f"Procesando archivo WinTAX: {archivo}")
        datos_registrados = registrar_datos_wintax(archivo)

        # --- Canonical mapping (si vm disponible) --------------------------
        if vm is not None:
            headers = datos_registrados.get("Variables", [])
            overrides = None
            base = os.path.basename(archivo)
            if header_map_overrides and base in header_map_overrides:
                overrides = header_map_overrides[base]

            mapping, datos_can, missing = vm.normalize_data(
                headers,
                datos_registrados["Datos"],
                units_map=None,
                alias_db=alias_db,
                overrides=overrides,
                platform=platform,
            )
            datos_registrados["Source"] = platform
            datos_registrados["HeaderMap"] = mapping
            datos_registrados["DatosCanonicos"] = datos_can
            if missing:
                datos_registrados["MissingRoles"] = missing

            # actualizar alias_db global (agrega headers confirmados)
            for role, head in mapping.items():
                if head:
                    lst = alias_db.setdefault("global", {}).setdefault(role, [])
                    if head not in lst:
                        lst.append(head)

        registros_locales.append(datos_registrados)
        all_data.append(datos_registrados["Datos"])

    # Guardar alias_db actualizado (si se usa mapeo)
    if vm is not None:
        vm.save_alias_db(alias_db)

    # --- Calcular varianzas sobre la tanda actual ---------------------------
    if not registros_locales:
        return []  # nada que hacer

    if len(registros_locales) > 1:
        variables = registros_locales[0]["Variables"]
        varianzas = {}
        for var in variables:
            # Sólo columnas que existan en todos los archivos
            series_list = []
            min_len = min(len(reg["Datos"].get(var, [])) for reg in registros_locales if var in reg["Datos"])
            for reg in registros_locales:
                vals_raw = reg["Datos"].get(var, [])[:min_len]
                vals = []
                for x in vals_raw:
                    try:
                        vals.append(float(x))
                    except Exception:
                        vals.append(np.nan)
                series_list.append(vals)
            values_matrix = np.array(series_list, dtype=float)
            varianzas[var] = np.nanvar(values_matrix, axis=0).tolist()
    else:
        # Una sola vuelta -> varianza 0.0
        reg0 = registros_locales[0]
        varianzas = {
            var: [0.0] * len(reg0["Datos"].get(var, []))
            for var in reg0["Variables"]
        }

    # --- Guardar JSON por archivo -------------------------------------------
    saved_jsons = []
    for reg, archivo in zip(registros_locales, archivos_wintax):
        reg["Varianza"] = {var: varianzas.get(var, []) for var in reg["Variables"]}
        nombre_base = os.path.basename(archivo)
        nombre_json = os.path.join(
            carpeta_salida, os.path.splitext(nombre_base)[0] + "_procesado.json"
        )
        with open(nombre_json, "w", encoding="utf-8") as json_file:
            json.dump(reg, json_file, indent=4)
        print(f"DEBUG: Datos guardados en: {nombre_json}")
        saved_jsons.append(nombre_json)

    # --- Actualizar global al final -----------------------------------------
    # (Si reset=True la global está vacía, así que simplemente extendemos)
    datos_registrados_WinTAX.extend(registros_locales)

    return saved_jsons

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

def procesar_archivos_canopy(
    archivos_canopy: List[str],
    header_map_overrides: Optional[Dict[str, Dict[str, str]]] = None,
    platform: str = "canopy",
    reset: bool = True,
):
    """
    Procesa una tanda de archivos Canopy y guarda un JSON por archivo.

    Parameters
    ----------
    archivos_canopy : list[str]
        Rutas absolutas de los CSV Canopy.
    header_map_overrides : dict | None
        Overrides rol->header por archivo (clave = basename CSV).
    platform : str
        Etiqueta de origen para el JSON ("canopy").
    reset : bool
        Si True (default), limpia datos_registrados_Canopy antes de procesar
        esta tanda para evitar contaminación de ejecuciones previas.
    """
    print("DEBUG: Starting procesar_archivos_canopy for files:", archivos_canopy)
    global datos_registrados_Canopy

    # --- Reset global state --------------------------------------------------
    if reset:
        datos_registrados_Canopy = []  # o datos_registrados_Canopy.clear()

    carpeta_salida = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../procesados")
    )
    os.makedirs(carpeta_salida, exist_ok=True)

    # Cargar alias_db una sola vez (si está disponible variable_mapping)
    try:
        import variable_mapping as vm
        alias_db = vm.load_alias_db()
    except ImportError:
        vm = None
        alias_db = None

    registros_locales: List[Dict] = []
    all_data: List[Dict[str, List]] = []

    for archivo in archivos_canopy:
        print(f"Procesando archivo Canopy: {archivo}")
        datos_registrados = registrar_datos_canopy(archivo)
        if not datos_registrados:
            print(f"DEBUG: No datos registrados para {archivo}")
            continue

        # --- Canonical mapping ------------------------------------------------
        if vm is not None:
            headers = datos_registrados.get("Variables", [])
            units_list = datos_registrados.get("Unidades", [])
            units_map = dict(zip(headers, units_list))
            overrides = None
            base = os.path.basename(archivo)
            if header_map_overrides and base in header_map_overrides:
                overrides = header_map_overrides[base]

            mapping, datos_can, missing = vm.normalize_data(
                headers,
                datos_registrados["Datos"],
                units_map=units_map,
                alias_db=alias_db,
                overrides=overrides,
                platform=platform,
            )
            datos_registrados["Source"] = platform
            datos_registrados["HeaderMap"] = mapping
            datos_registrados["DatosCanonicos"] = datos_can
            if missing:
                datos_registrados["MissingRoles"] = missing

            # Actualizar alias_db en memoria (sólo añadir nuevos encabezados)
            for role, head in mapping.items():
                if head:
                    lst = alias_db.setdefault("global", {}).setdefault(role, [])
                    if head not in lst:
                        lst.append(head)

        registros_locales.append(datos_registrados)
        all_data.append(datos_registrados["Datos"])

    # Guardar alias_db actualizado una vez
    if vm is not None:
        vm.save_alias_db(alias_db)

    # Nada procesado
    if not registros_locales:
        print("DEBUG: No Canopy files processed (registros_locales vacío).")
        return []

    # --- Calcular varianzas sobre la tanda actual ----------------------------
    # Tomamos como referencia el primer registro; sólo variables presentes en todos.
    ref_vars = registros_locales[0].get("Variables", [])
    variables = [v for v in ref_vars if all(v in reg["Datos"] for reg in registros_locales)]
    varianzas: Dict[str, List[float]] = {}

    if len(registros_locales) > 1 and variables:
        for var in variables:
            # Longitud mínima común
            min_len = min(len(reg["Datos"][var]) for reg in registros_locales)
            series_list = []
            for reg in registros_locales:
                raw_vals = reg["Datos"][var][:min_len]
                vals = []
                for x in raw_vals:
                    try:
                        vals.append(float(x))
                    except Exception:
                        vals.append(np.nan)
                series_list.append(vals)
            values_matrix = np.array(series_list, dtype=float)
            varianzas[var] = np.nanvar(values_matrix, axis=0).tolist()
    else:
        # Una sola serie (o sin intersección) -> varianza 0.0 por variable local
        reg0 = registros_locales[0]
        for var in reg0.get("Variables", []):
            varianzas[var] = [0.0] * len(reg0["Datos"].get(var, []))

    # --- Guardar JSON por archivo --------------------------------------------
    saved_jsons: List[str] = []
    for reg, archivo in zip(registros_locales, archivos_canopy):
        # Añadir varianza (usa varianzas.get(var, []) si var no en intersección)
        reg["Varianza"] = {var: varianzas.get(var, []) for var in reg.get("Variables", [])}

        nombre_base = os.path.basename(archivo)
        nombre_json = os.path.join(
            carpeta_salida, os.path.splitext(nombre_base)[0] + "_procesado.json"
        )
        with open(nombre_json, "w", encoding="utf-8") as json_file:
            json.dump(reg, json_file, indent=4)
        print(f"DEBUG: Datos guardados en: {nombre_json}")
        saved_jsons.append(nombre_json)

    # --- Actualizar global al final ------------------------------------------
    datos_registrados_Canopy.extend(registros_locales)

    return saved_jsons

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

