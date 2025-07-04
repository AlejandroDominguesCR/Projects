import dash
from dash import dcc, html, Input, Output, State
import os
import sys
import copy
import base64
import io
import json
import plotly.graph_objs as go
import numpy as np
from dash.exceptions import PreventUpdate
import procesar_datos
from Grafico_de_prueba_plotly import matrix_detailed_analysis_plotly
import webbrowser
import threading
from dash import MATCH, ALL, ctx
import dash_bootstrap_components as dbc
import plotly.io as pio
from dash import callback_context

UPLOAD_DIRECTORY = os.path.join(os.path.dirname(__file__), "uploads")
PROCESADOS_DIRECTORY = os.path.join(os.path.dirname(__file__), "../procesados")
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
os.makedirs(PROCESADOS_DIRECTORY, exist_ok=True)

# Placeholder for your data processing and plotting imports
# import procesar_datos
# import Grafico_de_prueba

def resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        return os.path.join(os.path.dirname(sys.executable), relative_path)
    return os.path.join(os.path.dirname(__file__), relative_path)

def ensure_default_conditions():
    from pathlib import Path
    config_path = Path(__file__).parent / "config_conditions.json"
    default_config = get_default_conditions()

    if not config_path.exists():
        try:
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            print("[INFO] config_conditions.json created with default conditions.")
        except Exception as e:
            print(f"[ERROR] Could not write default config: {e}")
    else:
        print("[INFO] config_conditions.json already exists, keeping existing configuration.")

def get_default_conditions():
    return {
        "mode": "percentage",
        "percentage_conditions": {
            "straight_line_points": {
                "v60": 0.6, "v75": 0.75, "vEOS": 1.0,
                "tol": 0.02, "tol_absolute": 2
            },
            "early_entry_ls": {
                "min_decel_percent": 0.9,
                "brake_threshold_percent": 0.02,
                "speed_lower_percent": 0.3,
                "speed_upper_percent": 0.6,
                "brake_lower_percent": 0.02,
                "brake_upper_percent": 0.9,
                "ax_lower_percent": 0.6,
                "ax_upper_percent": 1
            },
            "mid_corner_ls": {
                "throttle_threshold": 0.05,
                "lateral_acc_min": 0.05,
                "lateral_acc_max": 0.5,
                "long_acc_min": 0.01,
                "long_acc_max": 0.15
            },
            "exit_ls": {
                "throttle_increase_percent": 0.9,
                "longitudinal_acc_increase": 0.05,
                "lateral_acc_decrease": 0.05
            },
            "early_entry_ms": {
                "speed_lower_percent": 0.6,
                "speed_upper_percent": 0.8,
                "brake_lower_percent": 0.02,
                "brake_upper_percent": 0.6,
                "ax_lower_percent": 0.5,
                "ax_upper_percent": 0.7
            },
            "mid_corner_ms": {
                "throttle_lower_percent": 0.02,
                "throttle_upper_percent": 0.2,
                "lateral_acc_min": 0.35,
                "lateral_acc_max": 0.7,
                "long_acc_min": 0.1,
                "long_acc_max": 0.2
            },
            "exit_ms": {
                "throttle_increase_percent": 0.8,
                "longitudinal_acc_increase": 0.05,
                "lateral_acc_decrease": 0.05
            },
            "early_entry_hs": {
                "speed_lower_percent": 0.8,
                "speed_upper_percent": 1,
                "brake_upper_percent": 0.2,
                "ax_tolerance": 0.1,
                "throttle_lower_percent": 0.6,
                "throttle_upper_percent": 1
            },
            "mid_corner_hs": {
                "throttle_lower_percent": 0.5,
                "throttle_upper_percent": 1,
                "lateral_acc_min": 0.7,
                "lateral_acc_max": 1,
                "long_acc_min": -0.05,
                "long_acc_max": 0.05
            },
            "exit_hs": {
                "throttle_decrease_percent": 0.5,
                "longitudinal_acc_increase": 0.05,
                "lateral_acc_decrease": 0.05
            }
        },
        "absolute_conditions": {
            "straight_line_points": {
                "v60": 180, "v75": 225, "vEOS": 300,
                "tol": 2, "tol_absolute": 2
            },
            "early_entry_ls": {
                "speed_lower": 80,
                "speed_upper": 120,
                "brake_lower": 50,
                "brake_upper": 100,
                "ax_lower": -6,
                "ax_upper": -0.5
            },
            "mid_corner_ls": {
                "throttle_threshold": 5,
                "lateral_acc_min": 0.5,
                "lateral_acc_max": 2,
                "long_acc_min": 0.1,
                "long_acc_max": 1
            },
            "exit_ls": {
                "throttle_increase": 10,
                "longitudinal_acc_increase": 0.5,
                "lateral_acc_decrease": 0.5
            },
            "early_entry_ms": {
                "speed_lower": 130,
                "speed_upper": 180,
                "brake_lower": 5,
                "brake_upper": 20,
                "ax_lower": -4,
                "ax_upper": -2
            },
            "mid_corner_ms": {
                "throttle_lower": 2,
                "throttle_upper": 10,
                "lateral_acc_min": 1,
                "lateral_acc_max": 2.5,
                "long_acc_min": 0.2,
                "long_acc_max": 1
            },
            "exit_ms": {
                "throttle_increase": 8,
                "longitudinal_acc_increase": 0.5,
                "lateral_acc_decrease": 0.5
            },
            "early_entry_hs": {
                "speed_lower": 220,
                "speed_upper": 280,
                "brake_upper": 5,
                "ax_tolerance": 0.5,
                "throttle_lower": 20,
                "throttle_upper": 100
            },
            "mid_corner_hs": {
                "throttle_lower": 50,
                "throttle_upper": 100,
                "lateral_acc_min": 2,
                "lateral_acc_max": 3,
                "long_acc_min": -0.2,
                "long_acc_max": 0.2
            },
            "exit_hs": {
                "throttle_decrease": 20,
                "longitudinal_acc_increase": 0.5,
                "lateral_acc_decrease": 0.5
            }
        }
    }


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

overlay_controls = html.Div([
    html.Hr(),
    html.H5("Overlay de Gráficos (solapado con doble eje Y)", className="mt-4"),
    html.Div([
        dcc.Dropdown(
            id='overlay-plot-selection',
            options=[],
            multi=True,
            placeholder='Selecciona dos gráficos para solapar',
            style={'width': '60%', 'marginBottom': '10px'}
        ),
        html.Button("Overlay plots", id='overlay-btn', className="btn btn-info mb-3"),
        dcc.Graph(id='merged-figure'),
    ])], style={'marginTop': '20px'})

ensure_default_conditions()


def build_standard_tab():
    return html.Div([
        html.H1(
            "WinTAX/Canopy Data Comparison (Dash Version)",
            className="mb-4 mt-2 text-center",
        ),
        dbc.Container(
            [
                dcc.Upload(
                    id="upload-data",
                    children=html.Button(
                        "Upload up to 4 CSV Files", className="btn btn-primary mb-3"
                    ),
                    multiple=True,
                    max_size=4 * 1024 * 1024 * 10,
                ),
                html.Div(id="file-list", className="mb-2"),
                html.Div(id="file-type-selectors", className="mb-2"),
                html.Div(id="user-message", className="mb-2"),
                html.Div([
                    html.Button(
                        "Confirm File Types",
                        id="confirm-types-btn",
                        className="btn btn-success mb-3",
                    ),
                ]),
                dbc.Modal(
                    [
                        dbc.ModalHeader(dbc.ModalTitle("Edit Plot Conditions")),
                        dbc.ModalBody(id="conditions-modal-body"),
                        dbc.ModalFooter(
                            [
                                html.Button(
                                    "Save",
                                    id="save-conditions-btn",
                                    className="btn btn-success me-2",
                                ),
                                html.Button(
                                    "Cancel",
                                    id="cancel-conditions-btn",
                                    className="btn btn-secondary",
                                ),
                            ]
                        ),
                    ],
                    id="conditions-modal",
                    is_open=False,
                    style={"maxWidth": "2500px", "width": "90vw"},
                ),
                dcc.Loading(
                    id="loading-processing",
                    type="default",
                    children=[
                        dcc.Dropdown(
                            id="variable-dropdown",
                            placeholder="Select Y variable",
                            className="mb-2",
                            multi=True,
                            maxHeight=200,
                        ),
                        dcc.Dropdown(
                            id="canopy-variable-dropdown",
                            placeholder="Select equivalent Y variable for Canopy",
                            className="mb-2",
                            style={"display": "none"},
                            multi=True,
                            maxHeight=200,
                        ),
                        html.Div(id="show-conditions-area", className="mb-2"),
                        html.Div(
                            [
                                html.Button(
                                    "Export Conditions",
                                    id="export-conditions-btn",
                                    className="btn btn-warning me-2",
                                    style={"marginBottom": "20px"},
                                ),
                                dcc.Download(id="download-conditions"),
                                dcc.Upload(
                                    id="upload-conditions",
                                    children=html.Button(
                                        "Import Conditions", className="btn btn-secondary"
                                    ),
                                    multiple=False,
                                ),
                                html.Div(id="import-status-message", className="mb-3"),
                            ],
                            style={"marginBottom": "20px"},
                        ),
                        html.Button(
                            "Generate Plot",
                            id="plot-btn",
                            className="btn btn-info mb-3",
                            disabled=True,
                        ),
                        dcc.Graph(
                            id="plot-graph",
                            style={
                                "height": "600px",
                                "width": "100vw",
                                "minWidth": "1600px",
                                "marginBottom": "40px",
                            },
                        ),
                        html.Div(id="plots-and-dropdowns-block"),
                        html.Button(
                            "Add Plot",
                            id="add-plot-btn",
                            className="btn btn-secondary mb-3",
                        ),
                    ],
                ),
                dcc.Store(id="processed-files-store"),
                dcc.Store(id="plots-store", data={"figures": []}),
                dcc.Store(id="plot-state-store", data={"state": "idle"}),
                dcc.Store(id="plot-dropdowns-store", data=[]),
                dcc.Store(id="modal-just-saved", data=False),
                overlay_controls,
            ],
            fluid=True,
        ),
    ])


def build_detailed_tab():
    return dbc.Container(
        [
            dcc.Dropdown(
                id="detailed-variable-dropdown",
                placeholder="Select variable",
            ),
            dcc.Graph(id="detailed-graph"),
        ],
        fluid=True,
    )


app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    dcc.Tabs(id="main-tabs", value="standard", children=[
        dcc.Tab(label="Standard Analysis", value="standard",
                children=[html.Div(id="standard-tab-content")]),
        dcc.Tab(label="Detailed Analysis", value="detailed",
                children=[html.Div(id="detailed-tab-content")]),
    ])
])

@app.callback(
    Output('file-list', 'children'),
    Output('file-type-selectors', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)

def handle_file_upload(list_of_contents, list_of_names):
    if not list_of_contents or not list_of_names:
        raise PreventUpdate
    saved_files = []
    file_type_dropdowns = []
    for idx, (content, name) in enumerate(zip(list_of_contents, list_of_names)):
        content_type, content_string = content.split(',')
        decoded = base64.b64decode(content_string)
        file_path = os.path.join(UPLOAD_DIRECTORY, name)
        with open(file_path, 'wb') as f:
            f.write(decoded)
        saved_files.append(name)
        # Add a dropdown for each file to select its type
        file_type_dropdowns.append(html.Div([
            html.Label(f"{name} type:"),
            dcc.Dropdown(
                id={'type': 'file-type-dropdown', 'index': idx},
                options=[{'label': 'WinTAX', 'value': 'wintax'}, {'label': 'Canopy', 'value': 'canopy'}],
                placeholder='Select type',
                style={'width': '200px', 'display': 'inline-block'}
            )
        ]))
    # Only update file-list and file-type-selectors
    return (
        html.Ul([html.Li(f) for f in saved_files]),
        file_type_dropdowns
    )

@app.callback(
    Output('processed-files-store', 'data'),
    Output('variable-dropdown', 'options'),
    Output('user-message', 'children'),
    Output('variable-dropdown', 'style'),
    Output('variable-dropdown', 'placeholder'),
    Output('variable-dropdown', 'value'),
    Output('canopy-variable-dropdown', 'options'),
    Output('canopy-variable-dropdown', 'style'),
    Output('canopy-variable-dropdown', 'placeholder'),
    Output('canopy-variable-dropdown', 'value'),
    Input('confirm-types-btn', 'n_clicks'),
    State('upload-data', 'filename'),
    State({'type': 'file-type-dropdown', 'index': ALL}, 'value')
)

def process_files(n_clicks, filenames, file_types):
    if not n_clicks or not filenames or not file_types or len(filenames) != len(file_types):
        raise PreventUpdate
    import procesar_datos
    wintax_files = []
    canopy_files = []
    processed_jsons = []
    already_processed = []
    for name, ftype in zip(filenames, file_types):
        file_path = os.path.join(UPLOAD_DIRECTORY, name)
        base = os.path.splitext(os.path.basename(name))[0] + '_procesado.json'
        json_path = os.path.join(PROCESADOS_DIRECTORY, base)
        if os.path.exists(json_path):
            already_processed.append(name)
            processed_jsons.append(json_path)
        else:
            if ftype == 'wintax':
                wintax_files.append(file_path)
            elif ftype == 'canopy':
                canopy_files.append(file_path)
    # Process only new files
    if wintax_files:
        procesar_datos.procesar_archivos_wintax(wintax_files)
        for name in wintax_files:
            base = os.path.splitext(os.path.basename(name))[0] + '_procesado.json'
            processed_jsons.append(os.path.join(PROCESADOS_DIRECTORY, base))
    if canopy_files:
        procesar_datos.procesar_archivos_canopy(canopy_files)
        for name in canopy_files:
            base = os.path.splitext(os.path.basename(name))[0] + '_procesado.json'
            processed_jsons.append(os.path.join(PROCESADOS_DIRECTORY, base))
    # Extract variables from the first processed file of each type
    wintax_vars = []
    canopy_vars = []
    for json_path in processed_jsons:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if 'CarSpeed' in data['Datos']:
                wintax_vars = data.get('Variables', [])
            elif 'vCar' in data['Datos']:
                canopy_vars = data.get('Variables', [])
    # UI logic
    both_types = bool(wintax_vars) and bool(canopy_vars)
    # Prepare user message
    if already_processed and (wintax_files or canopy_files):
        msg = dbc.Alert([
            html.Strong("Warning: "),
            f"The following files were already processed and were not processed again: {', '.join(already_processed)}. "
            "Other files were processed successfully."
        ], color="warning", dismissable=True)
    elif already_processed:
        msg = dbc.Alert([
            html.Strong("Warning: "),
            f"All selected files were already processed: {', '.join(already_processed)}."
        ], color="warning", dismissable=True)
    else:
        msg = dbc.Alert("Files processed successfully!", color="success", dismissable=True)
    # Output logic for dropdowns
    if both_types:
        # Show two dropdowns, both enabled, no value selected
        return (
            processed_jsons,
            [{'label': v, 'value': v} for v in wintax_vars],
            msg,
            {'display': 'block'},
            'Select Y variable for WinTAX',
            None,
            [{'label': v, 'value': v} for v in canopy_vars],
            {'display': 'block'},
            'Select equivalent Y variable for Canopy',
            None
        )
    elif wintax_vars:
        # Only WinTAX files
        return (
            processed_jsons,
            [{'label': v, 'value': v} for v in wintax_vars],
            msg,
            {'display': 'block'},
            'Select Y variable',
            None,
            [],
            {'display': 'none'},
            '',
            None
        )
    elif canopy_vars:
        # Only Canopy files
        return (
            processed_jsons,
            [],  # No options for WinTAX dropdown
            msg,
            {'display': 'none'},  # Hide WinTAX dropdown
            '',                   # Placeholder for WinTAX dropdown
            None,
            [{'label': v, 'value': v} for v in canopy_vars],  # Canopy dropdown options
            {'display': 'block'},  # Show Canopy dropdown
            'Select Y variable for Canopy',
            None
        )
    else:
        # No variables found
        return (
            processed_jsons,
            [],
            msg,
            {'display': 'block'},
            'No Y variables found',
            None,
            [],
            {'display': 'none'},
            '',
            None
        )

@app.callback(
    Output("download-conditions", "data"),
    Input("export-conditions-btn", "n_clicks"),
    prevent_initial_call=True,
)

def exportar_condiciones(n_clicks):
    if not n_clicks:
        raise PreventUpdate

    config_path = resource_path('config_conditions.json')
    
    try:
        with open(config_path, "r") as f:
            current_config = json.load(f)
    except Exception:
        current_config = {}

    # Completar con defaults
    defaults = get_default_conditions()
    for mode in ["percentage_conditions", "absolute_conditions"]:
        if mode not in current_config:
            current_config[mode] = defaults[mode]
        else:
            for section, default_vals in defaults[mode].items():
                if section not in current_config[mode]:
                    current_config[mode][section] = default_vals
                else:
                    for key, val in default_vals.items():
                        if key not in current_config[mode][section]:
                            current_config[mode][section][key] = val

    # Incluir clave `mode` si no está
    if "mode" not in current_config:
        current_config["mode"] = defaults["mode"]

    return dcc.send_string(json.dumps(current_config, indent=2), filename="exported_conditions.json")

@app.callback(
    Output('conditions-modal-body', 'children', allow_duplicate=True),
    Output('conditions-modal', 'is_open', allow_duplicate=True),
    Output('show-conditions-area', 'children', allow_duplicate=True),
    Input('upload-conditions', 'contents'),
    prevent_initial_call=True
)

def import_conditions(contents):
    if not contents:
        raise PreventUpdate

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string).decode('utf-8')

    try:
        imported_config = json.loads(decoded)
        print("[DEBUG] Condiciones importadas correctamente:")
        print(json.dumps(imported_config, indent=2))
    except Exception as e:
        print(f"[ERROR] Fallo al importar condiciones: {e}")
        return dash.no_update

    # Completar con defaults
    defaults = get_default_conditions()
    for mode in ["percentage_conditions", "absolute_conditions"]:
        if mode not in imported_config:
            imported_config[mode] = defaults[mode]
        else:
            for section, default_vals in defaults[mode].items():
                if section not in imported_config[mode]:
                    imported_config[mode][section] = default_vals
                else:
                    for key, val in default_vals.items():
                        if key not in imported_config[mode][section]:
                            imported_config[mode][section][key] = val

    if "mode" not in imported_config:
        imported_config["mode"] = defaults["mode"]

    # Guardar al archivo oficial
    config_path = resource_path('config_conditions.json')
    with open(config_path, 'w') as f:
        json.dump(imported_config, f, indent=2)

    updated_body = build_modal_body(imported_config)
    updated_show_area = build_show_conditions(imported_config)
    return updated_body, True, updated_show_area

@app.callback(
    Output('conditions-modal', 'is_open'),
    Output('conditions-modal-body', 'children'),
    Input('show-conditions-btn', 'n_clicks'),
    Input('cancel-conditions-btn', 'n_clicks'),
    State('conditions-modal', 'is_open'),
    State('modal-just-saved', 'data'),  
    prevent_initial_call=True
)

def toggle_conditions_modal(show_click, cancel_click, is_open, modal_just_saved):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    trigger = ctx.triggered[0]['prop_id'].split('.')[0]

    if modal_just_saved:
        return False, dash.no_update  # Evita reapertura tras guardar

    if trigger == 'show-conditions-btn':
        config_path = resource_path('config_conditions.json')
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except Exception:
            config = {}
        print("[DEBUG] Modal abierto con condiciones activas:")
        print(json.dumps(config, indent=2))

        # Mode selector
        mode = config.get('mode', 'percentage')
        mode_selector = html.Div([
            html.Label('Condition Mode:', style={'fontWeight': 'bold', 'marginRight': '10px'}),
            dcc.RadioItems(
                id='condition-mode-radio',
                options=[
                    {'label': 'Percentage', 'value': 'percentage'},
                    {'label': 'Absolute', 'value': 'absolute'}
                ],
                value=mode,
                labelStyle={'display': 'inline-block', 'marginRight': '20px'}
            )
        ], style={'marginBottom': '40px'})

        # Get conditions
        cond_key = f"{mode}_conditions"
        conds = config.get(cond_key, {})

        # Build editable inputs
        fields_list = []
        for section, params in conds.items():
            if not section or not isinstance(params, dict):
                continue
            fields_list.append(html.H5(str(section).replace('_', ' ').title()))
            for key, value in params.items():
                if key is None or value is None:
                    continue
                fields_list.append(
                    dbc.InputGroup([
                        dbc.InputGroupText(str(key)),
                        dbc.Input(
                            id={'type': 'condition-input', 'section': section, 'key': key},
                            type='number',
                            value=value,
                            step=0.01,
                            style={'minWidth': '120px', 'width': '200px'}
                        )
                    ], className='mb-2')
                )

        fields = html.Div(fields_list, style={'maxWidth': '1800px', 'width': '98vw'})
        return True, [mode_selector, fields]

    elif trigger == 'cancel-conditions-btn':
        return False, dash.no_update

    return is_open, dash.no_update

@app.callback(
    Output('conditions-modal-body', 'children', allow_duplicate=True),
    Input('condition-mode-radio', 'value'),
    prevent_initial_call=True
)

def update_conditions_mode(selected_mode):
    import os
    import json

    config_path = resource_path('config_conditions.json')

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except Exception:
        config = {}

    config['mode'] = selected_mode

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    # --- reconstrucción del modal ---
    mode_selector = html.Div([
        html.Label('Condition Mode:', style={'fontWeight': 'bold', 'marginRight': '10px'}),
        dcc.RadioItems(
            id='condition-mode-radio',
            options=[
                {'label': 'Percentage', 'value': 'percentage'},
                {'label': 'Absolute', 'value': 'absolute'}
            ],
            value=selected_mode,
            labelStyle={'display': 'inline-block', 'marginRight': '20px'}
        )
    ], style={'marginBottom': '20px'})

    cond_key = f"{selected_mode}_conditions"
    conds = config.get(cond_key, {})
    sections = list(conds.items())
    mid = (len(sections) + 1) // 2
    col1 = []
    col2 = []

    for i, (section, params) in enumerate(sections):
        if not section or not isinstance(params, dict):
            continue
        target_col = col1 if i < mid else col2
        target_col.append(html.H5(str(section).replace('_', ' ').title()))
        for key, value in params.items():
            if key is None or value is None:
                continue
            target_col.append(
                dbc.InputGroup([
                    dbc.InputGroupText(str(key)),
                    dbc.Input(
                        id={'type': 'condition-input', 'section': section, 'key': key},
                        type='number',
                        value=value,
                        step=0.01,
                        style={'minWidth': '120px', 'width': '200px'}
                    )
                ], className='mb-2')
            )

    fields = dbc.Row([
        dbc.Col(col1, width=6),
        dbc.Col(col2, width=6)
    ], style={'gap': '40px'})

    return [mode_selector, fields]

def build_modal_body(config):
    mode = config.get('mode', 'percentage')
    cond_key = f"{mode}_conditions"
    conds = config.get(cond_key, {})

    mode_selector = html.Div([
        html.Label('Condition Mode:', style={'fontWeight': 'bold', 'marginRight': '10px'}),
        dcc.RadioItems(
            id='condition-mode-radio',
            options=[{'label': 'Percentage', 'value': 'percentage'}, {'label': 'Absolute', 'value': 'absolute'}],
            value=mode,
            labelStyle={'display': 'inline-block', 'marginRight': '20px'}
        )
    ], style={'marginBottom': '20px'})

    sections = list(conds.items())
    mid = (len(sections) + 1) // 2
    col1, col2 = [], []

    for i, (section, params) in enumerate(sections):
        if not isinstance(params, dict):
            continue
        target = col1 if i < mid else col2
        target.append(html.H5(section.replace('_', ' ').title()))
        for key, val in params.items():
            target.append(
                dbc.InputGroup([
                    dbc.InputGroupText(key),
                    dbc.Input(
                        id={'type': 'condition-input', 'section': section, 'key': key},
                        type='number',
                        value=val,
                        step=0.01,
                        style={'minWidth': '120px', 'width': '200px'}
                    )
                ], className='mb-2')
            )

    fields = dbc.Row([dbc.Col(col1, width=6), dbc.Col(col2, width=6)], style={'gap': '40px'})
    return [mode_selector, fields]

def build_show_conditions(config):
    mode = config.get('mode', 'percentage')
    cond_key = f"{mode}_conditions"
    conds = config.get(cond_key, {})

    phase_map = {
        'Straight Line': ['straight_line_points'],
        'Low Speed': ['early_entry_ls', 'mid_corner_ls', 'exit_ls'],
        'Medium Speed': ['early_entry_ms', 'mid_corner_ms', 'exit_ms'],
        'High Speed': ['early_entry_hs', 'mid_corner_hs', 'exit_hs']
    }

    columns = []
    for phase, keys in phase_map.items():
        col_items = [html.H6(phase, style={"textAlign": "center"})]
        for key in keys:
            params = conds.get(key, {})
            if not isinstance(params, dict):
                continue
            col_items.append(html.B(key.replace('_', ' ').title()))
            for k, v in params.items():
                if k is not None and v is not None:
                    col_items.append(html.Div(f"{k}: {v}", style={"marginLeft": "10px", "fontSize": "90%"}))
        columns.append(html.Div(col_items, style={
            "width": "24%", "display": "inline-block", "verticalAlign": "top", "padding": "0 1%"
        }))

    return html.Div([
        html.H5("Current Plot Conditions:"),
        html.Div(columns, style={"width": "100%", "display": "flex", "justifyContent": "space-between"}),
        html.Button('Edit Conditions', id='show-conditions-btn', className="btn btn-warning me-2"),
        html.Button('Reset to Default', id='reset-conditions-btn', className="btn btn-danger")
    ], style={"border": "1px solid #ccc", "padding": "15px", "borderRadius": "8px", "background": "#f9f9f9"})

@app.callback(
    Output('show-conditions-area', 'children', allow_duplicate=True),
    Output('conditions-modal-body', 'children', allow_duplicate=True),
    Output('conditions-modal', 'is_open', allow_duplicate=True),
    Input('reset-conditions-btn', 'n_clicks'),
    prevent_initial_call=True
)
def reset_conditions_to_default(n_clicks):
    if not n_clicks:
        raise PreventUpdate

    ensure_default_conditions()

    # Leer condiciones por defecto recién escritas
    config = get_default_conditions()

    print("[INFO] Condiciones reseteadas a los valores por defecto.")

    updated_show_area = build_show_conditions(config)
    updated_modal_body = build_modal_body(config)

    return updated_show_area, updated_modal_body, True  # Abrimos modal también si se desea

@app.callback(
    Output('conditions-modal', 'is_open', allow_duplicate=True),
    Output('conditions-modal-body', 'children', allow_duplicate=True),
    Output('show-conditions-area', 'children', allow_duplicate=True),
    Output('modal-just-saved', 'data'),  # Cuarto output
    Input('save-conditions-btn', 'n_clicks'),
    State({'type': 'condition-input', 'section': ALL, 'key': ALL}, 'value'),
    prevent_initial_call=True,
)
def save_conditions(n_clicks, values):
    if not n_clicks:
        raise PreventUpdate

    flat_states = []
    for group in callback_context.states_list:
        flat_states.extend(group)

    new_conditions = {}
    for state in flat_states:
        section = state['id']['section']
        key = state['id']['key']
        value = state['value']
        if section is None or key is None or value is None:
            continue
        new_conditions.setdefault(section, {})[key] = value

    print("[DEBUG] User applied changes to conditions:")
    print(json.dumps(new_conditions, indent=2))

    # Cargar config base con defaults
    config = get_default_conditions()

    # Recuperar el modo actual (por defecto 'percentage')
    config_path = resource_path('config_conditions.json')
    try:
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
            mode = saved_config.get('mode', 'percentage')
    except Exception:
        mode = 'percentage'

    config['mode'] = mode
    cond_key = f"{mode}_conditions"

    # Actualizar solo valores editados, manteniendo estructura base
    for section, values_dict in new_conditions.items():
        if section in config[cond_key]:
            config[cond_key][section].update(values_dict)
        else:
            config[cond_key][section] = values_dict

    # Guardar archivo completo con estructura correcta
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print("[DEBUG] Full config after save:")
    print(json.dumps(config, indent=2))

    updated_modal_body = build_modal_body(config)
    updated_show_area = build_show_conditions(config)

    return False, updated_modal_body, updated_show_area, True

def generate_plots(n_clicks, processed_files, wintax_vars, canopy_vars):
    if not n_clicks or not processed_files:
        raise PreventUpdate

    import Grafico_de_prueba_plotly
    import json

    # Cargar condiciones activas
    config_path = resource_path('config_conditions.json')
    try:
        with open(config_path, 'r') as f:
            condiciones_activas = json.load(f)
    except Exception:
        condiciones_activas = {}

    # Identificar los JSONs
    def is_wintax_json(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return 'CarSpeed' in data.get('Datos', {})
        except Exception:
            return False

    def is_canopy_json(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return 'vCar' in data.get('Datos', {})
        except Exception:
            return False

    wintax_jsons = [f for f in processed_files if os.path.exists(f) and is_wintax_json(f)]
    canopy_jsons = [f for f in processed_files if os.path.exists(f) and is_canopy_json(f)]

    print('Processed files:', processed_files)
    print('Selected WinTAX Y variables:', wintax_vars)
    print('Selected Canopy Y variables:', canopy_vars)
    print('Detected WinTAX JSONs:', wintax_jsons)
    print('Detected Canopy JSONs:', canopy_jsons)

    if wintax_jsons and canopy_jsons:
        fig = Grafico_de_prueba_plotly.main_comparacion_plotly(
            wintax_jsons,
            canopy_jsons,
            wintax_vars[0],
            canopy_vars[0]
        )
        y_labels = [wintax_vars[0], canopy_vars[0]]
    elif wintax_jsons:
        fig = Grafico_de_prueba_plotly.main_multiple_archivos_plotly(
            wintax_jsons,
            wintax_vars[0]
        )
        y_labels = wintax_vars
    elif canopy_jsons:
        fig = Grafico_de_prueba_plotly.main_multiple_archivos_plotly(
            canopy_jsons,
            canopy_vars[0]
        )
        y_labels = canopy_vars
    else:
        print('No valid files or variables for plotting.')
        return go.Figure(), {'figures': []}

    serialized = [{'fig': fig.to_plotly_json(), 'y_label': label} for label in y_labels]
    return fig, {'figures': serialized}

@app.callback(
    Output('show-conditions-area', 'children'),
    Input('variable-dropdown', 'value'),
    Input('canopy-variable-dropdown', 'value'),
    State('variable-dropdown', 'options'),
    State('canopy-variable-dropdown', 'options'),
    prevent_initial_call=True
)
def update_show_conditions(wintax_vars, canopy_vars, wintax_opts, canopy_opts):
    config_path = resource_path('config_conditions.json')
    try:
        with open(config_path, 'r') as f:
            condiciones_activas = json.load(f)
    except Exception:
        condiciones_activas = {}
    return show_conditions(condiciones_activas, wintax_vars, canopy_vars, wintax_opts, canopy_opts)

@app.callback(
    Output('plot-graph', 'figure'),
    Output('plots-store', 'data', allow_duplicate=True),
    Input('plot-btn', 'n_clicks'),
    State('processed-files-store', 'data'),
    State('variable-dropdown', 'value'),
    State('canopy-variable-dropdown', 'value'),
    State('plots-store', 'data'), 
    prevent_initial_call=True
)

def trigger_generate_plot(n_clicks, processed_files, wintax_vars, canopy_vars, current_store):
    fig, new_data = generate_plots(n_clicks, processed_files, wintax_vars, canopy_vars)
    
    if current_store is None:
        current_store = {"figures": []}
    
    # Añadir sin sobrescribir lo anterior
    updated_store = copy.deepcopy(current_store)
    current_store["figures"] += new_data["figures"]

    return fig, current_store


def show_conditions(condiciones_activas, wintax_vars, canopy_vars, wintax_opts, canopy_opts):
    # Validación de listas seleccionadas
    if wintax_opts and canopy_opts and wintax_opts != [] and canopy_opts != []:
        if not wintax_vars or not canopy_vars or len(wintax_vars) != len(canopy_vars):
            return None
    elif wintax_opts and wintax_opts != []:
        if not wintax_vars:
            return None
    elif canopy_opts and canopy_opts != []:
        if not canopy_vars:
            return None

    config = condiciones_activas or {}
    mode = config.get('mode', 'percentage')
    cond_key = f"{mode}_conditions"
    conds = config.get(cond_key, {})

    phase_map = {
        'Straight Line': ['straight_line_points'],
        'Low Speed': ['early_entry_ls', 'mid_corner_ls', 'exit_ls'],
        'Medium Speed': ['early_entry_ms', 'mid_corner_ms', 'exit_ms'],
        'High Speed': ['early_entry_hs', 'mid_corner_hs', 'exit_hs']
    }

    columns = []
    for phase, keys in phase_map.items():
        col_items = [html.H6(phase, style={"textAlign": "center"})]
        for key in keys:
            params = conds.get(key, {})
            if not isinstance(params, dict):
                continue
            col_items.append(html.B(key.replace('_', ' ').title()))
            for k, v in params.items():
                if k is not None and v is not None:
                    col_items.append(html.Div(f"{k}: {v}", style={"marginLeft": "10px", "fontSize": "90%"}))
        columns.append(html.Div(col_items, style={
            "width": "24%", "display": "inline-block", "verticalAlign": "top", "padding": "0 1%"
        }))

    return html.Div([
        html.H5("Current Plot Conditions:"),
        html.Div(columns, style={"width": "100%", "display": "flex", "justifyContent": "space-between"}),
        html.Button('Edit Conditions', id='show-conditions-btn', className="btn btn-warning mt-2")
    ], style={"border": "1px solid #ccc", "padding": "15px", "borderRadius": "8px", "background": "#f9f9f9"})


@app.callback(
    Output('plot-btn', 'disabled'),
    Input('variable-dropdown', 'value'),
    Input('canopy-variable-dropdown', 'value'),
    State('variable-dropdown', 'options'),
    State('canopy-variable-dropdown', 'options'),
    State('variable-dropdown', 'disabled'),
    State('canopy-variable-dropdown', 'disabled'),
    prevent_initial_call=True
)
def enable_plot_btn(wintax_val, canopy_val, wintax_opts, canopy_opts, wintax_disabled, canopy_disabled):
    # Solo habilitar si hay una variable seleccionada en el dropdown activo
    if not wintax_disabled and wintax_opts and wintax_val:
        return False
    if not canopy_disabled and canopy_opts and canopy_val:
        return False
    return True

@app.callback(
    Output('plots-and-dropdowns-block', 'children'),
    Output('add-plot-btn', 'style'),
    Output('plot-dropdowns-store', 'data', allow_duplicate=True),
    Output('plots-store', 'data', allow_duplicate=True),
    Input('add-plot-btn', 'n_clicks'),
    Input({'type': 'wintax-variable-dropdown', 'index': ALL}, 'value'),
    Input({'type': 'canopy-variable-dropdown', 'index': ALL}, 'value'),
    State('plot-dropdowns-store', 'data'),
    State('processed-files-store', 'data'),
    State('variable-dropdown', 'options'),
    State('canopy-variable-dropdown', 'options'),
    State('plots-store', 'data'),
    prevent_initial_call=True
)

def update_plots_and_dropdowns(n_clicks, wintax_vars, canopy_vars, dropdowns, processed_files, wintax_options, canopy_options, plots_data):
    import Grafico_de_prueba_plotly
    import json
    ctx_trigger = dash.callback_context.triggered[0]['prop_id'] if dash.callback_context.triggered else ''
    nuevas_figuras = []
    if dropdowns is None:
        dropdowns = []
    # Add new plot block if add-plot-btn was clicked
    if 'add-plot-btn' in ctx_trigger and ((wintax_options and len(wintax_options) > 0) or (canopy_options and len(canopy_options) > 0)) and len(dropdowns) < 4:
        dropdowns.append({'id': len(dropdowns), 'wintax': None, 'canopy': None})
    # Update variable selections for each plot block
    for idx, d in enumerate(dropdowns):
        if wintax_vars and idx < len(wintax_vars):
            d['wintax'] = wintax_vars[idx]
        if canopy_vars and idx < len(canopy_vars):
            d['canopy'] = canopy_vars[idx]
    blocks = []
    for idx, d in enumerate(dropdowns):
        # Plot (if variables selected)
        w_var = d.get('wintax')
        c_var = d.get('canopy')
        fig = None
        wintax_jsons = [f for f in processed_files if os.path.exists(f) and 'CarSpeed' in json.load(open(f, 'r', encoding='utf-8')).get('Datos', {})]
        canopy_jsons = [f for f in processed_files if os.path.exists(f) and 'vCar' in json.load(open(f, 'r', encoding='utf-8')).get('Datos', {})]
        if wintax_jsons and canopy_jsons:
            if w_var and c_var:
                fig = Grafico_de_prueba_plotly.main_comparacion_plotly(wintax_jsons, canopy_jsons, w_var, c_var)
            else:
                fig = None  # No se genera aún
        elif wintax_jsons and w_var:
            fig = Grafico_de_prueba_plotly.main_multiple_archivos_plotly(wintax_jsons, w_var)
        elif canopy_jsons and c_var:
            fig = Grafico_de_prueba_plotly.main_multiple_archivos_plotly(canopy_jsons, c_var)
        else:
            fig = None
        row = []
        if wintax_options:
            row.append(dcc.Dropdown(
                id={'type': 'wintax-variable-dropdown', 'index': idx},
                options=wintax_options,
                placeholder=f'Select WinTAX variable for plot {idx+1}',
                value=w_var,
                style={'marginBottom': '10px', 'width': '45%', 'display': 'inline-block', 'marginRight': '2%'},
            ))
        if canopy_options:
            row.append(dcc.Dropdown(
                id={'type': 'canopy-variable-dropdown', 'index': idx},
                options=canopy_options,
                placeholder=f'Select Canopy variable for plot {idx+1}',
                value=c_var,
                style={'marginBottom': '10px', 'width': '45%', 'display': 'inline-block'},
            ))
        dropdown_row = html.Div(row, style={'width': '100%', 'display': 'flex', 'alignItems': 'center', 'gap': '10px'})
        block = []
        if fig:
            block.append(dcc.Graph(figure=fig, style={'height': '600px', 'width': '100vw', 'minWidth': '1600px', 'marginBottom': '40px'}))
            nuevas_figuras.append(fig)
        block.append(dropdown_row)
        blocks.append(html.Div(block, style={'marginBottom': '40px'}))
    add_btn_style = {'display': 'block'} if len(dropdowns) < 4 else {'display': 'none'}
    # Obtener las figuras anteriores si existen
    existing_figures = plots_data.get("figures", []) if plots_data else []

    nuevos_items = []
    for fig, dropdown in zip(nuevas_figuras, dropdowns[-len(nuevas_figuras):]):
        y_label = dropdown.get('wintax') or dropdown.get('canopy') or "Variable desconocida"
        nuevos_items.append({'fig': fig.to_plotly_json(), 'y_label': y_label})

    # Concatenar con figuras anteriores
    all_figures = existing_figures + nuevos_items

    return blocks, add_btn_style, dropdowns, {'figures': all_figures}

@app.callback(
    Output('plot-dropdowns-store', 'data', allow_duplicate=True),
    Input('add-plot-btn', 'n_clicks'),
    State('plot-dropdowns-store', 'data'),
    State('variable-dropdown', 'options'),
    State('canopy-variable-dropdown', 'options'),
    prevent_initial_call=True
)

def add_dropdown(n_clicks, dropdowns, wintax_options, canopy_options):
    if dropdowns is None:
        dropdowns = []
    if ((wintax_options and len(wintax_options) > 0) or (canopy_options and len(canopy_options) > 0)) and len(dropdowns) < 4:
        dropdowns.append({'id': len(dropdowns)})
    return dropdowns

@app.callback(
    Output('overlay-plot-selection', 'options'),
    Input('plots-store', 'data')
)

def actualizar_dropdown_overlay(plots_data):
    if not plots_data or 'figures' not in plots_data:
        return []

    opciones = []
    for i, item in enumerate(plots_data['figures']):
        label = item.get('y_label', f'Gráfico {i+1}')
        opciones.append({'label': label, 'value': i})
    return opciones

def add_dropdown(n_clicks, dropdowns, wintax_options, canopy_options):
    if dropdowns is None:
        dropdowns = []
    if ((wintax_options and len(wintax_options) > 0) or (canopy_options and len(canopy_options) > 0)) and len(dropdowns) < 4:
        dropdowns.append({'id': len(dropdowns)})
    return dropdowns

@app.callback(
    Output('modal-just-saved', 'data', allow_duplicate=True),
    Input('conditions-modal', 'is_open'),
    State('modal-just-saved', 'data'),
    prevent_initial_call=True
)

def clear_modal_saved_flag(is_open, just_saved):
    if not is_open and just_saved:
        return False
    raise dash.exceptions.PreventUpdate
    return False

@app.callback(
    Output('merged-figure', 'figure'),
    Input('overlay-btn', 'n_clicks'),
    State('overlay-plot-selection', 'value'),
    State('plots-store', 'data'),
    prevent_initial_call=True
)

def generar_overlay(n_clicks, seleccionados, plots_data):
    import plotly.graph_objects as go
    from dash.exceptions import PreventUpdate

    if not seleccionados or len(seleccionados) != 2:
        raise PreventUpdate

    figuras_serializadas = plots_data.get('figures', [])
    if not figuras_serializadas or max(seleccionados) >= len(figuras_serializadas):
        raise PreventUpdate

    # Cargar figuras seleccionadas
    fig1 = go.Figure(figuras_serializadas[seleccionados[0]]['fig'])
    fig2 = go.Figure(figuras_serializadas[seleccionados[1]]['fig'])

    # Crear figura combinada
    merged_fig = go.Figure()

    for trace in fig1.data:
        trace.yaxis = 'y1'
        merged_fig.add_trace(trace)

    for trace in fig2.data:
        trace.yaxis = 'y2'
        merged_fig.add_trace(trace)

    # Configurar layout con doble eje Y
    merged_fig.update_layout(
        title='Gráfico Solapado con Ejes Y Dobles',
        height=700,
        xaxis=dict(
            title='Fase',
            tickvals=[5, 12, 20, 32, 40, 48, 60, 68, 76, 90, 98, 106],
            ticktext=[
                "0.60 Vmax", "0.75 Vmax", "vEOS", 
                "Early Entry LS", "Mid Corner LS", "Exit LS", 
                "Early Entry MS", "Mid Corner MS", "Exit MS", 
                "Early Entry HS", "Mid Corner HS", "Exit HS"
            ]
        ),
        yaxis=dict(title='Eje Y1 (Izquierda)'),
        yaxis2=dict(
            title='Eje Y2 (Derecha)',
            overlaying='y',
            side='right'
        ),
        legend=dict(orientation='h', y=-0.25)
    )
    return merged_fig


@app.callback(
    Output('standard-tab-content', 'children'),
    Output('detailed-tab-content', 'children'),
    Input('main-tabs', 'value')
)
def render_tabs(tab):
    if tab == 'standard':
        return build_standard_tab(), dash.no_update
    elif tab == 'detailed':
        return dash.no_update, build_detailed_tab()
    return dash.no_update, dash.no_update


@app.callback(
    Output('detailed-variable-dropdown', 'options'),
    Input('variable-dropdown', 'options')
)
def sync_variables(options):
    return options or []


@app.callback(
    Output('detailed-graph', 'figure'),
    Input('detailed-variable-dropdown', 'value'),
    State('processed-files-store', 'data'),
    prevent_initial_call=True
)
def plot_detailed(variable, processed_files):
    if not variable or not processed_files:
        raise PreventUpdate
    return matrix_detailed_analysis_plotly(processed_files, variable)

def open_browser():
    webbrowser.open_new("http://127.0.0.1:8050/")

if __name__ == '__main__':
    # Solo abre navegador si no está en modo reloader
    import os
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        import webbrowser
        webbrowser.open_new("http://127.0.0.1:8050")

    app.run(debug=True)