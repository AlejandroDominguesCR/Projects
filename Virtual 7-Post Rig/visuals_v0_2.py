import sys
import pandas as pd
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (QApplication, QMainWindow, QFileDialog, QVBoxLayout, QPushButton, QWidget,
                             QScrollArea, QSizePolicy, QToolBar, QFrame, QLabel, QListWidget, QDialog, 
                             QListWidgetItem, QGridLayout, QComboBox, QListWidget, QMessageBox, QLineEdit, 
                             QGroupBox, QHBoxLayout, QCheckBox, QAbstractItemView)
from PyQt6.QtCore import Qt
import random

class GraphApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Data Visualization')

        self.data = None 

        self.initUI()
        self.showMaximized()

    def initUI(self):
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)

        self.mainLayout = QVBoxLayout(self.centralWidget)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.setSpacing(0)

        self.scrollArea = QScrollArea()
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setFrameShape(QFrame.Shape.NoFrame)

        self.scrollWidget = QWidget()

        self.scrollLayout = QVBoxLayout(self.scrollWidget)
        self.scrollLayout.setContentsMargins(0, 0, 0, 0)
        self.scrollLayout.setSpacing(10)

        self.widget_titles = [
            'Velocidad',
            'Aceleración',
            'Fuerza de Amortiguador',
            'Altura de Suspensión',
            'Fuerza de Neumático',
            'Ángulo de Dirección',
            'Carga Aerodinámica'
        ]

        self.graph_widgets = {}

        for i, title in enumerate(self.widget_titles):
            self.create_graph_widget(i, title)

        self.scrollWidget.setLayout(self.scrollLayout)
        self.scrollWidget.setMinimumWidth(self.scrollArea.viewport().width())

        self.scrollArea.setWidget(self.scrollWidget)

        self.mainLayout.addWidget(self.scrollArea)

        self.toolbar = QToolBar('Tools')
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.toolbar)

        loadCSVButton = QPushButton('Load Sweep')
        loadCSVButton.clicked.connect(self.load_csv)
        self.toolbar.addWidget(loadCSVButton)

    def create_graph_widget(self, index, title):
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        frame.setStyleSheet('background-color: #272727; border-radius: 10px;')
        frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        frame.setFixedHeight(800)

        frameLayout = QVBoxLayout(frame)
        frameLayout.setContentsMargins(10, 10, 10, 10)
        frameLayout.setSpacing(5)

        toolbar = QToolBar()
        toolbar.setStyleSheet('background-color: #CCCCCC; border-radius: 5px;')
        toolbar.setMovable(False)

        editButton = QPushButton('Edit Graph')
        editButton.setStyleSheet('background-color: #6c757d; color: white; border-radius: 5px; padding: 8px 20px;')
        editButton.clicked.connect(lambda: self.open_trace_selector(index))
        
        toolbar.addWidget(editButton)

        titleLabel = QLabel(title)
        titleLabel.setStyleSheet('font-size: 16px; font-weight: bold;')
        titleLabel.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        plotWidget = pg.GraphicsLayoutWidget()
        plotWidget.setBackground('w')

        gridLayout = QGridLayout()

        frameLayout.addWidget(toolbar)
        frameLayout.addWidget(titleLabel)
        frameLayout.addLayout(gridLayout)

        self.graph_widgets[index] = {
            'frame': frame,
            'frameLayout': frameLayout,
            'plotWidget': plotWidget,
            'gridLayout': gridLayout,
            'subplots': 1,
            'traces': [[]]
        }

        self.scrollLayout.addWidget(frame)


    def open_trace_selector(self, index):
        if self.data is None:
            QMessageBox.warning(self, 'Warning', 'Load Simulation Sweep before editing charts.')
            return

        dialog = QDialog(self)
        dialog.setWindowTitle('Edit Channels')
        dialog.setFixedSize(600, 500)

        mainLayout = QVBoxLayout(dialog)

        x_axis_group = QGroupBox('Select X-axis')
        x_axis_layout = QVBoxLayout()
        x_axis_combo = QComboBox()
        x_axis_combo.addItems(self.data.columns)

        prev_x_axis = self.graph_widgets[index].get('x_axis', None)
        if prev_x_axis and prev_x_axis in self.data.columns:
            x_axis_combo.setCurrentText(prev_x_axis)

        x_axis_layout.addWidget(x_axis_combo)
        x_axis_group.setLayout(x_axis_layout)
        mainLayout.addWidget(x_axis_group)

        scrollArea = QScrollArea()
        scrollArea.setWidgetResizable(True)
        mainLayout.addWidget(scrollArea)

        scrollWidget = QWidget()
        scrollLayout = QVBoxLayout(scrollWidget)
        scrollWidget.setLayout(scrollLayout)
        scrollArea.setWidget(scrollWidget)

        subplot_traces = []
        trace_types = []

        def add_trace_item(trace_name, trace_list_widget):
            listItem = QListWidgetItem(trace_name)

            removeItemButton = QPushButton('×')
            removeItemButton.setFixedSize(20, 20)
            removeItemButton.setStyleSheet('font-size: 12px; color: red;')

            def remove_selected_trace():
                row = trace_list_widget.row(listItem)
                trace_list_widget.takeItem(row)

            removeItemButton.clicked.connect(remove_selected_trace)

            rowWidget = QWidget()
            rowLayout = QHBoxLayout()
            rowLayout.setContentsMargins(0, 0, 0, 0)
            rowLayout.addStretch()
            rowLayout.addWidget(removeItemButton)
            rowWidget.setLayout(rowLayout)

            trace_list_widget.addItem(listItem)
            trace_list_widget.setItemWidget(listItem, rowWidget)

        def create_subplot_selector(subplot_index):
            groupBox = QGroupBox(f'Sub-Plot {subplot_index + 1}')
            groupBox.setFixedHeight(220)  
            groupLayout = QVBoxLayout()

            removeButton = QPushButton('×')
            removeButton.setFixedSize(30, 30)
            removeButton.setStyleSheet('font-size: 14px; font-weight: bold; color: red;')
            removeButton.clicked.connect(lambda: remove_subplot(groupBox, subplot_index))

            addTraceButton = QPushButton('+')
            addTraceButton.setStyleSheet('font-size: 14px; font-weight: bold; color: green;')
            addTraceButton.setFixedSize(30, 30)

            headerLayout = QHBoxLayout()
            headerLayout.addStretch()
            headerLayout.addWidget(addTraceButton)
            headerLayout.addWidget(removeButton)
            groupLayout.addLayout(headerLayout)

            selectedTracesList = QListWidget()
            selectedTracesList.setFixedHeight(80)

            prev_traces = self.graph_widgets[index].get('traces', [[]])
            if subplot_index < len(prev_traces):
                for trace in prev_traces[subplot_index]:
                    add_trace_item(trace, selectedTracesList)

            def open_search_dialog():
                searchDialog = QDialog()
                searchDialog.setWindowTitle('Select Channels')
                searchDialog.setFixedSize(300, 400)

                dialogLayout = QVBoxLayout(searchDialog)

                searchInput = QLineEdit()
                searchInput.setPlaceholderText('Search channel...')
                dialogLayout.addWidget(searchInput)

                availableTracesList = QListWidget()
                availableTracesList.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)

                for column in self.data.columns:
                    availableTracesList.addItem(column)

                dialogLayout.addWidget(availableTracesList)

                addButton = QPushButton('Add')
                addButton.clicked.connect(lambda: add_selected_traces(availableTracesList, searchDialog))
                dialogLayout.addWidget(addButton)

                def filter_traces():
                    filter_text = searchInput.text().lower()
                    for i in range(availableTracesList.count()):
                        item = availableTracesList.item(i)
                        item.setHidden(filter_text not in item.text().lower())

                searchInput.textChanged.connect(filter_traces)

                searchDialog.setLayout(dialogLayout)
                searchDialog.exec()

            addTraceButton.clicked.connect(open_search_dialog)

            def add_selected_traces(availableTracesList, dialog):
                selected_items = availableTracesList.selectedItems()
                existing_traces = [selectedTracesList.item(i).text() for i in range(selectedTracesList.count())]

                for item in selected_items:
                    trace_name = item.text()
                    if trace_name not in existing_traces:  
                        add_trace_item(trace_name, selectedTracesList)

                dialog.accept()

            searchLayout = QHBoxLayout()
            searchLayout.addWidget(addTraceButton)

            groupLayout.addWidget(selectedTracesList)

            graphTypeLayout = QHBoxLayout()
            graphTypeLayout.addWidget(QLabel('Chart Type:'))

            traceTypeCombo = QComboBox()
            traceTypeCombo.addItems(['Plot', 'Scatter', 'Bars'])

            prev_types = self.graph_widgets[index].get('trace_types', ['Plot'] * len(prev_traces))
            if subplot_index < len(prev_types):
                traceTypeCombo.setCurrentText(prev_types[subplot_index])

            graphTypeLayout.addWidget(traceTypeCombo)
            graphTypeLayout.addStretch()

            groupLayout.addLayout(graphTypeLayout)

            groupBox.setLayout(groupLayout)
            scrollLayout.addWidget(groupBox)

            subplot_traces.append(selectedTracesList)
            trace_types.append(traceTypeCombo)

        def remove_subplot(groupBox, subplot_index):
            if len(subplot_traces) > 1:
                subplot_traces.pop(subplot_index)
                trace_types.pop(subplot_index)

                scrollLayout.removeWidget(groupBox)
                groupBox.deleteLater()

                update_pane_labels()
            else:
                QMessageBox.warning(dialog, 'Warning', 'There must be at least one sub-chart.')

        def update_pane_labels():
            for i in range(len(subplot_traces)):
                scrollLayout.itemAt(i).widget().setTitle(f'Pane {i + 1}')

        addSubplotButton = QPushButton('+')
        addSubplotButton.setStyleSheet('font-size: 14px; font-weight: bold; color: green;')
        addSubplotButton.setFixedSize(30, 30)
        addSubplotButton.clicked.connect(lambda: create_subplot_selector(len(subplot_traces)))
        mainLayout.addWidget(addSubplotButton)

        num_subplots = self.graph_widgets[index].get('subplots', 1)
        for i in range(num_subplots):
            create_subplot_selector(i)

        okButton = QPushButton('Apply')
        cancelButton = QPushButton('Cancel')

        def apply_changes():
            selected_traces_per_subplot = []

            for trace_list in subplot_traces:
                selected_traces = [trace_list.item(i).text() for i in range(trace_list.count())]
                selected_traces_per_subplot.append(selected_traces)

            self.set_selected_traces(index, len(subplot_traces), selected_traces_per_subplot, trace_types, x_axis_combo, dialog)

        buttonLayout = QHBoxLayout()
        buttonLayout.addWidget(cancelButton)
        buttonLayout.addStretch()
        buttonLayout.addWidget(okButton)

        okButton.clicked.connect(apply_changes)
        cancelButton.clicked.connect(dialog.reject)

        mainLayout.addLayout(buttonLayout)

        dialog.setLayout(mainLayout)
        dialog.exec()

    def set_selected_traces(self, index, num_subplots, selected_traces_per_subplot, trace_types, x_axis_combo, dialog):
        self.graph_widgets[index]['subplots'] = num_subplots
        self.graph_widgets[index]['x_axis'] = x_axis_combo.currentText()
        self.graph_widgets[index]['traces'] = selected_traces_per_subplot
        self.graph_widgets[index]['trace_types'] = [trace_types[i].currentText() for i in range(len(trace_types))]

        if 'trace_colors' not in self.graph_widgets[index]:
            self.graph_widgets[index]['trace_colors'] = {}

        for i in range(len(selected_traces_per_subplot)):
            for trace in selected_traces_per_subplot[i]:
                if trace not in self.graph_widgets[index]['trace_colors']:
                    self.graph_widgets[index]['trace_colors'][trace] = pg.mkColor(
                        random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
                    )

        self.update_graph(index)
        dialog.accept()

    def update_graph(self, index):
        widget_data = self.graph_widgets[index]
        gridLayout = widget_data['gridLayout']

        while gridLayout.count():
            item = gridLayout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        x_column = widget_data.get('x_axis', None)
        if x_column is None or x_column not in self.data.columns:
            QMessageBox.warning(self, 'Error', 'La columna X seleccionada no está en el archivo CSV.')
            return

        x_data = self.data[x_column].values
        widget_data['plots'] = []
        widget_data['vlines'] = []

        y_axis_widths = []

        for i in range(widget_data['subplots']):
            subplotContainer = pg.GraphicsLayoutWidget()

            subplot = subplotContainer.addPlot()
            subplot.showGrid(x=True, y=True)

            if i < widget_data['subplots'] - 1:
                subplot.getAxis('bottom').setStyle(showValues=False)

            widget_data['plots'].append(subplot)

            vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('white', width=1))
            subplot.addItem(vline)
            widget_data['vlines'].append(vline)

            gridLayout.addWidget(subplotContainer, i, 0)

            traces = widget_data['traces'][i]

            trace_types = widget_data.get('trace_types')

            trace_types = trace_types*len(traces)

            subplot.sigXRangeChanged.connect(lambda plot=subplot: self.sync_zoom(index, plot))

            checkboxContainer = QWidget()
            checkboxLayout = QVBoxLayout(checkboxContainer)
            checkboxLayout.setContentsMargins(10, 10, 10, 10)

            valueContainer = QWidget()
            valueLayout = QVBoxLayout(valueContainer)
            valueLayout.setContentsMargins(10, 10, 10, 10)

            all_y_data = []

            for j, trace in enumerate(traces):
                if trace in self.data.columns:
                    y_data = self.data[trace].values
                    all_y_data.extend(y_data)

                    if len(y_data) != len(x_data):
                        print(f"Warning: Column '{trace}' has different length from X-axis")
                        continue

                    color = widget_data['trace_colors'].get(trace, pg.mkColor(
                        random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
                    ))

                    graph_type = trace_types[j]

                    if graph_type == 'Plot':
                        plot_item = subplot.plot(x_data, y_data, pen=pg.mkPen(color, width=2), name=trace)
                    elif graph_type == 'Scatter':
                        plot_item = subplot.plot(x_data, y_data, pen=None, symbol='o', symbolBrush=color, name=trace)
                    elif graph_type == 'Bars':
                        plot_item = pg.BarGraphItem(x=x_data, height=y_data, width=0.5, brush=color)
                        subplot.addItem(plot_item)

                    checkbox = QCheckBox(trace)
                    checkbox.setChecked(True)

                    color_str = f"rgb({color.red()}, {color.green()}, {color.blue()})"

                    checkbox.setStyleSheet(f"""
                        QCheckBox {{
                            color: {color_str};
                            font-weight: bold;
                        }}
                        QCheckBox::indicator {{
                            width: 10px;
                            height: 10px;
                            border: 2px solid {color_str};
                            background-color: transparent;
                        }}
                        QCheckBox::indicator:checked {{
                            background-color: {color_str};
                        }}
                    """)

                    def toggle_trace(checked, plot_item=plot_item):
                        plot_item.setVisible(checked)

                    checkbox.toggled.connect(toggle_trace)
                    checkboxLayout.addWidget(checkbox)
                    checkboxLayout.setAlignment(Qt.AlignmentFlag.AlignTop)

                    gridLayout.addWidget(checkboxContainer, i, 1)

                    valueLabel = QLabel('0.00')
                    valueLabel.setFixedWidth(50)
                    valueLabel.setStyleSheet(f"color: {color_str}; font-weight: bold;")
                    valueLayout.addWidget(valueLabel)
                    valueLayout.setAlignment(Qt.AlignmentFlag.AlignTop)

                    gridLayout.addWidget(valueContainer, i, 2)

            y_axis_width = subplot.getAxis('left').width()
            y_axis_widths.append(y_axis_width)

        max_y_width = max(y_axis_widths)

        min_width = 50  
        final_width = max(max_y_width, min_width)

        for subplot in widget_data['plots']:
            subplot.getAxis('left').setWidth(final_width)

        xLabelContainer = QWidget()
        xLabelLayout = QVBoxLayout(xLabelContainer)
        xLabelLayout.setContentsMargins(10, 10, 10, 10)

        xLabel = QLabel(f'{x_column}')
        xLabel.setStyleSheet('font-weight: bold; font-size: 14px;')
        xLabelLayout.addWidget(xLabel)
        gridLayout.addWidget(xLabelContainer, widget_data['subplots'], 1)

        xValueContainer = QWidget()
        xValueLayout = QVBoxLayout(xValueContainer)
        xValueLayout.setContentsMargins(10, 10, 10, 10)

        xValueLabel = QLabel('0.00')
        xValueLabel.setStyleSheet('font-weight: bold; font-size: 14px;')
        xValueLayout.addWidget(xValueLabel)
        gridLayout.addWidget(xValueContainer, widget_data['subplots'], 2)

        widget_data['xValueLabel'] = xValueLabel

        for subplot in widget_data['plots']:
            subplot.scene().sigMouseMoved.connect(lambda pos, i=index: self.onMouseMove(pos, i))

        self.graphs_initialized = True

    def onMouseMove(self, pos, index):
        widget_data = self.graph_widgets.get(index, None)
        if widget_data is None or 'plots' not in widget_data or not widget_data['plots']:
            return

        x_column = widget_data.get('x_axis', None)
        if x_column is None or x_column not in self.data.columns:
            return

        x_data = self.data[x_column].values
        for subplot in widget_data['plots']:
            if subplot.sceneBoundingRect().contains(pos):
                mousePoint = subplot.vb.mapSceneToView(pos)
                x_cursor = mousePoint.x()

                for vline in widget_data['vlines']:
                    vline.setPos(x_cursor)

                for i, traces in enumerate(widget_data['traces']):
                    for j, trace in enumerate(traces):
                        if trace in self.data.columns:
                            y_data = self.data[trace].values

                            if len(y_data) != len(x_data):
                                continue

                            idx = np.abs(x_data - x_cursor).argmin()
                            y_value = y_data[idx]

                            valueLabel = widget_data['gridLayout'].itemAtPosition(i, 2).widget().layout().itemAt(j).widget()
                            valueLabel.setText(f"{y_value:.1f}")

                widget_data['xValueLabel'].setText(f"{x_cursor:.2f}")
                break

    def sync_zoom(self, index, plot):
        widget_data = self.graph_widgets[index]
        for p in widget_data['plots']:
            if p != plot:
                p.setXRange(*plot.viewRange()[0], padding=0)

    def load_csv(self):
        options = QFileDialog.Option(0)
        fileName, _ = QFileDialog.getOpenFileName(self, 'Load Sweep', '', 'CSV Files (*.csv);;All Files (*)', options=options)

        if fileName:
            try:
                self.data = pd.read_csv(fileName, encoding='utf-8', on_bad_lines='skip').dropna()
                if self.data.empty:
                    print('CSV is empty or only contains NaN values.')
                    return
                QMessageBox.information(self, 'Info', 'Simulation Sweep succesfully loaded.')
                    
            except Exception as e:
                print(f'Error loading Sweep: {e}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GraphApp()
    window.show()
    sys.exit(app.exec())
