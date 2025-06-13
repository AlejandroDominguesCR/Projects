from PyQt5.QtWidgets import QInputDialog, QApplication, QWidget, QPushButton, QVBoxLayout, QComboBox, QFileDialog, QTableWidget, QTableWidgetItem, QScrollArea, QHBoxLayout, QSplitter, QCheckBox, QGroupBox, QVBoxLayout, QListWidget
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import numpy as np
import matplotlib.colors as mcolors
import random
import sys
from scipy import signal
import pandas as pd
from scipy.signal import welch

class MatplotlibWidget(FigureCanvas):
    def __init__(self, parent=None):
        fig, self.ax = plt.subplots(17, 1, figsize=(8, 20))
        super().__init__(fig)
        self.setParent(parent)

        self.ax[0].set_title('Accu. Track Road Noise by Lap')
        self.ax[0].set_xlabel('')
        self.ax[0].set_ylabel('zTrack (mm)')
        self.ax[0].grid(True)

        self.ax[1].set_title('Heave Motion in the Car Coordinate System (zCoG)')
        self.ax[1].set_xlabel('Time (s)')
        self.ax[1].set_ylabel('Heave (m)')
        self.ax[1].grid(True)

        self.ax[2].set_title('Pitch Motion in the Car Coordinate System')
        self.ax[2].set_xlabel('Time (s)')
        self.ax[2].set_ylabel('Pitch (deg)')
        self.ax[2].grid(True)

        self.ax[3].set_title('Roll Motion in the Car Coordinate System')
        self.ax[3].set_xlabel('Time (s)')
        self.ax[3].set_ylabel('Roll (deg)')
        self.ax[3].grid(True)

        self.ax[4].set_title('Power Spectrum Density of Heave Motion')
        self.ax[4].set_xlabel('Frequency (Hz)')
        self.ax[4].set_ylabel('PSD Heave ($m^2$/Hz)')
        self.ax[4].grid(True)

        self.ax[5].set_title('Power Spectrum Density of Pitch Motion')
        self.ax[5].set_xlabel('Frecuency (Hz)')
        self.ax[5].set_ylabel('PSD Pitch ($deg^2$/Hz)')
        self.ax[5].grid(True)

        self.ax[6].set_title('Front Ride Height on Non Grip Limited Sections')
        self.ax[6].set_xlabel('')
        self.ax[6].set_ylabel('FRH RMS (mm)')
        self.ax[6].grid(True)

        self.ax[7].set_title('Rear Ride Height on Non Grip Limited Sections')
        self.ax[7].set_xlabel('')
        self.ax[7].set_ylabel('RRH RMS (mm)')
        self.ax[7].grid(True)

        self.ax[8].set_title('Contact Patch Load on Grip Limited Sections')
        self.ax[8].set_xlabel('')
        self.ax[8].set_ylabel('CPL RMS (N)')
        self.ax[8].grid(True)

        self.ax[9].set_title('Contact Patch Load on Braking Sections')
        self.ax[9].set_xlabel('')
        self.ax[9].set_ylabel('CPL RMS (N)')
        self.ax[9].grid(True)

        self.ax[10].set_title('Contact Patch Load on Traction Sections')
        self.ax[10].set_xlabel('')
        self.ax[10].set_ylabel('CPL RMS (N)')
        self.ax[10].grid(True)

        self.ax[11].set_title('Front Contact Patch Load on Braking Sections')
        self.ax[11].set_xlabel('')
        self.ax[11].set_ylabel('Front CPL RMS (N)')
        self.ax[11].grid(True)

        self.ax[12].set_title('Front Contact Patch Load on Traction Sections')
        self.ax[12].set_xlabel('')
        self.ax[12].set_ylabel('Front CPL RMS (N)')
        self.ax[12].grid(True)

        self.ax[13].set_title('Rear Contact Patch Load on Braking Sections')
        self.ax[13].set_xlabel('')
        self.ax[13].set_ylabel('Rear CPL RMS (N)')
        self.ax[13].grid(True)

        self.ax[14].set_title('Rear Contact Patch Load on Traction Sections')
        self.ax[14].set_xlabel('')
        self.ax[14].set_ylabel('Rear CPL RMS (N)')
        self.ax[14].grid(True)

        self.ax[15].set_title('Front Ride Height vs Contact Patch Load')
        self.ax[15].set_xlabel('CPL RMS (N)')
        self.ax[15].set_ylabel('FRH RMS (mm)')
        self.ax[15].grid(True)

        self.ax[16].set_title('Rear Ride Height vs Contact Patch Load')
        self.ax[16].set_xlabel('CPL RMS (N)')
        self.ax[16].set_ylabel('RRH RMS (mm)')
        self.ax[16].grid(True)

        plt.subplots_adjust(hspace=0.2)

        self.setFixedHeight(12000)

class SimVisualizationWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Sim Visualization')
        self.showMaximized()

        self.btn_cargar = QPushButton('Load Sweep', self)
        self.btn_cargar.clicked.connect(self.pedir_num_csv)

        self.combo_csvs = QComboBox(self)
        self.combo_csvs.currentIndexChanged.connect(self.mostrar_csv_seleccionado)

        self.tabla = QTableWidget(self)

        self.checkbox_group = QGroupBox('Select Sweep', self)
        self.checkbox_layout = QVBoxLayout(self.checkbox_group)
        self.checkbox_group.setLayout(self.checkbox_layout)

        self.canvas = MatplotlibWidget(self)

        self.toolbar = NavigationToolbar(self.canvas, self)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.canvas)

        self.splitter = QSplitter(Qt.Vertical)

        self.splitter.addWidget(self.tabla)
        self.splitter.addWidget(self.scroll_area)

        layout_principal = QVBoxLayout()
        layout_principal.addWidget(self.btn_cargar)
        layout_principal.addWidget(self.combo_csvs)
        layout_principal.addWidget(self.checkbox_group)
        layout_principal.addWidget(self.toolbar)
        layout_principal.addWidget(self.splitter)

        self.setLayout(layout_principal)

        self.datos_csvs = {}

    def pedir_num_csv(self):
        num_csv, ok = QInputDialog.getInt(self, 'Número de CSVs', 'Ingrese cuántos CSVs quiere cargar:', 1, 1, 10)
        if ok:
            self.cargar_multiples_csv(num_csv)

    def cargar_multiples_csv(self, num_csv):
        self.datos_csvs.clear()
        self.combo_csvs.clear()

        for i in range(self.checkbox_layout.count()):
            widget = self.checkbox_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        for i in range(num_csv):
            archivo, _ = QFileDialog.getOpenFileName(self, f'Seleccionar CSV {i+1}', '', 'Archivos CSV (*.csv)')
            if archivo:
                self.procesar_csv(archivo)

        if self.datos_csvs:
            self.combo_csvs.setCurrentIndex(0)

    def procesar_csv(self, archivo):
        nombre_csv = archivo.split('/')[-1]

        df = pd.read_csv(archivo)

        self.datos_csvs[nombre_csv] = df

        self.combo_csvs.addItem(nombre_csv)

        print(df)

        checkbox = QCheckBox(nombre_csv, self)
        checkbox.setChecked(True)
        checkbox.stateChanged.connect(lambda state: self.actualizar_graficos(df, nombre_csv))
        self.checkbox_layout.addWidget(checkbox)

        self.actualizar_graficos(df, nombre_csv)
    
    def mostrar_csv_seleccionado(self):
        nombre_csv = self.combo_csvs.currentText()
        if not nombre_csv or nombre_csv not in self.datos_csvs:
            return

        df = self.datos_csvs[nombre_csv]

        vector_tiempo = df['Time (s)']  
        datos = df.drop(columns=['Time (s)'])

        num_filas = len(df)
        num_columnas = datos.shape[1]

        self.tabla.setRowCount(num_filas)
        self.tabla.setColumnCount(num_columnas + 1)

        encabezados = ['Tiempo'] + list(datos.columns)
        self.tabla.setHorizontalHeaderLabels(encabezados)

        for i, tiempo in enumerate(vector_tiempo):
            self.tabla.setItem(i, 0, QTableWidgetItem(str(tiempo)))  
            for j, valor in enumerate(datos.iloc[i]):
                self.tabla.setItem(i, j + 1, QTableWidgetItem(str(valor)))  

    def actualizar_graficos(self, datos, archivo_csv):


        time = datos['Time (s)']
        sim_columns = [col for col in datos.columns if 'Sim' in col]

        simulaciones = {}


        for col in sim_columns:
            sim_number = col.split(' ')[1]

            print(col)

            if sim_number not in simulaciones:
                simulaciones[sim_number] = {'zTrack FL': [], 'rPedal': [], 'pBrake': [], 'Heave': [], 'Pitch': [], 'Roll': [], 'FRH': [], 'RRH': [], 'FL CPL': [], 'FR CPL': [], 'RL CPL': [], 'RR CPL': []}
            
            if 'zTrack FL' in col:
                simulaciones[sim_number]['zTrack FL'].append(col)
            if 'rPedal' in col:
                simulaciones[sim_number]['rPedal'].append(col)
            elif 'pBrake' in col:
                simulaciones[sim_number]['pBrake'].append(col)
            elif 'Heave' in col:
                simulaciones[sim_number]['Heave'].append(col)
            elif 'Pitch' in col:
                simulaciones[sim_number]['Pitch'].append(col)
            elif 'Roll' in col:
                simulaciones[sim_number]['Roll'].append(col)
            elif 'FRH' in col:
                simulaciones[sim_number]['FRH'].append(col)
            elif 'RRH' in col:
                simulaciones[sim_number]['RRH'].append(col)
            elif 'FL CPL' in col:
                simulaciones[sim_number]['FL CPL'].append(col)
            elif 'FR CPL' in col:
                simulaciones[sim_number]['FR CPL'].append(col)
            elif 'RL CPL' in col:
                simulaciones[sim_number]['RL CPL'].append(col)
            elif 'RR CPL' in col:
                simulaciones[sim_number]['RR CPL'].append(col)

        sim_list = []
        frh_ngls_values = []
        rrh_ngls_values = []
        cpl_gls_values = []
        cpl_gls_braking_values = []
        cpl_gls_traction_values = []
        front_cpl_gls_braking_values = []
        front_cpl_gls_traction_values = []
        rear_cpl_gls_braking_values = []
        rear_cpl_gls_traction_values = []

        for sim_number, columns in simulaciones.items():
            name = archivo_csv.split('ride_sim_')[1].split('.csv')[0]

            heave = datos[columns['Heave'][0]]  
            pitch = datos[columns['Pitch'][0]]  
            roll = datos[columns['Roll'][0]]
            frh = datos[columns['FRH'][0]]*1000
            rrh = datos[columns['RRH'][0]]*1000
            ztrack = datos[columns['zTrack FL'][0]]*1000
            rPedal = datos[columns['rPedal'][0]]
            pBrake = datos[columns['pBrake'][0]]
            flcpl = datos[columns['FL CPL'][0]]
            frcpl = datos[columns['FR CPL'][0]]
            rlcpl = datos[columns['RL CPL'][0]]
            rrcpl = datos[columns['RR CPL'][0]]

            dt = 0.01

            drPedal = np.diff(rPedal)/dt
            dpBrake = np.diff(pBrake)/dt

            f_low = 1
            order = 2
            b, a = signal.butter(order, f_low/(100/2), btype='high')

            frh_ngls = []
            rrh_ngls = []
            fl_cpl_gls_brk = []
            fl_cpl_gls_thr = []
            fr_cpl_gls_brk = []
            fr_cpl_gls_thr = []
            rl_cpl_gls_brk = []
            rl_cpl_gls_thr = []
            rr_cpl_gls_brk = []
            rr_cpl_gls_thr = []
            ztrack_ngls = []
            ztrack_gls_brk = []
            ztrack_gls_thr = []

            for j in range(1, len(time[:-1])):
                if rPedal[j] > 98:
                    frh_ngls.append(frh[j])
                    rrh_ngls.append(rrh[j])
                    ztrack_ngls.append(ztrack[j])

                if pBrake[j] > 5 and dpBrake[j] < 0:
                    fl_cpl_gls_brk.append(flcpl[j])
                    fr_cpl_gls_brk.append(frcpl[j])
                    rl_cpl_gls_brk.append(rlcpl[j])
                    rr_cpl_gls_brk.append(rrcpl[j])
                    ztrack_gls_brk.append(ztrack[j])

                if pBrake[j] < 5 and rPedal[j] < 98 and drPedal[j] > 0:
                    fl_cpl_gls_thr.append(flcpl[j])
                    fr_cpl_gls_thr.append(frcpl[j])
                    rl_cpl_gls_thr.append(rlcpl[j])
                    rr_cpl_gls_thr.append(rrcpl[j])
                    ztrack_gls_thr.append(ztrack[j])

            frh_ngls = np.array(frh_ngls)
            rrh_ngls = np.array(rrh_ngls)
            ztrack_ngls = np.array(ztrack_ngls)
            fl_cpl_gls_brk = np.array(fl_cpl_gls_brk)
            fr_cpl_gls_brk = np.array(fr_cpl_gls_brk)
            rl_cpl_gls_brk = np.array(rl_cpl_gls_brk)
            rr_cpl_gls_brk = np.array(rr_cpl_gls_brk)
            ztrack_gls_brk = np.array(ztrack_gls_brk)
            fl_cpl_gls_thr = np.array(fl_cpl_gls_thr)
            fr_cpl_gls_thr = np.array(fr_cpl_gls_thr)
            rl_cpl_gls_thr = np.array(rl_cpl_gls_thr)
            rr_cpl_gls_thr = np.array(rr_cpl_gls_thr)
            ztrack_gls_thr = np.array(ztrack_gls_thr)

            frh_ngls_filtered = signal.filtfilt(b, a, frh_ngls)  
            frh_ngls = frh_ngls_filtered - np.mean(frh_ngls_filtered)  

            rrh_ngls_filtered = signal.filtfilt(b, a, rrh_ngls)  
            rrh_ngls = rrh_ngls_filtered - np.mean(rrh_ngls_filtered)

            fl_cpl_gls_brk_filtered = signal.filtfilt(b, a, fl_cpl_gls_brk)
            fl_cpl_gls_brk = fl_cpl_gls_brk_filtered - np.mean(fl_cpl_gls_brk_filtered)
            fr_cpl_gls_brk_filtered = signal.filtfilt(b, a, fr_cpl_gls_brk)
            fr_cpl_gls_brk = fr_cpl_gls_brk_filtered - np.mean(fr_cpl_gls_brk_filtered)
            rl_cpl_gls_brk_filtered = signal.filtfilt(b, a, rl_cpl_gls_brk)
            rl_cpl_gls_brk = rl_cpl_gls_brk_filtered - np.mean(rl_cpl_gls_brk_filtered)
            rr_cpl_gls_brk_filtered = signal.filtfilt(b, a, rr_cpl_gls_brk)
            rr_cpl_gls_brk = rr_cpl_gls_brk_filtered - np.mean(rr_cpl_gls_brk_filtered)

            fl_cpl_gls_thr_filtered = signal.filtfilt(b, a, fl_cpl_gls_thr)
            fl_cpl_gls_thr = fl_cpl_gls_thr_filtered - np.mean(fl_cpl_gls_thr_filtered)
            fr_cpl_gls_thr_filtered = signal.filtfilt(b, a, fr_cpl_gls_thr)
            fr_cpl_gls_thr = fr_cpl_gls_thr_filtered - np.mean(fr_cpl_gls_thr_filtered)
            rl_cpl_gls_thr_filtered = signal.filtfilt(b, a, rl_cpl_gls_thr)
            rl_cpl_gls_thr = rl_cpl_gls_thr_filtered - np.mean(rl_cpl_gls_thr_filtered)
            rr_cpl_gls_thr_filtered = signal.filtfilt(b, a, rr_cpl_gls_thr)
            rr_cpl_gls_thr = rr_cpl_gls_thr_filtered - np.mean(rr_cpl_gls_thr_filtered)

            frh_ngls_rms = np.sqrt((1/len(frh_ngls))*np.sum(frh_ngls**2))
            rrh_ngls_rms = np.sqrt((1/len(rrh_ngls))*np.sum(rrh_ngls**2))

            if sim_number == '1':
                ztrack_total = np.cumsum(np.abs(ztrack)) * dt
                ztrack_ngls_rms = np.cumsum(np.abs(ztrack_ngls)) * dt
                ztrack_gls_brk_rms = np.cumsum(np.abs(ztrack_gls_thr)) * dt
                ztrack_gls_thr_rms = np.cumsum(np.abs(ztrack_gls_brk)) * dt

                width = 0.2
                category = [name]
                
                position = np.arange(len(category))

                self.canvas.ax[0].bar(position - 1.5*width, ztrack_total, width, label='Total Lap', color='black')
                self.canvas.ax[0].bar(position - 0.5*width, ztrack_ngls_rms, width, label='NGLS', color='blue')
                self.canvas.ax[0].bar(position + 0.5*width, ztrack_gls_brk_rms, width, label='GLS Braking', color='red')
                self.canvas.ax[0].bar(position + 1.5*width, ztrack_gls_thr_rms, width, label='GLS Traction', color='green')
                self.canvas.ax[0].set_xticks(position)
                self.canvas.ax[0].set_xticklabels(category)
                self.canvas.ax[0].legend()

            fl_cpl_gls_brk_rms = np.sqrt((1/len(fl_cpl_gls_brk))*np.sum(fl_cpl_gls_brk**2))
            fr_cpl_gls_brk_rms = np.sqrt((1/len(fr_cpl_gls_brk))*np.sum(fr_cpl_gls_brk**2))
            rl_cpl_gls_brk_rms = np.sqrt((1/len(rl_cpl_gls_brk))*np.sum(rl_cpl_gls_brk**2))
            rr_cpl_gls_brk_rms = np.sqrt((1/len(rr_cpl_gls_brk))*np.sum(rr_cpl_gls_brk**2))

            fl_cpl_gls_thr_rms = np.sqrt((1/len(fl_cpl_gls_thr))*np.sum(fl_cpl_gls_thr**2))
            fr_cpl_gls_thr_rms = np.sqrt((1/len(fr_cpl_gls_thr))*np.sum(fr_cpl_gls_thr**2))
            rl_cpl_gls_thr_rms = np.sqrt((1/len(rl_cpl_gls_thr))*np.sum(rl_cpl_gls_thr**2))
            rr_cpl_gls_thr_rms = np.sqrt((1/len(rr_cpl_gls_thr))*np.sum(rr_cpl_gls_thr**2))
 
            front_cpl_gls_brk_rms = 0.5*fl_cpl_gls_brk_rms + 0.5*fr_cpl_gls_brk_rms
            rear_cpl_gls_brk_rms = 0.5*rl_cpl_gls_brk_rms + 0.5*rr_cpl_gls_brk_rms

            front_cpl_gls_thr_rms = 0.5*fl_cpl_gls_thr_rms + 0.5*fr_cpl_gls_thr_rms
            rear_cpl_gls_thr_rms = 0.5*rl_cpl_gls_thr_rms + 0.5*rr_cpl_gls_thr_rms

            cpl_gls_brk_rms = 0.5*front_cpl_gls_brk_rms + 0.5*rear_cpl_gls_brk_rms
            cpl_gls_thr_rms = 0.25*front_cpl_gls_thr_rms + 0.75*rear_cpl_gls_thr_rms

            cpl_gls_rms = 0.5*cpl_gls_brk_rms + 0.5*cpl_gls_thr_rms

            heave_freq, heave_psd = welch(heave, 100, nperseg=1024)
            pitch_freq, pitch_psd = welch(pitch, 100, nperseg=1024)

            self.canvas.ax[1].plot(time, heave, label=f'{name} - Sim {sim_number}')
            self.canvas.ax[2].plot(time, pitch, label=f'{name} - Sim {sim_number}')
            self.canvas.ax[3].plot(time, roll, label=f'{name} - Sim {sim_number}')

            self.canvas.ax[4].semilogy(heave_freq, heave_psd, label=f'{name} - Sim{sim_number}')
            self.canvas.ax[5].semilogy(pitch_freq, pitch_psd, label=f'{name} - Sim{sim_number}')

            colores_csv = {}

            if archivo_csv not in colores_csv:
                colores_csv[archivo_csv] = plt.cm.tab10(len(colores_csv) % 10)

            sim_list.append(f'Sim {sim_number}')
            frh_ngls_values.append(frh_ngls_rms)
            rrh_ngls_values.append(rrh_ngls_rms)
            cpl_gls_values.append(cpl_gls_rms)
            cpl_gls_braking_values.append(cpl_gls_brk_rms)
            cpl_gls_traction_values.append(cpl_gls_thr_rms)
            front_cpl_gls_braking_values.append(front_cpl_gls_brk_rms)
            front_cpl_gls_traction_values.append(front_cpl_gls_thr_rms)
            rear_cpl_gls_braking_values.append(rear_cpl_gls_brk_rms)
            rear_cpl_gls_traction_values.append(rear_cpl_gls_thr_rms)

        color_csv = {}

        colores_disponibles = list(mcolors.TABLEAU_COLORS.values())
        color_csv[name] = random.choice(colores_disponibles)

        self.canvas.ax[6].plot(sim_list, frh_ngls_values, marker='o', label=f'{name}', color=color_csv[name])
        self.canvas.ax[7].plot(sim_list, rrh_ngls_values, marker='o', label=f'{name}', color=color_csv[name])
        self.canvas.ax[8].plot(sim_list, cpl_gls_values, marker='o', label=f'{name}', color=color_csv[name])
        self.canvas.ax[9].plot(sim_list, cpl_gls_braking_values, marker='o', label=f'{name}', color=color_csv[name])
        self.canvas.ax[10].plot(sim_list, cpl_gls_traction_values, marker='o', label=f'{name}', color=color_csv[name])
        self.canvas.ax[11].plot(sim_list, front_cpl_gls_braking_values, marker='o', label=f'{name}', color=color_csv[name])
        self.canvas.ax[12].plot(sim_list, front_cpl_gls_traction_values, marker='o', label=f'{name}', color=color_csv[name])
        self.canvas.ax[13].plot(sim_list, rear_cpl_gls_braking_values, marker='o', label=f'{name}', color=color_csv[name])
        self.canvas.ax[14].plot(sim_list, rear_cpl_gls_traction_values, marker='o', label=f'{name}', color=color_csv[name])
        self.canvas.ax[15].scatter(cpl_gls_values, frh_ngls_values, marker='o', label=f'{name}', color=color_csv[name])
        self.canvas.ax[16].scatter(cpl_gls_values, rrh_ngls_values, marker='o', label=f'{name}', color=color_csv[name])

        for i in range(len(cpl_gls_values)):
            self.canvas.ax[15].annotate(f'Sim {i+1}', (cpl_gls_values[i], frh_ngls_values[i]), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)

        for i in range(len(cpl_gls_values)):
            self.canvas.ax[16].annotate(f'Sim {i+1}', (cpl_gls_values[i], rrh_ngls_values[i]), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)

        for ax in [self.canvas.ax[6], self.canvas.ax[7], self.canvas.ax[8], self.canvas.ax[9], self.canvas.ax[10], self.canvas.ax[11], self.canvas.ax[12], self.canvas.ax[13], self.canvas.ax[14]]:
            for label in ax.get_xticklabels():
                label.set_rotation(45)
            ax.figure.canvas.draw()

        self.canvas.ax[1].legend()
        self.canvas.ax[2].legend()
        self.canvas.ax[3].legend()
        self.canvas.ax[4].legend()
        self.canvas.ax[5].legend()
        self.canvas.ax[6].legend()
        self.canvas.ax[7].legend()
        self.canvas.ax[8].legend()
        self.canvas.ax[9].legend()
        self.canvas.ax[10].legend()
        self.canvas.ax[11].legend()
        self.canvas.ax[12].legend()
        self.canvas.ax[13].legend()
        self.canvas.ax[14].legend()
        self.canvas.ax[15].legend()
        self.canvas.ax[16].legend()

        self.canvas.figure.canvas.draw()
        self.canvas.figure.canvas.flush_events()

def main():
    app = QApplication(sys.argv)
    ventana = SimVisualizationWindow()
    ventana.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
