import sys
import csv
from PyQt5.QtWidgets import QApplication, QWidget, QGroupBox, QPushButton, QFormLayout, QLineEdit, QToolButton, QLabel, QFileDialog, QSpinBox, QTreeWidget, QTreeWidgetItem, QMessageBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPalette, QFont
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import json
from tkinter import messagebox
from PyQt5.QtGui import QPixmap, QColor
from PyQt5.QtCore import QByteArray
from scipy.interpolate import interp1d
import numpy as np
from scipy.integrate import solve_ivp
import ast
import itertools
import os

def load_csv_1(sim_id):
    archive, _ = QFileDialog.getOpenFileName(None, 'Front Damper', '', 'CSV Files (*.csv)')
    if archive:
        try:
            with open(archive, mode='r') as file:
                reader = csv.reader(file)
                next(reader)
                values = list(reader)

                front_speed = [float(row[0]) for row in values]
                front_force = [float(row[1]) for row in values]

                parameters_damper_front[sim_id-1] = (sim_id, front_speed, front_force)

        except Exception as e:
            print(f'Error loading Front Damper: {e}')

def load_csv_2(sim_id):
    archive, _ = QFileDialog.getOpenFileName(None, 'Rear Damper', '', 'CSV Files (*.csv)')
    if archive:
        try:
            with open(archive, mode='r') as file:
                reader = csv.reader(file)
                next(reader)
                values = list(reader)

                rear_speed = [float(row[0]) for row in values]
                rear_force = [float(row[1]) for row in values]

                parameters_damper_rear[sim_id-1] = (sim_id, rear_speed, rear_force)

        except Exception as e:
            print(f'Error loading Rear Damper CSV: {e}')

def run_simulation(args):

    damper_front, damper_rear, parameters, track = args

    _, Speed_F, ForceD_F = damper_front
    _, Speed_R, ForceD_R = damper_rear

    frh0, rrh0, mCar, rWeightBalF, Icar, mHubF, mHubR, kinstf, kinstr, zCoG, wheelbase, tF, tR, kspringf, Disp_F, ForceB_F, freegapf, karbf, kspringr, Disp_R, ForceB_R, freegapr, karbr, fa0, fa01, fa02, fa03, fa10, fa20, fa30, fa11, fa12, fa21, fa22, faflap10, faflap20, faflap01, faflap02, ra0, ra01, ra02, ra03, ra10, ra20, ra30, ra11, ra12, ra21, ra22, raflap10, raflap20, raflap01, raflap02, a0, a01, a02, a03, a10, a20, a30, a11, a12, a21, a22, aflap10, aflap20, aflap01, aflap02, aero_frontflap, aero_rearflap, ktf, ktr = parameters

    times, ztrackFL, ztrackFR, ztrackRL, ztrackRR, Ax, Ay, vCar, _, _ = track
    
    damper_front = interp1d(Speed_F,ForceD_F,kind = 'linear', fill_value = 'extrapolate')
    damper_rear = interp1d(Speed_R,ForceD_R,kind = 'linear', fill_value = 'extrapolate')

    bumpstop_front_interp = interp1d(Disp_F, ForceB_F, kind='linear', fill_value='extrapolate')
    bumpstop_rear_interp = interp1d(Disp_R, ForceB_R, kind='linear', fill_value='extrapolate')

    Ms = mCar - 2*mHubF - 2*mHubR
    Ixx = Icar[0]
    Iyy = Icar[4]
    lf = wheelbase*(1-rWeightBalF)
    lr = wheelbase*rWeightBalF

    kFL = kspringf
    kFR = kspringf
    kRL = kspringr
    kRR = kspringr

    z_FL_free = freegapf
    z_FR_free = freegapf
    z_RL_free = freegapr
    z_RR_free = freegapr

    rho = 1.225

    def bumpstop_front(x):
        if x < 0:
            return 0
        else:
            return bumpstop_front_interp(x)

    def bumpstop_rear(x):
        if x < 0:
            return 0
        else:
            return bumpstop_rear_interp(x)

    def aero(xFront, xRear):

        cl_front = fa0 + fa01*xRear + fa10*xFront + fa11*xRear*xFront + fa12*xFront*xRear*xRear + fa21*xFront*xRear*xRear + fa20*xFront*xFront + fa02*xRear*xRear + fa22*xFront*xFront*xRear*xRear + fa03*xRear*xRear*xRear + fa30*xFront*xFront*xFront + faflap01*aero_rearflap + faflap10*aero_frontflap+ faflap02*aero_rearflap*aero_rearflap + faflap20*aero_frontflap*aero_frontflap

        cl_rear = ra0 + ra01*xRear + ra10*xFront + ra11*xRear*xFront + ra12*xFront*xRear*xRear + ra21*xFront*xRear*xRear + ra20*xFront*xFront + ra02*xRear*xRear + ra22*xFront*xFront*xRear*xRear + ra03*xRear*xRear*xRear + ra30*xFront*xFront*xFront + raflap01*aero_rearflap + raflap10*aero_frontflap+ raflap02*aero_rearflap*aero_rearflap + raflap20*aero_frontflap*aero_frontflap

        cd = a0 + a01*xRear + a10*xFront + a11*xRear*xFront + a12*xFront*xRear*xRear + a21*xFront*xRear*xRear + a20*xFront*xFront + a02*xRear*xRear + a22*xFront*xFront*xRear*xRear + a03*xRear*xRear*xRear + a30*xFront*xFront*xFront + aflap01*aero_rearflap + aflap10*aero_frontflap+ aflap02*aero_rearflap*aero_rearflap + aflap20*aero_frontflap*aero_frontflap

        return cl_front, cl_rear, cd
                
    def vehicle_model(t, z, frh0, rrh0):

        ztrack_FL = ztrackFL_interp(t)
        ztrack_FR = ztrackFR_interp(t)
        ztrack_RL = ztrackRL_interp(t)
        ztrack_RR = ztrackRR_interp(t)

        vx = vxx(t)
        ax = axx(t)
        ay = ayy(t)

        h, hdot, phi, phi_dot, theta, theta_dot, zFR, zFRdot, zFL, zFLdot, zRL, zRLdot, zRR, zRRdot = z

        cl_front, cl_rear, cd = aero(h - lf*phi + frh0, h + lr*phi + rrh0)

        h_2dot = ((1/(1/(kFR
                 + bumpstop_front(zFR - (-lf*phi - (tF/2)*theta + h) - z_FR_free)/(zFR - (-lf*phi - (tF/2)*theta + h) - z_FR_free))
                 + (1/kinstf)))*(zFR - (-lf*phi - (tF/2)*theta + h))
                 + (1/(1/(kFL
                 + bumpstop_front(zFL - (-lf*phi + (tF/2)*theta + h) - z_FL_free)/(zFL - (-lf*phi + (tF/2)*theta + h) - z_FL_free))
                 + (1/kinstf)))*(zFL - (-lf*phi + (tF/2)*theta + h))
                 + (1/(1/(kRL
                 + bumpstop_rear(zRL - (lr*phi + (tR/2)*theta + h) - z_RL_free)/(zRL - (lr*phi + (tR/2)*theta + h) - z_RL_free))
                 + 1/(kinstr)))*(zRL - (lr*phi + (tR/2)*theta + h))
                 + (1/(1/(kRR
                 + bumpstop_rear(zRR - (lr*phi - (tR/2)*theta + h) - z_RR_free)/(zRR - (lr*phi - (tR/2)*theta + h) - z_RR_free))
                 + (1/kinstr)))*(zRR - (lr*phi - (tR/2)*theta + h))
                 + damper_front(zFRdot - (-lf*phi_dot - (tF/2)*theta_dot + hdot))
                 + damper_front(zFLdot - (-lf*phi_dot + (tF/2)*theta_dot + hdot))
                 + damper_rear(zRLdot - (lr*phi_dot + (tR/2)*theta_dot + hdot))
                 + damper_rear(zRRdot - (lr*phi_dot - (tR/2)*theta_dot + hdot))
                 - 0.5*(cl_front + cl_rear)*rho*((vx)**2))/Ms

        theta_2dot = ((tF/2)*((1/(1/(kFL
                 + bumpstop_front(zFL - (-lf*phi + (tF/2)*theta + h) - z_FL_free)/(zFL - (-lf*phi + (tF/2)*theta + h) - z_FL_free)
                 + (karbf if np.abs(ay) > 0.3 else 0))
                 + (1/kinstf)))*(zFL - (-lf*phi + (tF/2)*theta + h))
                 + damper_front(zFLdot - (-lf*phi_dot + (tF/2)*theta_dot + hdot)))
                 - (tF/2)*((1/(1/(kFR
                 + bumpstop_front(zFR - (-lf*phi - (tF/2)*theta + h) - z_FR_free)/(zFR - (-lf*phi - (tF/2)*theta + h) - z_FR_free)
                 + (karbf if np.abs(ay) > 0.3 else 0 ))
                 + (1/kinstf)))*(zFR - (-lf*phi - (tF/2)*theta + h))
                 + damper_front(zFRdot - (-lf*phi_dot - (tF/2)*theta_dot + hdot)))
                 + (tR/2)*((1/(1/(kRL
                 + bumpstop_rear(zRL - (lr*phi + (tR/2)*theta + h) - z_RL_free)/(zRL - (lr*phi + (tR/2)*theta + h) - z_RL_free)
                 + (karbr if np.abs(ay) > 0.3 else 0))
                 + (1/kinstr)))*(zRL - (lr*phi + (tR/2)*theta + h))
                 + damper_rear(zRLdot - (lr*phi_dot + (tR/2)*theta_dot + hdot)))
                 - (tR/2)*((1/(1/(kRR
                 + bumpstop_rear(zRR - (lr*phi - (tR/2)*theta + h) - z_RR_free)/(zRR - (lr*phi - (tR/2)*theta + h) - z_RR_free)
                 + (karbr if np.abs(ay) > 0.3 else 0))
                 + (1/kinstr)))*(zRR - (lr*phi - (tR/2)*theta + h))
                 + damper_rear(zRRdot - (lr*phi_dot - (tR/2)*theta_dot + hdot)))
                 + ay*9.81*Ms*(h - zCoG))/Ixx

        phi_2dot = (lr*((1/(1/(kRR
                 + bumpstop_rear(zRR - (lr*phi - (tR/2)*theta + h) - z_RR_free)/(zRR - (lr*phi - (tR/2)*theta + h) - z_RR_free))
                 + (1/kinstr)))*(zRR - (lr*phi - (tR/2)*theta + h))
                 + damper_rear(zRRdot - (lr*phi_dot - (tR/2)*theta_dot + hdot))
                 + (1/(1/(kRL
                 + bumpstop_rear(zRL - (lr*phi + (tR/2)*theta + h) - z_RL_free)/(zRL - (lr*phi + (tR/2)*theta + h) - z_RL_free))
                 + (1/kinstr)))*(zRL - (lr*phi + (tR/2)*theta + h))
                 + damper_rear(zRLdot - (lr*phi_dot + (tR/2)*theta_dot + hdot)))
                 - lf*((1/(1/(kFR
                 + bumpstop_front(zFR - (-lf*phi - (tF/2)*theta + h) - z_FR_free)/(zFR - (-lf*phi - (tF/2)*theta + h) - z_FR_free))
                 + (1/kinstf)))*(zFR - (-lf*phi - (tF/2)*theta + h))
                 + damper_front(zFRdot - (-lf*phi_dot - (tF/2)*theta_dot + hdot))
                 + (1/(1/(kFL
                 + bumpstop_front(zFL - (-lf*phi + (tF/2)*theta + h) - z_FL_free)/(zFL - (-lf*phi + (tF/2)*theta + h) - z_FL_free))
                 + (1/kinstf)))*(zFL - (-lf*phi + (tF/2)*theta + h))
                 + damper_front(zFLdot - (-lf*phi_dot + (tF/2)*theta_dot + hdot)))
                 - ax*9.81*Ms*(h - zCoG)
                 - lf*0.5*cl_front*rho*((vx/3.6)**2)
                 + lr*0.5*cl_rear*rho*((vx/3.6)**2)
                 - (h - zCoG)*0.5*cd*rho*((vx)**2))/Iyy

        zFR_2dot = (-(1/(1/(kFR
                 + bumpstop_front(zFR - (-lf*phi - (tF/2)*theta + h) - z_FR_free)/(zFR - (-lf*phi - (tF/2)*theta + h) - z_FR_free)
                 + (karbf if np.abs(ay) > 0.3 else 0 ))
                 + (1/kinstf)))*(zFR - (-lf*phi - (tF/2)*theta + h))
                 - damper_front(zFRdot - (-lf*phi_dot - (tF/2)*theta_dot + hdot))
                 + ktf*(ztrack_FR - zFR))/mHubF

        zFL_2dot = (-(1/(1/(kFL
                 + bumpstop_front(zFL - (-lf*phi + (tF/2)*theta + h) - z_FL_free)/(zFL - (-lf*phi + (tF/2)*theta + h) - z_FL_free)
                 + (karbf if np.abs(ay) > 0.3 else 0 ))
                 + (1/kinstf)))*(zFL - (-lf*phi + (tF/2)*theta + h))
                 - damper_front(zFLdot - (-lf*phi_dot + (tF/2)*theta_dot + hdot))
                 + ktf*(ztrack_FL - zFL))/mHubF

        zRL_2dot = (-(1/(1/(kRL
                 + bumpstop_rear(zRL - (lr*phi + (tR/2)*theta + h) - z_RL_free)/(zRL - (lr*phi + (tR/2)*theta + h) - z_RL_free)
                 + (karbr if np.abs(ay) > 0.3 else 0 ))
                 + (1/kinstr)))*(zRL - (lr*phi + (tR/2)*theta + h))
                 - damper_rear(zRLdot - (lr*phi_dot + (tR/2)*theta_dot + hdot))
                 + ktr*(ztrack_RL - zRL))/mHubR
 
        zRR_2dot = (-(1/(1/(kRR
                 + bumpstop_rear(zRR - (lr*phi - (tR/2)*theta + h) - z_RR_free)/(zRR - (lr*phi - (tR/2)*theta + h) - z_RR_free)
                 + (karbr if np.abs(ay) > 0.3 else 0))
                 + (1/kinstr)))*(zRR - (lr*phi - (tR/2)*theta + h))
                 - damper_rear(zRRdot - (lr*phi_dot - (tR/2)*theta_dot + hdot))
                 + ktr*(ztrack_RR - zRR))/mHubR

        return [hdot, h_2dot, phi_dot, phi_2dot, theta_dot, theta_2dot, zFRdot, zFR_2dot, zFLdot, zFL_2dot, zRLdot, zRL_2dot, zRRdot, zRR_2dot]

    phi0 = np.asin((rrh0 - frh0)/(lf + lr))
    h0 = -0.02

    z0 = [-0.02, 0, phi0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    t_span = [times[0], times[-1]]
    
    samples = int((t_span[-1]/0.01) + 1)

    t_eval = np.linspace(t_span[0], t_span[1], samples)

    ztrackFL_interp = interp1d(t_eval, ztrackFL, kind = 'linear', fill_value = 'extrapolate')
    ztrackFR_interp = interp1d(t_eval, ztrackFR, kind = 'linear', fill_value = 'extrapolate')
    ztrackRL_interp = interp1d(t_eval, ztrackRL, kind = 'linear', fill_value = 'extrapolate')
    ztrackRR_interp = interp1d(t_eval, ztrackRR, kind = 'linear', fill_value = 'extrapolate')

    vxx = interp1d(t_eval, vCar, kind = 'linear', fill_value = 'extrapolate')
    axx = interp1d(t_eval, Ax, kind = 'linear', fill_value = 'extrapolate')
    ayy = interp1d(t_eval, Ay, kind = 'linear', fill_value = 'extrapolate')

    vCar = np.array(vCar)

    delay = wheelbase/vCar

    t_delay = times + delay

    interp_ztrackRL = interp1d(t_delay, ztrackFL, bounds_error=False, fill_value='extrapolate')
    interp_ztrackRR = interp1d(t_delay, ztrackFR, bounds_error=False, fill_value='extrapolate')
    ztrackRL = interp_ztrackRL(times)
    ztrackRR = interp_ztrackRR(times)

    sol = solve_ivp(vehicle_model, t_span, z0, args = (frh0, rrh0), method='RK45', t_eval = t_eval)

    return sol

def run_sweep(self):

    parameters = []

    for key, line_edit in self.line_edits.items():
        try:
            text = line_edit.text()

            print(line_edit.text())
            
            if text.startswith('[') and text.endswith(']'):
                values = ast.literal_eval(text)

                parameters.append([float(val) for val in values])
            else:
                parameters.append(float(text))
        
        except (ValueError, SyntaxError):
            print(f'Error transforming {key} to float.')

    track = []

    track.append(self.times)
    track.append(self.ztrackFL)
    track.append(self.ztrackFR)
    track.append(self.ztrackRL)
    track.append(self.ztrackRR)
    track.append(self.Ax)
    track.append(self.Ay)
    track.append(self.vCar)
    track.append(self.rPedal)
    track.append(self.pBrake)

    args = zip(parameters_damper_front, parameters_damper_rear, itertools.repeat(parameters), itertools.repeat(track))

    print((parameters))

    with ProcessPoolExecutor(max_workers=len(parameters_damper_front)) as executor:
        results = list(executor.map(run_simulation, args))

        directory = QFileDialog.getExistingDirectory(None, 'Select directory to save simulations')

        csv_filename = os.path.join(directory, f'{self.sweep_name.text()}.csv')

        t_list = [res.t for res in results]
        y_list = [res.y for res in results]

        max_len = max(len(t) for t in t_list)

        for i in range(len(results)):
            y_list[i]

        header = ['Time (s)']
        for i in range(len(results)):
            for j in range(15):
                if j == 0:
                    header.append(f'Sim {i+1} - pBrake (bar)')
                elif j == 1:
                    header.append(f'Sim {i+1} - rPedal (%)')
                elif j == 2:
                    header.append(f'Sim {i+1} - zTrack FL (m)')
                elif j == 3:
                    header.append(f'Sim {i+1} - zTrack FR (m)')
                elif j == 4:
                    header.append(f'Sim {i+1} - zTrack RL (m)')
                elif j == 5:
                    header.append(f'Sim {i+1} - zTrack RR (m)')
                elif j == 6:
                    header.append(f'Sim {i+1} - Heave (m)')
                elif j == 7:
                    header.append(f'Sim {i+1} - Pitch (deg)')
                elif j == 8:
                    header.append(f'Sim {i+1} - Roll (deg)')
                elif j == 9:
                    header.append(f'Sim {i+1} - FRH (m)')
                elif j == 10:
                    header.append(f'Sim {i+1} - RRH (m)')
                elif j == 11:
                    header.append(f'Sim {i+1} - FL CPL (N)')
                elif j == 12:
                    header.append(f'Sim {i+1} - FR CPL (N)')
                elif j == 13:
                    header.append(f'Sim {i+1} - RL CPL (N)')
                elif j == 14:
                    header.append(f'Sim {i+1} - RR CPL (N)')

        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)

            writer.writerow(header)

            vCar = np.array(track[7])
            vCar = vCar

            delay = parameters[10]/vCar

            t_delay = track[0] + delay

            interp_ztrackRL = interp1d(t_delay, track[1], bounds_error=False, fill_value='extrapolate')
            ztrackRL = interp_ztrackRL(track[0])

            interp_ztrackRR = interp1d(t_delay, track[2], bounds_error=False, fill_value='extrapolate')
            ztrackRR = interp_ztrackRR(track[0])

            for row_idx in range(max_len):
                row = []

                t_values = [t[row_idx] if row_idx < len(t) else 'NaN' for t in t_list]
                row.extend([t_values[0]])

                for i in range(len(results)):
                    for j in range(15):
                        if j == 0:
                            value = track[9][row_idx]
                        elif j == 1:
                            value = track[8][row_idx]
                        elif j == 2:
                            value = track[1][row_idx]
                        elif j == 3:
                            value = track[2][row_idx]
                        elif j == 4:
                            value = ztrackRL[row_idx]
                        elif j == 5:
                            value = ztrackRR[row_idx]
                        elif j == 6:
                            value = y_list[i][0][row_idx] - parameters[9] if row_idx < len(y_list[i][j]) else 'NaN'
                        elif j == 7:
                            value = (y_list[i][2][row_idx] if row_idx < len(y_list[i][j]) else 'NaN')*(180/np.pi)
                        elif j == 8:
                            value = (y_list[i][4][row_idx] if row_idx < len(y_list[i][j]) else 'NaN')*(180/np.pi)
                        elif j == 9:
                            value = y_list[i][0][row_idx] - (parameters[10]*(1 - parameters[3]))*y_list[i][2][row_idx] + parameters[0] if row_idx < len(y_list[i][j]) else 'NaN'
                        elif j == 10:
                            value = y_list[i][0][row_idx] + parameters[10]*parameters[3]*y_list[i][2][row_idx] + parameters[1] if row_idx < len(y_list[i][j]) else 'NaN'
                        elif j == 11:
                            value = parameters[70]*(track[1][row_idx] - y_list[i][8][row_idx])
                        elif j == 12:
                            value = parameters[70]*(track[2][row_idx] - y_list[i][6][row_idx])                        
                        elif j == 13:
                            value = parameters[71]*(ztrackRL[row_idx] - y_list[i][10][row_idx])
                        elif j == 14:
                            value = parameters[71]*(ztrackRR[row_idx] - y_list[i][12][row_idx])

                        row.append(value)
                
                writer.writerow(row)

class RideSimApp(QWidget):
    def __init__(self):
        super().__init__()

        base64_data = ('iVBORw0KGgoAAAANSUhEUgAADaMAAAMlCAYAAADU+Q7fAAAACXBIWXMAAC4jAAAuIwF4pT92AAAgAElEQVR4nOzd3W9cd34m+OdUiRJVpF7KTcnqbjlkFAVopbtoqtuAA3Qb4iDAZIA0YG0ws0i6gbECLJCZi0GrL3aBYLFoLZDGXka+27uRgf0D5L8g1EUuCghCSp1Ac5HVkrA6aBlCJDq25LYtcy+qaFIURVMUyV+9fD7AQb2wqvhcEDg8p37P+VYrKysBAAAAAAAAAAAAGGaPbp2fSXJ8Dz76YWN6fmEPPhcAAGDfVcpoAAAAAAAAAAAAMDwe3Tp/PMnMNl8+1d1exMuWui68xHvpWEqyuMP3PkzyIuW5he57vvZ1jen57bwOAADoYcpoAAAAAAAAAAAAsE8e3To/u8WPp7J18evrfp50imDHXiAS9JIbW/zs60pyc895frExPb+400AAAMDTlNEAAAAAAAAAAAAYaltMCttqgthW079M9oLet5zNy22L2Xyq3GYT4BTdAAAYOspoAAAAAAAAAAAA9IXnTBWbyubTwjZ77fEkr+9aIIBnLeXZMttmRba5DY8V2wAA6AvKaAAAAAAAAAAAAOyJTcpjm00ae95zSmPAsLux4fHchscbS24Ljen5jaU3AADYVcpoAAAAAAAAAAAAfOXRrfNTeXbS2OyGx5u9ZibJsT2IBMDOrC+zPUynvLZqMesmuDWm5+f2JREAAH1PGQ0AAAAAAAAAAGBAbFIk22zq2OyGx0pkAKy3lHVFtTw9kW39NLaHjen59QU3AACGgDIaAAAAAAAAAABAj9ikTLadx5N7mQkAtmE5T09em3vO/YXG9PzDAADQt5TRAAAAAAAAAAAAdtEmhbLZDS9Z//h4ktf3NBAA9KYb6+7PbXLf5DUAgB6kjAYAAAAAAAAAALCJR7fOz657OJWnC2Yz6RTJEoUyANhrS0kWu/cXkqxOV5vr3iquAQDsE2U0AAAAAAAAAABgoD26dX59cWwqz59aNpVkcj8yAQB75maeLastdrc0pufnNr4BAIDtU0YDAAAAAAAAAAD6xoZpZc8rmZlUBgB8nfXT1ua6t4vdzaQ1AIDnUEYDAAAAAAAAAACKUCwDAPrA6qS1xTxbXltoTM8/fPYtAACDSxkNAAAAAAAAAAB4aY9unT+eTqEs6RTINrufJBf2MxcAwD5YnbL2MMnqRLW5JGlMz88VSQQAsEeU0QAAAAAAAAAAgGc8unV+KmvTyba6P7lfmQAA+tjGCWur28PG9PzC894EANBrlNEAAAAAAAAAAGBIPLp1frZ7d6spZq/vcywAAJLldKaqPdxwa7oaANBTlNEAAAAAAAAAAKBPbZheNpNOmWzj/Qv7mwoAgD3wvLKayWoAwL5SRgMAAAAAAAAAgB7y6Nb59UWy2e6t6WUAAGxlY1ltcXVrTM8vFksFAAwcZTQAAAAAAAAAANhjGyaYzXZv1xfMppJM7mcmAACGys08PVVtLqaqAQA7oIwGAAAAAAAAAAA78OjW+Y1lsqnu/dnurQlmDIXPfr2y9CKvX3mSAx///coXO/19K19k9KO/W3l1p+/fjk8WBnNd3dhMtXef/Xr1LweO5/OdvPfgt6raodfy5XZfXx2qRkcmsqd/AwBDZinrJqmlW1hrTM/PFUsEAPQsZTQAAAAAAAAAAFjn0a3zM+kUyZK1YtlU1spmM0mO7Wsoht7n93Nv5bcrnz7v518sp/b4f6xsWeb5/H6OPvqnlebX/a4nHyef/rN1ZQyG0bNV6uNbv+bQ6erx6O/lw61eM/5GdaCqZ9MSpXIcMOCW0ymnLUZRDQCIMhoAAAAAAAAAAEPi0a3zs927U3m6WHY8ppixTSuf5/HnH65sWlrZqhD2dUWwT/95JU8+3q2UQC953kS6+pFk/PvVppMFD3+nqh049uy0uAPNarzWyDd2OSLATq1OVFtI8jDJXDpFtYWCmQCAPaaMBgAAAAAAAABAX3tOyWz9c5P7mYeyPvv1yjPFjpUnOfDx3688M9Fo5YuMfvR3K5tOM1IOA/rJZhPgnld226zoNnKyOlmN5PDepgSGzM2sFdQWu9tCY3r+YblIAMBuUEYDAAAAAAAAAKAnKZkNts/v597Kb1c+Xf/cbz9I7bN/eXqy2PMKY5//JvnsN9Y+AeyVzQpuje9WD0Ym8tHq49pYdWDse3mq7Dtysnq1Gsno/qQE+tByOpPUFrvbXJLFxvT8YrFEAMALUUYDAAAAAAAAAGBfPbp1fibJ8e420316/XOvF4rGFp58kgdPHq58tP65T/4xB7785OmJYx//w8rkk397+r2mjAEMr/p4p9i23tEfVveqA/mqkLxxWptCGwytm+lOUOtui43p+YWiiQCAZyijAQAAAAAAAACwKx7dOr9ZuWwqa1PNLux/KjZOIPtiObXH/+Pp6WObFcg+WbCuCIDesHFK28YJbWPnq9HawU65rTpUjY5M5JmJmkBfU1IDgB6ijAYAAAAAAAAAwNd6zjSz2e7tVJLJ/U81+J78W5affLTycPXxbz9I7bN/WSuSfX4/Rx/900pz/XuUyABgzcFTVUZOrT1eP5Xt4Leq2qHX1iayHfx25f8Z6C9KagBQgDIaAAAAAAAAAMCQe3Tr/Gz37maFM9PMXsLGMtkn/5gDX36y8sXq440TyT7955U8+Xh/MwIAz1pfYqsfSca/Xy2t/mz8jepAVc8XSTJysjpZjeRwmZTAc6wvqc2lU1JbLJgHAAaKMhoAAAAAAAAAwIB6dOv8+lLZatFsqrsdT/J6kWB95vP7ubfy25VPk2TlSQ58/PdrZbKNk8k+/03y2W+sxwGAYfS8AlttrDow9r1Oea06VI2OTOTVcilhqN3I05PUFhrT8w+3fAcA8AxlNAAAAAAAAACAPrShaDbbvV1fOJvc/1S9a+XzPP78w5UPVx+vn1C28kVGP/q7la8WhSuUAQD7oT6ejJ6tOvfXlddGTlajo7+bT5Okfrw6Wh9Lc4uPAV7OUjoFtbl0CmqLjen5hZKBAKDXKaMBAAAAAAAAAPQYRbOtffbrlaXV++tLZRunlH36zyt58nGJhAAAu2/0bJX6eOf+0R9W96oDncLakTe7hTbFNdhNpqgBwHMoowEAAAAAAAAA7LNHt87Pdu+uL5itbkNTNHvySR48ebjyUZKsPMmBj/++UypLko//YWXyyb917ptUBgDwYjYrrh38VlU79Fq+TJKRk9XJaiSHS2aEPrSUtXLaXDpT1BZLBgKAEpTRAAAAAAAAAAB20aNb5zcWzFannB1P8nqxYHts5fM8/vzDlQ+TrYtlppUBAPSO+ninuJYkje9WD0Ym8lFtrDow9r18kSitwTYsZ62ctjpBbbFkIADYa8poAAAAAAAAAADb9OjW+fXFspnu07Pd25kkxwrE2hPrp5b99oPUPvuXlS+T5NP/Nyd/e3flcJI8+bhTLgMAYLBtVlpbP2nt4LeroZnuC9t0I2tT1BYa0/MLhfMAwK5RRgMAAAAAAAAA6Fo31WzjdLOpJH29wPbz+7m38tuVT5Pkk3/MgS8/6UwuM7UMAIDdMnq2Sn08qR9Jxr9fLSXJ+BvVgaqeL0xZAwU1AAaDMhoAAAAAAAAAMLQe3To/1717oWSOnVj5PI8//3Dlw+T5k8s+/03y2W+sDQEAoHccPFVl5JTCGnStL6jNNabnF8vGAYCvp4wGAAAAAAAAAAylR7fOzyb529I51ttsetnKFxn96O9WXk2SJx93ppcBAMAgWy2sHTpdPR79vXxYG6sOjH0vX1SHqtGRibxaOh/soeV0i2lZm6C2WDIQAGykjAYAAAAAAAAADKVHt85fTfKzvf49qwWzlSc58PHfr3zRfe7oo39aaSamlwEAwE6MzVRJkqM/rO5VB/Lp4e9UtQPH8qXpagygpaybnpZOQe1h0UQADDVlNAAAAAAAAABgKD26dX4hyes7ee+TT/LgycOVj5LNJ5gpmAEAQFmjZ6vUx5PGd6sHIxP5SFmNAXMz6wpqjen5hcJ5ABgiymgAAAAAAAAAwNB5dOv88SQPNj7/2a9XlpLktx+k9tm/rHyZJB//w8rkk3/r/PyTBessAABgEDyvrHbw29Vk6WywQzfSnZyWTkHN9DQA9oQyGgAAAAAAAAAwdG7+YPqn9aP5fxJTzAC2cDNJqUXMCwV/92amutt+m0lyrMDvBRh6YzNVkuToD6t79WPVF2PfyxfVoWp0ZCKvFo4G27WUp8tppqcBsCuU0QAAAAAAAACAodNutq4lead0DqCvLSVZ3OZrF1/gtat2WsZ6+OaDX1lozFfazdZMkuMv+LapbL98t93XKtYBA+PgqSojp5JDp6vHo7+XD01Vo4+YngbAS1NGAwAAAAAAAACGTrvZWkxioSj0j6+b0PUwnQW1W9nOa5Jk4c0Hv7IoFwpoN1uzX/OSrX4+leeX4qZivw/so7GZKvUjyfj3q6WD36pqh17LlyMnq5PVSA6XzgYb3Ey3mJZkwfQ0ALZDGQ0AAAAAAAAAGCrtZmsqyf9XOgf0qa1KYVtN8vq6IpgCGFDEFgW4502U2+z1x5O8vkuRgAGnqEaPW06nmDaXTjltrmQYAHqTMhoAAAAAAAAAMFTazdblJH9TOgfsshvPef55BbGtymGKYQC74DlFt81Kbps9d2EvMgG9bbOi2sFvVyY7UtqNdAtqymkAJMpoAAAAAAAAAMCQaTdb15O8XToHQ+d5E8XmNnnueUWxh28++NVW08UAGFDtZut4OqW19TaW2DZ7jVIbDID6eDJ6tsqh09Xj0d/Lh+NvVAdqjerAyEReLZ2NoXQza9PT5hrT8y5kATBklNEAAAAAAAAAgKHSbrYslmArS0kWNzz3vHLYZlPHFMYA6FmbTGvb6vHxJK/vYRxgF4yerVIfT47+sLp36Heqzw69li9HTlYnq5EcLp2NoaGcBjBklNEAAAAAAAAAgKHRXYD9t6VzsKtubPLc3CbPbVYcW3jzwa8slASAbWg3W1NJptY9tX4y28apbDNJju1LMGBTm01TO/CNqlEfS7N0NgbeUp4upy2WDAPA7lNGAwAAAAAAAACGRrvZuprkZ6VzDLnlPDtlbDHPTiNTHgOAAdButjYrqq2W2KayVnAziQ32ydhMlfqRZPz71dLh71S1kZPVwZGJvFo6FwNrfTntuslpAP1PGQ0AAAAAAAAAGBrtZmshFjnv1FKeLYzNbXi8uMlrFMgAgBfWnWi7av399WU2E9hgF42erXLwm0pq7LmbeXpymuNFgD6jjAYAAAAAAAAADIXuVI4HpXMUcmPD441Txx5mw7SyNx/8am6PMwEA7Jp2s7W+pDa77ker901egx1SUmOPKacB9BllNAAAAAAAAABgKLSbrUtJ/nvpHDtwM08XxzYWyRbz9DSyh28++NVTxTIAANa0m62pJFPdh5uV2BTXYBtGz1ZpfKd6PPp7+XD8jerAgW9UjfpYmqVz0fduZK2YNlc2CgCbUUYDAAAAAAAAAIZCu9m6luSdff61y9kwcSydRXXPfWwiGQBA79hQXJvt3q5/bibJsX2MBD1vbKZK47vVg0OT1aOx7+WLkZPVyWokh0vnom+9n7VymguvAPQAZTQAAAAAAAAAYCi0m63FJJM7eOvGyWRzG37+1GNlMgCA4dNuto6nU0xLNi+tXdjfRNBbDp6qMnIqOfrD6l7je9VnIyergyMTebV0LvrOcpLrWSunLRZNAzCklNEAAAAAAAAAgIHXnWhxrftwsbvlOY8X3nzwq/XlMwAA2BXtZmsmyfHutrG8ZsoaQ2dspsqh09XjxnT1r6aosQM3s1ZMu144C8DQUEYDAAAAAAAAAAAAgB7xnClrqyU2hTUG3sYpaocmq/H6WJqlc9EXbqQ7Oa0xPb9QOgzAoFJGAwAAAAAAAAAAAIA+0p38O5W1gtr6AtuFMqlgb43NVGl8t3owNl19PHq2OjgykVdLZ6KnLadbTEtyvTE9bwI6wC5RRgMAAAAAAAAAAACAAdNutma7d1dvTVdj4IzNVDl0unrcmK7+dex7+eLgt6vJ0pnoWTfTKaddNzUN4OUoowEAAAAAAAAAAADAkGk3W+vLaRtvldXoW6NnqzS+o6DGlkxNA3gJymgAAAAAAAAAAAAAwFM2maw2m05Z7fUCceClKKjxNUxNA3gBymgAAAAAAAAAAAAAwLa1m63VCWpTGzZT1egbCmo8x1K6E9OSzJmaBvAsZTQAAAAAAAAAAAAAYNd0p6qtFtbWF9eUfehpYzNVGt+tHoxNVx+Pnq0Ojkzk1dKZKO79dMtpjen5xbJRAHqDMhoAAAAAAAAAAAAAsC/azdZqQW02a0U1E9XoWWMzVY7+sLrX+F712aHJarw+lmbpTBRzM51i2rXG9PxC4SwAxSijAQAAAAAAAAAAAADFbZioNtXdLpRLBM86eKrK6O8n49+vlsbfqA4c+p3q26UzUcRykuvdba4xPf+wcB6AfaOMBgAAAAAAAAAAAAD0rHaztX6C2upUtakkk+VSwZrRs1Ua36keH32ruj96tjo4MpFXS2di372ftWLaYuEsAHtKGQ0AAAAAAAAAAAAA6EvtZmt1itr6spppahRVH+8U1I7+sLp35A+rLw5+s3qlGsnh0rnYNzeTXEtyXTENGETKaAAAAAAAAAAAAADAQGk3W1PplNRmu7dTUVKjoPXT0xqtarw+lmbpTOyLpXQmpl1rTM8vlA4DsBuU0QAAAAAAAAAAAACAobCupLY6SW0qSmoUUB9Pxs5XGf9+tTT+RnXg0O9U3y6diT23Wkyba0zPXy8dBmCnlNEAAAAAAAAAAAAAgKFmkhq9YGymSuO71YOjP6o+Pvz71UQ1ksOlM7FnltMppl1XTAP6jTIaAAAAAAAAAAAAAMAmuiW1mTw9Se31gpEYIqNnqzS+Uz0++lZ1v9GqxutjaZbOxJ5QTAP6ijIaAAAAAAAAAAAAAMALaDdbs+kU09YX1Y4VjMQQOHiqyvgb1ePGdPWv4z+oDoxM5NXSmdh1q8W0uXTKaQ/LxgF4ljIaAAAAAAAAAAAAAMBL6k5Rm0oyG1PU2Af18WTsfJWjb9V+rZw2sN7P2tQ0xTSgJyijAQAAAAAAAAAAAADske4UtdVy2kySCyXzMLiU0waeYhrQE5TRAAAAAAAAAAAAAAD2UbvZmkmnmLZ+O1Y0FANHOW1gLWetlHa9dBhg+CijAQAAAAAAAAAAAAAU1m62prJWTJuNghq7TDltICmmAftOGQ0AAAAAAAAAAAAAoAcpqLGXlNMGjmIasC+U0QAAAAAAAAAAAAAA+kS3oDabtZLahZJ5GBwHT1UZf6N6fPSt6n6jVY3Xx9IsnYkdW0qnmHatMT2/UDoMMFiU0QAAAAAAAAAAAAAA+li72do4Pe31ooEYCKvltOafVPcP/341UY3kcOlM7MhSkmvpFNMWy0YBBoEyGgAAAAAAAAAAAADAgGk3W7N5eoLaZMk89L/Rs1WOvFk9aP6H6tGh36m+XToPO3IznWLadcU0YKeU0QAAAAAAAAAAAAAABly72ZrK09PTLpTMQ/87+laVo2/Vfj3+g+rAyEReLZ2HF/Z+kuvpFNMelg4D9A9lNAAAAAAAAAAAAACAIdSdnra+oGZ6Gjty8FSV8Teqx0ffqu6P/6CaqEZyuHQmtm05a6W066XDAL1PGQ0AAAAAAAAAAAAAgI3T02aTvF4wDn1s9GyV5h9X9478YfXFod+pvl06D9u2lE4x7Vpjen6hdBigNymjAQAAAAAAAAAAAACwqe70tNVtJsmxgnHoQ/XxZOx8lVd+XPug0arG62Npls7EttxMci2dYtrDwlmAHqKMBgAAAAAAAAAAAADAtrSbrfWT02aSTJbMQ/8ZPVvlyJvVg+Z/qB6ZmtY33k+nlHa9dBCgPGU0AAAAAAAAAAAAAAB2pN1sTWWtnDYb5TRewPqpaeM/qCaqkRwunYktLWdtWtpC4SxAIcpoAAAAAAAAAAAAAADsCuU0Xsbo2SrNP67uHZut1UdOZqJ0HrZ0M2vFtIeFswD7SBkNAAAAAAAAAAAAAIA90S2nzaRTTLsY5TS26eCpKuNvVI+bf1LdP/z7pqb1uPeSXG9Mz18vHQTYe8poAAAAAAAAAAAAAADsC5PT2Kmjb1Vp/ofavbHz1cH6WJql87CppaxNS1ssGwXYK8poAAAAAAAAAAAAAAAUoZzGToyerdL84+resdlafeRkJkrnYVM30imlXSsdBNhdymgAAAAAAAAAAAAAAPSEbjntYtbKaccKxqEPHDxV5dgfVQ+O/qj6uPEH1Wul8/CM5XSmpV01LQ0GgzIaAAAAAAAAAAAAAAA9qd1szWStnHahbBp6XX08OTZbe3z0rer++A+qiWokh0tn4immpcEAUEYDAAAAAAAAAAAAAKAvtJut9VPTXi8ahp539K0qr/y49oFiWs8xLQ36mDIaAAAAAAAAAAAAAAB9p91sTaVTSlstqB0rGIceN3q2ysSfVvePztbq9bE0S+fhK6alQZ9RRgMAAAAAAAAAAAAAoO+1m62ZdIppF2NqGltQTOtJS+lMS7tmWhr0NmU0AAAAAAAAAAAAAAAGSrvZOp61iWkXY2oaz6GY1pPeT3K1MT0/VzoI8CxlNAAAAAAAAAAAAAAABpqpaWzHwVNVjv1R9eAbF2tPRk5monQespTkSpLrjen5h4WzAF3KaAAAAAAAAAAAAAAADI12szWVtYlpszE1jU0opvWU5STXk1xpTM8vFs4CQ08ZDQAAAAAAAAAAAACAodVutlZLaReTTJZNQy8aPVtl4k+r+0dna/X6WJql8wy5G0muNqbnr5cOAsNKGQ0AAAAAAAAAAAAAAJK0m62ZdIppl5K8XjQMPUkxrWcsJbma5Fpjev5h6TAwTJTRAAAAAAAAAAAAAABgg3azNZW1iWlvFw1DT1JM6wnLSa6lMy1tsWwUGA7KaAAAAAAAAAAAAAAAsIV2s3U8a8W0i0mOFQ1Ezzn6VpVXflz7YPwH1UQ1ksOl8wyp99Mppc2VDgKDTBkNAAAAAAAAAAAAAABeQLvZWi2lzSaZLJuGXnP0rSonflL7oPEH1Wulswypm+mU0q6VDgKDSBkNAAAAAAAAAAAAAAB2qN1szSS5lE45TTGNr9THk2Oztcff+I/VR6O/W71aOs8QWkpyLZ1i2sPCWWBgKKMBAAAAAAAAAAAAAMAuUEzjeQ6eqnLsj6oHJ/6sdqB+NEdK5xkyy0muJ7nSmJ5fLJwF+p4yGgAAAAAAAAAAAAAA7DLFNJ5n9GyViT+t7h//97WxaiSHS+cZMu8ludaYnp8rHQT6lTIaAAAAAAAAAAAAAADsIcU0nufoW1Ve+XHtgyN/WL1WOsuQuZHkamN6/nrpINBvlNEAAAAAAAAAAAAAAGCfKKaxmfp4cmy29vjkO7VPRk5monSeIbKU5Epjev5a6SDQL5TRAAAAAAAAAAAAAACggHXFtEtJjhUNQ884eKrKyf9c3T86W6vXx9IsnWdILCW5muRaY3r+Yekw0MuU0QAAAAAAAAAAAAAAoLB2s3UxnWlpF6OYRtfRt6qc+Entg8YfVK+VzjIkltMppV1VSoPNKaMBAAAAAAAAAAAAAEAP6RbTLiV5u3AUekR9PHnl7dqDE39eq9WPKCvug+Uk15NcaUzPLxbOAj1FGQ0AAAAAAAAAAAAAAHpQu9k6ns6ktEtJLpRNQ68YPVvl1b+o3TvyZnW0Gsnh0nmGwHtRSoOvKKMBAAAAAAAAAAAAAECPazdbU+kU0y4nmSybhl5QH0+OzdYen3yn9snIyUyUzjMElNIgymgAAAAAAAAAAAAAANBX2s3WTDqltItJjhWOQw8wLW1f3UinlDZXOgiUoIwGAAAAAAAAAAAAAAB9qt1sXUqnlPZ24Sj0ANPS9pVSGkNJGQ0AAAAAAAAAAAAAAPpcu9maSqeUdjnJZNk09ALT0vaNUhpDRRkNAAAAAAAAAAAAAAAGSLvZmkmnlHYxybHCcSisPp688nbtwYk/r9XqR/w97CGlNIaCMhoAAAAAAAAAAAAAAAygdrN1PGvT0l4vHIceMDZT5cSf1z448ofVa6WzDDClNAaaMhoAAAAAAAAAAAAAAAy4drM1lU4p7VJMSxt6B09VeeXtavmVt2tf1sfSLJ1nQCmlMZCU0QAAAAAAAAAAAAAAYIi0m61L6ZTSLpRNQi945ce1x9/4j9VHo79bvVo6y4BSSmOgKKMBAAAAAAAAAAAAAMAQMi2N9UbPVnn1L2r3jv5IKW2PKKUxEJTRAAAAAAAAAAAAAABgiLWbreNJLqZTTHu9cBwKq48nJ35aW37l7dqX9bE0S+cZQEpp9DVlNAAAAAAAAAAAAAAAIEnSbrZm0imlvVM6C+W98uPa45Pv1D4ZOZmJ0lkG0HvplNIWSweBF6GMBgAAAAAAAAAAAAAAPKU7Le1SOsW0ybJpKG1spsqJP699cOQPq9dKZxlASmn0FWU0AAAAAAAAAAAAAADgudrN1sV0SmkXSmehrIOnqpz8z9X94/++NlaN5HDpPAPmvSSXG9PzD0sHga0oowEAAAAAAAAAAAAAAF+r3WxNJbmS5GKSY0XDUFR9PDnx09ryK2/XvqyPpVk6zwBZTnI1yVWlNHqVMhoAAAAAAAAAAAAAALBt7WbreJJL6UxLmyybhtJe+XHt8an/UvusfkRBcRcppdGzlNEAAAAAAAAAAAAAAIAdaTdbl9Ippl0om4TSxmaqfOty7d7o71avls4yQJaSXGlMz18rHQRWKaMBAAAAAAAAAAAAAAAvpd1szaQzKe2d0lkoa/RslVf/onbv6I+U0nbRUpJLjen5udJBQBkNAAAAAAAAAAAAAADYFe1mayqdUtqlJMeKhqGog6eqfPO/KaXtshvpTEqbKx2E4aWMBgAAAAAAAAAAAAAA7Kp2s3U8a6W0ybJpKKk+npz4aW154j/VDlYjOVw6z4B4L51S2mLpIAwfZTQAAAAAAAAAAAAAAENtcjsAACAASURBVGDPtJutS0muRCltqCml7Yl30ymlPSwdhOGhjAYAAAAAAAAAAAAAAOy5drN1MZ1paRdKZ6Gc+njyytu1Byf+vFarH8mx0nkGwHI6hbSrpYMwHJTRAAAAAAAAAAAAAACAfdNutmbTmZSmlDbkXvlx7fGp/1L7TCltVywludyYnr9eOgiDTRkNAAAAAAAAAAAAAADYd+1mayadSWnvlM5CWUppu+pGOqW0hdJBGEzKaAAAAAAAAAAAAAAAQDHtZmsqnUlpSmlDTiltV72XTintYekgDBZlNAAAAAAAAAAAAAAAoDilNFYppe2a5SRXG9PzV0oHYXAoowEAAAAAAAAAAAAAAD1DKY1VSmm7ZimdKWnXSweh/ymjAQAAAAAAAAAAAAAAPUcpjVVKabvmRpJLjen5xdJB6F/KaAAAAAAAAAAAAAAAQM9SSmOVUtqueTfJlcb0/MPSQeg/ymgAAAAAAAAAAAAAAEDPU0pj1am/rC1P/KfawWokh0tn6WPLSS43puevlQ5Cf1FGAwAAAAAAAAAAAAAA+oZSGklSH09O/FQpbRfcSKeUtlA6CP1BGQ0AAAAAAAAAAAAAAOg73VLatSQXyiahpNVS2omf1I6VztLn3k1ypTE9/7B0EHqbMhoAAAAAAAAAAAAAANC32s3WbDqT0pTShtjBU1W++d9q947+qHq1dJY+tpzOlLRrpYPQu5TRAAAAAAAAAAAAAACAvtctpV1N8nrhKBR08FSV039Vuz82U02UztLHbqRTSlsoHYTeo4wGAAAAAAAAAAAAAAAMjHazdSmdSWmTZZNQ0thMlW9drt0b/V2T0l7Cu0muNKbnH5YOQu9QRgMAAAAAAAAAAAAAAAZOu9m6kuRykmOFo1DQsQvVo2//r/XP60f8HezQUjpT0q6XDkJvUEYDAAAAAAAAAAAAAAAGUrvZOp5OIe0XpbNQ1sSf1T499b/UVqqRHC6dpU+9n04pbbF0EMpSRgMAAAAAAAAAAAAAAAZau9maSnIlyTtlk1BSfTw58dPa8omf1ExJ25nlJFca0/NXSwehHGU0AAAAAAAAAAAAAABgKLSbrdl0SmkXyiahpIOnqpz+q9r9sZlqonSWPnUzyaXG9PxC6SDsP2U0AAAAAAAAAAAAAABgqLSbrUvplNImyyahpLGZKq/97/X7IyejlLYz76YzKe1h6SDsH2U0AAAAAAAAAAAAAABg6LSbreNJLne3Y4XjUNArP649/tblWlWNZLR0lj60lM6UtLnSQdgfymgAAAAAAAAAAAAAAMDQajdbU0muJnm7cBQKqo8nJ35aWz7xk5pi4s68l+SyKWmDTxkNAAAAAAAAAAAAAAAYeu1mazbJtSSTZZNQ0sFTVSZ/Wbs/eraaKJ2lDy2nMyXteukg7B1lNAAAAAAAAAAAAAAAgK52s3U5yZUkJmQNsbGZKpN/XV+uH/F3sAPvp1NKMyVtACmjAQAAAAAAAAAAAAAArNNuto4nuZrkndJZKOvUX9aWJ/7n2kh1II3SWfqMKWkDShkNAAAAAAAAAAAAAABgE+1mazbJtSSTZZNQ0sFTVU7/Ve3+2Ew1UTpLHzIlbcAoowEAAAAAAAAAAAAAAGyh3WxdSXI5ybHCUShobKbK5F/Xl+tH/B28IFPSBogyGgAAAAAAAAAAAAAAwNdoN1tT6UxJu1A2CSXVx5MTP60tn/hJTSHtxZmSNgCU0QAAAAAAAAAAAAAAALap3WxdTKeUpow0xA6eqjL5y9r90bPVROksfcaUtD6njAYAAAAAAAAAAAAAAPAC2s3W8SRXkvyscBQKe+XHtcffulyrqpGMls7SZ0xJ61PKaAAAAAAAAAAAAAAAADvQbrZm05mSNlk2CSXVx5PTf1W/d/RH1auls/SZpXQKaXOlg7B9ymgAAAAAAAAAAAAAAAAvod1sXUnyi9I5KGtspsrkX9eX60dyrHSWPvNuY3r+cukQbI8yGgAAAAAAAAAAAAAAwEtqN1sz6UxJe71wFAqqjycnflpbPvGTmkLai7mZzpS0hdJB2JoyGgAAAAAAAAAAAAAAwC4xJY0kGT1bZer/qt8fOZmJ0ln6zM8b0/NXS4fg+ZTRAAAAAAAAAAAAAAAAdpEpaaw69ZempO3AjSQXG9PzD0sH4VnKaAAAAAAAAAAAAAAAAHvAlDQSU9J2aDnJpcb0/PXSQXiaMhoAAAAAAAAAAAAAAMAeMSWNVaak7ci7Sa6YktY7lNEAAAAAAAAAAAAAAAD2WLvZuprkZ6VzUJYpaTtyM50paQulg6CMBi/k3JnTM0mOl84BwPbcvnN3brc/89yZ01NJpnb7cwG24eHtO3cdSAMAQNcW52tn9zkKsHMPk2x2rLt4+87dxX3OAj3n3JnTx5PMlM4BbMk+i6HgO1KAnuT7Y4A+1m62ZtOZkjZZNgmlmZL2wpaTXG5Mz18rHWTYKaMxtNYtVJjK2gmz9YsXjscYVIB+duP2nbuzu/2h586cXoj9A9B7bqazgG+9uXX3Ny7uW7h9566R5UPi3JnTc0kulM4BwEtZSrK4xc/nNnluIU//f2CBJD1v3eLG9Qvv15+znYovZWFYbdwXznVv1x/v2tfR0zYUy2a7txvLZjNJLDyBwbDxnO1i1vZl6/dfFpFTzLp902b7qKk4/gLodT+/fefu1dIhANi5drN1PMnVJO+UzkJZpqTtyPvpTEmzBq4QZTQG1rqFC7Pdp1ZvLcIEGHzLSWZ2e/HJuTOnryT5xW5+JkAPWF0UsX4BxOridQv5+ty5M6cvJ/mb0jkA6EnrF0YuZm1R5Fcltr2YNg0bztuuLnicikWOwO660b2d696u7t9cnIU91b0Y5lTW9m+rm/0csB2rx2mL3W31nK3CGi+lexw2s247HmtnAPrdnlygGYAy2s3WxXSmpLlI0ZA7/b/V7jf/pKaQtn1LSS42puedNylAGY2+t8lJs6mYWAMw7P7i9p2713bzA7uLCOZ38zMB+shy1i3cy9piCIv4elj3WGkhTlYC8PI2Loj8alNcZyvnzpyezdPnbS12BHrFjWw4xlXC5kX4fhIoZPU87WJ3m4vjMjboTjubTWf/tHrrHDHAYNmTCzQDUFa72ZpKp5Dmu5QhNzZTZfKv68v1I47lXsDPG9PzJsbuM2U0+kr3i53ZrH2xY4cLwEbv375z9+Juf+i5M6cXYjEBwPOsX8S3kM4CCFecKezcmdNzccwEwP5YLavNxYL+obVuweNsnLsF+tdS1i3uj30asagf6Bs38nRJzYXEhsSGY7HZ+D4TYBj8/PaduxZbAwyodrN1JckvSuegrPp4MvnL+v2xmcqUtO17L8nlxvS88yH7RBmNnta9cu7q5osdAL7OcpKp3f5y7dyZ01fiAA9gJ1YXQCyks/hhrmiaIXLuzOnLSf6mdA4Aht76Bf2r/w8sFszDLuuev70YCx6BwXczaxdgcXw74NZdHHN1myyXBuClLGVt/zVn/zU4zp05PZPOsdjFOBYDGDY3bt+5O1s6BAB7q91szaYzJc15qSF37EL16LX/o15VIzlcOkufuJnkUmN63kXU94EyGj1lQ/nMlXMBeFH/0+07d6/v5gd2v8yZ383PBBhyqwv45tJZALFYNM0A6u675uJiHgD0puWs/S+wuiDS1en6yLkzp1cXPF6M/zeA4XYzT+/PFoum4aV092+z6ezfLPIBBtmNPH1+1vFYn7CvAiCdc6szjj8BhkO72TqeTiHt7cJRKOzgqSqTv6zdHz1rSto2LaczIe1a6SCDThmNorpXFlw9YTYbixcA2Ln3b9+5e3E3P/DcmdPH0/kyzhUFAfbOcroLH9JZ/ODKNC/p3JnTC7HvAqC/rC7mn4vFkD1JAQ1gW5by9P5ssWQYtrbhO0oLeoBhdiOdfdd152Z7T/fCY5eigAZAx89v37l7tXQIAPZXu9m6nORvSuegvFN/WVs+8ZOa7+m2793G9Pzl0iEGmTIa+27dybLZWCAJwO5YTjK12wsWz505fTXJz3bzMwH4WstJrmdtAYTF6C/g3JnTV5L8onQOAHhJq+W067fv3J0rG2V4dRfpX+puFj0CvLjVctr1KFv3hHUFtEvxHSXAZlbPzdp3FdS9WOal2F8B8LQbt+/cnS0dAoAy2s3WTDrHar6vGXJjM1Um/7q+XD/i4pHbdDPJbGN63jmOPaCMxr5wtSYA9ti/2+0FiufOnJ5N8re7+ZkA7MjNJNfSWYy+WDZKb+sed82XzgEAu+yrxZC379y9XjrMMOgeD1+OKTEAu231+NZU8H3UXdB/MZ19mwX9AC/m/XSOxa6VDjIMuud3Lyd5p3QWAHrOcpIZ35UCDLd2s3U8nfOLvr8ZcvXxZPKX9ftjM9VE6Sx9YjnJxcb0/FzpIINGGY0907264OUooAGwt969fefuro7S7S5QWIj9F0CvUUzbwrkzpxdiYR0Ag221mHbNxLTdd+7M6UuxUB9gvyxlbZ+mmLYHuuXqS7GgH2A3OBbbQ91jsUtJLpRNAkAP+/ntO3evlg4BQG9oN1tXkvyidA7Km/iz2qff/K+10dI5+sjPG9Pz/qfaRcpo7CpXFwRgny2lc/WnXR2he+7M6atJfrabnwnArruZ5Go6xbShH6V+7szpK3GyEYDhspROSf2akvrL6S58vBIXZAEoxT5tl3S/p7yUzveU9msAe2MpnfOy15yXfTmOxQDYphu379ydLR0CgN7SbrZm07loyLHCUShs9GyVM1fry/Uj/ha26b0klxvT885p7AJlNHbFuTOnZ9L5YsfVBQHYT/9ut6/A2L1i7t/u5mcCsKdWr8p7dVivJt89HpsvnQMACnovrtD/wrrHv1fjomIAvcSFV3bg3JnTU+ks5r8YC3AA9svqedkrytQvRgkNgBewnM4FmhdLBwGg97Sbral0jst8zzPk6uPJ5C/r98dmqonSWfrEzSSzCmkvTxmNl9I9SWYKGgAlvHv7zt3Lu/mB3SvnLsSXPwD96kY6pbTrpYPsp3NnTi/EMRkAJJ3/Ba4opW2tu2D/WpILZZMAsAUL/LehW6y+FBfLBCjtvdhnfa1zZ05fTKd07ntIALbr57fv3L1aOgQAva3dbF2L82Mkmfiz2qff/K+10dI5+sRyOoW0obzw+W5RRuOFdRfqX07nyx0nyQAo4ebtO3dndvtDz505fTXJz3b7cwHYd0vpLH64VjrIXjt35vSVJL8onQMAeoxS2ibWndf1vwNAf7Ff26BbQrsSxWqAXqOUtgn7LQB26MbtO3dnS4cAoD+0m61LSf576RyUN3q2ytn/u/5pNRKltO35i8b0/LXSIfqVMhrbtm6xwuUkxwrHAWC4nb995+6uXpGg+0XQ3+7mZwJQ3FKSS4O6YO/cmdMzSeZL5wCAHvZ+kssWQn51zHstLi4G0M+G5sIrz2O6J0Df+D+TXL195+7D0kFK6q6xuRoTCgB4cctJZpzXBOBFtJutmSRzscZ/6NXHkzPv1u+Pnq0mSmfpE+/+/+zdT2wcaXrn+Z+kqkGV+o/Egjwe2OERO9BeIxagRJ5ygDWmsi87hzYg1mUPvijlPdinFQWMDSw8aAVnMVjAPojEAsb44FVosfDMnooE2sdFJTGzBx4WIpW7iAG6EE32RI+77UKlslGtqnW1xD1EUCIpUiIZb+TzvpnfD0BIVa1+80FRjHjjjefP5RuPl6yDCBHFaHgritAAAJ5Zzosydblgfa/bEkl5ADCpNlQVpe1YB+IK9y4AAE5tpCpxf8U6EAv1niEVU8ABYJJMdOOV49RFaKlI5geAkOyqag6yZh2IhSSOeqoK0cixAQCcx71pPc8EADSzOTM3K2lN0k3jUOCB6E8ufjbz/YsUpJ3OuqTe5RuPp7qxzllRjIY3SuIoFUVoAAB/bOdFOe960SSOVkRiHgBMA+cFzVa4dwEAcGYTV5z+NvUU1Uy8cASASbWhKsl/yzqQNvGuEgCCt67qWWwqkrmY4gkAcGAjL8qudRAAgHBtzswxpRkvffB7F7/8jaWLuvCu3reOJQDbkhYv33i8Yx1IKChGw7GSOFpUdSOiyz4AwBcjSV3XyRVJHHUlfeJyTQCA17ZVJT8Em6zHvQsAgHMbqdoHTHxn/iSOliQ9sI4DADAWq6qmgE5Ukn/97JuJd5UAMAl2JS2GfCZ7GkxDAwA4MJI0P00NtQAA7dmcmUsl3beOA/be++4Fzf7Plz579x+LKWlvN5LUvXzj8USfYbhCMRoOoUsTAMBj9/KiXHG5YBJHVyVtiYQGAJhGd/KizKyDOCvuXQAAOOH8+dIX9V6BbpcAMH0mpuC6vpelYho4AEyiIM9k36a+d2WSbhmHAgAI38SeWwIAbGzOzPUkPbSOA/YufVOKVy999t53L1CQ9nYjSUuXbzzOrAPx3UXrAOCPJI5SST8WhWgAAP9stHTglopkfgCYVg+TOMqsgziHVNy7AABo6kGg+4A3qhuN9UUhGgBMoyuSPk7iaKVOiA9SPQ1tSxSiAcCkCvVM9kRJHM2rundRiAYAaKqtvBgAwBTrDAeZpAVVxTWYYs+/kH703z+/NvybF59ZxxKAK5IePnuysGQdiO+YjIb9FzuZSGgEAPhpJGk+L8odl4vW979PXK4JAAjShqTFvCifWgfyNty7AABw7lFelD3rIFyoEyD7ql6OAACm27aq59wd60DOIomjFVGEBgDTYiKexZI46qmaTM1zGACgqVbyYgAA2Lc5MzevqlbgpnEo8MAHv3fxy9+4d3Hvwju6bB1LAB5dvvG4Zx2EryhGm3L1NLT71nEAAPAG91x3f6q7A++Il0MAgMq2pK7PBWn1vWtLNBEBAMC11bwog+5qVydAPrSOAwDglZGq59wt60Depp7suSYSYQBg2gRdkEYRNQDAMed5MQAAHLU5M3dVVWNDzuGg9757Qd/9t5e+uvCu3rOOJQCPJC1dvvHY27wyKxSjTSle7AAAArGRF2XX9aJJHK1JuuV6XQBA0Lbzopy3DuIkJDcAANCqO3lRZtZBnAeFaACAt/D6HpfE0aKqjsw0DQOA6RRcQVrdNCwT7xkBAO60khcDAMBJNmfmMkm3reOAvUvflH77f33n83d/XR9YxxKAbUldCtIOu2gdAMavfrGzJQrRAAB+G0ladL1ofR/kBREA4KibSRxl1kEcp753UYgGAEB7VurmXUGpi9UpRAMAvMnDunDZO0kcLUn6WBSiAcA0u+3rfeo4dSFaX7xnBAC4M5LUsw4CADBdOsNBT9WkJ0y5519IP/qDX30w2tj73DqWANyU1H/2ZGHWOhCfUIw2ZZI4SsWLHQBAGHp5UTrtInCgWyEAAMe5XSfDeYN7FwAAY3FFgd1v6yJ6itUBAKfxMIkjryaB1/exB9ZxAAC84N196jgHCtFo+gwAcCnNi3LHOggAwPSpC9LuWMcBe8+/kH7yg+cfDP/mxWfWsQTgpqStZ08WvD/HGBeK0aZI/WLnvnUcAACcwnpelGstrJuJgmwAwJulnk1GycS9CwCAcfgwlI789Tnvbes4AABB6fvwrJvE0dUkjtbEfQwAcFhmHcCbUIgGAGjJRl6UK9ZBAACmV2c4yERBGmrln7249tM/f/GldRwBuKJqQhoFaaIYbSrUL3a2xIsdAEAYRpJ6rhdN4mhR0i3X6wIAJo43k1G4dwEAMHapdQBvQyEaAOCczJ91DyTy85wLADjqZhJHqXUQx6EQDQDQklbyYgAAOKu6IO17qu5NmHKf//DF+8Xd59r7WhSlvdl+QVrPOhBrFKNNOA7GAAAB6uVF+dTlgvX9MHO5JgBgon2YxFHXMgDuXQAAmLju83S0JI5WRCEaAOD8PkziaMnig3lfCQA4haX6fuEN7l8AgBaleVHuWAcBAIAkdYaDvqSuKEiDpF9u7enTP3r+/t7X+so6Fs9dkfRw2gvSKEabYByMAQACtJoX5VoL62aqNn8AAJxWavz5mbh3AQBgwSRJ/23qIrm71nEAAIKXGiX698X7SgDAm12RR89j5NsAAFq0kRflinUQAAAc1BkOtkRBGmpffbqnfPFX7339c31uHUsAprogjWK0CcXBGAAgQLtqIfE/iaNFSbdcrwsAmHhm09G4dwEAYOpmEkez1kEcVBeiPbSOAwAwEcae6J/EUSbeVwIATsebYjRVzcK4fwEAXBtJ6lkHAQDAceqCtHlJ29axwN7zL6Qf/cGvPqAg7VQePnuy4NOZxthQjDa5+uJgDAAQll5elE9dLlgXZ2cu1wQATJXeuD+QexcAAF5YtA5gX10cTyEaAMCl3rg+KImjFUm3x/V5AIDgXakbdZmq7180CwMAtCHNi3LHOggAAE7SGQ52VE1IoyANev6F9J/+u199MNrYoyDt7R48e7KQWQcxbhSjTSA6DAIAArSaF2W/hXUzVd1+AQA4D4vEh0zcuwAAsNa1DkCSkjial7RmHQcAYOJcr+8xraone95t+3MAABPHtBitLobj/gUAaMNGXpQr1kEAAPA2neHgqShIwwE/+cFzCtJO5/a0FaRRjDZhkjhaEh0GAQBh2ZaUul60fllE10IAQBNXxpGgt69O1OPeBQCAvbHd/09ST0tdE0XqAIB2tJroXz9Lk2QJADgPs2K0JI5mVTULAwDAtZHGOKUaAICmKEjDUT/5wfMP/v6vX4ys4wjAVBWkUYw2QeoXOw+s4wAA4Ix6eVE+dblgnbSXuVwTADC1xpKMXic6kKgHAIAfrlt+eP1M27eOAwAw0bptLXzgbJaCagDAeVypz0ot0BAEANCWNC/KHesgAAA4CwrScNTP/vLFlZ/++YsvreMIwNQUpFGMNiEOdMoFACAky3lRbrWwbiZeFgEA3Jgd0+dk4t4FAAAqK5JuWgcBAJhoV1tcOxX3MQBAM7Pj/sAkjlJx/wIAtGMjL0oaUgIAgkRBGo76/Icv3qcg7VSmoiCNYrTJkYpOuQCAsGznRZm6XjSJoyVJt1yvCwCYWrNtf0B97/qw7c8BAACnl8RR1+hzlyTdtvhsAMBUaSXZvr5/3m1jbQDAVOmO88OSOJqXdH+cnwkAmBojST3rIAAAaIKCNBy1X5C29ys9s47FcxNfkEYx2gTgxQ4AIFA91wsmcTSrqkAbAABXZttcnHsXAADYVydAPrCOAwCA80ji6Kqqqd8AAIQmsw4AADCx0rwod6yDAACgKQrScNTnP3zx/qd/+Pzy3tdiStqbTXRBGsVogatf7KxZxwEAwBkt50W51cK6maQrLawLAEBbMnHvAgDAR0/H+WH1OW9/nJ8JAIBjK5KuWwcBAMBZJHGUqqWJoQCAqbeRF+WKdRAAALhCQRqO+urTPX36R8/fpyDtrSa2II1itPBlInERABCWjbwoU9eLJnG0JOlD1+sCANAW7l0AAPirpQYqb7ImznkBAOOz4XKxJI4WJd12uSYAAG2rp1Pft44DADCRRpJ61kEAAOAaBWk4ioK0U5vIgjSK0QJWv9i5ZR0HAABn0MqBWxJHs5JS1+sCACCplUR07l0AAGBf3YmfAnUAwDg5mwBaT/fMXK0HAMAYZdYBAAAmVpoX5Y51EAAAtIGCNBxFQdqpTVxBGsVogeLFDgAgUG0duGWigzwAoB3OEvSOyMS9CwAAX43t5VkSR13RiR8AMH59h2tl4vkWABCYuinITes4AAATaSMvyhXrIAAAaBMFaTiKgrRTu/3sycLE7BUpRgtXJl7sAADC0sqBWxJHS6KDPACgPTuuF2T6CQAA3tsZx4fQcAwAYKjvYpEkjhYl3XKxFgAAB2y1uXgSR/OiKQgAoB0jST3rIAAAGIcDBWm7xqHAExSkndrdZ08WetZBuEAxWoDqpHte7AAAQtLKgVsSR7OSUtfrAgBwgNPEBxIdAAAIQquJjwdkkq6P6bMAANi3mxdl43sdRdUAgBY9bXn9rOX1AQDTK82Lcsc6CAAAxqUuSFtUlR8KUJB2eg8noSCNYrTAkHQPAAhUr6UDt0xMCgUAtGfkIkHviMzxegAAwL1+2x/AJBkAgKHM0Tpr4mwWANCCvCj7ba2dxFEq6WZb6wMAptpGXpQr1kEAADBuneFgS9WENArSIImCtDMIviDtHesAcGaZeLEzyUYaX+dlABiXrbwo11wvWk+WkaQN12sDgKR5se+G40R0Eh2Csq32OzBPo6viZyAE7K9P50PrANCaUZuJj9LLhmNZm58Bb3GNRUi4102uxsmRSRwtib8joduVtGMdBNASzh/Ctt3WwvW7xfttrY+xO3qGu6Xjz3SvqnrnAwBt61kHAACAlc5wsLU5M9dVlWdDvhVeFqR9999e+vLCu3rfOh6PPXz2ZGHr8o3HQdaPUIwWEF7sTIz9grO+qpc8O6oKNUh0BIAzqCfVdK3jADD56hfU+y8rr6q69lCsNh36rhYi0cFbu6q+z31JO20XH6BS/zwsSloS11JL2zr89z/Iw01L9d/lrqq/y9dto4FD/TF8Riauf5NuQ9XfpS1xjUXA6imOi5JuW8cCZx41fR9VF1WnTqLBOIz06p7UV3Vf2jGMBxiLJI6uqrqHpeJ5LTT9FtfOWlwb7dk/w93a/yK/BgAAAPBPXZC2JOmhdSzwAwVpp9Z/9mShG2JB2oW9vT3rGHAKdXJLXyQphGpddYIXiQcAAACToU6+2k9CXxRJDZPoO64StJI42hIdmX3ySNIKz2e26sSwvvjZGKddVZMw1khAdSuJo66q/7b8fQ7fnbwos7YWrxuOPWhrfZjZlbSm6vzX+XR4wFp9n1sT76hCN5I066AYrS+aZ/puW1XBBe8lAbEHD9BHbeypkzhKRbOwkOzfyzjDAgAAAAKzOTPXEwVpOOC9714QBWlvNZIUXEEaxWiBIHExSBt6dThGVyYAAIAJVxen9eovCtPCt5EXZdfFQiQ6eGVd0hIJDP6op218bB3HFNiVlLZZYIMKSY7Bc5KkfxIajk2kgxjD0AAAIABJREFUR6rOfylAw8TjGjYRlvOiTJsswF7Ha7uq3ktmPPMCr0viqCcS4UIwyovyqutF633MY9frohU0EQMAAAAmwObMXCryZHDAe9+9oN/+q0vWYfhuW1VBWjB1JxSjBYDExeA8UpXgtWMdCAAAAGzUxRVLolN4yJxMRSHRwRsjST0Sxf2UxBGHU+1aVXVOEcyBZehI1g/ao7woe20tTsOxiTFSley/whkwpk09Ie0T6zhwLtt5Uc43WaBuwrMl9ji+2VBVgJZZBwL4LomjNUm3rOPAG7XyTMazWBDIswEAAAAmzObMXCbptnUc8McHv3fxy9/844tMR3uzoArSKEbzHImLQeFwDAAAAIfUHXdXRKJWaJx14CXRwQvbkhZ5VvNTEkdXJQ2t45hQI1V/9/vWgUwjCtKCtdBW93Uajk2MR6qmrAbxAghoQxJHfdF4JUSN73E833pnQ9V7yb51IEAoyL0Iwkeum0nxLOa9DVVNxHasAwEAAADg3ubMXF+cJ+MACtJO5dHlG4971kGcxkXrAPBWmXUAeKttVS/xOCADAADAIXVX6llVe0aEY8XFIkkcrYhEPWvbkro8q3mt0XQGnGhb0jyJqXbqZO8l6zhwJhstFqLNi+TH0G1I+k59BkwhGqadk+cljNWyg0K0VDzf+mKkapp7l/0+cDb1tXDXOg6caLeFQjSexfw1knSvvp/tWAcDAAAAoDWLImcKB3z+wxfv//TPX3xpHYfnbj97shDEuxiK0TxG4mIQlvOinG8rUQUAAADhq5NVu+JwJRQjOUiuTOKoK+lu42jQxH4hGgnjflu0DmACUYTpiboofcM6Dpxa1sai9QRIpwmVGKuDCf871sEAPnCdJI7WbeRFmTZZgER+r6xKmq33mQDOZ8c6AJwoC2RNNLffRCmIxDoAAAAA59cZDp6qygkYWccCf3z+wxfvD//mxWfWcXju7rMnCz3rIN6GYjRPkbjovZGkj5q+wAMAAMB0OFCQxuGK/1aaFi/VSeeZm3BwThSihaNnHcCE4e++f1LrAHAquy0mdaeSrre0Ntq1oSpBMrMOBPAQU2XCMJKb/XbmYA00M5L0vbwol9jrA43RZNZPThqEHcRUT289qhs+71gHAgAAAGA8OsPBjqqcKeCl8s9eXBtt7H1uHYfnHj57stC1DuJNKEbzEImL3hupSuyi+ycAAABOrU4WWrKOA2/kKukhFUnnlvaf2UjQ81wSRz1JV6zjmCAUonkoL8q+mI4agrSNRWk4FrRlpqEBb7RjHQBOZanpdYxEfi9sq5qG1rcOBJgQs9YB4FiNG4QdxFRPb93Ji7JnHQQAAACA8esMB1uS7ljHAb/85AfPP/jl1h4T0t5s7dmThXnrIE5CMZqfUpG46Kv9pEY6pgEAAODM6qkKTEfzV+pgKlpXJJ1bohAtLKl1ABOEv/t+61sHgDfaaGPyFQ3HgrU/eSY1jgMAmlpven8jkd8L+9Nj2OcD7ly1DgCvcT4VTTyL+WakqhAtsw4EAAAAgJ3OcJBJWrWOA37Z/dPn177+uZiQdrIrkrJnTxa8PNOiGM0zSRwtisRFX1GIBgAAABfYT/ppOy/KRkkPJJ17YYlntjDUU9FoxOMGhWj+61sHgDdKW1o3E9e50GxLmmfyDHAqs9YB4I1GknpNFqifb9ecRIPzWmZ6DNAKbztJTzHXU9FSMdXTJ/vnVpl1IAAAAADsdYaDJUnr1nHAH8+/kH70B7/64Ou/ExPSTnZTnuajUYzmERIXvdcjqREAAAAO9K0DwLGWHKyRiqRzS49IaghKah3ABKEI038UCvprvY3Co7rh2C3X66JV26oSJHesAwECwXOP3xYdJPWn4vts6Q5TOgH36omPV6zjwCG7cjgVjame3qHhMwAAAIDj9FS9lwEkVQVpO//j82t7X+tL61g8duvZk4XUOoijKEbzSyYOP321nBclHSABAACAybTaNBE9iaOumHJtaVtuCgoxBnWHahJb3VilCBM4t5FauHfQcCxIj/KinGfCJHA6dZI3/MXzbfjusMcHWtOzDgCvSR3vwzOHa6E5GigBAAAAeE1nOHiq6hl9ZBwKPPLVp3v69I+ev28dh+fuP3uysGgdxEEUo3mCbrle26b7IAAAABzqWgeAQ3bVcEITSefmRqomWZNAHoAkjmZF4aAr23lR8t8SOL+VlqZgrYmGYyF5lBdlzzoIIDBd6wBwom3xfBs6Jn4D7fIqWQfacHnNq5sf3XS1HhqjuBoAAADAiTrDwZZoGoMjvvp0Tz/98xdMR3uz7NmTBW+aBlKM5gFe7HivZx0AAAAAJoo3D4SQ5KaIKRNTniyldNgNyooo0nBhJJLoQtK1DgCvaaX5VBJHS5I+dL0uWkMhGnA+XesAcCIXz7epeL61ss59CWhP3RyY65tfnDXYqSe33ne1HhpbphANAAAAwNt0hoM1ScvWccAvn//wxft//9cvmJp3siuqCtKuWgciUYzmC7rl+useSY0AAABwJYmjrtj7+2Q1L8p+kwWYcm1uIy/KFesgcDr8vDi11NJEJ7SDQnT/9FwvWE9+TF2vi9ZQiAacQ91ckf2cn5abvs+qzyzuugkHZ7QtmmMCbUutA8Ahje9bR2QO10Izj9po/gIAAABgMnWGg1TShnUc8MvP/vLFldHG3ufWcXjspqpG0OYoRjNGt1yvkdQIAAAA13rWAeClXTVMQmHKtTkmQwWEnxen1ukuHQ6S9r3kOulxHw3HwkEhGnB+7L/9tNE06bves6w5iQZnNZKbqXYATlDnZNy0jgMvOZ1UncRRKr6/vtiWw4l3AAAAAKbGoqocHuCln/7Z8w++/rkoSDvZ7WdPFnrWQVCMZohuuV4biURhAAAAOFTv/29bx4GXXCR6ZSLp3BLJemHJxM+LC5xXhIekfb84TXrcR/JjUEiOBJrh58c/rvaHmdivW0lbKpQHIHIyPNVztVASR/OS7rtaD43sSupyXgsAAADgrDrDwVPxThVHPP9C+tEf/OqDva/1lXUsHlt59mRh3jIAitFsZeLFjq96eVHuWAcBAACAiZJaB4CXlvOi7DdZIImjRTHpxtJ6XpR07Q8EPy9OUYQZntQ6ABzSc70gyY9B2RbJkcC5JXHUFYW3Pmr8Pov9uqmNvChXrIMAJlwmcjJ8cs9xAW7mcC2c30jSIs9aAAAAAM6rMxxsSbpnHQf88vwL6dM/ev6edRweuyIpe/Zk4apVABSjGUniaEnSh9Zx4FiPSGoEAACAS3XSHlPR/NB4IkoSR1dFooMlJkMFpO5AnhmHMSkowgxMff533ToOvOQ66XFf1sKacI/kSKC51DoAvKbx/pDnW1M82wItIyfDO04LcJlQ7ZUeUz4BAAAANNUZDlYkrVvHAb989emefvrnL760jsNjNyWZNTyjGM1A3S33gXUcONaupCXrIAAAADBx6HLtj56DNTLRUdkSk6HCkomfFxdIVA1MXYiZGoeBV1qZOpLE0YpIfgzBSNVEtB3rQIBQ1Q1WSOb3y654vg1d46l2AE5W52Sk1nHgJafnGkyo9sodmicBAAAAcKin6uwTeOnzH754f7Sx97l1HB67/ezJwqLFB1OMZiOzDgAnIqkRAAAATpGk7JXlph1akzhalHTLUTw4OyZDBYQO5E5xXhGeTCR2+6KVYs66MOOu63XRiiW69AONpdYB4DWN94dJHPXE860Vnm2B9mXimcwnqeMC3MzhWji/R3lRZtZBAAAAAJgcneHgqWjSimP85AfPP/j656Ig7WTZsycLs+P+UIrRxiyJo1QkovpqOS/KvnUQAAAAmBwkKXtlOy/KtMkCSRxdFYkOlpgMFRCmQjm1SqJqWCjE9I7rpEf2BGFZJTkSaIapaF5abfo+q96vM8XdhqupdgBOQE6Gd5xOqub7642NvCh71kEAAAAAmDyd4aAvadk6Dvin+B+ef7D3tb60jsNTV2Tw/ppitDFK4mhe0n3rOHCsxompAAAAwEEkKXun52CNNdFR2RKTocKSiZ8XF3ZFUV9QKMT0jtOkxwNWJF1vYV24tZEX5ZJ1EMAEyKwDwCHbjq5tmdivW+HZFmhRXURNToY/nDaXIufGG9uSFq2DAAAAADC5OsNBKmnDOg745R9+tqcf/8vn71vH4bEPnz1ZSMf5gRSjjQmJqF6juz4AAADakIokZV/cy4tyq8kCTLkxx2SogPDz4hSJquHJRGK3L1o580viaFHSbdfrwrmRSI4EGqsnj/Bc65de0wXYr5tqPNUOwMnIyfDSkuNJ1ZnDtXA+I0mLnFcBAAAAGIOeqmcQ4KVfbu3p7//6BX8vTnb/2ZOF+XF9GMVo45NKumkdBI6VNk1MBQAAAA6qk5TvWscBSQ4mojDlxhyToQLCz4tTJKoGhsRu7/QcJz2S3BoWkiOBhpg84iUXjVZmxX7dyrb4bw+0LRVF1D5Zz4syc7VYXSRPzo29rutnbQAAAAA4Tmc42JG0ZB0H/POzv3xx5atP9z6zjsNj2bg+iGK0MUjiqCsSUX3VODEVAAAAOIgkZa+4moiSiSk3lpgMFZY18fPiAkWYgakT9lPrOPDSeksTNTNxjQvBMsW8gBOZdQA4xNX7rEzcy6zwbAu0iOZg3nE6qZoieW/codEzAAAAgHHqDAeZpHXrOOCf4u7za3tf6yvrODx189mThXQcH0QxWstIRPXaSNKidRAAAACYOJlI7PJF2rRLK1NuzDEZKiB0qHaKRNXwZOL+7wunSY/76j3BLdfrwrmNvChT6yCA0CVxtCL2dT5xcm+r9+s839pYJnkfaA85GV5yfa6ROVwL57PsctIdAAAAAJxBT9UZKfDS8y+kH//L5+9Zx+Gx+8+eLMy3/SHvtP0B0Iqk69ZB4FgkdgEAAMCpJI56IknZF427xidxNCum3FhiMlRA6FDtFEWYgaEQ0zvOz/zYEwSjlUJEYNokcdQVk2V803PQaIX9uh0KpYH2ZaI5iE+cTqrmmdsLj7iXAQAAuLE5Mzcv6eob/shs/eWD/lv+963OcEAONlrXGQ6ebs7M9SR9bB0L/PLLrT39/V+/GP3a71/kXOh4maRWC9Iu7O3ttbn+VEviaFFc+Hz1KC/KnnUQAAAAmBx1kvKWSHzwwUjSvINkvb7oGm/pexTkhCOJoy2RGOTCrqrrFy9uAlEndj+2jgMvredFueh6UfYEwfjIZdIrMI3qyTI74rnWJ07ubezXzTg5nwBwsnqC8QPrOPDSSNKsq3MNnrm9sC2py1kVAACYVpszc90j/+roP1/V8Yn+0/xOYVvS0f3j1pF/t1N/vfznznBw8J+B12zOzK2JBuE4xu/8u0ujf/QbF3ivcbzlyzcep20tzmS0ltQv7DLrOHCsXUlL1kEAAABg4mQiYc8XSw4K0ZY03QfE1pgMFRA6VDvFFPfwZNYB4KVdtTAViz1BMB5RiAY4kYnnWp84ubexXzfV+HwCwMmYYOylRcfnGpnDtXB2u6IQDQAATIgjRWUHp5PN6vAkMt4HNHfcOdRb/7tuzswd/MeNA7/f0avCtUO/p4Bt6vREMzUc48f3Xlz5r/73S19eeFfvW8fioaVnTxayyzce77SxOMVo7cnExc5XJHYBAADAKZKUvbKeF2XWZIG6427qJBqcx7b47x+M+uflvnUcE2KZIsywkNjtHednfvU1jikL/qP5GOBA/VxLV1m/NL63sV831fh8AsBbrYmcDJ84bS7FM7e5kdwXFwIAADizOTN3cCLZweKy7oE/Rg5HuD484feHHChgOziNrV//ulN/qTMc9IXgdYaDp5szcz1JH1vHAr/8w8/29J//p+d7//RfX7IOxUdXVNU1ddtY/MLe3l4b6061JI4WxYXOV8t5UabWQQAAAGBy1Ildj63jgKTqBfmsg2S9LZHoYGkhL8ot6yDwdvVU+L74eXFhOy/K+bf/MfiC+793VvOidF6MxJ4gGOwdgIbq+1pfJPT7xMm9jXuZGSfnEwBOVhcqUWzrj11J866uezxze+Ejpk8DAAALR4rMuvWvs3o1vYwCMzSxX7S2U389lbQlUbAWks2ZuTXRWA3HiFcvffaN+QvXrOPw1EeXbzx2/pzPZDTHkjiaVVU9CP9sU4gGAACAFmTWAeAlF13jU5GoZ2mZZPKgpOLnxZWedQA4vboQM7OOAy/tqoWJmuwJgsHeAWjowH2NQjR/bDsqRFsR9zIrzie2AniFqY9ecn3dyxyuhbO7QyEaAABowwmFZvtTzWYlXR9/VJgy+2dlrxU1HpiytlH/2j/4K8VqXumpKibkTBuH7P7p82u/8+/fGV36Fn83jrHy7MlC//KNx07PrSlGcy8TFzcfjSQtWgcBAACAyUKSslfWm74gJ5HFHA1EApLEUVfSXes4JgSFFOFJxf3fJ86TvetrHHsC/22wdwCcSMV9zTe9pguwXze1SgI/0B6ag3hpNS/KvqvFOHM39ygvysw6CAAAEKYDxWbH/coeD6H48Miv96WXxWojVZPUdo58bXWGAxoTjUlnOHi6OTO3JOmhdSzwy/MvpN1/9fxKvHrJOhQfXZe0JMdNXi/s7e25XG+qJXG0JOmBdRw41r28KFesgwAAAMDkqAuXHlvHAUnVgd+sg6loW+IQ3NICBTlhqBO/tkR3Qhe286Kcf/sfgy/qxO5PrOPAS6suJsccxDUuGE72f8C0S+JoUdLH1nHgkMbvs7iXmdqVNM/9CWhPPfWRYlt/OL3uceZubiMvyq51EAAAwG+bM3P7xWVdUWwGHHS0UG1L0k5nOCAHoiWbM3N9HTPlDoj+5OJnM9+/eM06Dk995/KNxzuuFmMymiNJHM3KcaUgnFmnEA0AAAAu1YlddLn2R+OJKHTcNcdkqLCkIrHVlZ51ADg9OvB7Z9t1IVotFde4EDifiAdMG+5rXtpw9D4rFfcyK4vcn4D2MPXRS86ue5y5m9uWtGgdBAAA8MPmzFxXrwrNZuuveUlXzIIC/HdFVWHUoeKoeqLatqSnkvqqi9U6w0F/rNFNpp6kH1sHAf/87V+8uPbtf35xdOlb3LeOkakqKHeCYjR3MrHR8tFIJHYBAADAvVQkdvliNS/KRkkKdcfd+47iwdlt50WZWgeB0yHxyymKMMOTivu/T3quF6wnBHGN81/j/R8ASVWyN++1/OHkfRb7dVPs74EWUajkJdfXvVQ8c1sZiYJqAACmzpEJZ7Oi4Axo035j5JeFanWR2q6q4rR+/esWk9ROrzMc7GzOzC2LfB8c8fwLafdfPb8Sr16yDsVHHz57stC9fONx38ViFKM5UHfQZ8yjn+iQCwAAAKdI7PLKrtxMqM4crIHz61kHgNNheoZTFGEGhvu/d5wne3ONC8a23Oz/gKmWxNGSeK/lm15elDtNFuBeZor9PdC+TCTF+sTpdY9nbnPdpvsQAADgp82ZuaPTzbqqCtBunvh/AjBO1+uvo0Vq25K29KpQbaszHJALf7wVVTknNDfBIb/c2tPwb158NvP9i9esY/FQpmpf0BjFaA3RQd9rdMgFAACAU3Tg9U7j5hN1cxEO2+3QOT4sK+IQ1wWmuAeGxG7vtJXsnYnk1hDQfAxoqH6v9cA6Dhyy7uh9Vib26xbY3wMtqycY37KOA4f0XC3EM7c5zmcBAJgAB4rO9gvP9n/PmTcQppt6lcdyX5I2Z+ZGqgrU+vWvW53hYMciOJ90hoOnmzNzS5I+to4F/vnbv3hx7dv//OLo0re4Hx5x/dmThd7lG4+zpgtd2NvbcxDP9EriaEskLvpoV9I8iQkAAABwKYmjNZH44IvVvCiXmixQJ2E+dhQPzm4jL8qudRA4nTrxiwNcN+7lRbliHQROj/u/dxZamIrGNS4MXD+Bhupk7754r+UTJ++zuJeZ4v4EtCiJo1lVSXYkDflj2fFUtBUxFc3Kbl6Us9ZBAACA06PoDMARFKjVNmfm+jowYQ7Y9435C4pXL1mH4aORpNnLNx43ejfBZLQG6KDvtUUK0QAAAOASHXi9sispbbIAU+7M0Tk+IHSodmqDRNWwcP/3zr0WCtFmxTUuBFw/ATdS8V7LNy4mfrNft8P9CWhfJhJrfeJ0UnUSR11RiGYptQ4AAACcbHNmrquq4GxWUlcUnQF43RVVBVgvi7COTFDrqypQm4Z8+p6kH1sHAf/8cmtPv9za++wb8xeuWcfimSuSltTwbIBitHOqO+jft44Dx1p2nZQCAACA6UaSsncaJ+upepi+7iAWnE+aF+WOdRA4tUy83HKBIszAkNjtnbaSvTNxjfPdSNKidRBA6OoCa5K9/bKcF2XfwTqZuJdZ4P4EtCyJoyXR1dwnTs81eOb2Qt86AAAAIG3OzM3qcMHZrGgmBOD8Dhao3ZekzZm5Xb2antbvDAcTl2PfGQ52NmfmlkVtB46x+6fPryVr73x14V29Zx2LZ5aePVlYaTIdjWK0c6CDvtecduICAAAAaplI7PJF42Q9Ou6ao3N8QJgK5RRFmOHJxP3fF60Uc5LcGgwXjQiAqUaDFS85eZ/Fft0U9yegRXVz4AfWceCQ1HFT4FQ0CzPFORUAAOO3OTM3r1cFZ10x7QzAeFyXdLv+0ubMnCRt6HCB2iScc62omvTEdRWHPP9C+i8rL/Z+848vWofim8bT0S7s7e05i2ZaJHG0IhIXfTSSNM+BGQAAAFyqk5RJfPDDdl6U800WqJuLbIlEBys8twWk/nnZEYe1LmzkRdm1DgKnVyd2f2wdB16657qQuU5u7YtrnO9W86Jcsg4CCF0SR31RfOuTkaRu04R+9uumHuVF2bMOAphkSRxtiWkQPnF6rlE3C/vE1Xo4twXHBYYAAOCAzZm5rl4Vns2LsxkAfttWXZimqjhtxzSac9qcmetJemgdB/z023916bP3vnvhmnUcnhlJmj3vdDQmo50RHfS9tkRCIwAAAFyqu8enxmHglZ6DNVJRiGaJyVBhyURiqwutTHRCe+rE7sw6DrzU1kTNTFzjfLct9uJAY0kcpSLZyTeuJsusiXuZhV1V3WIBtKRuDkwhmj+cnmvwzO2VVNKidRAAAEyCA4Vn+1/sZwGE5mb9tT89bVeHJ6cF0ciiMxxkmzNzS+I6jGPs/umLa7/zf1yyDsM3jaajMRntDOig77X1vCg5JAMAAIBTdOD1ynJelGmTBei4a47JUAFJ4qgnOoa54nyiE9rF9BivtDJRsy7MuO9yTTjnZGoQMO3qKZCPrePAIU6ei5jibup7eVH2rYMAJhXnd15yeq5RFxvSANofTPsEAOCMKDwDMKVGqqemyfPitPo6zdkCjvVP/vDi6Nd+/yJN3g4793Q0itHOIImjNUm3rOPAa0aSZvOiPNd4QAAAAOA4JCl7ZTsvyvkmC9BcxFwrxQRoRz0VcktMWXCBIszAkNjtnTt5UWYuF6QwIxgU8gIN8QzkJSfvs9ivm1rNi5KpaEBLuHd5yem5BsWG3tpVNbk1sw4EAADfbM7MHSw6mxeN7ABg30jSml4Vp+2YRnPE5sxcX1yzcYxL35R+59+/M7r0Lc7Xj1i+fONxetb/E8Vop5TE0aKkj63jwLHoQAgAAACnSFL2zkLTqRh03DX3UV6Ua9ZB4HSYCuUMzXMCQ2K3d9bzolx0uSDJrcFw/r0HplESR5mk29Zx4BAnz0Xs1800bpQD4M24d3nH6bkGz2NB2E8m3dr/4lwLADBNNmfmZvWq6Kwrnr0B4Cx29Wpy2lpnODB9lqiv6T+2jAH+uvLhhWf/9F9fumwdh2fONR3tnZaCmSj1oVhmHQeOtUohGgBMnvreu/9if7b+AoBxIvnVH/ccFKItikI0S+sUooWjngrFyzU3eiTsBCcThWi+GEnqtbBuKhIffdfW9x6YKvUzEMn8fnnkqBCN/bqdnnUAwCTj3uUl1+caS+J5zHdXVP0cvvxZTOJIqp7TGp3RA/BG/8DvtyQ9lbSTF+WOSTSAsc2Zua6qorP9AjT2KgBwftf16nni4ebM3LZeFab1xx1MZzjY2ZyZeyTOGnCM0cbe5a9+vPfz975z4detY/HIFVVnN+lZ/k9MRjuFJI7WJN2yjgOv2ZbUJbELAMJVTx6aV1Vs1pV0VdJNw5AAAH7ZyIuy22SBusB5RxQXWGEyVECYCuUUU30CUyd2P7COAy85n6iZxFFX0icu10QrvkfzMaAZ9nRe2pU03/S5qD5L7YvvrYV7eVGuWAcBTCrO77zk9Fyjvoc9drUeAKAV26rux1uS+pzPYNLUE3K6ejX1jNwkABivdb0qTtsZxwcyHQ1v8t53L+i3/+qSdRi+2b184/HsWf4PFKO9RRJHPUkPrePAsRaaTkgAAIxX/bJpUYyzBwC83UhVst5Ok0VoLmLOeTEB2pPEUV/s0VygCDMwJO17x3kxJ8mtwVjNi3LJOgggdOzpvOSk0DaJoy2RLGehcaMcAG/Gvcs7zs81uIcBQLD2J5pk5OghNPXUs/3Cs644GwYAn+xKWpPU7wwHreaTbM7MZWI6Gk5w/d9c+vm3f5fpaEfcuXzjcXbaP/xOi4EEr05Eocudn5Z5yAWAMCRxtKiqAG1RHO4AAE4vdVCItigK0SytU4gWjnoqFIlfbvQoRAvOmnhW8cVIUq+FdTPxPfbdNoVoQHNJHKViT+ebZUeFaKlI4rfQ1t4EQI3zCC85Pdeov8fcwwAgTDfrr7tJHI1UnaOukLMH32zOzF3Vq6KzebG/BADfXZd0V9LdzZk5qd2paUsibxUn+Nv/5cWvf/t3mY52xJKqd+unwmS0N6ADl7foQAgAnqsnoLGRBwCcV+M9PxNQzDEZKiD13u2xdRwTwvlEJ7SrTuy+bx0HXnIyOeagujj9Y5drwrmRpC6JTEAzSRx1JX1iHQcO2c6Lcr7pIuzXTd3JizKzDgKYVEyp9pLTcw3OaAFgYm2rKkrLrAPBdNqcmZvVq+KzrqqiBgDAZHg5mbUzHDh5b7RNQUMtAAAgAElEQVQ5M5eK98E4QfQnFz+b+f7Fa9ZxeOZ7l2887p/mDzIZ7QR04PIWHQgBwGNJHPVUFaHR4RAAcF6u9vyZSHKwxGSosGTWAUwIziwCUyd28+LBH6stFKJdFde4EKQUogHNcL3zksu9YeZoHZzNOsm1QOuYUu2XXbk/11gR32MAmEQ3JT2sG32l7JvRts2ZuXkdLj5jfwEAk+vlZNbNmbn9yaxrneFgrcGaK6pyWrl/4DV/+xcvrl39by9+eeFdvW8di0d6qopC34rJaMegA5fX6EAIAB6qi9BS0W0IANBc4z0/E1DMPcqLsmcdBE6HqVBOfZQXZZNDcIxZEkdbopGGL3YlzbsuZE7iqC8ajvmOiZKAA0kcrUm6ZR0HDrmXF+VK00XYr5th2jfQMq5vXnI6qZqprQAwVXZVNSnsWweCybA5M9dVVXS2X4RGHi8AYKSqOGa/OO1M53ZMR8Ob/JM/vDj6td+/yH7jsJnLNx6/9eeMYrRjkIjiLRITAMAzFKEBABxrvOevJwLsiJcSVlopJkA76qlQj63jmBAUYQaGxEfvOE16lKQkjpYkPXC5Jpwj0R9woD6fe2gdBw7ZyIuy23QR9uumnO9NALzC9c1Lq3lRLrlckOYgADCV1lUVpXHWgzM5UHzWFfsHAMDprKsqTOt3hoOdt/3hzZk5colwokvflJK1d5iOdti9yzcev7XhHsVoR5CI4i0SEwDAI/WLwhVxCAQAcMfJnp+JAOZI2AsIzXicoQgzMCQ+eqeNpMdZSVvihZLv2DcADXG985KrZ9urqr63NAEbP+d7EwCvcH3zkvNzjSSOFiV97Go9AEBQdiUt5kW5ZR0I/EXxGQDAsW1JmaqJaTsn/SGmo+FNoj+5+NnM9y9es47DI7uXbzyefdsfemcMgQSjTkThIuOnRZK6AMBe/ZIwlXTXOBQAwORp3CmxnghAIZqdVRLKw1E346EQzQ06vYYnsw4AL22resZ0bU0UZvhumX0D4ATXO/+42humolDDQlt7EwCvpOL65ps2zjXe2jkbADCxrkt6nMTRnbwoM+tg4AeKzwAALbsp6YGkB5szc28qTFsRdSI4wd/9b3vXZr5vHYVXrj97stC9fONx/01/iMloB9AR3Ft0IAQADyRx1FW1UeclIQDAtfW8KBebLMBEAHNMhgoIU6Gc4swiMEkcrYjmGj5ZcN0luS625UWS37bzopy3DgIIHdc7Lz3Ki7LXdJH6HPaTxtHgPJzvTQC8wrQsL7Uxqbon6aHLNQEAwXLyjITwbM7MzUtaFMVnAABbrxWmbc7MZZJu24UEn13/N5d+/u3fvfDr1nF45NHlG497b/oDFKPVeGnnLRITAMAD3CcBAC0aSZp1MBWtL15mWPoe003CUE+63RINBlygCDMwJHZ7ZzkvytTlghTbBmGk6tq5Yx0IEDLuaV5ysjdkv27K+d4EwCv19W1HNJLySSu5GEkc7Yj7GADgFQrSpkBdfNY98MWeDwDgm/3CtL54l4gTvPfdC/rtv7pkHYZPRpJmL994fOJ7j3fGGIy36pd2JNj7qWcdAABMs/rlYCbplnEoAIDJ1XOQrLckCtEsrVKIFpRUJAS50vj6hfE58GwDP2y3UIjG9zgMSxSiAc1wvfOWq71hKvbrFpzvTQC8JhNJyb7puV6wnorGfQwAcNDtJI522G9Pls2ZuVlVRWf708/Y5wEAfHdT0oP69yNx78Ixvvp0T1//nT579x/rmnUsnriiar+XnfQHpr4YjZd2XruXF+WWdRAAMK3qjvKZqo04AABtWM2Lcq3JAkkczapK1oONXfHfPxh1M5671nFMCIoww5OKhDif9FpYMxXPr75bz4sysw4CmACZuKf5ZtnF3jCJo0WxX7cwUvVCHUBL6gIlmh76ZbmlXIy0hTUBAOG7n8TRVtN3grCzOTN3VYeLzziXAACEjEI0nOjvHr34xm/+8UXrMHzyxmK0C3t7e+MLxUNJHK2IFzs+2siLsmsdBABMq7oQrS823gCA9uxKmncwFa0vpqJZWqCJSBjqZjxb4gWhC06uXxifuhDzE+s48NJyC1PRuuJ77DuunYAD9VToB2/9gxin7bwo55suUu/Xd8R5rIV7eVGuWAcBTKq6kdSWuL75xMm966i66PCh63UBABNjJGmWs6FwbM7MdfWq+IwmYAAAYCpc+qaUrL3z5YV39b51LB6ZuXzj8bH7+KmejEZHcG+N1E53ZADAKdQdeDPxYhAA0K6eg0K0JVGIZqmtDspoRyoK0VxpfP3C+NSJ3Zl1HHhpo4VCNL7HYeDaCTRUN49KrePAIS7fZ2XiPNbCOoVoQOsycX3zTS+wdQEAk+GKpCXxXOutzZm5eVWFZ10x1RYAAEyp519IX/zfe599659d+C3rWDxy4nS0qS1Gq5MUGP3sp15elDvWQQDANKJrIQBgTFbzouw3WaDuqpy6CAbnsu26mADtoRmPU42vXxi7FVGI6Yu2GlDxPfbfMtdOwIlMJPP7JnXRoKNuDkai3/jRHBNoWRJHqWgk5ZtWmkvVRfN8rwEAb7OUxNEKDYv8sDkzd1WvJp91xRkrAACAJOnv/92L3/rWP7tkHYZPKEY7RiZe2vloPS9KigQBwACFaACAMdmVmyKyTDzTWepZB4DTYWKQU9t5US5ZB4HTqxO7b1vHgZdS1w2o+B4HgQJ2wIEkjlYk3bSOA4c4majFft0UUzuBFtXFSfet48AhzidVH8B5CQDgNK7oDYmsaN/mzFxXVeHZojhnAAAAONYvt/b0/JcaXvqGZqxj8cStZ08Wrl6+8fi18/SpLEajw6C3dkVCIwCYqF8KNk6eAADgFBone9FV2VwrHZTRmkx0s3SlZx0ATo/Ebu9suEjYP4jvcRBGqhJbADTAlFsvuZyolYlGKxYe0RwTaF1mHQAOaXsaJPt+AMBpdcU+YWw2Z+Zm9ar4rCuePwEAAE7lF/9x79nMv7hAMdorxzaVmLpiNJIUvEYHQgAwUBei9cWhEwCgfct5UfabLEBXZXNMNwkIzXicoggzPJl4xvFFW0mPmfge+27J9TQ8YNrU77QomPGPk/dZSRwtif26hV0xwQdoFRM9veR8UvW++vyJZzMAwGnNWwcw6erpZ/vFZ+zJAAAAzuEX/+HFb878i0vWYfikK4rRJJGk4KvVpkmpAICzO5DQwr0RANA2V0VMmYM1cH496wBwOjTjcYoizMBQiOkd5wVJJO8H4VFelJl1EMAEyMS5nW9WXUzUSuJoVlLaOBqcB80xgRYx0dNLzidVH8FUNADAWVAc5RjTzwAAANz7xX/Y097X+vLCu3rfOhZPHHv+M1XFaCQpeGs7L0o6EAKAjb6k69ZBAACmQq/pAkkcpeIllSUmQ4UlEy8cXelZB4DToxDTO+uuC5JI3g8CE2cAB3in5aVdubsHZWK/bqHxxHYAJ+N5zEttTao+iGI0AADGjOlnAAAA7fvyR3ufXf6vL/yWdRyeuPLsyUL38o3H/YP/8qJRMGNHkoLXetYBAMA0SuJoRRxKAQDGo3ERUxJH85LuO4oHZ7fBZKhwMBXKKYoww5OJxG5ftJX0mInvse8WmTgDNFM//6TWceA1Tq5vdaHhhw7iwdkw8RhoXyYaIPrG+aTqg+o9C89nAACM0ebM3KKqfdddkfMDAADQml/8x71vWsfgme7RfzE1xWgiScFX90jqAoDxqxOU71rHAQCYCq6SvTIHa+B8xtFBGY7QhdwpklUDwwQZ7/RcFyTVU1JJ3vcbRbxAQwf2c7zT8ouT6xvNM031rAMAJhmNcbzkfFL1Mbotrw8AAI7oDAdrneFgVtJHkjaMwwEAAJhYo/9zb8Y6Bs8sHv0XU1GMRodBb23kRbliHQQATJs64SEzDgMAMD16TReoE8/p7GcnbbODMpxbE4nLrvSsA8DpkdjtnfW8KNdcLsiU1CAwSRVwIxXPP75x2aQgE/t1CzTHBFrEeycvjau51OwYPgMAMFl2rQOYFHVRWlfSgqRHxuEAAABMnH/42Z72vtaX1nF45OazJwtXD/6LiS9Gq5MUUus48JqRjqmOBACMRSYSHgAA49E42YvEc3M0EQkIzXicIlk1PJl4zvFFW0mPWQtrwh0mqQIO1FNl7lrHgUOcvc9iwqcZnmuB9mXiecw3zidVn2B+DJ8BAJgsO9YBTJrOcLDVGQ56kr4jaVXVcywAAAAc+PJHe59Zx+CZ7sF/mPhiNHHw6atxHX4CAA4gQRkAMEaukr0yB2vgfEgqDwhToZwiWTUwPOd4Z9H1uR9TUoPQY5Iq0EwSR1fF84+PnEyKptGKGZ5rgZbxPOYl55OqAQBwiCZwLekMBzud4WBJ1eTSe2IKHQAAQGPP/p+9f2Qdg2cONSaa6GI0khS89YjDTwAYPxKUAQBj5CTZK4mjFfFMZ8lJ0iXGJhPNeFwgWTUwPOd4ZzUvyr7LBZM46orkfd9x3gu4sSb2c75Zd9ikIHO0Ds6GYmmgRXWhbWodBw4Z97kGhYgAgLOiGK1lneHgaWc4WOkMB7OS7kjaMA4JAAAgWL/4v/Z+3ToGz3QP/sPEFqPRYdBbu5KWrIMAgClFQgsAYFwaFzHVied3nUSD82AyVEDoQu4URZjhycRzji925TgRlSlBQeC8F3CA/ZyXnCXz0zzTDJOBgPZl4nnMNz3Xk6oBAHCMYrQx6gwHWWc46Er6nqRHxuEAAAAE56tP96xD8M2hd1kTW4wmkhR8xeEnABgg4QEAMEaNi5hIPDfHZKiAMBXKKYowA0PivnfaOPdLJV13vCbcWuS8F2imbq74wDoOvMbJfY3mmWZ2xXMt0CreO3lp1aAId3vMnwcACNtuXpQUoxnoDAf9znDQk/QdVUVpI9uIAAAAwvD8C+sI/PPsyUJ3//cTWYyWxNGKOPj00XJelH3rIABg2pDwAAAYI1dFTKlIPLe0xGSooGSiC7kLFGEGhsR976y6PvdL4mhRTEn13TIJREAzNOLwlpNkfr6/pmiOCbQoiaOueO/kG+eTqk+Jay0A4Cz61gFMu85wsFMXpc1KWla1hwAAAMAb/MNP99gzHTa//5uJK0arDz5JUvDPdl6UqXUQADClMusAAABTo3ERE8905tbzosysg8Dp1F3ImQrlRkoRZnAy6wDw0rYcJz2SvB+EDc57ASdS0VzRNy6T+VPx/bXgvEgewCvs1b1lVYTbN/hMAEC4+tYBoNIZDp52hoO0MxzMSrojpp0CAACc6P/7z5NXc9XQZBajcfDpLbqLA4CROkGZhAcAwDg0LmLimc4cz24BYfqtUxt5Ua5YB4HT4znHO20kPWZi6qPPRpIWrYMAQscESG8turiv0WjFjPMieQCvSSVdtw4Ch1gW4TaeJAoAmCrcNzzUGQ6yznAwL+l7kjas4wEAAPDNP/yXvRfWMXhmdv83E1WMJg4+fZXmRbllHQQATBsSlAEAY+SqiCkVz3SWrDoo43wy6wAmBEWYgeE5xzvLrs/9kjjqSbrlck04x54BaIhGHN5ycl/j+2uKexTQIgqpveRyoueZ1ffNXavPBwAE5RF7db91hoN+ZzjoSvqOpEfG4QAAAHjj68/0besYPPPh/m8mphiNDoPeors4ANjJrAMAAEyNxslePNOZW8+Lko6UgWAqlFO9vCh3rIPAmWTWAeCl7bwoU5cLJnE0K4mzRL+tsmcAnFgTEyB94/K+lopGKxacF8kDeIVCW285mejZ0JLx5wMAwsB5UiA6w8FOZzjoSZqRtKyqqR8AAMDUevb/7s1Yx+CbZ08W5qUJKUarDz55YPHPSNKidRAAMI1IUAYAjFHjIiaSWcwxGSogTIVyiiLMwPCc451eC2tmojjDZ9synLoATIr6fvbh2/4cxsrZ+ywarZjZcF0kD+A1mdir+8aLItz6bIXpKQCAN9nlLD48neHgaWc4SDvDwVVJd8Q0VAAAALwyK01IMZo4+PRV4+kIAICzqxMeSFAGAIyDqyKmTHSNt8SzW1gy6wAmBEWYgeE5xzvOkx6TOFoSxRm+Y88ANERjAW+lLqbl0mjFDHt7oGX1Xv2WdRw4xPmk6ibyouyJgjQAwMlWrANAM53hIOsMB7OSPpK0YRwOAADAWD3/wjoCL03GZLQkjhbFwaePVuloAgDjR8IDAGDMGick80xnjslQAUniaEVMhXKFgoqA8JzjHeeTR+riDKdrwrl7PkxdAEJW38/Ye/tnPS9KV4mRmWi0YsFJMSGA4yVxNCv26j7qWQdwFAVpAIATjMTZ7sToDAdrneGgK2lB3PcBAMCU+OrTPesQfDQrBV6MRiKKt3bFgTQAWElFwgMAYDwaN6Dgmc7crjxMXMHx6qlQd63jmBAUYYYnFc85vmhr8kgm6UoL68KNDYeFGsA0WxH3M984u6/RaMWMy2JCAMdbE3t13zifVO1KXZD2kapzRwAAJGmFxnCTpzMcbHWGg56k70haVvV8DQAAgOkxKwVejCYOPn21yEMkAIwfCcoAgDFy1YAiE890lpgMFQgKN51qq5AGLeE5xzvOJ48kcZSKqY8+G0latA4CCF1dqHTbOg68xskzEft1M+ztgZaxV/fStutJ1a7VDYDmRWI6AKC6D9A8YoJ1hoOdznCQqkpGviMK0gEAAKbFvBRwMVoSR0uSPrSOA6/xtgsXAEwyEh4AAGPWOGGPrvHmVvOi7FsHgVNLxRQNVyjCDAjPOd5xPh0riaN5SfddrgnnuG4CDSVxNCvuZz5qPO37gEw0WrHAPQpoEXt1LwVThJsX5dO8KNO8KK+qSkxft44JAGAiZc8+HTrDwdPOcJB1hoNZVff+DeOQAAAA0K4rknRhb2/POpAzq1/cbYkXO77Zzoty3joIAJhGSRytiGkBAIDxWM2LcqnJAnVxwY54prOyK2meF4BhqKdCfWIdx4RYz4uS6T4B4TnHKyNV944dVwvW+4EtUWzrs8b7PgBSEkd90VzRN9uSuo6moi1K+rh5SDgj7lFAi+q9el9MRfPNPdcNQsatPufa/5oX58MAMMl286KctQ4CdjZn5rqqCumZFA8AAII3t/GOdQg++k6o/1UycSjlm5EkEroAwED94oYETQDAOOyqmtDUVCae6SzRPT4QTIVyaleBdA9HpU7s5jnHH0suC9FqqShE89m23Oz7gKmWxFEqCtF85OSZiKl3ZlydTQA4WSoK0XzjfFK1hbwo+6oKHSW9PPualzRbf0lVodq+WfHcCACh6lkHAFud4aAvqb85M5dKWlL1d4L3wwAAAJNjNrhitCSOlsSLOx+lLSSkAADeon5Js2YdBwBgajRO2KuLC245igdnt1onfSAMqUi4cYUizIBQiOmd9bwoM5cL0lTFeyNx3QQaS+JoXtJ96zjwmuW8KLccrZWJRDoLi9yjgPawV/fSSBOa0F9fz/tN1qj/zgKYLLOqClMXxX47VOu8i8K+znCwI2npSFEa774AAAAmQFDFaPWLuwfWceA165PQhQsAApWJA1gAwHgsN31xRHGBObrHB4SpUE5RhBmeTDzn+MJ50iNNVYKQOizUAKYS1zpvbeRFmbpYiOaZZlwWEwI4gvuXt2gM/Aac+QATK6vvS0uiyUdoRqq+b8AhneHgqar3lOnmzFxP1d8TpvECAACEqxtUMZpIWvTRxHbhAgDfMVnGWxuStiQ9PfArgPNbE8noPth2lLDH99MWE04CQeGmUxRhBobnHO+0ce/IxH7AZzQeA9zIRJdv3zh7n5XE0azYY1pwdTYB4GSZ2Kv7ZoP9OYBpVZ9JpUkc9SV9YhwOTo8iarxVZzjIJGWbM3NdVUVpvBMAAAAIUDDFaEkcpaITgo9IZgQAAyQoe2Wkqrgio/si4Fb9DEDygx96TRega7y5xpPtMFaZuP65wrlFQHjO8c56XpROpyJQbOg9Go8BDiRx1BPXOh8tOUyIzMR+fdy4RwEtY6/upZGkResgAMBaXpT9JI7uSXpgHQveiiJqnElnOOhL6m/OzM2qavpy2zIeAAAAnM1F6wBOI4mjeTFy20errhNSAACnlomEB2sjScuSZvOi7JHgD7jFM4BXlvOi3GqyAF3jzdE9PiAkfzm1yh4tOJl4zvGF84Rvig2DsEgBL9BM/exD4p1/1vOizFwsRKMVM2nTswkAJ6vvX5lxGHgdDXYAoFYXOG1Yx4E3ooEEzq0zHOx0hoOepBlVeTgj24gAAABwCvPeF6ORpOCtXZFMCgAmSFD2wrqqIrSUF4FAazLrACDJXRFTJooLLPWsA8DpcAbiFOcWgWGKjHfaKEpaE/sBn1HAC7jBtc4/zhIi68Y5qYu1cCZMVwDal4n7l2+cT6oGgAmQWQeAN3I5jRpTqjMcPO0MB2lnOLgq6Y6qdz0AAADw01Xvi9FUvdS5aR0EXkOXXAAwQIKyF5bzouQ+CLQoiaNUPAP4otd0AbrGm2s82Q5jlYnkL1foHh4Qpsh4x3lREvsB723nRblkHQQQOp5lveXyHC8T+/VxG0latA4CmGTs1b3EZBkAOB5Fuv5yNo0a2NcZDrLOcDAr6XtiMiIAADD0jfkL1iF4y+titCSOupLuWseB15DMCAB2MpHwYOmOowlBAE5Qdxm/bx0HJEn3mu776+KC1Ek0OA9Xk+0wBky/dYrpPuHJxHOOL5xPFWQ/4D0SXQEH6vdZPMv6x9m+kGJDMzSZAFpUn8U+sI4Dr+HaBwDHqK+NI+s48BrOltCqznDQ7wwHXUnfkfTIOBwAAAAc4G0xGpNfvLVBMiMA2Ki7U5KgbOce3byAscisA4Ckat/vYkJNJooLLPWsA8Dp1IUamXEYk2JbFL0EhS783mkj6XFN7Ad8ltJ4DGiG91necrYvpHGOmfW8KJl+AbQrsw4Ar+HaBwBvxhmGfyiixlh0hoOdznDQU1WUtiyKUwEAAMx5W4ym6gXRdesgcAidTADACJ3kzT1yVJQB4A3oMu4NJ/t+igvMMdE6LJko1HCFF98B4TnHO86nCrK/8946z7qAE5l4n+Ujl/vCzNE6OL1d8U4SaFUSRytir+4brn0A8HZXrQPAIRRRY+zqorRU0qykO6r2UAAAAK359n9z4efWMfjKy2K0JI4WJd21jgOvWcqLcsc6CACYUplIULaynRdlzzoIYNIlcdQVXcZ9kTbd99dd41Mn0eA8tploHQ4KN52iCDM8mXjO8YXzqYJMkfEejccAB5I46km6ZR0HXuNsX0hhtRmaTOD/Z+9+YuM48zTPP0lKHjktl5ge1djdjrJSAbmrorGkqL3kANuAUqc9uADRwOygd/ug1KmrCliYOlQBc3LSwB72ZPLSXYtFtYPALFDbF5PA7tnJI09JIg95i05i4lKA4VQOFuXCHNp7iKDEkigpyXwjf29kfD+A0C5bDj5oWoyMN35/UKD8LJZ6DP/wsw8A3ozP5v7gbAmmWuPB09Z4ELfGg6akTyUdGEcCAAALqnZFf7LO4CvvmtGiMFgREwZ9tD9M0tg6BABUEQXKpjhABeaAZwCvHDjajhGL5gIrE0kb1iEwHbZCOUUTZsnwnOMdp0WPfL4rhQ0KXYHZ5E23bBf0z4Grz4UMzjHjfFsrgOf4rO4tfvYBwBvk5+nwB03U8EZrPNhrjQdtSQ8k7RrHAQAAC+atv6x513PlCx//HxOLokXfUIgPAEYoUDbXZbsGMBddSbesQ8DN536mxpubebMd5ioWZyCudKwDYHps0PROEVsFu+LzgM+2KHQFnIjFZznfOHufRbOGmeNhkm5ahwAW3LY4i/XNiXhGBoBptK0D4Jn9YZLuWYcAXtQaD3qt8aAj6bakHWXnBAAAADP5Nz/Rv1pn8JVXzWhRGGxIemidAy9hSi4A2IlFUYsVV9uBALxGPmX8M+sckCRtztrElDcXMDXeDveuEmErlFNFNNKgWLF4zvGF862CfL7zHpskAQeiMNgWTbc+mvm59oyuaNaw0LEOACyyvB7jkXUOvITNMgAwnY51AEhiqD1KoDUejFrjwaakpqQnypr/AQAALmV5pfYj6wy+8qYZLd/8EhvHwMt2mJILADYoUDbFASowB0wZ98r+MEljB9dxcQ1cDveuEmErlFM0VZQMGzS903F5MT7feW8iacM6BFB2NN16y9VzLd9jO08YMgEUh8/q3qIeAwCmkNcVUjvhB5qoURqt8eBpazzYbo0HTUmPJR0bRwIAACW0/I4a1hl85U0zmpiI7KNjURwHACbyAuUvrXNUWMfhFGUAr9YVU8Z94KSJieYCc13uXaUSizMQVzrWATA9Nmh6p4itgtvi853PXG4MAiopL+Tfs86BlzgbzkGzhhk2fQPF2xNnEb45HibppnUIACiJ2DoAJGVN1DwTo5Ra40HcGg/WJT2QtG+dBwAAlMO1OzXrCF7zohmNzS/eYpIJANiJrQNU2D4HqEDxmDLulZk/99NcYI6ivRKhcdOpIhppUKzYOgCeOXC9VTAKgw1Jj1xeE0452xgEVFwsCvl9tOHwfRaN1fPHpm+gYNRjeKtjHQAAyiAKg464j/ngRAy1xwJojQe91niwIem2pF1lz6QAAADneusvrBP4zbwZLV+j3TWOgZdR0AUARihQNkXhAzAHTBn3iqsG3NjBNXA53LtKhMZNp5w30qBYPOd4xfm9g8933jsRnxeAmeWF/A+tc+AlO8Mk7bm4EI3VZtjcCRSIegxvUY8BAFPIz9QZxucHhtpjobTGg1FrPOhIakraEk1pAADgHNf/+9qJdQafmTejiSmSPqKgCwCMUKBszuUUZQCv1hVTxn3gpBCd5gJzFO2VS2wdYEHQhFkyPOd4p1vAvSMWZ7w+o1gImFF+L+ta58BLjuXo+0JjtRk2dwLF2xOf1X1zTD0GALxZ3lDdE/cxHzgbAgL4pjUePG2NB93WeLAi6bGywV4AAACSpLd/VvOh38pbpv/PyYsWWaPtFwq6AMBWbB2gwjhABeYgCoO2pCCVdKEAACAASURBVM+sc0CSg6JkmgvMUbRXIjRuOlVEIw0Kkhd2u9jCCTcOhknqdJo0m4K8t8WzLuBELAogfeSy2TYW3+N5450kUDDOIrzVsQ4AAL47c6bIZ3R7J2I4CyqiNR7ErfGgKemBpAPjOAAAwAP/5lbtunUGj/XMmtEoWvQWU/UBwAgvBU1xgArMAVPGvbIzTNKZGgNoLjBH0V6JcAbilPNGGhSuKzai+sL5vSOfUN11eU04xcYFwIEoDLbFmZ2PngyT9MjFhaIw2BCN1RY22NwJFIezCG9tubp/AcCiyt8/9cRzmC9cDgEBSqE1HvRa40Fb0m1Ju8ZxAACAkeXr0vI7aljn8JnlZrTY8GvjfEzVBwAjvBQ0R+EDMB9dUYzuA1cNuF3x/bTEy7+SoBHXKZowS4aNqN4pYghVLCZU+2oiacM6BFB2eZMS9zL/OBtQwOd1Mzts7gSKwxApbx0wLAIApsJAEH/wuR2V1hoPRq3xoKOsKW1L2ZkrAACoiGt3atYRvGfSjMbmFy+diIIuADBBwYM5plACc0AxuldmbmLi+2luf9bNdpirrjgDcaXLNvfy4DnHO86HUOVnvPddXhNOFdF8CFQK9zJvuR5QsCcaq+fN1ZAcAK/WFUOkfMOAHQCYQhQGsaRH1jkgic/twDN5U1pXUlPSY2V/PgAAwIL70f9Q+4N1Bs8dzb0Zjc0v3mKqPgDY6YoCZSvHTKEEikcBn1dmnmDI99MchSslQuOmU862X2BuuqL40RfO7x2c8XrPefMhUFE0Kfmp46rZNgqDTdFYbWGDd5JAcdjq6S0G7ADAG0Rh0BGNaD7hczvwgtZ48LQ1HsSt8aAp6VNJB8aRAABAger/Xe2/WWfw3NO5NqPlRYtMb/cPK7UBwAgFyqYo5gfmpyuK0X3gaoJhV3w/LTFIpCRo3HSKz20lw3OOd4q4d8SOrwd3+JkJOECTkrecbYnOG6u/dHEtXMjWMEmPrEMAi4qzCG8xYAcA3iBvRPvKOgee4XM78Aat8WCvNR60JT2QtGscBwAAFODtj2s3rTN4br7NaKJo0UfHwyTdtA4BAFXES0FzXQ5QgeJRjO6VmQvR+X6ac1Z4ibnoijMQVzaZHl4eDKPyjvN7RxQGXbHd22c0rgMzoknJWydy1GzL5xUzx8Mk7VqHABZcLLZ6+oZhEQDwBvkzGI1o/uBzO3ABrfGg1xoPOpJuS9pR9vkPAACU3LU7NdWu6m3rHD6rr/WP5taMRtGitzrWAQCgwrqiQNkKUyiBOaDp1itbs25D5vtpjsKVEuEMxKn9YZLG1iFwIbEofvSF83tH/vPtc5fXhFM0rgMzoknJay6bbWNxLmuhYx0AWGT5RpmH1jnwEgbsAMBr5I1oPesc+DMd6wBAGbXGg1FrPNiU1JT0RNlQHQAAUFLvtmpj6wxlMJdmNIoWvfWEjTAAYIMCZVMU8wPz0xXFXT5wNcGwK76fljbYclIOnIE4xee2konCYEMUP/rE6b2Dn2+lsGkdAFgA2+K5x0c7sw5YOUWzhpkd3kkCxYnCoKnsHga/MGAHAF7jzDAQBlv5Y4vP7cBsWuPB09Z4sN0aD5qSHks6No4EAAAu4Ud/U/v/rDN47kCaUzOaeHnnIzbCAIARCvjMbTOFEigeTbde6cx6Ab6f5pwVXmIuOANxx+X2CxSM5xzvFHHv6Iqfbz7b4lkXmE3epPTIOgdecjxMUifNtvnWBd6Nzd9E2ecIAMWJRSG/bxiwAwCvkZ8l9sRZk09cDbcEkGuNB3FrPFiX9EDSvnUeAAAwvbc/rt20zuC5p9IcmtHyici8vPMLB58AYKsrDlWtnHCAChSPYnSvzDzB8MxkStg4EUV7pcEZiFP7wyTlZ0+5xKL40RfO7x35zzca0/01Ec0VwExoUvJax+G1YvF5xcImQyaA4kRh0JV03zoHXsKAHQB4vT1Jd61D4M90rAMAi6o1HvRa48GGpNuSdpWd5wIAAE9du1NT7arets7huSOp4GY0ilC91WFKLgDYoIDPXNc6AFARbAXyg6sJhrEo1rNE4UpJcAbiFEN0SiZ/znlonQPPOL138POtFLb5vABc3pmfczz3+OfJrANWTkVhsC2KXS2cDJM0tg4BLKq8mfpz6xx4CQN2AOA1ojCIRSO1b2YebgngzVrjwag1HnQkNSVtiaY0AAC81Pgfa3+wzlACI6n4zWixeHnnGw4+AcAIBXzmKHwA5oCtQF7pzHoBmgvM7QyTtGcdAlOLxRmIKzRhlkgUBk3xnOOTIu4dsfj55jO2ogGzi0WTko8Ohknq5OdbFAZtMSDMStc6ALCoeOfkLQbsAMBr5Bs9eY/oF1fDLQFMqTUePG2NB93WeLAi6bGkE+tMAADguRvtpWXrDCUwkgpsRovCoCOKFn1zIg4+AcBSLAr4LHWtAwCLjgIIr8w8PZ7vp7kTce8qDRo3nWKITvnE4jnHF8dyfO/gjLcUYhp4gcuLwmBT/JzzkbNC/vzZls+XNhgOBhSrK5qpfcSAHQB4hfyciY2efplI2rAOAVRZazyIW+NBU9IDSQfGcQAAqLzl69LVf6eb1jl8V1/r96SCmtHyichMY/UPB58AYIQCZXMTUXQCzEMsitF94Gp6fCy+n5Z4fisJGjedYnp4yeQF/Petc+AZp/cOznhLg+8RcEn5ed2X1jlwrs4wSUeOrhWLZ1srXesAwKJi46O3dhiwAwDny5+/vrLOgZd0HT57AZhBazzotcaDtqTbknaN4wAAUFk32kvfW2cogePTvyhqM1osXuz4ZmuYpD3rEABQRRQoe2GPgn6gWDTdesNJIwffT3M7PL+VSizOQFzZ4DNbeeSNSl3jGHhua9atqOeIxc833x1TMARcThQG6+K8zlfONuWy+c4Uw8GAgvDOyVsn4hkZAM7F85e3XA23BOBQazwYtcaDjrKmtC1lz9cAAGBOGp/UvrXOUAKj079w3ozGRGQvHQ+TtGsdAgAqLBYFfNZ61gGARUYBhFdmnmDI99Pc8TBJN61DYDoUtzpFE2b5xOI5xxfOz/444y2N2DoAUEb5M09P3Md8dCJHm3JpnDfHcDCgOLGkW9Yh8BKnm6oBYFHkn8t74vnLN06GWwIoTt6U1pXUlPRY2ZkJAAAo0PJ1qf7XtZ9Y5yiBZ0NynTaj8WLHWxQyAoARNst4o2cdAFhwsXiJ5ANXEwxj8f201LEOgOlwBuIU08NLhkYl73RcXiyfVt11eU0Uho0zwAXRiOY9l4X8sfg+W+IeBRSAd07eYsAOAJwjf/7aE5/LfTTzcEsA89EaD562xoO4NR40JX0q6cA4EgAAC+tGe+l76wwl0Tv9C9eb0XiA9A8HnwBgJC9Qjo1jQBIHqUBxKIDwhpMJhlEYdMT309LWMEmP3vzb4IlYnIG4wvTwEqER0ztF3Dti8fOtDE541gUu5kwj2l3jKDifs/dZNM7bGyYpzWiAY7xz8hYDdgDgHDx/ec3VcEsAc9YaD/Za40Fb0gNJu8ZxAABYOI1Pat9aZygJ95vRojDoigdI30zEwScAWIpFAZ8Pjq0DAIsqf5EUW+eAJAcTDPOCFl4+2TkeJmnXOgSmQ3GrUwzRKR+GUfnjwPW9gzPeUqGBHbgACiG9dzxM0k0XF6Jx3gtMaQeKEYtnMR9tMGAHAM61LZ6/fORkuCUAW63xoNcaDzqSbkvaUfZnGwAAzGD5ulT/69pPrHOUwHF9rf/sLMxJM1oUBuuSPndxLTi1zcEnANigQNkr3AuB4sSiAMIH+44mGMbi+2mpYx0A06G41Smmh5cMjUpecV44whlv6dCMBkyJRrRS6Di8Viyeba1xjwIcy5/FeOfknyI2VQNA6UVhEEt6ZJ0D55p5uCUAf7TGg1FrPNiU1JT0RNl7NwAAcAk32kvfW2coiT87C3O1GS12dB24c8JUfQCwQYGyd9atAwCLKAqDDUkPrXPATSE6TdTmKFwpl1gUt7rSYYhOedCo5B2nhSN5o8aeq+sBgC9oRCuFJ66eh/KzCp5t7fEZH3Ao/9nGs5h/jqnHAICXRWHQEY1ovjpwNNwSgGda48HT1niw3RoPmpIeSzo2jgQAQOn82/9Q+6/WGUqid/Z/zNyMxkRkb/HwCAB2YlGg7BO+F4BjeTFfbJ0Dkhw0ctBEbY7ClRKhcdOpnWGS9qxD4EJi6wB4pojCka6kW46vCQCm8mednniH5TNn9zTOKgAsonwoSGydA+fqWAcAAN/kjWhfWefAuSaSNqxDACheazyIW+PBuqQHkvat8wAAUAZvfVDTtdu1961zlETv7P+YqRktCoO2mMLlo4k4lAYAExQo+ymfHArAnVg0evpgf5ikLjaYxOL7acXJZjvMR14A9qV1jgVxLJpgS4VhVF5xfu/Iz3g/c3lNALCWf3Y7Evcvn7m+p22KZ1sAC+RMky0/2/yz5WqrJwAsChrRvDfzcEsA5dIaD3qt8WBD0m1Ju8rOYQAAwDnee1jjPjmdk/paf3T2b1y6GY0Jg17b4wESAOaPzTJeoxkNcCRv7nxonQNuivZoojbXpXClVGLrAAuEF98lkhfzM4zKH5vDJB25ulh+xuuiuR3zt2IdAPBV/pzTF8X7vuu4uqfl57J8XvEH9yhgRvnn9J5oqvbR8TBJu9YhAMAnNKJ5z9VwSwAl1BoPRq3xoCOpKWlLNKUBAPCS9x4u/at1hpLovfg3ZtmM1pV0a4Z/H8XZtg4AABUViyIXXz3Ki1IAzICBFF6ZuZGDJmpzB8Mk5dmtJNgK5RTTw0uEe7939odJGju+ZiyeY8uqbR0A8E0UBitRGOyJbbZlsOu4GDJ2eC3Mrm0dACgzGtG85nxTNQCUHY1o3uPeBUCS1BoPnrbGg25rPFiR9FjSiXUmAAB88M56TcvvqGGdoyReeq9zqWa0KAzakj6bNQ0KMaGoCwDmLy9QZrOM3yj4B2YXi2JlH7gq2ovF99MKL/9KhK1QTjE9vHy6ogDSF87vHWy8Lb27DF0BnsvfWx2Jn2tlcCJp09XF8u8957J+4R4FXBKNaN7rUosBAM9FYbAtGtF8N/NwSwCLpzUexK3xoCnpgaQD4zgAAJj68f+89F+sM5RI78W/ceFmNCYie69nHQAAqiZ/sU6Bsv8e5pPZAFwCxcrecFK0F4XBpijWs9QdJunIOgSmFlsHWCAd6wCYHsOovOO0cCR/jo1dXQ9mOtYBAGv5NrRY0jeSbhnHwXRcF0N2HV4L7nStAwBlk39G74lGNF8dDJOUoYcAoGfPYT1xdui7fccbqQEsmNZ40GuNB21J9yTtGscBAGDu3vqgpnf/fe0n1jlK4ri+1n/p3c5lNqPF4oWez5jEBQDzF1sHwNS+oiENuDgGUnhlY9aivXzL05eO8uDiKFwpkXz7LUVgbmwxPbw8uPd7p4jCkVhsSF0En+ef7YBKyodsjCQ9Mo6C6W0Nk7Tn6mJsRfPaI+5RwPTObPjkDMJPzjdVA0BZ5feskfgc7jvuXQCm1hoPjlrjQUfSbUlbyn6GAACw8N57WOOeN734vL95oWY0tiGUQs86AABUSd7YxEFruXyVFysBmF4sipV98MRRI0fs4Bq4HF7+lUj+Up3tt24cD5O0ax0CF9IVw6h84fzewYbUhRPnDaRAZURh0InCYKRsyAbPquVRxGdC19eDW3vco4A3ywfhfCPuaT7rDpN0ZB0CACy9sJWae5b/XG+kBlABrfFg1BoPupKakh5LOjENBABAwd57uPSv1hlKpHfe36z98MMPU/3b+cuCkXig9N0Dl1MlAQCvlt8bj0SRZlntS9rkBSLwevlAiq+tc0D7wyTdmPUieXELzTV2nrAVrRz4nOfcPbailUfeiPmNdQ484/SsLwqDprKfb5zxLpZjSW2KjLDI8s9nG6JhuqwmktZdnsPxmaU0jpUVwvI8ALwg/zm2Lbah+e5gmKRt6xAAYCl/t7QpzpPKYneYpB3rEAAWw2FjtaNsYB4D7gAAC+W9ny99/+Gvl962zlESJ/W1fvO8f3CRzWixeKgEAOCsTVH8UmYPJR1FYdBlQi9wvrxYOTaOgbxwa9aLRGGwLhrRLO3TiFYqXfE5z5UtCk/LI/9cHFvnwDM7BQyd2hNnvIvorrLn27Z1EMC1KAzW8+n7I0lfic9oZdUpYCBUx/H1UIy7knr5ZlYAys5cz2yWoRHNbxNlzfAAUDn5JrTNfCv15+I8qSxOlNWxAIATrfEgbo0HbUkPJO0axwEAwJl/+x9q/9U6Q4nsveofTLUZLX9B8KXLRCgMm9EAYA7YGLpwJso+MG1TLA08F4VBT0y4suZsenwUBkeiwMXKRFKTbSXlwIYFp46HSbpuHQLTi8JgT9nQBtg7UfYZwNm9gw2plbEvqcuzLcosH6TRUVb8TfNZ+e0Mk9RpMWQ+POdfXF4Tc3Gi7B4VWwcBLOTnDR1Jj2yT4AI+HSbpK4ttAGAR5c9jm8qex6iFKB/qBgEU6rCx2lR2n+iI+wQAoKTeWa8p3Fm2jlEm9+pr/XPfPb+xGS1/oXMkPjiUxWNe4gBA8SjiW2gnyhrT9jioRZUxkMIb91wUEnPfMkfhSknkAweORNGzC86aaTEfURhsSPraOgeecVo4khcS9V1dD6VwrGzT4R4/i+G7/GfUuqR2/ovPYoujkOEEPOOW3kTP71E92yhAsWiwLrX9YZKyFQ1AJeTngm1xvyo754NAAOBVDhurK3relMa9AwBQKs3/ffm/vPvvaz+xzlESJ/W1fvNV/3CaZrSe2IZQJjxYAsAcRGHwVDRqV8WxsqL0kaSepKdMmMeiYyCFN5wMmqD43ByFKyUShcG2pM+scyyIJ8Mk3bYOgemw+dk7Ts/38u9vT2xIrbITZf8NHOW/RjSowUK+EUbKihxXlDWg8f5pcRW2IToKg5EodFokB8ruU6P81xGbxVFG+RlcU9n97bTJmmescirsHgYAlvL3f01l96mmsnsV50WL4UTZcDjuXQDm7rCx2lHWmMY9BQDgvbc+qOmn/zdb0S5gp77Wf2Xtwmub0diGUEonwyRtWocAgEUWhUFH0lfWOeCFibJCPmDRNEVRlzVnG4+jMDgSB79WKFwpkbw4+hvrHAviYJikbesQmF4UBnuSHlrngKRsGEbb5b2DRlu8wYmywn+gCCviWaTKnGz6fhEDVyrnwDoA8Abc6xaT003VL8rfM3aKuj4AvKAp3vlVQaH3LgCYxmFjta2sKY33TQAAb93635b/8KO/qb1vnaNEbtfX+qNX/cNXNqOxDaHUnBWuAgBeRlE/AKBgW8Mk7bq4EMXn5j4dJumedQi8GVuhnJoom8A6sg6C6URhsCHpa+sceMZp4T6NtgAAI4W9p+I5FwBQMKebql+UN1X3xBkUAMCdQu9dAHBRh43VpqSupEe2SQAA+HPL16W//n+vWMcok+P6Wn/9db9h6TX/LBYHYGXVzQvpAACO5c3aNKIBAIqy67ARrS0K9Czt04hWKrE4A3GlSyNaeeTnR7F1Djyz5bgRje8vAMDCTsEDEzcKvDYAoNqOC25EWxGNaAAAtwq9dwHAZbTGg1FrPOhIakjaUjbIEgAAcz/+uyXuSRcTv+k3nNuMFoVBR9J9x2EwP7dEoQkAFIViBwBAUXaHSdpxcSGKz81NJHWsQ2A6+Vaoh9Y5FsTBMEm3rUPgQvZEEZwvjl01pJ8RKzsnBABgXnYLLuJvinsbAKAYE0ntoi5OIxoAoAATUb8CwGOt8eBpazzotsaDFUmPJZ1YZwIAVNfydenm/7T0lnWOkonf9BteakbLD8G6BYTBfD2MwiC2DgEAC6hjHQAAsJCcNaLluqJAz9LGMEmfWofAm9G46RRNmCUThcGmGEblk47Li9FoCwAwcCyp6In87YKvDwCornbB53nbku4WeH0AQPVsDJN0ZB0CAKbRGg/i1njQlPSppAPjOACACvrx3y1Nalf1tnWOEtmtr/XfeFZ23ma0TVG0uCgeRWHQy4vrAAAzyn+e8qIIAODaE5eNaFEYtCV95up6uLCdYZL2rENgarGYSO1Klxff5ZFvFekax8BzW8MkPXJ1MRptAQAGjlV8Eb8krRd8fQBANT12+Uz2oigMupIeFXV9AEAlPeFdFIAyao0He63xoC3pnqRd4zgAgIpgK9qlxNP8pj9rRssLFYqeWoj5ui9plE+7BgDMpm0dAACwcB4Pk3Tb1cUoPjd3Ipo7SoOtQU7tu/xZhrmIRSOmLw6GSdp1fM1YfH8BAPNzovk0okk0owEA3NsaJmlc1MWjMOhI+ryo6wMAKmmX83gAZdcaD45a40FH0m1JW5ImtokAAIuMrWgXdlJf6/em+Y0vbkbbFIUKi+iGpC+jMBhFYdDNp18DAC6OYgcAgCsTSZ8WUOjQFZuuLXXmVICJGdG46dREUsc6BKaXDyy6b50Dkgr485N/f2m0BQDMy0TSxhyfgzifBQC4tFvAcJBnojBYl0SzAADApYNhknasQwCAK63xYNQaD7qSmpKeKBt6BACAM2xFu5Spz7NqP/zww7P/EYXBSBQuVsWxpD1JR5KOhkk6so0DAP6LwmBPFPUBAGY3UTY1/sjlRaMwaEv6xuU1cSE7wyRlI3VJRGHQE804rnw6TNI96xCYTj6g6EgMo/LFE8cbUpvi+wsAmJ9Cnm1fJwqDH978uwAAmMpukcX8+SCkkXg+AwC4c6z5baUGADOHjdWOskF+vMsFAMzsg79fmvz4f1nifOZiGvW1/lTPHVdO/yIKgw3RiFYld/NfkqQoDKTsoZUHVuBinkqKKb6sjBXrAACA0ivkRRFbnsydKNtKhxJgK5RT+zwLlU4sCuF8ceCyES0Xi+8vAGA+LBrRmvP6WgCAhTePRrSeeD4DALhDIxqAymiNB7Gk+LCx2lbWlPbIMg8AoLze+qCmm/9x6ap1jpLZnbYRTTrTjKbspo1qu/vm3wLgHA+jMDgtQN7j8AcAALzCrqTNgj4rbIvhIpY6fAYsh7yAtWscY1FMxFlSqURh0BWNmL5w/ueH7y8AYI7m3oiWa8756wEAFlOhjWi5bVF7AQBwh0Y0AJXUGg96knqHjdWusve7G2LgAwDgAv7if136Q+2K3rfOUTLdi/zmJenZZKaHRaQBgIq4JekrSaMoDLpMaQUAAC94PEzSQhqW8i3XTAOzszNM0p51CEwtFi8pXKEJs0SiMFiX9Ll1DjyzOUzSkauL8f0FAMzRsaR1g0Y0AABcKLwRLR8UwlktAMAVGtEAVF5rPBi1xoOOskFFW5JOTAMBAErhrQ9q+tHf1GhEu5iD+lp/dJF/YSn/v23nUQCgmm4oKwD7lygMYprSAACovBNJ94ZJGhdx8XywSCHXxlSOxZat0ojCYFNsDXJlf5ike9YhcCGxdQA8s1/A5wLX1wMA4DynRZAj6yAAAFzCPBrROmJQCADAHRrRAOCM1njwtDUedFvjQVPSY9GUBgB4jeA/LX1rnaGEuhf9F2hGA4DiPFLWlNaLwqBtHQYAAMzdroqfGB+LLU+W2AxVEvmQiK5xjEUxkdSxDoHp5VPZ71rngKQC/vxEYbAtvr8AgOIdiCJIAEB57cyhEW1d0naRXwMAUCn74hkMAF6pNR7EeVPaA2XnVgAAPPPOek3vrNduWucomeP6Wr930X/pSv5/206jAADOui/pmygMTiR1i9qMAgAAvDFR1qRU6NagKAw2JD0s8mvgtbYKbjSEW3uicdMVmjBLJC+GYyq7P5z++ckH33zm6noAALxC4ZtkpsTzFwDgMh4X/W42CoMVST1x9gQAcMOXZzAA8F5rPOhJah82VpvKBpM+sswDAPDDX24u/UHS+9Y5SuZSQ5ZON6MxPRcAindL0ldRGDyNwqCbv5hAufSsAwAAvLcvqTmHRrQVZVvRYON4mKRd6xCYDluhnNop+ucbnIutA+CZfZd/fvgsAACYkye+FEEyEAEAcEETSQ/mNCS0JxrRAABuPPblGQwAyqQ1Hoxa40FHUkPSlrLnAQBABb3386Xvr92u0Yh2MSf1tX58mX9xKZ+gCwCYnxvKJtOPozCIozBoGufB9Ji+CwB4lYmkT4dJujGnArlYFDhY6lgHwHTYCuXUibKJgiiJKAy2RSOmLyZyf+/oKht6AwBAEU6fcS81CbNAJ9YBAAClcCxpfZikvaK/UBQGsXj2BgDMbp5N1ACwsFrjwdPWeNCV1JT0WJwlAUClLF+XPvjF0n+zzlFC3cv+i0tv/i0AgAI9kvQvURj0aA4uhZ51AACAl3Y1h21op6Iw2JD0cB5fC+faGiYpDerlEVsHWCAdtlGUR/58+Zl1DjzjtFk9/yzA9xcAUJTTAn4fN+KOrAMAALy3K6k9TNJR0V8oCoNNZe96AQCYxYGy94w96yAAsCjyprS4NR40JX2q7GctAGDB/fjvlibL7zLc/YIuvRVNyprR2s6iAAAu676kb6IwGEVh0LEOg/PlxYP71jkAAN44VjalcG4NGlEYrIjmGkvHwyTtWofAdKIw6IrJ1K7s8CK8PLhXeMfpnx++vwCAgs2tgP+SfGyQAwD44XSr51zOavMhIV8W/XUAAAtva5ikbQbBAUBxWuPBXms8aEu6p+zsCwCwgN76oKab/3HpqnWOEurO8i+zGQ0A/HJL0ld5U1o3LzKDXyh4AABMJD0ZJum6QXPGnsQEF0Md6wCYThQG65I+t86xIE404+ET5q6r7NkS9or48xOLzwIAAPfmWsA/A85mAQDnOdAct3rm507xPL4WAGBhnSgbeNk1zgEAldEaD45a40FH0m1JO8rOwwAACyL4T0vf1q6obp2jZGbaiibRjAYAvrqlrHh2FIVBHIVB0zgPcsMkjZUdDAIAqmlLUnOYpNvz/sJRGGwq26YKG0+GSXpkHQJTi60DLBDfC5JxRhQGbUmfWefAM07//OSb1B+6uh4AALm5FvDPIt/YdmCdAwDgjdOhYXPbuppk+wAAIABJREFU6pkPEmVgGABgFjvKnsF61kEAoIpa48GoNR5sSmpKeiLqAAGg9N5Zr+md9dpN6xwl1J31ArWf3f6wK6aFA0AZ7Eva5kDKXhQGG5K+ts4BAJirXUndeRU1vChvTD8SRQ5WDoZJ2rYOgelEYbAtmnFc2Rkm6aZ1CEwnL4g7ElvRfOH0zw+fBQAABZgoe86d+7CVWeTN999Y5wAAmDtQNgBkNM8vGoXBkaS78/yaAICFcSxpk5ofAPDPYWO1I2lTfNYHgFL66//nymT5Xd6jX9Bxfa2/PutF2IwGAOXxUNI3URgc5dPQYSSfErxvnQMAMBe7km4Pk3TuhQ0viEXxuZWJpI51CEyHrVBOncjBFCTM1bZoRPPFsdz/+YnFZwEAgDun29BK1YgmSXnhJtvRAKC6TiR9Os9taKeiMIhFcSoA4OImkraGSco2NADwVGs8iFvjwbqkB6ImEABK5YO/X6IR7XKcDNat/ez2h2x3AYByOlFWjLY9TNKnxlkqJ986MBLFgACwqEw3oZ0VhcGmpC+tc1TYkzIWaFYRW6GcuzdM0iPrEJgO25u94/TPD58FAAAOTZRN4o+tg8yCjaEAUFlbMnovynMZAOCSvHnfCACY3mFjtals6OCGOH8CAG+99UFNf/V/Lf+xdkV16ywlc1Bf67ddXKj2s9sftiV94+JiAAATE0l74gBr7qIwWJfUt84BAHDKq5dCFNiZOxgmads6BKYThcG22IrmytYwSbvWITAdBmV4x+mfH547AQAO7Sh73l2IwWY04wNApZie2XLPAQBcwoGye1fPOggA4PIOG6sryjbHbIr3cADgnY9/t/zttTu1m9Y5Suhefa3vZLhu7We3P1yRNHZxMQCAuV1JMQda8xOFQUfSV9Y5AAAzmUjaVnYPHRln+TNRGPQk3bfOUVETSeu+/TeB80Vh0BaDdlw5HibpunUITC8Kgz1JD61zQFIBf36iMDiSdNflNQEAlXMgqbOIzzaczQLAwjMfHJYPCOmJwlMAwHRoQgOABXXYWO0o25Z2yzYJAECSbtyv/fGjL5bZiHZxu/W1fsfVxWo//PCDojAYiRskACySA2UF9bF1kCqg6AEASutE2WHhno+T4aMw2JT0pXWOCnsyTNJt6xB4s3wr1JE413Dl3jBJnUxAQvGYzu4dp39+ojDoSvrc1fUAAJVTiSJIzmYBYCGZN6FJnDkBAC6kEs9fAADpsLHaVlZnwlBhADCyfF2K9q78qXZV16yzlMxEUrO+1ndWJ3najBZLeuTqogAAb5zo+aYX74rsFwmTEQGgVLzfJBqFQVNZoQP3FRsHwyRtW4fAdDjTcGprmKRd6xCYTl4UNxL3Cl84/fOTP2P2XV0PAFAplSuCzBvStsXnIgAoOy+a0E6xqRoAMIV9SdtVev4CAGQOG6vrkjbFe2oAmLvgN0vfNj5Zummdo4S26mv9rssLnjajMUUZABbbRFKs7BBsZBtlceWNA3vixRQA+KhUDdoUOpiaSGqW4b8TsBXKseNhkq5bh8D0ojDYk/TQOgckOf7zw/R9AMAlVa4J7ay8kXtP3D8BoGxO9PwdpjfncQw/AgC8xkTZs4c3DdQAADuHjdWmpI6yxjQGJQFAwd5ZryncWbaOUUYn9bV+0/VFaz/88IMkKQqDp+JGCABV4P02mLKLwqAr6XPrHACAZy+DtodJemQdZlrcR8x9OkzSPesQeDO2Qjl3r0w/K6suCoNNSV9a54Ck7PPGusvCkygMtiV95up6AICFx3lvLn9G6Ir7KACUwYGy+1dsHeRFPHMDAF7hWNngyz2fGqgBAH44bKyuSNpQdjbFsCQAKMDydemnv78yWX6XOqFLeFBf6/dcX/RsM1osJjsBQJUcKCvOp9i6APkk3m1J962zAEDFnDag7ZXxHpffP/rWOSpsf5ikG9YhMB22Qjm1NUzSrnUITCffyHwkGjF98WSYpNuuLhaFQVvSN66uBwBYWBM93/49Ms7infx+GovCHwDwzekWNG/vX1EYbEj62joHAMAbpRx8CQCwddhY7SjblkbdIAA4FPxm6dvGJ0s3rXOU0H59rV9IPdzZZrSmpH8p4osAALx2omwiB9ObCpC/tNoWhQ8AUKRSN6CdFYXBkaS71jkqaiKpyeehcqAwyKnjYZKuW4fA9KIw6ImXV744GCZp29XF2PgIAJiCt1tkfJRvtumKeysAWDo9u/V+i2c+KKwn7hsAgGwDdenfOwIAbB02VtvKmtJYFAMAM3pnvaZwZ9k6RhlNJK3X1/qjIi7+rBlNopgFACqOaboFisKgI9ZwA4BLJ8oKAxbmRVAUBl1Jn1vnqLBPF+W/pUVHs4ZTE0ltJrqWR15Q/aV1DkjKD21dPj+z8REA8Aonys5t9zi3vbj8+WEz/8UzBADMR+mGh+X3iyPxHg8Aqur03tUTg5wBAI4dNlabyuoGN8T5FABc2PJ16ae/vzJZfpefoZfwpL7W3y7q4i82ozXFdjQAQDblaZuiVPeiMGgrK3ygwBAALu5A+YugRbtH5VN3+9Y5Kmx/mKSFrCOHezRrOPVkmKSFHTrBrfzc7ki8pPKF0z8/bHwEALzgRM+3yCzU86+VvMmgo+xslkYDAHDvWFkBfynvXQxuBoBKWrjBlwAAvx02Vk+HJnXE+RQATC34zdK3jU+WblrnKKHj+lp/vcgv8GfNaJIUhcG2pM+K/KIAgNI4UNaUxsGbY3khaUc8XALA65wWMPSUNaAt7BTCKAyOJN21zlFRE0nNRf7va5Hk22a/ss6xIA6GSdq2DoHpURjnFadNzDQaAgByx3q+RaZ0RfxlkjeBd8SQCwCYxWkBf0/Z2e3IMswsojCIJT2yzgEAmIt9Pb938dwFADBz2FjtKGtMo04EAF7jnfWawp1l6xhlda++1i/0uee8ZrQVZcUPFMYDAE6dKFsVvUehtnv5NpyOslXc3H8BVFllms/OisKgK+lz6xwV9imN9+VAs4ZTE0nrZS4Uq5ooDDYlfWmdA5IKaGKm0RAAKmuifPu3Sl7EX1b5M8ZpYxqFPwDwemebz44WpYCfwUcAsNAmyt4p9JQ9c/VM0wAAcI7DxmpbWV0m74kA4AXL16Wf/v7KZPld6oQuYae+1t8s+ou81IwmSVEYtCV9U/QXBwCUzkTStrJtaZVoEJi3vDGtrawIgodMAItsorxwQRV+AZT/3O9b56iwnWGSFv7gDTdo1nDqyTBJt61DYDrcK7zjtImZRkMAqJTT5+CemMLvHRrTAOAlB3p+fnu0iE3T1MUAwMI5Vn7fEs9cAICSOWysNpU1pbG1GQBywW+Wvm18snTTOkcJnUhar6/1C6/zP7cZTaIQAgDwRruSuov48skn+Yuw018UXwMoqxNJIy148cJFsZXa3ImyzVA02JcAZxROHQyTtG0dAtOLwuBIFET7Yn+YpBuuLsbGRwBYeKfbv0+fgymELIn8Ht1W1pzWFvdqAIvtxbPbURXuWfngl574GQ8AZXW28eyoqkMvAQCL57CxuiJpM//F8wqAyrpxv/bHj75YrlvnKKkH9bV+bx5f6JXNaJIUhUEsuqwBAK93oKwprWcdpAryl2Nnf9GgBsAnEz1/8TPS8xdANPucIwqDbUmfWeeosAd8fikHmjWcmihrwhxZB8F0ojDoSvrcOgckZX9+mi4/19BoCAAL4+yz8GkRf880EZzKz2TbZ37xbAKgjI4lPVXWgDXKf1Xy7DYfEtYTz2MAUAYHyu5fZ5+3Fr5pGgCAvCltQ9m2NAYcA6iU5etStHflT7WrumadpYR262v9zry+2Gub0SQa0gAAUztR1pQWWwepmrxAu6msEGJFzxvVKIoAUITTIrvTFz8jUWh3Yfnmy2+sc1TYzjBJN61DYDpRGPREA74rT4ZJum0dAtPJi5771jnwzKfDJN1zdTEaDQGglM4W8T8rhqxiEX/V5eexpw1qnMUC8MXphrPTe9Sz81uG0vw5zpoAwCsv3r9GqnDDNAAA5zlsrG4o25TGcwyASrjzfy4/ffuvaivWOUpoIqlZX+vP7Vnqjc1oEg1pAIALmUjalrTN4aC9vID1tEFtRVnTWjP/xxRJAHjRaaOZ9LxgYZT/esqkQTfyybtHYnqVlRNlm6H4nFICURhsSvrSOseCOBgmads6BKbH1iyvOG1iptEQALx02mgmZc1m0plCfp4f8CYvDAw7/WvOXwHM6ux5rfT8HvX0zN/nPnUB1L4AwNycNplJz981Ss/vZTRLAwBwQYeN1bakjnimAbDAbv7t0p/+4pdLbES7nE/ra31nA3anMVUzmsShHADgUnaVbUsbWQfB6+Ubek6dNq+d1darMXUF8MOLhQln9V7436cFdRJNZnMXhcG2pM+sc1TYAzb5lUNezHkkijddmEhqUpxWHmzN8orTJua8Kb0nGg0BwJXXPQuP9Lzg8VTv7D/n3BLzcGZgWFPPB4W1z/yWFfHZAFgEZxubzzPSy/cl6ZyzW57fixGFwYakr61zAICnzjaPnefsu8VX/T3uYQAAzMFhY7WpbFNaR7xLB7BArt2p6c7/sfzH2hXVrbOU0H59rb8x7y86dTOaJEVh0JH0VWFpAACL6kBZU1rPOggAAAAAAFZoSjexP0zSuR+8AwAAAAAAAAAAFOWwsbqirCFtU9It2zQAMJvl69LHX1359uq/003rLCV0Imm9vtaf+3CQCzWjSc+m9+2JGxcA4OKOJW0PkzS2DgIAAAAAwDzlG6m/sc5RMWyGBAAAAAAAAAAAC+2wsdpR1pR21zgKAFxK8JulbxufLNGIdjkP6mv9nsUXvnAzmiRFYbAiqSum+AIALudEUqysMY2CMAAAAADAQsvPU4/EgK95+3SYpHvWIQAAAAAAAAAAAIp22FhtK2tKe2gcBQCmduN+7Y8ffbFct85RUjv1tf6m1Re/VDPaqXya77bopAYAXM5E2bbN7jBJR8ZZAAAAAAAoRBQGe+LF37ztD5N0wzoEAAAAAAAAAADAPB02VpvKls48sk0CAK/31gc1/dV/Xv6+dlVvW2cpoRNJ6/W1vtlSmJma0U5FYdBRdtNisi8A4LL2lW1K61kHAQAAAADAlSgMNiR9bZ2jYiaS1hl8AwAAAAAAAAAAquqwsbqibFPapqQbxnEA4CU/++cr3119X+9Z5yipe/W1/pFlACfNaJIUhcGKpI6yGxZNaQCAyzpW1pQWWwcBAAAAAGAW+ZnpSLzgm7cnwyTdtg4BAAAAAAAAAADgg8PGakcsngHgkeA3S982Plm6aZ2jpLbqa/2udQhnzWhn5ZvSOpLuO784AKAqTiTFyhrTzFaIAgAAAABwWVEY7El6aJ2jYg6GSdq2DgEAAAAAAAAAAOCbw8bqhrLFM9T4AzBz437tjx99sVy3zlFSB/W1fts6hFRQM9qpKAyaym5YG6KTGgBwORM9b0ob2UYBAAAAAGA6URhsSvrSOkfFTCStc34AAAAAAAAAAADwaoeN1XVlNf6PrLMAqJa3Pqjpr/7z8ve1q3rbOksJTSSt19f6I+sgUsHNaGdFYbCurCltQ9LduXxRAMCi2ZUUD5O0Zx0EAAAAAIBXyYd0HUm6YRylap4Mk3TbOgQAAAAAAAAAAEAZHDZWm5I6yhrTeK8FoFDL16WP/+nKd1ff13vWWUrq0/paf886xKm5NaOdFYXBiqR2/mtdrPoEAFzMgbKmtNg6CAAAAAAAL4rCoCfOPOftYJikbesQAAAAAAAAAAAAZXPYWF3R86a0W7ZpACyqj75Y/u7G/RqNaJezW1/rd6xDnGXSjHaefFpwU1mD2oqyJjWJog0AwKudSNpW1pj21DoMAAAAAABRGHQlfW6do2ImktaHSTqyDgIAAAAAAAAAAFBmh43VjrLGNGr4ATjz3s+Xvv/w10tvW+coqWNJ7fpa36taeW+a0aYVhUHbOgMAwDtHNKMBAAAAAKxFYbAuqW+do4KeDJN02zoEAAAAAAAAAADAojhsrLaVbUp7aBwFQMldu1PTx79bto5RVhNljWhH1kFeVLpmNAAAAAAAAADwURQGR5LuWueomINhkratQwAAAAAAAAAAACyiw8ZqU1JX0oakG6ZhAJTO8nXpp7+/Mll+l58fl/Skvtb3cjArzWgAAAAAAAAAMKMoDLYlfWado2ImktaHSTqyDgIAAAAAAAAAALDIDhurK8o2pW2KpjQAU/r4d8vfXrtTu2mdo6T262v9DesQr0IzGgAAAAAAAADMIAqDtqRvrHNU0JNhkno5BQ4AAAAAAAAAAGBRHTZWO8q2pd2yTQLAZ8Fvlr5tfLJEI9rlHEtq19f6T62DvArNaAAAAAAAAABwSVEYrEg6Ei/b5u1gmKRt6xAAAAAAAAAAAABVddhYbStrSrtvmwSAb977+dL3H/566W3rHCU1UdaIdmQd5HWWrAMAAAAAAAAAQIl1RSPavE0kdaxDAAAAAAAAAAAAVFlrPOi1xoO2pHuSdo3jAPDEtTs1/eWTJbZmXd6m741oEpvRAAAAAAAAAOBSojDYkPS1dY4KejJM0m3rEAAAAAAAAAAAAHjusLHaVDZQcFPSDdMwAEwsX5d++vsrk+V3+RlwSbv1tX7HOsQ0aEYDAAAAAAAAgAuKwmBF0ki8SJu3g2GStq1DAAAAAAAAAAAA4HyHjdUVSRuSupJu2aYBME8/++cr3119X+9Z5yipY0nt+lr/qXWQadCMBgAAAAAAAAAXFIXBnqSH1jkqZiJpfZikI+sgAAAAAAAAAAAAeLPDxmpH2ba0+7ZJABTtoy+Wv7txv0Yj2uVMJK3X1/oj6yDTohkNAAAAAAAAAC4gCoOOpK+sc1TQk2GSbluHAAAAAAAAAAAAwMUcNlbbyprSHtkmAVCEm3+79Ke/+OXSNescJfZpfa2/Zx3iImhGAwAAAAAAAIApRWHQlHQk6YZxlKo5GCZp2zoEAAAAAAAAAAAALu+wsdqU1JW0Id63AQvhxv3aHz/6YrlunaPEtupr/a51iIuiGQ0AAAAAAAAAphSFQU/SfescFXR7mKQj6xAAAAAAAAAAAACY3WFjdUXSprJtabds0wC4rGt3arrz2+Xva1f1tnWWktqvr/U3rENcBs1oAAAAAAAAADCFKAw2JX1pnaOCtoZJ2rUOAQAAAAAAAAAAAPcOG6sdZdvSaEoDSmT5uvTT31+ZLL/LlsNLOpbUrq/1n1oHuQya0QAAAAAAAADgDaIwWJfUt85RQcfDJF23DgEAAAAAAAAAAIBiHTZW28qa0u7bJgEwjZ/985Xvrr6v96xzlNREWSPakXWQy1qyDgAAAAAAAAAAJRBbB6iojnUAAAAAAAAAAAAAFK81HvRa40Fb0m1Ju8ZxALzGR18s04g2m06ZG9EkmtEAAAAAAAAA4LWiMOhKumudo4K2hkla6gN4AAAAAAAAAAAAXExrPBi1xoOOsqa0LWUbhAB44oO/X5rcuF+jEe3yntTX+nvWIWZV++GHH6wzAAAAAAAAAICXojBoS/rGOkcFHQ+TdN06BAAAAAAAAAAAAGwdNlZXJG1I6kq6ZZsGqLb3fr70/Ye/XnrbOkeJ7dbX+h3rEC7QjAYAAAAAAAAA54jCYEXSkXipZeEeW9EAAAAAAAAAAABw1mFjtSOpI+m+bRKgeq7dqenj3y1bxyizY0nt+lr/qXUQF5asAwAAAAAAAACAp7qiEc3CFo1oAAAAAAAAAAAAeFFrPIhb40Fb0gNJu8ZxgMq4dqemO79d/t46R4lNtECNaBKb0QAAAAAAAADgJVEYtCV9Y52jgo6HSbpuHQIAAAAAAAAAAAD+O2ysNiVtKtuWdsM0DLCglq9LP/39lcnyu/wZu6TTRrSFGshKMxoAAAAAAAAAnBGFwYqkkXhhZeEeW9EAAAAAAAAAAABwEYeN1RU9b0q7ZZsGWBzL16WP/+nKd1ff13vWWUrscX2tH1uHcI1mNAAAAAAAAAA4IwqDPUkPrXNU0NYwSbvWIQAAAAAAAAAAAFBeh43VjrLGtLvGUYDS+/h3y99eu1O7aZ2jxLbqa/2udYgi0IwGAAAAAAAAALkoDDYkfW2do4KOh0m6bh0CAAAAAAAAAAAAi+GwsdqW1JV03zYJUE4ffbH83Y37NTaiXd5ufa3fsQ5RFJrRAAAAAAAAAEBSFAZNSUeSbhhHqaJ7wyQ9sg4BAAAAAAAAAACAxXLYWG0qa0p7ZJsEKI/gN0vfNj5ZYiPa5R3X1/oLPYx1yToAAAAAAAAAAHgiFo1oFrZoRAMAAAAAAAAAAEARWuPBqDUedCQ1JG1JmtgmAvz23s+XvqcRbSbHktrWIYrGZjQAAAAAAAAAlReFwaakL61zVNDxMEkXeiIcAAAAAAAAAAAA/HHYWF2RtKFsW9ot2zSAX977+dL3H/566W3rHCU2kdSur/UXfhgrzWgAAAAAAAAAKi0Kg3VJPbEVzcI9tqIBAAAAAAAAAADAwmFjdUPSpqT71lkAa9fu1PTx75atY5TdvSo0oknSknUAAAAAAAAAADAWi0Y0C1s0ogEAAAAAAAAAAMBKazzYa40HbUkPJO0axwHMXLtT053fLn9vnaPkHlelEU1iMxoAAAAAAACACovCoCvpc+scFXQ8TNJ16xAAAAAAAAAAAADAqcPGalPZprSOGGaJijhtRKtd1dvWWUrsSX2tv20dYp5oRgMAAAAAAABQSVEYrEvqW+eoqHtsRQMAAAAAAAAAAICPDhurK3relHbLNg1QnOXr0k9/f2Wy/C7NlzPYra/1O9Yh5o1mNAAAAAAAAACVE4XBiqSepLvGUapoa5ikXesQAAAAAAAAAAAAwJscNlY7yhrTeK+IhbJ8Xfr4n658d/V9vWedpcQq2YgmSUvWAQAAAAAAAADAQFe8MLJwImnbOgQAAAAAAAAAAAAwjdZ4ELfGg3VJDyQdWOcBXKARzYljZY2qlcRmNAAAAAAAAACVEoVBW9I31jkq6sEwSXvWIQAAAAAAAAAAAIDLOGysNpUNvnxkmwS4vJ/9M41oMzqW1K6v9Z9aB7FCMxoAAAAAAACAyojCYEXSkaRb1lkqaGeYpJWdDAcAAAAAAAAAAIDFcdhYXVG2FWlT0g3jOMDUPvpi+bsb92s0ol3eRNJ6fa0/sg5iack6AAAAAAAAAADMUSwa0SycKJsQCQAAAAAAAAAAAJReazx42hoPupKakh4rex8GeI1GtJlNlG1EG1kHscZmNAAAAAAAAACVEIXBhqSvrXNU1INhkvasQwAAAAAAAAAAAABFOWysbijblHbfOgvwIhrRnLhXX+sfWYfwAc1oAAAAAAAAABZeFAYrkkaSbhhHqaKdYZJuWocAAAAAAAAAAAAA5uGwsdqW1JH0yDYJkKERzYnH9bV+bB3CF0vWAQAAAAAAAABgDvZEI5qFE0ld6xAAAAAAAAAAAADAvLTGg15rPOhIui1pR9LENhGq7ObfLv2JRrSZ0Yj2AjajAQAAAAAAAFhoURhsSvrSOkdFPRgmac86BAAAAAAAAAAAAGDlsLG6ImlT2ba0W7ZpUCXv/Xzp+w9/vfS2dY6Se1Jf629bh/ANzWgAAAAAAAAAFlYUBk1JR2IrmoWdYZJuWocAAAAAAAAAAAAAfHHYWO0oa0y7axwFC45GNCd262v9jnUIH9GMBgAAAAAAAGBhRWHQk3TfOkcFnUhaHybpU+sgcOcf/vG365JWrHMAUxj96pe/GFmHkKR/+MffNiU1jWMAkvT0V7/8xZF1iEX2D//42xVJ69Y5gGn86pe/6FlneJP8s+eGdQ5HRr/65S9i6xCv8w//+Nu2dQZUHp9VSoCfFXiDo1/98hecBQIAcAGHjdW2pK54l4kC0IjmBI1or0EzGv5/9u7mt63z3Pv9j6KV2rJbm6kbpzvekbbhAscFqMg4Aw2tjneBqB0URSdhRiU5iTzYZ7YRuniK4gAbiPQMSE6eXXp+gC3/BaUGHRAoKslq4Q5Sg0LdIgmMyEz9IltW1hmsxWjJFvW6yPteF78fQNCyzQJXWK7F9XL97gsAAAAAAAAw6dqVyxVJH7uuY0j96N79B03XReDwombfCYVN9PFmeh6AIo2u+9DIGgVTViSNu64FeMWqpLbCz+eKpCZNo/uLBc0mtPv78oJYwRrpc6tcKlZcF7GfaJ9rys7+9ZNyqbjouoheqrV6QdJvXdcBxKwrPFfp/jRFyKXvYgtpzIj7Aji+jsJ9tilp0ZeFYgAA8F0rl59QGEr7wG0lsIIgWiIIoh2AMBoAAAAAAAAAc65duTwladl1HUNq4d79B3Oui0BvUfBsRmFj2ZTsNPkCknSzXCrOuy5Ckqq1+qKk913XARzSqqSGaBiV9M3Uj+7PhAiVwo7Vcqno/QS/aq0+L+kj13Uk5Ha5VCy4LqKXKHyyIum841KAw1hVFHLxOeCZBtG+P6Od+wIEztAvtyXN+7BgDAAAadDK5S9Imot+uE7DsRBES8SqpJmxyWUWRdkHYTQAAAAAAAAA5ly7cnlFBGxcWJc0de/+A27MeySaLDEb/cyIB5iwa6lcKs64LkJiwghS77akyjCF0qKgdvd7kmZsWNWRNOX7vl2t1Wcl/Y/rOhKyKmnG52lO1Vqda2ek1brCIP28z/uYL6L7AjPaOd8haI9BW1B4jcH+CgDAIUShtFmF09I4d8OhEURLBEG0QyKMBgAAAAAAAMCUa1cuW1pFP21+dO/+g6brIrArgDYnmksxHDqSJnxobGPCCAxZKJeKZqedRgG0gsLvS5p6MAx+4vskoegcti0736HXfZ4EU63VK5I+dl0HcEIdhYG0iutCfPPKwjRMbIYPvA9pAwDgo1Yu333WwwJK2BdBtEQQRDsCwmgAAAAAAAAAzLh25fKMpN+5rmNILdy7/8Bsw3paRI31c5I+cF0LMGA/KpeKTddFSFK1Vm+KxgDYYa5hNJpcSFgbw+Z2uVQsuC7iIMa+Q29bZ1LaAAAgAElEQVSWS8V510X0Uq3VZ8S1M2xZlTTr+/THQYgmTBZEAA1+Mnd9AQDAoLRy+RmF53k8/8FrCKIlgiDaERFGAwAAAAAAAGDCtSuXLyicRMNki8FblzR17/4Dbs47EjWTVmSneRc4Cm+mNzFhBEaZaBit1upzCkNonCti2KRiH4720U9c15GQpXKpOOO6iF6iaUlcO8OijsLjnbcTCfsl2q/nFDYns2/Dd6kIyQMA4KtWLj+hnXM/K5PFcQIE0RLRkTQ1Nrncdl1ImhBGAwAAAAAAAGDCtSuXG2I1QFd+dO/+g6brIoZRtVafkNQQITQMr9VyqTjlugjpm8mEy67rAPrEm9DnUUWT0CqiMRvD67rvwQxj36EdSRM+h/+qtXpDXDvDrnVJUz7vg0mK7gnQiIw08ma6OQAAadXK5S9IWhTPh4YaQbREdBRORPP6/pmPTrkuAAAAAAAAAABO6tqVy7Oimc6VBYJogxdb9ZwJTBhmHYVNl85F++Si6zqAPvqoWqs3fA+0xEXhlnnRkIPhdtP3/dbgd2jB5xBMtVbn2hnWjSsMoacyRH9Y0bGzIukjx6UAxzUnqem6CAAAUo77XkOOIFoiCKKdwIjrAgAAAAAAAADgJK5duXxB4WQoDN66wuYnDFDUXL8igmjAnEcN9hUxdQn2paapu1qrVxROWaIhB8NsqVwqzrsu4hDmZec7dKFcKnobrItNVQas+ygKa5kUnee0RRAN6fa+5f0UAIB+auXyF1q5/IpYaGSoEURLBEG0E2IyGgAAAAAAAIC0a0g677qIITV37/4Db1f9t6haqxck/dZ1HYAH7pRLxYbrIqRvJozQCIphMOO6gIPEJiwRQsOw60iadV3EQYxN6VqV/wt1NMS1M4bHrIyFL6u1+ozC/yYrAV5gRramowIA0HetXH5K4fcn54RDjCBaIgiiJYDJaAAAAAAAAABS69qVywVJ77uuY0jduXf/AQ0jA1St1RsiiAZI4VTGgusipG+CLw3XdQADMu7z9ILY5FCCaIA0Wy4VvV40wuCUroLP73k0SYnjI4bJlOsCklKt1S9Ua/V5Sb8TTcewxcx+CgDAILRy+YKkpjgnHGoE0RJBEC0hhNEAAAAAAAAApNK1K5cnJM27rmNIdeRJEGRYREE0K1MjgJPyqcF+UUwYwXDxsmE0CqI1RTMOIEkL5VKx6bqIQ2jIznfozXKp6G0DU3SM/Nh1HcCAeXnOclTRNLQVMYkZAABgqLVy+XmFixVauY7GMRBESwRBtASdcl0AAAAAAAAAABxTQzx0caVw7/4DX4Ig5hFEA3a55Uuzd7VWnxMTRgDnYkE0zgsBabVcKs65LuIgxqZ0LZVLRW8XSYkmWjLRGkih6FhJkBQAAGCItXL57jWdlWtoHBNBtMTMEURLDmE0AAAAAAAAAKlz7cplAgDu3Ll3/wHNjANCEA3YZalcKlZcFyF9E36puK4DGHYE0YBdOpJmXRdxEGNTutLwns+LqZFAqkQh0oak9x2XAgAAAIdaufyUwiAa13RDjiBaYj4cm1xuuC7CkhHXBQAAAAAAAADAUVy7cnlK0ieu6xhSHUkF10UMi2jqEkE0IOTb8achwi8YTt6sGhub9sO+CITmyqVi23UR+zE4pWu2XCp6OzG6WqvPiusJDK+m6wKOo1qrTyisnSAaAADAEGvl8gWF54UE0YYcQbTEEETrAyajAQAAAAAAAEibhusChljh3v0H3jZbWlKt1WdE6BKIK/jSYF+t1eclvee6DsAFz0IXDdGQA3TdKZeKDddFHIKlKV0L5VKx6bqIXqJAS8NxGYBLbdcFHBUTXzGEvFnoAgAAn7Ry+YZYWAQiiJYggmh9QhgNAAAAAAAAQGpcu3K5IgIArty5d/+BpSkG3oomRjRc1wF4ZKFcKnpx/ImCoh+5rgNwZMl1AV3R9FAmhgChdfk1PXRPxqZ0rZZLxTnXRRygIQItGG5N1wUcBUE0DCnCaAAAxLRy+QsKzwl5DgqCaMkhiNZHhNEAAAAAAAAApMK1K5dnJH3suo4h1VEKGlwNacjOxAjgpFYlVVwXIREUBST5EgqdEtNDgbhZz6YWvsbYlC7vr42iwO4N13UADq36MlX5MAiiYUitp2k/BQCg31q5POeE+MbbvxzpfO8XI3wWTo4gWp+NuC4AAAAAAAAAAA5y7cplAgBuFe7df+B1g6sV1Vq9ICa9AF0dSQWPGuwbIiiK4eY8jEYoFHjNrXKpmIapIg3Zaair+PyeE9gFJKXoXIEgGoaY82sLAAB80crlC5KWxTkhJL37q+yXBNESQRBtAJiMBgAAAAAAACANKiIA4Mqde/cf0CAyANHEiHnXdQAe8abZm6AooNueTC6oSHrPdRGAJ5bKpWLFdREHqdbqFdmZ0nWnXCp6e75OYBf4RiruYRBEw5Dz9vsUAIBBaeXyFxR+J37guhb44d1fZb88fyPzpus6DCCINiCE0QAAAAAAAAB47dqVyzOSPnJdx5DqSCq4LmKILIomNKDLm2ZvgqKAOgpDYE5Va/UZcU4IdHUkzbou4iBR0OJj13UkJA3XRhUR2AV8CdDvKwqPcg8Aw2opDfspAAD91MrlJxSeD3INB0kE0RJEEG2ARlwXAAAAAAAAAAC9XLtyuducAzcK9+4/eOS6iGEQTYzgoSMQ8q3ZuyGaRDHcKq6bRWMN2wBChXKp6PV5usH9dtbn97xaq8+KwC4gpWcRiaakcddFAI5UXBcAAIBLrVx+RtKKeCaECEG0xBBEGzDCaAAAAAAAAAB81hABAFfu3Lv/wFLzqLeiSS9WJkYASfCm2TsKit5wXQfg0JInUwob4pwQ6Fool4ppOE+fl52gxUK5VGy6LqKXKPjXcF0H4IGlcqm44rqIg1Rr9XnReIzhteTzdyoAAP3WyuUrkn4n7nNBUvac9IP/k31IEC0RBNEcOOW6AAAAAAAAAADYy7Url2clve+6jiHl21Qis2gcBV5zy5fGtGqtPiWCohhuXpwPVGv1OXFOCHStKgXTRKIpXR+4riMhq+VScc51EQdYFI2MgOTBectBmGII+L+fAgDQD61cvvssiHtckBQF0f771Jejl3TRdS0GEERzhMloAAAAAAAAALxz7crlCRHQcalw7/4DL6YSDYGG7EyMAE5qtVwqVlwXIX0TFE3D1BmgnwrlUrHtsoAoFFpxWQPgkY7C/dLr8/RqrT4hO9dyHUmzrovYTxTYZYorIN12fd5yEGPHR+A4bvm+nwIA0A+tXH5KUlME0RCJBdGYiHZyBNEcIowGAAAAAAAAwEcNsbK7K3fu3X9AAGMAmPQC7OJbs3dFBEUx3G6XS0Ufzgca4pwQ6KqUS8UV10UcgqUpXXM+N81Hgd1PXNcBeKAjyfcJhhLnNRhuq5LmXRcBAMCgtXL5gsIg2ntuK4EvCKIlpiOCaM6dcl0AAAAAAAAAAMRdu3KZld3d6UgquC5iGAzppJeOpJXo55GkdvQDSNIjz5q9F8VkNOyYkTQR/R6GkOK6PGjortbq8xq+Rp1Vhd+N3e/KNASPMCDlUrHpuoaDVGv1iuzst3fKpWLDdRG9RFNcG67rGID4cbEtrh8Oa0bhQg9W9seDVFIwNbIi+/e6VhU2WrfFOUxSGrJz/eH9dFcAAJLWyuXnJX3kug744/TVjK7Ws5uZUYJoJ9SRNDM2ucx1h2OE0QAAAAAAAAB449qVy8MY0PHJ3L37D2gMGYyG7K+Ivq6wEa0pqelZ0AjYVxoa/jFQze5GtVafVbiiv5Wm0L3Mum4UrdbqMxqOZp1VhcHXJscdpF20337suo6ErMv/RToqshs0uq3wu3fR9fdRijUlVaL90tK0wr0slUtFr6ctRYvRWDk+vqo77Yr9NWFRgNHKNcfNlEx3BQAgEa1c/oKYhoZXREG0Z5lRnXFdS8oRRPNIJggC1zUAAAAAAAAAgCTp2pXLK+LhjCtL9+4/mHFdxDCIJr1YbbBfV9js2KDRCIBV0TSaFdlpDo27VS4VKy4LiN7ftuw2zi8pmr5IUBtWGDwu/sjngGgUMPqd6zoS1lEYaJkn0JKsKAi17LqOPulImvL5+9Tg8bFrSeFEuqbrQiwydpxfKpeKM66LAABgUFq5/IzsLwiBIyKIlhiCaJ5hMhoAAAAAAAAAL1y7crkigmiudOT/yv8mGJ70clth4ygPgACYVy4VH1Vr9YbsTbhYch1EizRkr2FnXeF/V8PnhnngBBqyE7S45XO4Igq2LLquI2ELCkMthND6oFwqrlRr9TuS3nddSx8UUvC9am2i7rrC973puhCrouN8w3UdCelImnVdBAAAg9LK5eckfeK6DvjlzR+PPHvnP0bOSATRToggmocIowEAAAAAAABw7tqVy1Oy19CdJpV79x+0XRdhndHG0dsKG0fbrgsBgAFruy4gYV4E06u1+pxsNct3JM2VS8WG60KAfqnW6gXZ2W9XPQnl7qchO4HdjqRZQi0DsSI7+2nX7XKp6PX1dbVWn5X0ges6EkRwdDAashNgnOXzAgAYBq1cvhsmt3bOjROKBdFwMgTRPEUYDQAAAAAAAIAPZiUtuS7imCaU7iaRpXv3H8y7LmJINETjKABYYW2Ff+eTRaq1+pSkissaErYkGnBhXLVWn1A49ccC76e3GAvsrkqa4RiJY1qVNOe6iP1Ex8eG4zKS9CHh+v4zFvBe4J4RAGAYtHL5KYXnfe85LgWeufjzkc3vlwiiJYAgmscIowEAAAAAAABw7t79BxXXNQD9ZKxxtKOwcZQHPwCGUtRcbOWYLvkzWaQhO6Ht2+VSseC6CGAAFmVnv51zHcrdj7HA7roIog3ajOsCElZIwefH0vGRINoARMd5KwHv1XKp6HVgFACAJLRy+YLC728r531IyLu/yn55/kbmTdd1GLAuaZYgmr8IowEAAAAAAAAA0EfGGkclgmgAYKmxcl0e/PdUa/V52VlBepUgGoZBtVavyM5+eycFQYuGbDQ4dics+x4kMqNaq1+QdMN1HQm66fv1qLHjI0G0wWnI0HHedREAAPRbK5dvSPrAdR3wD0G0xKwqnIjG/QOPEUYDAAAAAAAAAKBPosa/hmw0FEnSLd8b/wCgn6q1+oykj1zXkSDngQBj72lHUsF1EUC/Rfvtx67rSMi6PN9vjQV2K1xPDFzDdQEJWiqXil5PjjJ2fExDUNcEY8d5ryeNAgBwUq1cfkLhFFwr391ISPacdGUh+/D01cxF17UYQBAtJQijAQAAAAAAAADQPxXZeSi5VC4VK66LAABXYgFjK5xPFone00WXNSSMkAXMM3gsdB7K3Y+xwO4d34NE1lRr9VlJ77uuIyHeT1sydnwkYD8gBo/zDddFAADQL61cfla2Fh9EQrLnpB/896kvRy+JINrJEURLkRHXBQAAAAAAAAAAYFHU+GeloYhGNAAIA8bjrotIiC+TRRqy08BDyALDoiE7x0Kvp/4aC+xyPTFg1Vp9QnaCUZLnwdFIQ3aOj4UUvN+pZ+w47/2kUQAATqKVy89L+h/ZuY+FhJy+mtEPfnvq4eglvem6FgMIoqUMk9EAAAAAAAAAAEiYsRXRpbARre26CABwxdjEAi8mi1Rr9TnZmtZScF0E0G/VWr0gO/ttGqb+NmSn0TENQSJrGrLz+Vkol4pN10Xsx9jx8U65VLQSkPLdouzspxznAQAmtXL5bnj8huta4J/TVzO6Ws8+y4wyES0BtyXNEURLF8JoAAAAAAAAAAAkz1JD0W0a0QAMM6MBY6cP9au1+pTCSXNW0HwL86IpS1am/3kfIDUW2PU+SGRNtVavyE6z7Gq5VJxzXcR+OD7iOKLjvJX91OtJowAAHFcrl5+RrWc9SNDZqYz+7b+yzzKjOuO6FgNuj00uF1wXgaMjjAYAAAAAAAAAQIKMNRStSvK68Q8YlCiQNBX7qylJFxyVg8GakjTuuoiELHgSMG7ITiMPIQsMC0sNeF5P/Y2CLRXHZSRlVXb+W1IhCnx/7LqOhKQlGGXp+DhHwL7/ov30E9d1JCQNk0YBADiyVi5fkZ3zaiTszR+PPHvnP0bOSATREkAQLcWchtFaufy8dh7aNWP/tCKpe2G7Mr2xxkUuAAAAAAAAAMB7xhqKJA+m5wCDFgudzUS/JyS957AkICleBAKqtfq87OxT3k9rAZIQTVmyst+mYeqvpWAL1xMDFJ3H+v75PoqK79OWjB0f18ulYsN1EdYZm7qclsAoAACH1srlu+fUVhYcRMIu/nxk8/ulEUJoySCIlnKuJ6NNaedg3fOg3crlu5vrktrR9iOFobWuZndjemOtKQAAAAAAAAAABshYQ5Ek3fS98Q9IShQkLSgMoFlpJgVe5TwQUK3VZyR95LKGBHUkzbouAui3aL+1shr8ujyf+mss2ML1xODNy8401zvlUnHedRH7MXZ8lHZ68tBflhZm8HrSKAAAR9XK5acU5hGsLA6ChL37q+yX529k3nRdhxEfjk0uN1wXgZNxHUZr6mjJ4XHtvmnyfmz7m4v7WHhNClf46z5UaWvnwjm+/Wh6Y40bYAAAAAAAAACAk6jITkPRku+Nf8BJVWv1CYUN6bOy07QL9OI8EGBwWssczbewzuBiC7OuQ7n7MRZs4XpiwKq1+qykD1zXkZB1eT5tyeB5jSRNVWv1Cz4fJ9PO2H66kIJJowAAHForl5+T9InrOuCn7Dnp3V9lO+f+b4JoCSGIZoTrMNogxB/8H2b6miQtxbZXtBNmi2+3pzfW2gnUBwAAAAAAAABIuaihiEkvQApEjd4VHW3BRCDNfAkENGRnZek75VKx4boIYAAashPYvuU6lLsfY8E/ricGzNjnR/JgmushNGTnvKbrvKR2NKFxkdB9sqLFUBqOy0jKqsJragAAUq+Vy3fPpd8/4KUYUtlz0g/++9SXo5dEEC0ZBNEMcR1Ga8rPVZ1u9NjeJRZg6ygMqnU1e2yvTG+s+X6zBAAAAAAAAABwBMYaiqR0NP4BRxbbVwmhYZh4EQio1upzstPU4/20FiAJxvbbpXKpWHFdxAEashP843pi8BZlJxh1q1wqNl0XsR9jx8dXnVc4EeSTaq0uhaEj9udkTMjGftoRx3kAgBGtXH5K4bm0lWsxJOz01Ywm/t/s56MXdcl1LQZ0JM2MTS57u1AQjs51GM2K8+odYNsVtosF2NYltaPtR9oJs8W3Nb2x1kyuTAAAAAAAAABAHzRko6FIkhbKpeKi6yKApEWr+/u4QCLQb84bRaMgaMVlDQlz/p4C/Vat1adkZ7/tyPMAabVWL8hOsOU21xODFQWjrCy2sOp7cNTY8fEw3nNdALxT8XnSKAAAh9XK5ecUhvCBPZ2+mtHVevZZZpQgWgIIohnlNIw2vbHWjIWzhs24diep97yx+Mr7E19tpq2dMFt8+9H0xho7KgAAAAAAAAAMQBRwMdP4p+FqqsMQiEIwi6KJEsPJl4Ax01qA9GnIzn5bKJeKbddF9BKdq8y7riMh65LmXBcxTKJglJUGWi+muR5CQ3aOj8BR3SmXila+swAAQ6qVy19QeE5nZUEQ9MGbPx559s5/jJyRdMZ1LQYQRDOMyWjpEn9Q2rO54ZUA21Jse0U7Ybb4dnt6Y62dQH0AAAAAAAAAMDSixj9Lk5aY9AJTqrX6jGyFYICj8CJgHIW2rYRBvZ/WAiShWqvPy85+60sodz+WzlVmuZ4YnGqt3m2itWLO5+CoZO74CByV95NGAQA4SCuXn1J4DTZ+0GsxvN7+5Ujne78YsXKd7tqqwiAa9wqM8iGMtiQ7q8b66EaP7V1iAbaOwqCaFIbV4inUZmx7ZXpjjQMDAAAAAAAAgKEUNf753th6FDfLpSKrEsKMaq1ekPRb13UADjkPGEeBUCuh7bRMawFOJNpvP3JdR0K8COXux1hg9xbXEwNXkZ3Pz+1yqdhwXcR+jB0fgeMgcAwASLVWLj8nO1OF0Sfv/ir75fkbmTdd12EEQbQh4EMYDX45r92htfgY0l0Pi2IBtnVJ7Wi7HdveFWab3lhrJlUkAAAAAAAAADg2LzurZy6VS8V510UASSGIBrgPGBuc1lLwfVoLcFIGF1twHsrdj7Epy0yOHLBqrT4rO8GodUlzrovYj8HjI3BUt8qlYtN1EQAAHEcrl++eyzE4Bz1lz0k/+O9TX45eEkG0ZNyRVCCIZp/zMNr/9f+dWhkZ0+T2o+CrYFunHv8heClJWw/1nad/DnKStPWZ9OKzwG2h2M+4dpouDjN9TQrTrt0DzEpsu61YmG16Y42VswAAAAAAAAB4JWr8+8B1HQlh0gtMiRq7CaJhmPkSMG7ITmj7drlUpAEdw6ChcPFaC5yHcvdjLNjC9cSAGQx8p2HaUkN2jo/AURE4BgCkViuXn1F47cW5HHo6fTWjK/PZTvbbBNEScntscrnguggMhvMw2uj39EhSLns2k5Okb72b2ff1Ww/1efA82CS4lnrvxbYPG2Bbim03Y9u7wmzTG2vtE9YGAAAAAAAAAHuq1uoTovEP8FK0fzYdlwG45EUgIJpO+L7rOhLi/bQWIAnVWn1OdvZbX0K5+7E0ZXmOyZEDZ6mZ9pbPwVHJ3PEROCovri8AADiOVi5fkZ1p1OiTs1MZ/dt/ZZ9lRs1cY7lGEG3IOA+jHdXoRV2SwsDaQcG1F38P1iXpZUcjz/4SfC1Jm3/VW88fBGckafPTQNuP+1svEnWjx/YusQBbR2FQTQrDavEbWM3Y9sr0xhrNFgAAAAAAAAAOoyE7jX8L5VKx6boIIEEN2dk/geMouA4YR6FQ30MgR0FoG+ZFU0UrrutIiPdN88amLN8pl4oN10UMkygY1bNfJmWWfJ+2ZOz4CBxHgcAxACBtWrl8dxK1lfNm9MmbPx559s5/jJyRdMZ1LUZ8ODa53HBdBAbLhzBa31a4eeOdzHj4Wxr74dGDa4//GIxv/zP89ycrTFtLofPafTIRX6loV9o9FmBbl9SOttux7V1htumNtWZSRQIAAAAAAABIh2qtXpGdB5ir5VKRSS8ww9j+CRzHQrlUXHRdhJjWAqRRQ3b2W+eh3P0Ym7K8LqnguohhYiwY1ZHnn59qrX5Bto6PwFHd9uT6AgCAQ2vl8jOydW8KffLur7Jfnr+RedN1HYYQRBtSPoTRvLgRuFdw7bs/ff11wZaebX0RfCFJz/+mkRf/ILhmzHj0Ix1u+pokrWrnc7wS224rFmab3ljjYRUAAAAAAACQUtVafUavLHKVYt43/gFHETXmWtk/geNYlQfN6VEo9D3XdSTE+2ktQBKqtfq87Oy3voRy99OQnaZIr4N/RjVk6/PTdl3EASqyc3wEjmpdEgsYAQBSpZXLz0v6yHUd8Fv2nHRlIfvw9NXMRde1GNGRNDs2udx0XQjccB5Giz58mad3r89EfzUl6YKkiejngjy6uM+M6kw8uCb1Dq5tP9HG9qPgK0l68ied+vpJ8DJ4qdNf/T64JEnbj6XNTwmuGRD/fB4mwNbR7omAzdj2rjDb9MZa++TlAQAAAAAAADiJ2IroVlSY9AIrov3T98ZzoN+cBwIIbQPpE+23Vhr1vAjl7sfYFNeFcqnYdF3EMCE4OljVWn1Wdo6PwHHMur6+AADgsFq5/ITC+8NWzpfRJ6evZjTxm+zD0bdEEC0ZHUkzY5PLPG8cYs7DaF2xRGSz12v2CaxNaGealTeyZ5XLns3kpN3BtYs/e/21BwXXtj6TXnxGcM2I89p9k/kwAbZ1xaasaZ8w2/TGGjcDAAAAAAAAgGQ15OE96GO6Uy4V510XASSoIjv7J3AcN10HjA2GttMwrQU4EWNh7o48COXux9gU19Vyqci0nAEiODpY0fGRa2YMM+fXFwAAHFYrl5+VrQnC6JOzUxn9239ln2VGCaIlZFXhRLS260LgljdhtMM4KLD29O71CwqDapI0E/2OB9e8fRh5lODa1kN9HjwPNoNtnXr8h+Bl9HffefrnICcRXDNqXLs/v+/HtnfdtI4F2FYVm7KmHmG26Y21ZmJVAgAAAAAAAMZUa/WCdt+PSzMmvcAUJhYAWvIkYNyQx89hj8j7aS1AQhqy06zn9dRfi8E/10UME2OfH8nz4GhkTnbOa4Cj8uX6AgCAfbVy+e4CAh+4rgX+e/PHI8/e+Y+RM5LOuK7FiFWFE9F8v7bDAKQqjHaQ6EPdjP7Y3Os1scBar+Ca9zdcRy/qUjes9q13M/u+thtce9nRyLO/BF9L0uZf9dbzB8EZSdr8NND2435XDEfiI2cPM31NkpZi2yvqEWab3ljz9mY+AAAAAAAAkJRqrT4hWyuiz6ag8Q84FIOTmICj6kiadV2EsdC299NagCRUa/U52dlv0zD1d152gi1eB/+MaigFfUyHlJZpSwXXBQCOeHF9AQDAQVq5/JTC8+T3DngpoHd/lf3y/I3Mm67rMOS2pDmCaOgyFUY7jFcCa3uuHvT07vUJhZPUUhtY6+oG1954Rxr74f7BtRd/D9YlqVdw7ckK09aGwI0e27vEAmwdxaasaXcIdFeYbXpjrX3y8gAAAAAAAICBWlSK7gcfYKFcKjZdFwEkyNL+CRyH88kiBkPbzt9ToN+qtfqU7IQuvZ/SFU1xtbJSfxqCf6YQHB28aq0+IzvhUeCoWMAIAOC9Vi4/J+kT13XAf9lz0g/++9SXo5dEEC05t8cmlwuui4Bfhi6Mdhhjk8tt7UyBOiiw1v2JB9d6hnh89sY7mfHw9/7BtWBLz7a+CL6QpOd/08iLf4TBtcd/DMa3/xm+huDa0DivowfY1hWbsqYeYbbpjbWmAAAAAAAAAEeqtXpFdlbWXC2XinOuiwCSEjXmpvJZDJCQhXKpuOczzAGzFApNy7QW4KQasrPfet00b2yKq/fBP2uiwHfFcRlJSdPnZ8Z1AYAjLGAEAPBaK5fvXl9ZWawBfXT6akZX5rOd7LcJoiXo5tjksvcLjDwVssIAACAASURBVGDwCKMd0yuBtT09vXt9SmFIbUK7A2sXlOImhsyozsSDa1IYXPvuT19/7V7BteClTn/1++CSJG0/ljY/Jbg2ZMa1eyWp+Mnhx92NWHhNklYVm7KmvcNsj6Y31nhICAAAAAAAgBOLVkP/+KDXpUSaGv+AAxmb6AIcx6o82AeMhbaX0jCtBTipaq0+Lzv7bRqa5i0FdpkcOXiWPj9eB0dfMeG6AMABFjACAHitlcvPyNb5MfrozR+PPPuXmyNB5hSflwR9ODa53HBdBPxEGK2PxiaXDwzGPL17fSba3Cu4lvobwb2Caxd/9vprt59oY/tR8JUkPfmTTn39JHhJcA2R+L5wmOlrkrQU217R3mG29vTGWlsAAAAAAABAxNgEA0maY9ILjGmIxgMMN+eBAIOh7VnXRQD9Fu23H7muIyHeN80bm+LqyzTOoWEs8J2G4CgwzFjACADgtVYuX5Gde1Dos7d/OdL53i9GeHaQnI6k2bHJ5abrQuAvwmiOxXbQZq/X7BNYm9DuCVOplj2rXPZsJicdHFzbeqjPg+fBZrCtU4//ELyM/u47T/8c5CRp6zPpxWcE14bcjR7bu8QCbB3tTFmTdu+T8TDbyvTGWlpW7QIAAAAAAMDRNWTnvuudcqnYcF0EkBRjE12A47jpOmBsMLSdpmktwLFE+62VMJH3TfPRFNdPXNeREC+mcQ4TY4Fv74Oje2i7LgAYMBYwAgB4qZXLTyi8juVeMA6UPSeN/zr78OxU5qLrWgzpSJo5zGAmDDfCaClwUGDt6d3rFxQG1SRpJvodD65ZaZz4xuhFXeqG1b71bmbf1+4VXNv8q956/iA4I4XT1rYf97tipMB5HT3Atq6dm5GP1CPMNr2x1hQAAAAAAAC8F00weN91HQlZl+eNusBRGJvoAhzHUrlUnHddhGyFtpnWgmGxKDtTRSs+N80bDOw6n8Y5TIx9frwPjvawKDthQOAgLGAEAPBSK5cvSJqXnetY9NHpqxlN/Cb7cPQtEURLzqrCIBr3A3AgwmgGRDt7M/pjc6/X9AisdX9PyfCX9lGCay/+HqxL0suORp79JfhaIriGfY1r9wPXeKPSNzcoY+E1KfyS7n5Bt7V3mO3R9Maatw9RAAAAAAAALIomGFRc15EgGkdhhrHGXOA4OpJmXRdRrdULshPaTuO0FuDIosUWei7CmTJ3PAnl7qciOyv33/I5+GdUQ3YC36mctlQuFVeqtfqS7Bw3gV5YwAgA4J1WLt+9B2zl3hP67PyNzNN//c9sJjNKEC1BS5JmCaLhsAijDYlDBtYmFE5S6zVpzWxgreuNdzLj4W9p7IeHC649/5tGXvwjDK49/mMwvv3P8N+frAR9rRWpFX/4cJjpa1L45d61or3DbO3pjbW2AAAAAAAAcFwN2bkHeotJLzCmITuNuXtZUvjspq3YPd9yqdje++U4SBRgXJGdz82s64BxtVafULgqtQVpndYCHEm02MInrutIiPf7bbVWn5WdKa5L5VKx4rqIYWIs8J32aUsFheeRVu4PAHtxfn0BAEBcK5efUjil1sq9PPTZ278c6XzvFyOcsyfr9tjkcsF1EUgXwmj4xtjkcls7DzkX93pNLLDW/YkH14ZqZaB4cK07ee27P339dcGWnm19EXwh7R1c234cTlwD9nGjx/YusQBbRztT1qTdAdT49sr0xho3lwAAAAAAwNCr1urzsjPBYJXGUVhirDE37rakxXKpuOfzGJxYRXaaVxY8CRgvyk5TdiWN01qAozA4VdTrpnlj77f3wT9rjAW+Uz9tqVwqtqu1+ozCfdrKfQIgjsmXAACvtHL5iqSPXdeBdMiek8Z/nX14dirDNLRk3RqbXK64LgLpQxgNR/JKYG1PT+9en1IYUpvQkAfWJCkzqjOHDa5tP9HG9qPgK0l68ied+vpJ8DJ4qdNf/T64JBFcw6GdV+8A266T9liAbV07+/Yj7YTZ4tua3lhrJlcmAAAAAACAe1GTmZUJBh1Js66LAJJirDG367bCIEzbdSFWGZtMs1ouFedcF1Gt1Suy04x9p1wqWjuuAHupyM5+m4apv5YCuwXOUwbO2ufH2+DoYZVLxZXoXsFc9GPl/x+AyZcAAG+0cvkJhQsADF1fOY7n9NWMJn6TfTj6lgiiJevDscnlhusikE6E0ZC4scnlA1dPeXr3+ky0uVdwzcpN8SPLnlUuezaTk3YH1y7+7PXX7hVc23qo7zz9c5CTpK3PpBefEVzDoY1r9yqxe640HAuvSdKqwrCaFAbZ2ntsP5reWGNFJQAAAAAA4KVogoGlqURzNI7CmIbsNH52FE51abouxDIm0ySvWqtPyc7q1F68p0C/GQzlVlwXsZ9qrT4nO82Tt5naOljGAt9pCI4eWhSqq0ST1AsKF36xsq9jOHEuDADwRiuXLyhchMzKvV/02fkbmaf/+p/ZTGaUIFqCOpJmDpP7AHohjAYnxiaXm9Fms9dr9gmsTWh3aGYo9Qqu7WXroT4PngebwbZOPf5D8DL6O4JrOKn4TfGeN11fCbAtxbZXtBNmi2+3pzfW2gnUBwAAAAAAcJCG7DzsvFMuFRuuiwCSEjXmWmn27EiaKZeKPNTtv4bsHNcrrj8zBkPbsxamtQD7MRjK9XrqbxTY/cR1HQlZVzgBCgNiLPDtfXD0uKJzh/nopztdvdvHNOOsMAyCtQXVmXwJAHCulct3r1n3HJYA7OXtX450vveLESv3fH2xLmmWIBpOijAavHVQYO3p3esXFN7gkXZu8MSDa0MfWOsavahL3bDat97tHVqTpBd/D9Yl6WVHI8/+EnwtSZt/1VvPHwRnJGnz00Dbj/tbL0y70WN7l1iAraMwqNbV7LG9Mr2xxgNkAAAAAABwaNEEAysPPNfF6tYwxFhjLkG0ATF2XL9TLhXnXRehsOnayvM2U9NagH0syk4oNw1TfxuuC0hQgcDu4BgLfHsfHE1SdD7RdFwGBiCaimcljLbA5EsAgGutXH5G4TWUlXtN6LPsOenKQvbh6asZpqEla1XhRDTuAeDECKMhtaKDYDP6Y3Ov1/QIrHV/T8nOjfjEvPFOZjz8LY398OjBtcd/DMa3/xn++5MVpq3hxM6rd4BtVzNKLMC2LqkdbT/STpgtvq3pjbVmcmUCAAAAAIA0iYIuFdd1JIhJLzDDWGOuFDZ2E0Trs2qtPiE7x/WOPAgYV2v1WUkfuK4jIWantQBxUSjXylRR76f+GgspENgdPEuBb6YtwZzoXPgj13UkZFV2rpUAACnVyuUrsrP4GAbg9NWMrsxnO9lviyBasm5LmiOIhqQQRoNphwysTSicpNZr0hqBtR72Cq5996evvy7Y0rOtL4IvJOn53zTy4h8E19BX49p9437PlXBj4TUpvPnWPblaiW23FQu2TW+s0TQCAAAAAEDKRUGXhuzc97tF0AXGWGrMZfX5wbE0Cch5wDgK9zVc1pCgoZrWguEVLbbwies6EuL91N9qrT4jQyEFAruDZSzwfZvzXVhj8FyYyZcAAGdaufyUwu9VKwt5YADe/PHIs3+5ORJkTpm53+uLhbHJ5TnXRcAWwmgYemOTy23thE32vEkWC6x1f+LBNSury/VNZlRn4sE16XDBtSd/0qmvnwQvg5c6/dXvg0uStP1Y2vyU4BoSF7/Y6blPvxJgW4ptN2Pbu8Js0xtr7RPWBgAAAAAAklWRnQefSzSOwhJjjbmr5VKRB7sDUK3VK7JzXF/wZDKNpXDfHNNaYF1ssQUrvG6aNzbFlcDugBkLuaxL4nwXFlk6F66wgBEAwJVWLj+n8HmMle9VDMC7v8p+ef5G5k3XdRj04djkcsN1EbCHMBpwCK8E1vb09O71KYUhtQkRWDu2XsG1iz97/bXbT7Sx/Sj4Sto7uLb1mfTiM4Jr6JsbPbZ3iQXYOgqDalIYVovf8GvGtlemN9a8fcAGAAAAAECaRUEXKxMMOvJ8YgRwFMYa6WnsHpBoMs3HrutIiBcBRmPhvjvlUrHhughgACqys9/e8iSUu5+G7DRTVgjsDlxDdj4/zqe5AkkzeC4877oIAMDwaeXy3QU86JvGob3xdkZX/nf2y9FLIoiWrI6k2bHJ5abrQmATYTQgIWOTyweuJPP07vWZaHOv4JqVmxkDkz2rXPZsJicdHFzbeqjPg+fBZrCtU4//ELyM/u47T/8c5CSCaxiI89p9gfV+bHtXs0QswLaunSBsO7a9K8w2vbHWTKpIAAAAAACsMhZ0kcKJEW3XRQAJsrT6PPvnABg7rnsRMDYW7luXB+8p0G/GFltY9X3qb7VWn9PuZ3xpRkhhwKKQi5WG3FtMW4I1xs6Fvbi+AAAMn1YuPytbCzBgAM5OZfRv/5V9lhkliJawdYVBNK7d0DeE0YABiiWLm71eQ2CtP0Yv6lI3rPatdzP7vrYbXHvZ0cizvwRfS9LmX/XW8wfBGUna/DTQ9uN+VwxIksajH+lw09ckaVVhWE0KA2vd7bZiYbbpjTVOMAEAAAAAw8ZS0OV2uVRcdF0EkJSosdtKYy775+DMa+f+adpVXDd0R+E+S5/dAtNaYJ3BUK7XU0WrtfqUwil0FhBSGLDo82Ml5LLke3AUOKpoH7V0LszkQgDAQEXT0OYlfeC6FqTL278c6XzvFyPnJZ1xXYsxq5JmxiaXOSdEXxFGAzxzUGDt6d3rFxQG1SRpJvodD65ZefDqTDe49sY70tgP9w+uvfh7sC5JBNfgkXho9bABtqXYdjO2vSvMNr2x1j5hbQAAAAAAOGEs6LIuac51EUBSoqa/T1zXkSAmjAxANAnISnOLL5NpGrIT2r5VLhWbrosABsDSYgtzKZgq2pCd95uQwgARHAX8Fl2TNmXnGM+5MABgoFq5/IzC8116l3Fo2XPSlYXsw9NXMxdd12LQ7bHJ5YLrIjAcCKMBKROllJvRH5t7veaAwNqU7NxAce6NdzLj4e/9g2vBlp5tfRF8IUnP/6aRF/8Ig2uP/xiMb/8zfM2TlaDv9QI93OixvUsswLau2JQ1haG1rmZse2V6Y40HWQAAAAAApwwGXWgchTUN1wUkqON6utUwMNjQXXBdRBTaft91HQlhWguGgrHFFm6XS8WG6yL2U63V57V7Qcg0WyCkMHAV2fn8MHkUplRr9RnZCnevci4MABikVi4/L+kj13UgXU5fzejKfLaT/bYIoiXv1tjkcsV1ERgehNEAgw4ZWJtQOEmtV3DNyo0WL2RGdSYeXJPC4Np3f/r6a/cKrgUvdfqr3weXJGn7cThxDXBoXLtXMok3KXwcf2EswLaq2JQ19QizTW+sNROrEgAAAAAw9IwFFqRwdWuCLjDDWGO3tHvRJvSPpWZR5wFjg6HtiusCgH4ztt96P/U3CipYaa5cLZeKXr/f1kTTXK18fhbKpeKi6yKAJET3i+ZlZ9qyxORCAMAAtXL5KYXPXizd28UAXPz5yOb3SyOnZef+ri86kubGJpcbrgvBcCGMBgypscnltnbCIHveMNwjsBYPrllZac87vYJrF3/2+mu3n2hj+1HwlSQ9+ZNOff0keElwDZ6KX3geZvqaJC3FtlfUI8w2vbFGkw8AAAAAoJeK7DwMZdILTDHW2N01dfBLcBLGJgE5n0xjMLQthaGWpusigH4xuN86D+XuJ3q/LYVvCq4LGCbG9tdVEfiGAdF+ORf9WGuAniuXim3XRQAA7Gvl8hW9smA9cJDsOWn819mHZ6cyTENLXkfSzNjkMn20GDjCaAB6OmRgrRtSm9Drk9asPBD2VvasctmzmZx0+OBasK1Tj/8QvJSkrYf6ztM/BzlJ2vpMevEZwTV45UaP7V1iAbaOdq8+3Yxt7wqzTW+stU9eHgAAAADAZ8ZWoO+IxlEYYrCxu+t8tVafK5eK864LsSiaBFRxXUdCfJlMY206oSS9X63VVyQVmCYKoyztt2mY+tuQnbDCzRS839ZYmebaUfi96m1wFDhIdC0xp3BymIX98lW3y6Viw3URAADbWrn8hMJzXCvXpBiQ01czujKf7WS/LYJoyVtVGETjeg1OZIKA4AGA/ooF1vYKrnFi6qGth/o8eB5s9gqubX4aaPux2xqBhKwrNmVN+4TZpjfWOGEHAAAAgJSo1uoTCq/xrDQY/aRcKloM7mBIVWv1RUnvu66jjxYkzbMqfbKigJGFZwodSTOuAwHVWr0g6bcuaxiAJYVBkkUa6GFBtNjC/7iuIyFL5VJxxnUR+zF2nPT+/bYmmub6ies6EnKTxRaQNtECKDPRz6ykcZf19Nm6pCnOdwEA/dTK5ecULhJl5ZkLBuTiz0c2v18aOe26DqPuSCoQRINLhNEAeOHp3esz0SaBtZR58fdgXZJedjTy7C/B15K0+Ve99fxBcEYiuAaTVhWbsqYeYbbpjbXmIIsCAAAAAOyo1upN7TNlO2UWPJmeAyTCWGP3QZYU3i/iYfDJTclOgNF5Q7fB0PZhrCq8l9u9h8u+ia6VNDRvG9tvOwqb5tuuC+nF4Ps9kYbPuRXRBKZl13Uk5E65VJx1XUQv0b5acFwG/NFdJHtCtsNnr7rueqELAIBd0TS0huw8b8GAZM9J7/w/2S/P38i86boWoxbGJpd5dgjnTrkuAAAkaWxyuRltNvf696d3r3dvGknhykXS7uDaMN1I8sob72TGw9/S2A8z+762G1x7/jeNvPhHGFx7/MdgfPuf4b8/WSEgjVSIB2R7Xmi3cvn4H5di2/Emh7Z2wmzt6Y21tgAAAAAAJ1Kt1Suy82B0VeFqo4AJUbPoME1VuCE7xyMk447rIFpkUTYCFkfxXvRjJdSIZKyWS8Wpg1/mhYbs7LcFn4NoEUvHyQJBtIFruC4gIR35H/RaFIsbY7jdJIgGAOiXVi5fUHgv18q1EQbk9NWMJn6TfTj6li66rsWoD8cmlxuuiwAkwmgAUiIaI9qM/tjc6zUHBNamxEmxc/HgmhQG177709dfF2zp2dYXwRcSwTWYcaPH9i6xAFtHsSlr2n3ci4fZVqY31niACAAAAACRaAX6j13XkSAaR2GNpcZu4Ki8aOiu1urzomkbkDzZJw/D2GILt8ul4qLrIvYTvd9WjpPev9/WGPuenfX5etTYvgocx5InC10AAIxp5fIXFC6wwII+OLKLPx/Z/H5p5LREEK0POpJmxiaXWYwA3iCMBsCMQwbWJhROUusVXKMRwgOZUZ05bHBt+4k2th8FX0nSkz/p1NdPgpfBS53+6vfBJUnafixtfkpwDalzXkcPsK1rZ8raI/UIs01vrDUFAAAAAAZVa/ULCoMuVrC6NUyp1upzolkUw815Q3e1Vp+R9JHLGgCPVNJwrmVssYV1SXOui9hPdJzk/caxGPueXSiXik3XRfRibF8FjqMjadZ1EQAAe1q5/KxsTebGgGTPSeO/zj48O5UhhNYfq5JmxyaX264LAeIyQUCDPgDE7RFYiwfXrKw6OJT2Cq5tPdR3nv45yEnS1mfSi8/4XsRQWdXOlLW29g6zPZreWPP+gTwAAAAAVGv1hqQPXNeRkKVyqTjjugggKVFYtC2aGDC8FsqlotNAAPshsMudcqnofQN3tN+uSBp3XUtCrvscAOT9xkkY+55dLZeKUwe/zA1j7zVwXD9h8iUAIElMQ8NJnL6a0ZX5bCf7bc7R++SOpEI0sAXwCpPRAOAVUXK8Hf1xz5s3T+9e74bUJvT6pDUCa57KnlUuezaTk3ZPXNvL1kN9HjwPNoNtnXr8h+Bl9HcE12BNfDXyw0xfk6Sl2PaK9g6ztac31toCAAAAgAGp1uqzshNEY3VrWDQnmkUxvFZdB9Eii2I/BKTwXKvguohDmpedYFQapv42ZOf9vpWC99uahmx8z6bherQhG+81cFwLBNEAAEliGhpO4uLPRza/Xxo5LT4//bIwNrnsw71lYE+E0QDgGMYmlw+8eb9HYK37c0G7AyDw0OhFXeqG1b71bu/QmiS9+HuwLkkvOxp59pfga0na/Kveev4gOCNJm58G2n7c33qBAbrRY3uXWICto50pa5LUjG3Hw2wr0xtrrN4BAAAA4MiqtfqEwgelVhTKpSLXR7Cm4LoAwBEvQi/VWn1OLKQHdM2m4VzL2GILS+VScd51Efup1uoF2ZkAsFQuFSuuixgm0feslc/PXLlUbLsuohdj7zVwHEueLHQBADCAaWg4iew5afzX2YdnpzIXXddi2Idjk8sN10UA+yGMBgB9csjA2ky0udekNQJrKfHGO5nx8Lc09sOTBdeerDBtDeac19EDbOvambL2SDthtvi2pjfWmgnVCAAAACDdGrKz4iKrW8OcqJneypQR4KgqrifTVGv1KUmfuKwB8MitcqnYdF3EQYwttuD9lKXo/fY6LHcEXoSgh0n0PVtxXUdC7pRLxYbrInrhnAbQqjz/TgUApAfT0HASZ6cyGv9f2U722yKI1h8dSTOH6UEHXCOMBgAOjU0uN6PNZq/XxAJr3d/x4BpNHClz2OBasKVnW18EX0jS879p5MU/wuDa4z8G49v/DF9DcA2GjWv38W3PFXhi4TUpvPneXc22rZ0wW3z70fTGGhdpAAAAgCHVWr0iO5NeVmWniRGIo1kOw+qO60lA1Vr9giRCzkBoNUXTohqy0xCYhqm/i7Lzfns91cqohmx8ftblcZAxOqdpuK4DcGhV0kwKvlMBAJ5jGhpO6u1fjnS+94uR87JxHeSjVYVBNM77kAqE0QDAcwcF1p7evX5BYUBNIrBmRmZUZ+LBNSkMrn33p6+/Nh5ce/Innfr6SfAyeKnTX/0+uCRJ24+lzU8JrsG8+DTJw0xfk6Sl2PaKdsJs8e329MZaO4H6AAAAACSsWqvPSPrYdR0JSkOjLnAcM64LABzwZTJNQzwjAKQUTOfqMrbYgvdTf6P3+72DXpcSXk+1sqhaq8/LzufH9+vRiuy818BR3ZH/+ygAIAWYhoaTyJ6TrixkH56+mmEaWv/cljRHEA1pQhgNAFIuOvFoRn9s7vWap3evTygMpvUKrnGBkWK9gmsXf/b6a7efaGP7UfCVtHdwbesz6cVnBNcwNG702N4lFmDrKAyqdTV7bK9Mb6xxUQgAAAD0kcFV0W+WS0UmOcMqgjAYRrOum0WrtXpBrHINdKViWlS1Vp+SncUWvJ/6a+z99nqqlUXR4igfua4jIbfKpWLTdRG9VGv1Wdl5r4GjWiiXinOuiwAApBvT0HBS529knv7rf2ZHMqMiiNY/t8YmlyuuiwCOKhMENJwDAA4MrFlZgRFHsPVQnwfPg81gW6ce/yF4Gf3dd57+OchJBNeAQ1qX1I62H2knzBbf1vTGWnOgVQEAAAApV63VF2XnwemdcqmYikkdwFFFTbq/c10HMGDOG0artfqEwntPLEQHpORcK1psYUV2QtzXfV5sweD7/SOfw0TWGPv8LJVLxRnXRfTCOQ2GWEdhmL7huhAAQLoxDQ0nkT0nfb888jD37yOE0PqnI6kwNrns9WR5oBcmowEAJEljk8tt7QQm9jyxeXr3+pTCsNqEXg+uEVgzZvSiLnWnrH3r3cy+r90ruLb5V731/EFwRpI2Pw20/bjfFQNeGtfuh5F7NsvGpq9J4Yqx3ZWz29o5Nse3H01vrHn7IB8AAADoJ2OTXjpiggEAWLLqOogWmRdNRoCUrmlR87IRbJHSMfXX0vu9QBBt4Bqy8flJw/VoQ5zTYPisSiqk4LsUAOAxpqHhpE5fzWjiN9mHo28xDa2P1iXNjk0uc96H1CKMBgA4tMOc9OwTWLsg6b0+lgeHjhJce/H3YF2SXnY08uwvwdcSwTXgFfFjZc+g7ysBtqXY9op2wmzx7fb0xlo7gfoAAAAAZ6JV0edd15Gg2XKp+OjglwEAUqAjyfn0pWgiIY1GQKiQhnOtaq0+K+kD13UkZKlcKnp9vm7s/fYlBD00jC2OUiiXim3XRfRSrdUrYkFcDJ8FSZU0nL8AAPzFNDSc1MWfj2x+vzRyWiKI1kdLCoNonPch1QijAQASdcjA2ky0uVdwjcCacW+8kxkPf0tjP+wdXAu29Gzri+ALSXr+N428+EcYXHv8x2B8+5/ha56sBH2vF0iJGz22d4kF2DoKg2pSGFaLH7ubse2V6Y01LnoBAADgk0XZeYDKBAOYVy4Vm9Va3XUZwKDMedLQXXBdAOCJW2k414oWW2g4LiMpXoRy92Pw/S64LmKYGFsc5Xa5VFx0XUQvUbj+Y9d1AAPUURgQ9Xa/BAD4j2loOKnsOenKQvbh6asZQmj9dXtscrngugggCYTRAAADNza53Iw2m71es0dgLR5cG+9bcfBGZlRn4sG17uS17/709dceFFzbfhxOXAPwjfPaHVqL34ja9XAvFmBbl9SOttux7V1htumNtWZSRQIAAABx0aroVhaxYYIBhklHdkKkQC93yqViw3URESvTfoCTWC2XihXXRRxSQ3a+J9Mwia4hO+93pVwqHrhIKBJlZXGUdUneXo9Wa/VuEzUwLJiGBgA4sVYuPyepIhvnq3Dg/I3M03/9z+xIZpRpaH324djkcsN1EUBSCKMBALx0UGDt6d3r3YCaJM1EvwmsDamjBNe2n2hj+1HwlSQ9+ZNOff0keBm81Omvfh9ckgiuAfsY186x9TDT1yRpVWFYTQoDa93ttmJhtumNNR6YAwAAYF/GVkVnggGGzaIIx8C2dXlyXI++L4Fh5/10rq5osYWe91pTZsH3aS7G3u875VLRyoSuVDC2OMqs56GXhnjWj+GwqnC6ctN1IQCA9Grl8hMKz5+sXOtgwLLnpO+XRx7m/n2EEFp/dSTNjE0u0yMHUwijAQBSaWxy+ZF2gmrNvV4TC6z1Cq6xEsgQyp5VLns2k5N2B9cu/uz113aDa8G2Tj3+Q/BSkrYe6jtP/xzkJGnrM+nFZwTXgH3EH8weNsC2FNtuxrZ3hdmmN9baJ6wNAAAAKWFwVXQmn3rUiwAAIABJREFUGGDYEEaDdWmYBAQMk0K5VGy7LuIgxhZb8H7qb7VWn5Kd95vFLQbM2P560+fr0WqtPifpfdd1AH3WUXhviFAxAOBEmIaGkzp9NaOJ32Qfjr7FNLQ+W1UYROMeMswhjAYAMOuVwNqeqzE+vXt9QuEkNQJreE08uPatdzP7vnbroT4PngebvYJrm58G2n7c95IBC2702N4lFmDrKAyqSWFYLf4QtRnbXpneWOOiHgAAIJ0asrMqOhMMMHTKpeJitVZfl539GIi7xSQDwCu3fZ/OJZlbbMH7YFT0fnv/uTgCQtADZGx/XfL5ejQKjVZc1wH0UUfSvKR5juMAgJNo5fJTCr9TmIaGY3v7lyOd7/1i5LxEEK3PbkuaI4gGqwijAQCG2tjkcltSO/pjr8Bad7rahF4PrnFRB0nS6EVd6k5ZOyi49uLvwbokvexo5Nlfgq8lafOveuv5g+CMRHANOKLz2n0sjq+YuWul0liAbV07x/52bHtXmG16Y62ZVJEAAAA4HmOronvfqAv0UUHS71wXASRstVwqVlwX8Yq26wIAh9YleT2dK6YhOyHtNEz9rcjO+72QhsClMfOy8fnpSJp1XUQvsdAfC8XCIkJoAIDEtHL5iuxM7YUDb7yd0fivRx6evpohhNZ/N8cml71dEARIQiYIAtc1AACQevsE1i5Ies9ZYUi9vYJrj/8YjG//M/z3JyucywEDshTbXlEYXJNeCbNNb6z53ngBAACQGtGq6E3ZaUb7EdNzMMyqtXpFNErAjo6kqXKp2HZdyKuqtXpTLKKG4XQ9BaEoVWv1gqTfuq4jIXfKpaK34RZJqtbqs5L+x3UdCVmVNEOQYXCMfX5+4nOQsVqrz0v6yHUdQMLWFYYsCaEBAE4smobWEH2IOIE3fzzy7F9ujgSZUxpzXYtxHUmzY5PLTdeFAP1GGA0AgAF5evf6TLS5V3CNC0WcSLClZ1tfBF9I0vO/aeTFPwiuAY51FJuyprCJumtXmG16Y609oJoAAABSpVqrr8jO9fJCuVRMy6QOoG+qtXpD0geu6wAS8GG5VGy4LmIvBsPcwGHc8nBS4WuqtfqEwnuDFvbPjqQJn5vro0lLbdl4v6WUBC6tMLa/en09aiz0B0jhApcNX68XAADp0srlLyic9kxwH8eWPSeN/zr78OwU09AGYFVhEK3tuhBgEAijAQDgkX0CaxOSxl3UBHu2n2hj+1HwlSQ9+ZNOff0keBm81Omvfh9ckqTtx9Lmp5wjAg6sKzZlTfuE2aY31rxt8gAAAEiCsVXRV8ul4pTrIgBfVGv1OUmfuK4DOIE0TAKakrQo7iljOCyVS8UZ10UchrHFFryf+mtsUuTNcqk477qIYWLo8+P19aix0B+GW3cKWsPH6ckAgHRq5fIzCr9fuL+DYzs7ldG//Vd2MzOq065rGQK3Jc2NTS7T04Whccp1AQAAYEdsNG9zr39/evf6BYVBNUmaiX7Hg2tcfOJA2bPKZc9mcpL0xjuSlJEkXfzZ66/dK7i29VDfefrnICdJW59JLz4juAYkZFy7j+Pvx7Y/jr+wlct3N1cVm7KmHmG26Y21ZmJVAgAA9Fm1Vp+RnSBaR5LXgQVg0Mql4nzU3DsvGw2+GC7rkgquizhIuVRciQJpc9EPDd6wqqMU7JOSVK3VK7ITRFtIQRBtTnbOM5YIog1WtL9a+Pyk4RjZEOcpSK9VhQtALDK5EgCQpGgaWkO7e0aAI8mek75fHnmY+/eRixJBtAG4OTa5zLU7hg6T0QAAMKZHYK37e0rc0EefbD3U58HzYDPY1qnHfwheRn9HcA3wz1Jse0U9wmzTG2s8OAMAAANXrdUvKDwnsXLt+mG5VGy4LgLwVRQ+LUj6wG0lwKF5PwnoVdF362z0QxMTrPlJuVRcdF3EQaLvu9+5riMhXk9Zkr6ZDrnsuo6EdCRNMWVncIx9fryeqBeF/j4+6HWAR9YVLirclNTk2AwA6IdWLl9QuIiXlWckcODsVEbj/yvbyX6bz9EAdCTNxoZQAEOFMBoAAEPo6d3rEwonqfWatMaFCPqqG1x72dHIs78EX0vS5l/11vMHwRlJ2vw00PZjtzUC+EZHsSlr2j29c1eYbXpjrT2gmgAAgFHVWn1Rdhrl75RLRaaiAYcQC8vMRD/j+70ecORWuVSsuC7ipKJQzIzC+8ATsjOpCcNnoVwqzrku4iDRd9yKbHy3dSTN+Dz9JXq/m7JzbEtF4NIKY/ur19ejxkK6sGlV4WJJKwq/V1bKpeKj/f4HAACcRCuXn1A4Dc3ChF449PYvRzrf+8UIvZ+DsaowiNZ2XQjgyinXBQAAgMGLToDb0R/3fIgVC6x1f+LBNS58cSKjF3VJyuiNd6SxH2b2fe2LvwfrktQruPZkhcUVgD47r93H/Z7fAa1cvru5rtiUNfUIs01vrDUFAAAQiZrRLmj3JNe0eqRw2hOAQ4ia+hrRj6Rdx4Tu/aip6M+AC48sBNEkKZrs1nz176N9rov9DWng7bSfVxQU3idrO60iGQ2fg2iRGYXn4hauKVYIog3cjOzsrwXXBRygIBv7KdKtGdvuLr7YZuIZAGDQWrl8RUyMxQmdvprRxG+yD0ff0kXXtQyJ25LmxiaXWbAAQ43JaAAA4Nie3r3ebUqYEIE1OBZs6dnWF8EXkvT8bxp58Y8wuPb4j8H49j/D1xBcA7y2qtiUNe0dZns0vbHme8MLAAAAAAAAAAAAAABAT61cfkbhQitWJjvDEaahDdzNscnltCySBPQVYTQAANBXscDaXsE1LqbhxF7BteClTn/1++CSJG0/ljY/5TwZSIH4yqndVSul3WG29vTGWlsA8P+zd3+vcd95nu9fVbIzsuzYroxju7unx2rIuRmQYjEXtRc9RLCwZ2ECbZYzszvTcNrNOcsMA8N4LpbDMHs4btjhnMvkP+j+D9L/gXyRC0GIZFUCuXBAwkmwjdmSHP2Wyt9z8ZVa5UTyr9j1LVU9HiDq06UKvC/6wvrW9/l9AwAAAAAAAFRotjFxPsnNJP9U8Sgcc29cruXK/1e/P/qz2qWqZxkSK0mujU3OzVQ9CPQLMRoAULn1hanpvaNgjb7TWUu7s1w8SpK1z3Li8VqxK1yDY2slB1vWkmSm69wds803263lAAAAAAAAAAC8ArONiWtJfpvEFit+kAv/pb55+b/WH9dOZKzqWYbE7ZQh2mLVg0A/EaMBAH1vfWFqf7NakkzvvXaHa1d6PxV8387D3C+2is2ikxOrnxS7e++dXf+8aCTJzr1k+55/f8Mxs5SDLWvLOSJma7ZbMwEAAAAAAAAA6DLbmBhPGaG9V+0kHHdvXK7lyr/VH46+U7tQ9SxD5HdJboxNznmoNXyHGA0AGAjPCNauxhNl6DOHhWubX+bi1lfFqaTcttZZrXZG4KXdzsGWtcUcHrMtN9ut+QAAAAAAAAAAA2m2MXEzyf9T9Rwcf7ahVeKfxybnPqh6COhXYjQAYGisL0yNp9ykdlS4JlijL21/XSwlye5K6htfFI8T4RoMoFtd5/kcxGzd58Vmu7XYy6EAAAAAAAAAgBcz25iYTrkN7Uq1k3Dc2YZWiZUk02OTcx4yDU8hRgMA6HJIsNYdrlmVTt/bD9e27qa+/U0Zrq1+WlzpfFv+fm3ev/9hQKzkYMtakswccZ5vtlvLAQAAAAAAAABeq9nGxPmUEdovKh6FAWAbWiVupwzR3GsDzyBGAwB4QesLU/uR2ni+v2lNsMaxUOxkY+dB8SA5PFzrrJYb14CBspRkce+8nIOYrfucZrs109OpAAAAAAAAAOCYm21M3EhyM8m5ikfhmLMNrTK/G5ucu171EHBciNEAAF6DQ4K1/Z/zSd6taCx4KZ21tDvLxaMkWfssJx6vFbvFbkYffVxcSoRrMOBup4zVkjJkWzzkvNxst7q3tAEAAAAAAADAUJhtTFxNuQ3NPWH8YLahVWIlyY2xybnfVj0IHCdiNACAiqwvTE3vHQ/btObiBMfSfrhWdHJi9ZNiN0l2Hubs+udFI0l27iXb9/wNAgPuVtd5PgcxW/d5sdluLfZyKAAAAAAAAAB4VWYbE+dTbkL7p4pHYQDYhlaZpSTXxibnPIQZXpAYDQCgj3UFa/uv3eHald5PBK/OzsPcL7aKTeEaDL2VlKHavpkjzvPNdms5AAAAAAAAAFCh2cbE9SQfJDlX8SgMANvQKnMrZYjmXhR4CWI0AIBjbH1h6nzKQC0RrDHAtr8ulpJkdyX1jS+Kx0my+WUubn1VnEqSzTtFOqtVTgj02FKSxb3zcg5itu5zmu3WTE+nAgAAAAAAAGBgzTYmrqaM0N6rehaOP9vQKvWbscm5m1UPAceZGA0AYMCtL0yNpwzTjgrXPKGHgXJYuLb6aXGl8235+7V5fwPBkLqdMlZLymBt/7yYrrCt2W7NBwAAAAAAAAD2zDYmzie5meSfKh6FAWEbWmVWklwfm5z7qOpB4LgTowEAIFhjKBU72dh5UDxIkq27qW9/I1wDvudW13mm6/xEzNZstxZ7NRAAAAAAAAAAvTPbmLiechua+6f4wWxDq9TtJNfGJucWqx4EBoEYDQCA57K+MHU1Zaw2nu+Ha1bPM9A6a2l3lotHSbL2WU48Xit2i92MPvq4uJQkndVk846/rYCspAzVkjJW696yNtN1nm+2W8sBAAAAAAAAoC/NNiaupozQ3BfFK2EbWqV+l+TG2OScezXgFRGjAQDwyjwlWDuf5N3KBoMeela4tnMv2b7n7zDgD5aSLO6dF7vOT8RszXZrpoczAQAAAAAAAAyl2cbE+SQ3k/xTxaMwIEbfqWX8/x15ePJibEOrxq/HJud+W/UQMGjEaAAA9NT6wtT03vGwcE2wxlDZeZj7xVaxWXRyYvWTYnfvvbPrnxeNRLgGHOl2ylgtKYO1/fNiumK2Zrs1HwAAAAAAAACey2xj4kbKEO1cxaMwIC7/XX3l7b+t+/9TNZaSXBubnHPvBLwGYjQAAPrOIcFad7h2pZqpoFr74druSuobXxSPk2Tzy1zc+qo4lSSbd4p0VqudEehrt7rOM13nJ2K2Zru12KuBAAAAAAAAAPrBbGNiOskH8SBtXpHTV2v56b/ahlahWylDtOVnfhJ4KWI0AACOnfWFqf1ALUmm914Fa7Bn++tiKUmEa8APsJSuLWspo7V9M13n+Wa75eItAAAAAAAAcOzMNibGU25C+1W1kzAoRs4kP/qH+sPGX9ZFaNX5zdjk3M2qh4BBJ0YDAGAgHRGs7b9eTWL9OUOv2MnGzoPiQZJs3U19+5syXFv9tLjS+bb8zNq8vxmB53Y7XVvWckTM1my3Zno5FAAAAAAAAMB3zTYmbia5EfcQ8YqcvlrLlf8xsjLypv9PVWQlyfWxybmPqh4EhoEYDQCAobW+MDWecpPaUZvWXBiAPYeFa8VuRh99XFxKks5quXEN4AXd6jrP54iYrdludW9mAwAAAAAAAHgps42Ja0k+SHKl6lkYDCNnkiv/NvLw9NWabWjVuZ3k2tjk3GLVg8CwEKMBAMBTdAVr+z/d4dp7VcwE/a6zlnZnuXiUJGuf5cTjtWJXuAa8Iivp2rKWZKbr/ETM1my3Fns0EwAAAAAAANDnZhsTV1NGaO734ZV56/36xo9v1Gu1kxmtepYh9ruxybnrVQ8Bw0aMBgAAP9D6wtTVlJHaeARr8EJ2HuZ+sVVsFp2cWP2k2N177+z650UjSXbuJdv3/N0K/CBL6dqylqfEbM12azkAAAAAAADAwJhtTJxPcjPJP1U8CgPkjcu1XPm3+sPRd2xDq9BKkhtjk3O/rXoQGEZiNAAA6IH1hanpveNh4dq7lQwFx8yzwrXNO0U6q9XOCAyM2+naspYjYrZmuzXTy6EAAAAAAACA5zfbmLiRMkQ7V/EoDJDLf1dfufCf6ydqIzld9SxD7HaS62OTc/PP/CTwWojRAACgTwjW4NXZ/rpYSpLdldQ3vigeJ8nml7m49VVxKhGuAa/Fra7zfA6P2Rab7dZiAAAAAAAAgNdmtjExneS3Sa5UOwmD5PTVWn76ryMPT16MbWjV+n3KEG35mZ8EXhsxGgAAHBPrC1PnU4ZqSTK999odrrmABi9hP1zbupv69jdluLb6aXGl8235+7V5fzcDr8VKurasJZnpOnfHbPPNdstFdAAAAAAAAHiG2cbEeMoI7b1qJ2GQjJxJfvQP9YeNv6yL0Kr3z2OTcx9UPQQgRgMAgIHyjGDtapJzFYwFA6HYycbOg+JBcni41lktN64BvEZLOdiytpwjYrZmuzUTAAAAAAAAGBKzjYnzST5I8quqZ2GwnHuvtv7T/3ukXjuZ0apnGXJLSa6NTc7NP/OTQE+I0QAAYMisL0yNp9ykdlS4JliDH6izlnZnuXiUJGuf5cTjtWK32M3oo4+LS4lwDeip2znYsraYw2O25Wa75aI9AAAAAAAAx85sY+Jmkhtxvwuv0BuXa7nyb/WHo+/UbEOr3q2UIdryMz8J9IwYDQAA+J5nBGvv9X4iGFyHhWs7D3N2/fOikSQ795Lte/52B3rqVtd5PofHbIvNdmsxAAAAAAAAUIHZxsT1JDeTXKl2EgbN5b+rr1z46/rJ2omMVT0L+c3Y5NzNqocAvk+MBgAAvJT1hamrKWO18Xw/XBOswWuw8zD3i61is+jkxOonxe7ee8I1oGorOdiyliQzXefumG2+2W55Wh0AAAAAAAAvbbYxMZ0yQnNvCq/U6au1/PRfRx6evBjb0Kq3knIb2kzVgwCHE6MBAACvzSHB2v7P+STvVjQWDIXtr4ulJNldSX3ji+Jxkmx+mYtbXxWnkmTzTpHOapUTAkNuKQdb1pZzELN1n9Nst2Z6OhUAAAAAAAB9abYxMZ7kgyS/qHgUBszImeRP/mXk/tmf1y5VPQtJklspQzQPOoU+JkYDAAAqtb4wNb13PGzTmmANeuCwcG310+JK59vy92vzrh0AfeF2DrasLeYgZus+Lzfbre4tbQAAAAAAABxjs42J8yk3of1TxaMwgN56v77x4xv11E7mVNWzkCT5cGxy7kbVQwDPJkYDAAD6Xlewtv/aHa5d6f1EMJyKnWzsPCgeJMnW3dS3vxGuAX3vVtd5PgcxW/d5sdluLfZyKAAAAAAAAJ5uL0K7sfdzruJxGDCj79Ty0/9evz/6M9vQ+sRKkutjk3MfVT0I8HzEaAAAwLG3vjB1PmWglgjWoC90h2trn+XE47Vit9jN6KOPi0tJ0llNNu+4JgH0nZWUodq+mSPO8812azkAAAAAAAC8crONiespt6G534NXauRM8qN/qD9s/GX9QtWz8Ae3k1wbm5xbrHoQ4PmJ0QAAgKHQFawdFa55ihZUpLOWdme5eJQcHq7t3Eu277l+AfStpSSLe+flHMRs3ec0262Znk4FAAAAAABwzMw2JqaT/DYiNF6Dt96vb/z4Rr1WO5nRqmfhDz4cm5y7UfUQwIsTowEAAOxZX5gaT7lJTbAGfWrnYe4XW8Vm0cmJ1U+K3b33zq5/XjQS4RpwbNxOGaslZci2eMh5udludW9pAwAAAAAAGEh7EdrNJO9VOwmD6I3LtVz5t/rD0XdqtqH1j5Uk18cm5z6qehDg5YjRAAAAXkBXsLb/0x2uuSgKfeSwcG3zy1zc+qo4lSSbd4p0VqudEeAF3Oo6z+cgZus+LzbbrcVeDgUAAAAAAPCyZhsT4ykjtF9VOwmDaORM8vYv6ytv/23dw6f7y+0k18Ym5xarHgR4eWI0AACAV2x9YepqykhtPII1OBa2vy6WkmR3JfWNL4rHiXANOPZWUoZqSRmrdW9Zm+k6zzfbreUAAAAAAAD0iAiN1+3ce7X1n/xfI1sjp9Ooehae8OHY5NyNqocAfjgxGgAAQAXWF6am946HhWvvVjIU8EzFTjZ2HhQPkmTrburb35Th2uqnxZXOt+Vn1uZdawGOraUki3vnxa7zEzFbs92a6eFMAAAAAADAgJhtTJxPcmPvx7YqXrnRd2r56X+v3x/9We1S1bPwhJUk18cm5z6qehDg1RCjAQAA9KmnBGvjSa5UMRPw/A4L14rdjD76uLiUJJ3VcuMawDF3O2WslpTB2v55MV0xW7Pdmg8AAAAAADCURGi8biNnkh/9Q/1h4y/rF6qehe+5neTa2OTcYtWDAK+OGA0AAOCYWl+YOp8yVEuS6b3X7nBNsAbHRGct7c5y8ShJ1j7Licdrxa5wDRhgt7rOM13nJ2K2Zru12KuBAAAAAACA12O2MXE9yc24h4HX5K336xs/vlGv1U5mtOpZ+J4PxybnblQ9BPDqidEAAAAG2DOCtavxxDE4dvbDtaKTE6ufFLtJsvMwZ9c/LxpJsnMv2b7neg8wUFZShmpJGat1b1mb6TrPN9ut5QAAAAAAAJUTofG6nb5ay49v1O+P/qx2qepZ+J6VJNfHJuc+qnoQ4PUQowEAAAy59YWp8ZSb1I4K1wRrcEztPMz9YqvYPCpc27xTpLNa7YwAr8lSksW982LX+YmYrdluzfRwJgAAAAAAGHizjYlrST6ICI3XZORM8if/MnL/7M9FaH3qdpJrY5Nzi1UPArw+YjQAAACe6ZBgrTtce6+aqYBXafvrYilJdldS3/iieJwkm1/m4tZXxalEuAYMhVtd5/mU4VrynZit2W51b2YDAAAAAACSzDYmplNuQnMPAa/N27+sty/9uj5aO5lTVc/CoT4cm5y7UfUQwOsnRgMAAOCVWF+Y2o/UxvP9TWsuNsMA2Q/Xtu6mvv1NGa6tflpc6Xxb/n5t3vUmYCispGvLWpKZrvMTMVuz3Vrs0UwAAAAAANBTIjR64fTVWq78j5GVkTdzrupZONRKkutjk3MfVT0I0BtiNAAAAHqmK1g7LFx7t7LBgNei2MnGzoPiQSJcA9izlK4ta3lKzNZst5YDAAAAAAB9SoRGL7xxuZYr/1Z/OPpO7ULVs3Ck20mujU3OLVY9CNA7YjQAAAD6yvrC1PTeUbAGQ6SzlnZnuXiUJGuf5cTjtWK32M3oo4+LS0nSWU0277iOBQyl2+naspYjYrZmuzXTy6EAAAAAABhOIjR6YeRMcvFX9fsX/rp+qepZeKoPxybnblQ9BNB7YjQAAACOna5gbf+1O1y70vuJgF46LFzbeZiz658XjSTZuZds33PNCxhqt7rO8zkiZmu2W92b2QAAAAAA4EgiNHrlrffrGz++Ua/VTma06lk40kqS62OTcx9VPQhQDTEaAAAAA2d9Yep8ykAtEazBUNt5mPvFVrFZdHJi9ZNid+894RrAk1bStWUtyUzX+YmYrdluLfZoJgAAAAAA+oAIjV45fbWWn/7ryMOTF3Oh6ll4qttJro1Nzi1WPQhQHTEaAAAAQ6krWDsqXDtXwVhAhfbDtd2V1De+KB4nyeaXubj1VXEqSTbvFOmsVjsjQB9ZSteWtRwRszXbrZkAAAAAAHDsiNDolTcu1/In/1J/ePpqTYTW/z4cm5y7UfUQQPXEaAAAAHCE9YWp8ZSb1ARrwBO2vy6WkuSocG1t3jU3gEPcTteWtRwesy032635AAAAAABQCREavTJyJrn4q/r9C39dv1T1LDzTSsptaDNVDwL0BzEaAAAA/ADrC1P729XG8/1wzcV5GHLFTjZ2HhQPkmTrburb35Th2uqnxZXOt+VnhGsAT3Wr6zyfw2O2xWa7tRgAAAAAAF6aCI1euvBf6puX/896UTuZU1XPwjPdShmiLT/zk8DQEKMBAADAa/aUYO18kncrGwzoK93h2tpnOfF4rdgtdjP66OPiUpJ0VpPNO67lATzDSg62rCXJTNe5O2abb7ZbvjQFAAAAAIaeCI1eOvPntbU//c3I7sibOVf1LDyX34xNzt2segig/4jRAAAAoA+sL0xN7x0PC9cEa8ATOmtpd5aLR4lwDeAVWMrBlrXlHBGzNdutmQAAAAAADAgRGr00+k4tP/3v9fujP6tdqnoWnstKym1oM1UPAvQnMRoAAAAcE08J1saTXKliJqD/7TzM/WKr2Cw6ObH6SbG7997Z9c+LRpLs3Eu277lGCPCCbudgy9piDo/Zlpvt1nwAAAAAAPqICI1eeuNyLT/6x/r9sz8XoR0jv09yfWxybvmZnwSGlhgNAAAABsT6wtT5lKFakkzvvXaHa4I14KkOC9c2v8zFra+KU0m5ba2zWu2MAMfYra7zfA5itu7zYrPdWuzlUAAAAADAcJhtTFxPciPJuxWPwhAYOZO8/cv6yoW/rp+snchY1fPw3P55bHLug6qHAPqfGA0AAACGyBHB2v7r1STnejwScExtf10sJcnuSuobXxSPE+EawCu2koMta0kyc8R5vtlueTopAAAAAHCovQjtZjy8lB556/36xo9v1Gu1kxmtehae21KSa2OTc/PP/CRAxGgAAADAd6wvTI2n3KR21KY1wRrw3IqdbOw8KB4kydbd1Le/KcO11U+LK51vy8+szbtGCfCKLCVZ3Dsv5yBm6z6n2W7N9HQqAAAAAKCnZhsT55NciwiNHjr3Xm39J/9tZGfkTfcUHDO/S3JjbHLOg++A5yZGAwAAAF5YV7C2/9Mdrr1XxUzA8fescK2zWm5cA+CVup0yVkvKkG3xkPNys93yNFQAAAAA6HN7EdqNvR9BED1x+motP75Rvz/6s9qlqmfhhaykjNB+W/UgwPEjRgMAAABei/WFqaspI7XxCNaAV6yzlnZnuXiUJGuf5cTjtWK32M3oo4+LS4lwDeA1u9V1ns9BzNZ9Xmy2W4u9HAoAAAAAhtVsY2I8yfWI0OihNy7X8if/Un94+mrtQtWz8MJuJ7k+NjnnQXTASxGjAQAAAJXpCtYOC9ferWwwYKDsh2tFJydWPyl2k2TnYc6uf140kmTnXrJ9z3VSgNdoJWWotm/miPN8s91aDgAAAADwXPYitJtJflXtJAyTkTN53lcoAAAgAElEQVTJn/zLyP2zP7cJ7Zj6cGxy7kbVQwDHmxgNAAAA6GvrC1PTe0fBGvDa7TzM/WKr2DwqXNu8U6SzWu2MAENiKcni3nmx67ycrrCt2W7N9HAmAAAAAOgLs42J6ZRb0H5R8SgMkZEzydu/rK9c+Kv6G7WTOVX1PLywlSTXxibnZqoeBDj+xGgAAADAsba+MLW/WS1Jpvdeu8O1K72fChgG218XS0myu5L6xhfF4yTZ/DIXt74qTiXCNYAK3E4ZqyVlsLZ/XkxXzNZst+YDAAAAAMfQbGPiWsoI7b2qZ2G4XP47EdoxdytliLb8zE8CPAcxGgAAADDwBGtA1Q4L11Y/La50vi1/vzbvOi1ARW51nWe6zk/EbM12a7FXAwEAAADAd802Jq4nuRnfa9Jjb71f37j89/XtkTdzrupZeGn/PDY590HVQwCDRYwGAAAAkGR9YWo8ZZh2VLjm4jrw2hU72dh5UDxIkq27qW9/I1wD6DMrKUO1pIzVureszXSd55vtlifMAgAAAPDSZhsT51NuQbsR31XSY+feq63/5L+N7IjQjrXbSa6PTc7NP/OTAC9IjAYAAADwnJ4RrL3X+4mAYdZZS7uzXDxKkrXPcuLxWrFb7Gb00cfFpSTprCabd1z/BegDS0kW986LXecnYrZmuzXTw5kAAAAA6FOzjYnxlFvQflXtJAyj01dr+fGN+v3Rn9UuVT0LP8iHSW6OTc55aBrwWojRAAAAAF6h9YWpqyljtfF8P1wTrAGVeFa4tnMv2b7nWjFAH7mdMlZLymBt/7yYrpit2W55oi0AAADAgJhtTEynjNB8p0jPidAGxkrKbWgfVT0IMNjEaAAAAAA99pRg7XySdysbDCDJzsPcL7aKzaKTE6ufFLt7751d/7xoJMI1gD52q+s803V+ImZrtluLvRoIAAAAgGebbUxcTxmhXal2EobRG5dr+ZN/rd87PVm7XPUs/GC3klyzDQ3oBTEaAAAAQB9aX5ia3jseFq4J1oC+sB+u7a6kvvFF8ThJNr/Mxa2vilNJsnmnSGe12hkBONJSuraspYzW9s10neeb7ZabFwAAAABeodnGxHiS60luJDlX6TAMpTcu1/Kjf6zfP/tzm9AGxD+PTc59UPUQwPAQowEAAAAcU4cEa93hmicnAn1l++tiKUmOCtfW5l2rBjgGbqdry1qOiNma7dZML4cCAAAAOC5mGxPTKSO0X1U7CcNKhDZwbie5PjY5N//MTwK8QmI0AAAAgAG1vjC1H6glyfTeq2AN6GvFTjZ2HhQPkmTrburb35Th2uqnxZXOt+VnhGsAx8qtrvN8jojZmu2WmyUAAACAgTXbmLiecgvauxWPwpASoQ2kD8cm525UPQQwnMRoAAAAAEPsiGBt//VqknM9HgnguR0WrhW7GX30cXEpSTqryeYd18ABjpmVdG1ZSzLTdX4iZmu2W4s9mgkAAADghc02JsZTbkG7Ed+5UZGRM8nbv6yvXPjr+snaiYxVPQ+vxFLKbWgzVQ8CDC8xGgAAAABPtb4wNZ5yk9pRm9Z8eQb0vc5a2p3l4lGSrH2WE4/Xil3hGsDAWErXlrU8JWZrtlvLAQAAAHiNZhsT0ykjtF9VOwnD7A8R2l/V36idzKmq5+GV+X3KEM11TqBSYjQAAAAAfrBDgrXucO29aqYCeDk7D3O/2Co2i05OrH5S7O69d3b986KRJDv3ku17rq0DHHO307VlLYfHbMvNdms+AAAAAM8w25g4n+RakptJrlQ7DcNMhDawVlJGaB9VPQhAIkYDAAAAoEfWF6b2I7XxfH/TmmANOJYOC9c2v8zFra+KU0m5ba2zWu2MALwyt7rO8zk8ZltstluLAQAAAIbCbGPiapIbKUO0cxWPwxAToQ20WylDtMWqBwHYJ0YDAAAAoG90BWuHhWvvVjYYwCuw/XWxlCS7K6lvfFE8ToRrAANuJQdb1pJkpuvcHbPNN9ut5QAAAADHxmxj4nqS6/HARSomQht4/zw2OfdB1UMAfJcYDQAAAIBjZX1hanrvKFgDBtZ+uLZ1N/Xtb8pwbfXT4krn2/L3a/Ou7QMMqKUcbFlbzhExW7PdmgkAAADQU7ONifGUW9CuxxY0KiZCG3i3U25Dm3/mJwEqIEYDAAAAYOB0BWv7r93h2pXeTwTwehQ72dh5UDxIDg/XOqvlxjUABtrtHGxZW8zhMdtys91y4woAAAC8BFvQ6CcitKHwm7HJuZtVDwHwNGI0AAAAAIbO+sLU+ZSBWiJYA4ZEZy3tznLxKEnWPsuJx2vFbrGb0UcfF5cS4RrAkLnVdZ7P4THbYrPdWgwAAAAMIVvQ6DcitKGwlHIb2kzVgwA8ixgNAAAAAA6xvjA1njJMOypc88UjMLAOC9d2Hubs+udFI0l27iXb93y/ADBEVnKwZS1JZrrO3THbfLPdWg4AAAAcQ7ONifNJrsUWNPqICG1ofJjk5tjknGtrwLEgRgMAAACAlyRYA0h2HuZ+sVVsFp2cWP2k2N17T7gGwFIOtqwt5yBm6z6n2W7N9HQqAAAA+I7ZxsTVlFvQrsV3O/SJNy7X8qN/rN8/+/Papapn4bVaSXLNNjTguBGjAQAAAMBrtL4wdTVlrDae74drnqoJDJXtr4ulJNldSX3ji+Jxkmx+mYtbXxWnkmTzTpHOapUTAlCx2znYsraYg5it+7zcbLe6t7QBAADAC9vbgnY9ZYR2pdpp4IAIbaj8Psl129CA40iMBgAAAAAVe0qwdj7Ju5UNBlChw8K11U+LK51vy9+vzft+A4Akya2u83wOYrbu82Kz3Vrs5VAAAAD0p9nGxLWUEdovKh4FniBCGyorKSO0j6oeBOBlidEAAAAA4BhYX5ia3jseFq4J1oChVexkY+dB8SBJtu6mvv2NcA2Ap1pJGartmzniPN9stzyVGgAAYADMNiaupgzQric5V+kw8B2nr9by9t/U777572o/rXoWesI2NGAgiNEAAAAAYEA8JVgbT3KlipkA+kl3uLb2WU48Xit2i92MPvq4uJQkndVk847vTQD4nqUki3vn5RzEbN3nNNutmZ5OBQAAwJFmGxPncxCgeagffef01Vp+fKN+f/RnNqENCdvQgIEiRgMAAACAIbG+MHU+ZaiWJNN7r93hmmANYE9nLe3OcvEoOTxc27mXbN/zHQsAR7qdMlZLypBt8ZDzcrPd6t7SBgAAwA8025i4nuRakl9UPAoc6uxf1HLp/xChDZlbSa7ZhgYMEjEaAAAAAPAHzwjWriY5V8FYAH1t52HuF1vFZtHJidVPit29986uf140EuEaAM/tVtd5pus8n66wrdluLfZqIAAAgONgtjExnXID2rX4HoM+9db79Y3Lf1/fHnnT/0eHyEqSm2OTcx9UPQjAqyZGAwAAAABeyPrC1HjKTWpHhWu+RAM4wn64truS+sYXxeMk2fwyF7e+Kk4lyeadIp3VamcE4NhYSRmqJWWs1r1lbabrPN9stzx5GwAAGCizjYmrOQjQrlQ7DRxu5ExyblqENqRuJbk+Njm3WPUgAK+DGA0AAAAAeOUOCda6w7X3qpkK4HjZ/rpYShLhGgCv0FKSxb3zYtf5iZit2W7N9HAmAACA5zLbmBhPGZ9dT/JupcPAU4ycSd7+ZX3lwl/V36idzKmq56GnbEMDhoIYDQAAAACoxPrC1H6kNp7vb1oTrAE8p2InGzsPigdJsnU39e1vynBt9dPiSufb8jNr874PAuCl3U4ZqyVlsLZ/XkxXzNZst+YDAADwis02Js6njM+uR4BGn3vjci0/+sf6/TebtbMitKFkGxowNMRoAAAAAEDfOiRY2/85H186A7yww8K1Yjejjz4uLiVJZ7XcuAYAP9CtrvNM1/mJmK3Zbi32aiAAAOD42AvQru39/KLiceCZRt+p5dKv6/fP/rx2qepZqIRtaMDQEaMBAAAAAMfa+sLU9N7xsE1rgjWAl9RZS7uzXDxKkrXPcuLxWrErXAPgNVlJGaolZazWvWVtpus832y3lgMAAAwcARrH0emrtfz4Rv3+6M9EaEPMNjRgKInRAAAAAICB1xWs7b92h2tXej8RwGDZeZj7xVaxWXRyYvWTYnfvvbPrnxeNJNm5l2zf850UAK/UUpLFvfNi1/mJmK3Zbs30cCYAAOAFCNA4rt56v75x+e/r2yNv5lzVs1CZlSQ3xibnflv1IABVEKMBAAAAAENvfWHqfMpALRGsAbxWzwrXNu8U6axWOyMAA+tW13k+ZbiWfCdma7Zb3ZvZAACAV0iAxnE1ciZ5+5f1lQt/VX+jdjKnqp6HSv0+5TY029uBoSVGAwAAAAB4Dl3B2lHhmqdfArxi218XS0myu5L6xhfF4yTZ/DIXt74qTiXCNQBeu5V0bVlLMtN1fiJma7Zbiz2aCQAAjp3ZxsR4yuvpAjSOnTcu1/Kjf6zff7NZOytCG3orKSO0j6oeBKBqYjQAAAAAgFdkfWFqPOUmNcEaQI/th2tbd1Pf/qYM11Y/La50vi1/vzbvOzEAemIpXVvW8pSYrdlueYI6AAADay9Au5bkepJ3Kx0GXsLpq7Vc/rv63bE/q/206lnoC7ahAXQRowEAAAAA9FBXsLb/0x2uvVfFTADDpNjJxs6D4kEiXAOgb9xO15a1HBGzNdutmV4OBQAAL2q2MXE1ZYB2LQI0jqm33q9vXPxVfe3kxVyoehb6gm1oAIcQowEAAAAA9Jn1hamrKSO18QjWACrTWUu7s1w8SpK1z3Li8VqxW+xm9NHHxaUk6awmm3d81wZAJW51nedzRMzWbLe6N7MBAMArN9uYuJZkOmWAdqXaaeDljJxJ3v5lfeXCX9XfqJ3MqarnoW98mOSmbWgA3ydGAwAAAAA4htYXpqb3joeFa544C9Bjh4VrOw9zdv3zopEkO/eS7Xu+lwOgMivp2rKWZKbr3B2zzTfbLTfZAQBwpNnGxPmU4dn03uu5SgeCH2D0nVou/bp+/+zPa5eqnoW+spRyG9pM1YMA9CsxGgAAAADAgBKsAfSnnYe5X2wVm0UnJ1Y/KXb33hOuAdBvltK1ZS1HxGzNdmsmAAAMtNnGxNUcxGfvVTsN/HBvvV/f+OP/rfZo9GciNL7nN0k+sA0N4OnEaAAAAAAAQ2p9Yep8ylAtKW8kSJ4M1670fioAum1/XSwlye5K6htfFI+TZPPLXNz6qjiVJJt3inRWq5wQAJ5wOwdb1hZzeMy23Gy35gMAQF+bbUx0bz9zrZhjb+RM8vYv6ytv/aL+eOR0GlXPQ9+5nXIbmr9XAZ6DGA0AAAAAgCMJ1gCOj2eFa2vzvhcEoC/d6jrP5/CYbbHZbi0GAIDXZrYxMZ4yPJtO8otKh4FXaPSdWi79un7/7M9tQeNQKyk3od2sehCA40SMBgAAAADAD7K+MDWeMkw7Klw71/OhADhSsZONnQfFgyTZupv69jdluLb6aXGl8235GeEaAH1sJQdb1pJkpuvcHbPNN9ut5QAAcKjZxsT5lNdxp2P7GQPorffrGxf+c+1//tGf1n5S9Sz0rVspt6EtVj0IwHEjRgMAAAAA4LV7RrD2Xu8nAuB5dIdra5/lxOO1YrfYzeijj4tLSdJZTTbv+L4RgL63lIMta8s5ImZrtlszAQAYYLONias52H7muiwD543Ltbz1i9rKhb+qv1E7mVNVz0PfWklyY2xy7rdVDwJwXInRAAAAAADoC+sLU1dTxmrj+X645sYIgD7XWUu7s1w8Sg4P13buJdv3fDcJwLFxOwdb1hZzeMy23Gy35gMA0KdmGxPjOdh8Np3kXIXjwGtz+motl/+ufnfsz2o/rXoW+t7vU25Ds0kb4AcQowEAAAAAcGw8JVg7n+TdygYD4IXsPMz9YqvYLDo5sfpJsbv33tn1z4tGIlwD4Ni61XWez0HM1n1ebLZbi70cCgAYHrONifM5CM+mk1ypch54nUbOJG/9ot5++2/q9ZE3hZY801LKCG2m6kEABoEYDQAAAACAgbK+MDW9dzwsXBOsARwzh4Vrm1/m4tZXxakk2bxTpLNa7YwA8JJWcrBlLUlmjjjPN9stT+0HAL5nLz6b7vpx/ZOBN/pOLZf/a/3um//OFjSe22+SfGAbGsCrI0YDAAAAAGDoHBKsdYdrnhYMcExtf10sJcnuSuobXxSPE+EaAANnKcni3nk5BzFb9znNdmump1MBAD0hPmNYjZxJzk3XNy7+qr528mIuVD0Px8atJDfGJufmn/lJAF6IGA0AAAAAAL5jfWFqP1BLyps6EsEawMAodrKx86B4kCRbd1Pf/qYM11Y/La50vi0/szbve1QABsbtlLFaUoZsi4ecl5vtlhs0AaDPzDYmxlNel5yO+IwhNPpOLZd+Xb//ZrN2tnYyp6qeh2NjJcnNscm5D6oeBGBQidEAAAAAAOAlHBGs7b9eTXKuxyMB8Bo8K1zrrJYb1wBgwNzqOs/nIGbrPi82263FXg4FAINuLz6b7vrxUCyGji1o/EC/T3J9bHJu+ZmfBOClidEAAAAAAOA1WV+YGk+5Se2oTWuCNYAB0llLu7NcPEqStc9y4vFasVvsZvTRx8WlRLgGwEBbSRmq7Zs54jzfbLfcFAoAe2YbE9N5cvOZ64UMLVvQ+IGWUkZoM1UPAjAMxGgAAAAAAFChQ4K17nDtvWqmAuB12w/Xik5OrH5S7CbJzsOcXf+8aCTJzr1k+57vcgEYaEtJFvfOi13n5XSFbc12a6aHMwHAa7O39Ww/PLsa1/4gI2eSt35Rb//xtXrHFjR+gN+MTc7drHoIgGEiRgMAAAAAgD63vjC1H6mN5/ub1ty0AjDgdh7mfrFVbB4Vrm3eKdJZrXZGAOiR2yljtaQM1vbPi+mK2Zrt1nwAoGLf2Xp2NcmVKueBfnL6ai1v/0397pv/rvbTqmfhWLuVchvaYtWDAAwbMRoAAAAAAAyArmDtsHDt3coGA6Cntr8ulpJkdyX1jS+Kx0my+WUubn1VnEqEawAMpVtd55mu8xMxW7PdWuzVQAAMntnGxNU8GZ65Hgff8cblWt76RW3lrV/UH4+cTqPqeTjWVlJGaB9VPQjAsBKjAQAAAADAkFhfmJreOwrWADg0XFv9tLjS+bb8/dq875IBGEorKUO1pIzVureszXSd55vt1nIAGDpd4dn+z3vVTgT97a336xuNv6w9HPszW9B4JT5McnNscs6/xQEqJEYDAAAAAAD+oCtY23/tDteu9H4iAKpW7GRj50HxIEm27qa+/Y1wDQC+YynJ4t55sev8RMzWbLdmejgTAK/AbGNiOuV1MeEZvIDRd2q58J9qD8//h/rp2smcqnoeBsKtJDfGJufmn/lJAF47MRoAAAAAAPDc1hemzqe88SYRrAHwHZ21tDvLxaMkWfssJx6vFbvFbkYffVxcSpLOarJ5x3fUAJDkdspYLSmDtf3zYrpitma75WZbgB6YbUzsX/OazkF89m6FI8GxM3ImeesX9fYfX6t3Tl7MharnYWCspIzQflv1IAAcEKMBAAAAAACv1PrC1HjKm3aOCtfO9XwoAPrOYeHazsOcXf+8aCTJzr1k+57vswGgy62u80zXuTtmm2+2W8sB4EizjYn9LWfjOYjPPGAJXtLZv6jl7b+t3x37s9pPq56FgfNhkptjk3P+fQvQZ8RoAAAAAABAzwnWAHgROw9zv9gqNotOTqx+UuzuvSdcA4CnW8rBlrXFo87NdmsxAANoLzobT3mtaf9s2xm8AqPv1HLp1/X7bzZrZ2snc6rqeRg4t1JuQ7MlGKBPidEAAAAAAIC+tL4wdTVlrDae74dr71UzFQD9bj9c211JfeOL4nGSbH6Zi1tfFaeSZPNOkc5qtTMCQJ8SrwHHkugMeuONy7Wc+/e19tt/U6+PvOlhYrwWKykjtN9WPQgATydGAwAAAAAAji3BGgA/xPbXxVKSHBWurc37Ph0AnkK8BvTMbGNi/5rP/rWg6ZTXgq5UNxUMvpEzybnp+saF/1z7n3/0p7WfVD0PA+3DJDfHJueWqx4EgGcTowEAAAAAAANtfWFqeu94WLjmKdkAPFOxk42dB8WDJNm6m/r2N2W4tvppcaXzbfkZ4RoAPNPzxGvLzXZrvoczAX2kKzgbz8Gms/PxwCHoubN/Ucvbf1u/O/ZntZ9WPQsD71bKbWj+DQhwjIjRAAAAAACAofeUYG08nrANwAvoDtfWPsuJx2vFbrGb0UcfF5eSpLOabN7xPT0APIeVJPs3JS8fcU6z3Zrp7VjADzHbmBjPk9ddBGfQJ05freXtv6nfPfPntQu1kzlV9TwMvJWUEdpvqx4EgBcnRgMAAAAAAHiG9YWp/SdzJ8n03mt3uCZYA+CFddbS7iwXjxLhGgC8Ire6zvMpw7Xvnheb7dZiL4eCYdK13ey7r+Nx/QT6zug7tVz4T7WHZ6frIyOn06h6HobGb5J8MDY5t/zMTwLQl8RoAAAAAAAAr8AzgrWrSc5VMBYAA2LnYe4XW8Vm0cmJ1U+K3b33zq5/XjSSZOdesn3P9/8A8IK6t68lyUzXeXHvJ0mWm+1W9+dgaM02Jqb3jrbLwzH1xuVaLv7vAjQqcSvJ9bHJucWqBwHghxGjAQAAAAAA9Mj6wtR4ypuzjgrXBGsA/GCHhWubX+bi1lfFqaTcttZZrXZGADjmbudg09pyngzaurewpdluzfRuLHh5s42J/bis+5pF93vvVjQa8Aq8cbmWc/++1v7ja/XOyYu5UPU8DJ2llBHaTNWDAPBqiNEAAAAAAAD6yDOCtfd6PxEAg2z762IpSXZXUt/4onicCNcA4DX67ia2xRxsXzv0fzfbre7/Dc9ttjHRfV0hObi20P3+eGwzg4ElQKMPrCT5YGxy7mbVgwDwaonRAAAAAAAAjpn1han9J5OP5/vhmmANgNdiP1zbupv69jdluLb6aXGl8235+7V59x8AwGv23Zgt+c4mtj0z3/0PbWg7vmYbE+Mp//bf993/vX+NYP934jIYYgI0+sjvktwYm5z77r9TABgAYjQAAAAAAIAB9JRg7XySdysbDICBV+xkY+dB8SA5PFzrrJYb1wCAyt3O90O25JCYbc9h4du+5Wa79d1Qbugcso2s2/Qh743nybAse//9uVc2FDDwBGj0mVspI7Sh/3cBwCATowEAAAAAAAyp9YWp6b3jYeGaYA2A166zlnZnuXiUJGuf5cTjtWK32M3oo4+LS4lwDQAGzGGb3Z7X8g/4bw/TvU3sRYzH5jGgDwjQ6ENLKSO0j6oeBIDXT4wGAAAAAADAkQ4J1rrDNTfgAdAz++Fa0cmJ1U+K3STZeZiz658XjSTZuZds33MPBAAAMJgEaPSplSQfjE3O3ax6EAB6R4wGAAAAAADAS1tfmNoP1JJkeu9VsAZApXYe5n6xVWwK1wAAgONs9J1aGv9r7f656fqIAI0+9LuU29CWqx4EgN4SowEAAAAAAPBaHRGs7b9eTXKuxyMBwB9sf10sJcnuSuobXxSPk2Tzy1zc+qo4lSSbd4p0VqucEAAAGCaj79Ry4T/VHp6dro+MnE6j6nngELdSRmjzVQ8CQDXEaAAAAAAAAFRufWFqPOUmtaM2rQnWAKjcYeHa6qfFlc635e/X5t2DAQAAvLizf1HLW+/X757589qF2smcqnoeOMJSkutjk3MzVQ8CQLXEaAAAAAAAABwLhwRr3eHae9VMBQDfV+xkY+dB8SBJtu6mvv2NcA0AADgwciY5N13faPxl7eGp/0WARt9bSXJzbHLug6oHAaA/iNEAAAAAAAAYGOsLU/uR2ni+v2lNsAZA3+mspd1ZLh4lydpnOfF4rdgtdjP66OPiUpJ0VpPNO+7tAACA4+6Ny7Wc+/e1duM/1tb/6E9rP6l6HnhOv0nywdjk3HLVgwDQP8RoAAAAAAAADJWuYO2wcO3dygYDgGd4Vri2cy/Zvuc+EAAA6Benr9bS+A+1h2en6yMjp9Ooeh54Ab9LuQ1tsepBAOg/YjQAAAAAAAD4jvWFqem9o2ANgGNp52HuF1vFZtHJidVPit29986uf140EuEaAAC8DiNnknPT9Y2zf1F7eObPaxdqJ3Oq6pngBd1KGaHNVD0IAP1LjAYAAAAAAAAvoStY23/tDteu9H4iAHg5++Ha7krqG18Uj5Nk88tc3PqqOJUkm3eKdFarnREAAPrV6Du1vNmstRv/sbb+R39a+0nV88BLWkpyXYQGwPMQowEAAAAAAMBrsL4wdT5loJYI1gAYENtfF0tJclS4tjbvPhQAAAbbyJnk9FQtb71fvzs2UTszcjqNqmeCH2Ap5Sa031Y9CADHhxgNAAAAAP7/9u5uN6oDT/fwu/yFKTs2jBDQk56xpc5ILXVP4eppaUbKtLC0b2RfSvYdsO+A3AG5AzjIWRQ7pUTKAZFs5UPAoI1N7ML4g7UPVjkUwQ4B7Fr18TxSaS1sH7ynSPXTHwCgJj3B2mnh2mINswDgvZUHeXbwqHyUJM+/z8T+T1W4tvNluXT0c/U3wjUAAIaF62eMoO0kt5LcajTXtuoeA8BwEaMBAAAAAADAAOu0W8upLqkJ1gAYSSeFa+VhZp9+Xl5LkqOdZO++77cAANA/rp8x4v5PRGgAvAcxGgAAAAAAAAy5nmDt+NMbrt2sYxMAnIej3Tw52iqfJsnu15l6sVseCtcAADgLcytFFj4uHi6uTkxOX82VuvfAOfg0ySeN5tpG3UMAGG5iNAAAAAAAABgDnXZrJVWkthzBGgBj4OBxHpbPy73yKFM7X5SH3Z8tdL4pLyfJwYNk/4HvzQAAjKuZ60UW/1fxZOG/i52L/1ZcKaZzse5NcE7uJfnfIjQAzooYDQAAAAAAAEjySrB2Urh2o7ZhAHDO3hSu7d0vc7RT70YAAN7P5HyyuDrxbOEfxePGvxfzk3O5XPcmOGf3Ul1Cu1v3EABGixgNAAAAAAAA+K5OtCQAABL8SURBVN067dZq91WwBsBY2v+x3EySw+1MPPu2fJEke9/l6vMfyouJcA0AYFBMzidzrSIL/5j4cf4/iqnpK7lW9ybok81Ul9Du1j0EgNEkRgMAAAAAAADOTKfdOr6sliSr3WdvuLbU/1UAUI/jcO3595nY/6kK13a+LJeOfq5+v7vuezsAAGdpbqXIwsfFw8XVicnpq7lS9x7os81Ul9Bu1z0EgNEmRgMAAAAAAAD6SrAGAK8qD/Ls4FH5KDk5XDvaqS6uAQDwquP47IP/Kg4v/GvxYd17oCYiNAD6SowGAAAAAAAADJxOu7WcKkw7LVxb7PsoABgAR7t5crRVPk2S3a8z9WK3PCwPM/v08/JaIlwDAEab+AxesZ3kVqO59kndQwAYL2I0AAAAAAAAYCi9IVi72f9FADBYTgrXDh5nofNNeTlJDh4k+w98dwgAGEyT88lcq8jCPyZ+nPtrDmc+LFxSh8p2klupQrStuscAMH7EaAAAAAAAAMDI6rRbK6liteW8Hq4J1gCg6+BxHpbPy73yKFM7X5SH3Z8J1wCAvpm5XmT+78WzRrP4f/P/UUxNX8m1ujfBgBGhATAQxGgAAAAAAADAWPuNYO1Skhu1DQOAAbX/Y7mZJIfbmXj2bfkiSfa+y9XnP5QXk2TvfpmjnToXAgDDYPajIh/8Z/FkrlnsNP69mJ+cy+W6N8GAEqEBMFDEaAAAAAAAAABv0Gm3VruvJ4VrgjUAOMVJ4drOl+XS0c/V73fXfXcJAMbB5HwVny18XDz84L+Kw5k/FP9UTOdi3btgwInQABhIYjQAAAAAAACAM/AbwdpykqU6NgHAsCgP8uzgUfkoSZ5/n4n9n4RrADDMZj8q0vhz8WzhH8Xj2Y+KmekruVb3Jhgy/zfJJyI0AAaRGA0AAAAAAACgDzrt1qVUoVqSrHafveGaYA0AfofecG3360y92C0Py8PMPv28vJYkRzvJ3n3fiQKAfnH1DM7Up6kitI26hwDAacRoAAAAAAAAAAPiDcHaSpLFGmYBwNA62s2To63yaXJyuHbwINl/4PtTAPA25laKNP5SPFn472LnwlIxPzmXy3VvghEgQgNgaIjRAAAAAAAAAIZIp91aTnVJ7bRwTbAGAO/g4HEels/LvfIoUztflIfdny10vikvJ8I1AMbTcXg21yx2Zj8qZqav5Frdm2CEbCe5ExEaAENGjAYAAAAAAAAwYt4QrN3s/yIAGC0nhWt73+Xq8x/Ki0myd7/M0U69GwHgbQnPoG+2k9xKcqvRXNuqewwAvC0xGgAAAAAAAMAY6rRbK6liteW8Hq4J1gDgjOz/WG4myeF2Jp59W75IhGsA1E94BrUQoQEwEsRoAAAAAAAAAJzohGDt+HMpyY2aZgHASCoP8uzgUfkoSZ5/n4n9n6pwbefLcuno5+pvdtd91wuAtzNzvcj09WTh4+Jh46/F/oWlYn5yLpfr3gVjRoQGwEgRowEAAAAAAADwzjrt1mr39aRLa4I1ADgHJ4Vr5WFmn35eXkuSo53q4hoA42VupciFPxbPZv+UR/N/L6Zm/lD8UzGdi3XvgjEmQgNgJInRAAAAAAAAADhXJwRrveHaUj2rAGA8HO3mydFW+TRJdr/O1Ivd8lC4BjDcZq4Xmf23ZP5vxebFPxcT01eLmekruVb3LuAXm6kitNsiNABGkRgNAAAAAAAAgFp12q3jQC1JVrtPwRoA9NlxuFYeZWrni/IwSQ4eZ6HzTXk5SQ4eJPsPfN8MoF9mrheZvp4sfFw8vPCvxf6Ff8mLmQ8L/z+CwbWZ5JNGc+123UMA4DyJ0QAAAAAAAAAYeKcEa8fPlSSLfZ4EAGPt4HEels/LvdPCtb37ZY526t0IMCxmPyoy84fq0tnMPxcTojMYOiI0AMaKGA0AAAAAAACAkdBpt5ZTXVI77dKaYA0AarD/Y7mZJIfbmXj2bfkiSfa+y9XnP5QXE+EaMB4m56vo7MIfi2ezf8qj+b8XUxONYmr6Sq7VvQ14Z/dSRWh36x4CAP0kRgMAAAAAAABgbJwQrPWGazfrWQUAHDsO155/n4n9n6pwbefLcuno5+r3u+u+7wYMtrmVIpMfvHrlbPpqcbWYzsW6twFnRoQGwFgTowEAAAAAAABAj067dRypLef1S2uCNQAYAOVBnh08Kh8lwjWgv44vnAnOYCx9mipC26h7CADUSYwGAAAAAAAAAG+pJ1g7KVy7UdswAOA1R7t5crRVPk2S3a8z9WK3PCwPM/v08/JakhztJHv3fY8OeGn2oyKT80njL8WT6St52vhLcTQ5n8mZD4ulurcBfbed5HaSWyI0AKiI0QAAAAAAAADgHHTardXuq2ANAIbESeHaweMsdL4pLyfJwYNk/4Hv3MGw+3VsdvHPxcTUoutmwCu2k9xKFaFt1T0GAAaJGA0AAAAAAAAAatITrB0/e8M1VxcAYIAdPM7D8nm5Vx5laueL8rD7M+Ea1GzmepHp68nkB8n834rNJPngP4vqdy6bAW+2meSTRnPtdt1DAGBQidEAAAAAAAAAYEB12q1LqQK1RLAGAENr/8dyM0kOtzPx7NvyRZLsfZerz38oLybJ3v0yRzt1LoTBNzlfXTRLXl40m5grpub+msNEaAa8t3upIrS7dQ8BgEEnRgMAAAAAAACAIdYTrJ0Wri3WMAsAeEdvCtd2133nj9Ey+1GRyfnqfeHj4mExlb3eyGz6anG1mM7FOjcCI+3TJLcazbX1uocAwLAQowEAAAAAAADAiOu0W8upLqkJ1gBgRJQHeXbwqHyUJM+/z8T+T1W4tvNluXT0c/U3wjXq0BuXXfhj8Wz2T3mUJDP/XExc+Je8SJLJS8XC5Fwu17cSGHPbSW4lud1orm3UvAUAho4YDQAAAAAAAADoDdaOP73h2s06NgEAZ6M8TOfgYfk/yasX1w4eZ6HzTflLECReo9fcSvHL++QHyfzfis3jf/eGZUky82Gx1Od5AO9iM8knSe40mmtbNW8BgKElRgMAAAAAAAAAfpdOu7WSKlJbjmANAEba/o/lL+FR7+W18jCzTz8vrx3/7mgn2bvve4iDpDciO7bwcfGwmMre8b9/HZMlgjJgpN1LcqvRXLtT9xAAGAViNAAAAAAAAADgzHTardXu60nh2o1aRgEAfXO0mydHW+XT43+XR5na+aI87P2bX19k++XnD5L9B6P1ncbJ+WT2o9fjsFf+5ldXx35t/u/FVDGZw1//fPpqcbWYzsUzmAkwiraT3EnySaO5tlHzFgAYKWI0AAAAAAAAAKCvBGsAwLvovdZWFwEYwMDbTHIrye1Gc22r7jEAMIrEaAAAAAAAAADAQOm0W5dShWpJstp99oZrS/1fBQAAwAC7l+RWo7l2p+4hADDqxGgAAAAAAAAAwNARrAEAAIy97SR3knzSaK5t1LwFAMaGGA0AAAAAAAAAGEmddms5VZh2Wri22PdRAAAAvK/NJLeS3G4017bqHgMA40aMBgAAAAAAAACMLcEaAADA0Pgsya1Gc+1u3UMAYJyJ0QAAAAAAAAAAfkOn3VpJFast5/Vw7WY9qwAAAMbCZpLbqa6gbdQ7BQBIxGgAAAAAAAAAAO9NsAYAAHCm7qUK0G7XPQQAeJUYDQAAAAAAAACgDzrt1mr39aRw7UYtowAAAAbHdqoraLdcQQOAwSVGAwAAAAAAAAAYEL8RrC0nWapjEwAAwDlzBQ0AhogYDQAAAAAAAABgSHTarUupQrUkWe0+e8M1wRoAADAMXEEDgCElRgMAAAAAAAAAGCFvCNZWkizWMAsAACBxBQ0Ahp4YDQAAAAAAAABgzHTareVUl9ROC9cEawAAwFnZTHUF7bYraAAw/MRoAAAAAAAAAAC85g3B2s3+LwIAAIbMp0nuNJprd+oeAgCcHTEaAAAAAAAAAADvpNNuraSK1ZbzergmWAMAgPHzVV5eQduqeQsAcA7EaAAAAAAAAAAAnJvfCNYuJblR2zAAAOCsbOdlgLZe8xYA4JyJ0QAAAAAAAAAAqFWn3Vrtvp4UrgnWAABgMH2WKkC7U/cQAKB/xGgAAAAAAAAAAAy83wjWlpMs1bEJAADG0Fd5eQVtq+YtAEANxGgAAAAAAAAAAAy9Trt1KVWoliSr3WdvuCZYAwCAd7OZ5E6SW43m2kbNWwCAmonRAAAAAAAAAAAYC6cEa8fPlSSLfZ4EAACDajtVgHa70Vy7W/MWAGCAiNEAAAAAAAAAAKCr024tp7qkdtqlNcEaAACj7LNUAdqduocAAINJjAYAAAAAAAAAAG/hhGCtN1y7Wc8qAAB4Z5+luoJ2p9Fc26p7DAAw2MRoAAAAAAAAAABwxjrt1nGktpzXL60J1gAAqNtXSW6nCtA26p0CAAwTMRoAAAAAAAAAANTghGDt+HMpyY2aZgEAMLoEaADAexOjAQAAAAAAAADAgOq0W6vd15MurQnWAAB4EwEaAHCmxGgAAAAAAAAAADDETgjWesO1pXpWAQBQIwEaAHBuxGgAAAAAAAAAADDCOu3WcaCWJKvdp2ANAGC0CNAAgL4QowEAAAAAAAAAwJg7JVg7fq4kWezzJAAA3kyABgD0nRgNAAAAAAAAAAB4o067tZzqktppl9YEawAA5+9ekjsRoAEANRGjAQAAAAAAAAAAZ6InWDv+9IZrN+vYBAAwAj7LywBtq+4xAMB4E6MBAAAAAAAAAAB902m3VlJFassRrAEAnGQ73fgsyV0BGgAwSMRoAAAAAAAAAADAQDkhWDv+XEpyo6ZZAADnaTMvr5/drXkLAMCpxGgAAAAAAAAAAMDQ6bRbq93Xky6tCdYAgGFwLy8DtI2atwAA/C5iNAAAAAAAAAAAYCT1BGvHz95wban/iwCAMbedKj67mypA26p3DgDA2xOjAQAAAAAAAAAAY6nTbl1KFaglgjUA4Hx8lSo+u91orq3XvAUA4L2J0QAAAAAAAAAAAE7RE6ydFq4t1jALABhcvdfP7jaaaxu1rgEAOGNiNAAAAAAAAAAAgPfQabeWU11SE6wBwHj6KlWAdsf1MwBg1InRAAAAAAAAAAAAzllPsHb86Q3XbtaxCQB4Z5upLp/dSXX9bKveOQAA/SNGAwAAAAAAAAAAGACddmslVaS2HMEaAAyS7VTx2d1U18826hwDAFAnMRoAAAAAAAAAAMCQ6AnWTgrXbtQ2DABGz728jM/Wa94CADAwxGgAAAAAAAAAAAAjpNNurXZfBWsA8Psdx2d3G821u/VOAQAYXGI0AAAAAAAAAACAMdMTrB0/e8O1pf4vAoC+E58BALwDMRoAAAAAAAAAAACv6LRbl1IFaolgDYDRID4DADgDYjQAAAAAAAAAAADeWqfdWk4Vpp0Wri32fRQAVLZThWfrEZ8BAJwpMRoAAAAAAAAAAADnQrAGQJ9s5tX4bL3eOQAAo0uMBgAAAAAAAAAAQG067dZKqlhtOa+HazfrWQXAgLuXbniWKj7bqncOAMD4EKMBAAAAAAAAAAAw0ARrAGOt9+rZeqO5drfWNQAAY06MBgAAAAAAAAAAwNDrtFur3deTwrUbtYwC4G1tp+fiWar4zNUzAIABIkYDAAAAAAAAAABgLAjWAAZKb3h2fPVso85BAAC8mRgNAAAAAAAAAAAAknTarUupQrUkWe0+e8O1pf6vAhgJwjMAgBEhRgMAAAAAAAAAAIDfSbAG8EabSTYiPAMAGEliNAAAAAAAAAAAADhDnXZrOVWYdlq4ttj3UQDn416q8Gw9L8OzrVoXAQBwrsRoAAAAAAAAAAAA0GeCNWDIfJVXo7ONRnNtvdZFAADUQowGAAAAAAAAAAAAA6jTbq2kitWW83q4drOeVcCIE50BAPCbxGgAAAAAAAAAAAAwpARrwDu6lyo624joDACAtyBGAwAAAAAAAAAAgBHWabdWu68nhWs3ahkF9MNXSbaS3O0+15OsN5prW3WOAgBguInRAAAAAAAAAAAAYMz9RrC2nGSpjk3A73IvL0Oz4+dGo7m2UecoAABGlxgNAAAAAAAAAAAA+E2ddutSqlAtSVa7z+NwzYU1OD/H1816Y7OtRnPtbp2jAAAYX2I0AAAAAAAAAAAA4Ez0XFhb7n6Sl/HaclxZg16bSTbyMjJLkrtJIjYDAGBQidEAAAAAAAAAAACAvuqJ1novri3nZcC2HOEaw+1e97nR/STd0CzJeqO5ttXnPQAAcCbEaAAAAAAAAAAAAMDA6rRbvcFauu+Xuu8n/W6xT9MYP1+lumKWvAzLeq+abTSaaxt93gQAAH0lRgMAAAAAAAAAAABGUs8FtuT1cO3X/06Sm+e9iYFxr+d9Iy+vlyUvQ7M0mmt3AwAA/EKMBgAAAAAAAAAAAHCCTru1nGT5Vz/uvcx27KS/S8Rt52Ezr4ZjyesxWdITlHWtN5prWwEAAN6LGA0AAAAAAAAAAACgj06J3HqdFLyd5KTrboNgK8n67/i7jbwekf3yu0Zz7bTfAQAANfn//iUo6WV+iP8AAAAASUVORK5CYII=')

        self.image_data = QByteArray.fromBase64(base64_data.encode('utf-8'))

        self.pixmap = QPixmap()
        self.pixmap.loadFromData(self.image_data)
        self.pixmap = self.pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        self.image_label = QLabel(self)
        self.image_label.setPixmap(self.pixmap)
        self.image_label.move(80, 0) 

        self.image_label.resize(400, 160)

        self.setWindowTitle('Ride Sim')

        self.title = QLabel('Ride Sim Scan', self)
        self.title.setFont(QFont('Arial', 30, QFont.Bold))
        self.title.move(800,40)
        self.title.show()

        self.label = QLabel('Number of Simulations in Sweep:', self)
        self.label.setFont(QFont('Arial', 12))
        self.label.move(1150, 250)
        self.label.show()

        self.spinbox = QSpinBox(self)
        self.spinbox.setRange(1, 15)
        self.spinbox.setValue(1)
        self.spinbox.setFixedWidth(60)
        self.spinbox.valueChanged.connect(self.refresh)
        self.spinbox.move(1450, 250)
        self.spinbox.show()

        self.car_button = QPushButton('Load Car', self)
        self.car_button.setFixedWidth(150)
        self.car_button.setFont(QFont('Arial', 12))
        self.car_button.clicked.connect(self.load_json)
        self.car_button.move(150, 200)
        self.car_button.show()

        self.track_button = QPushButton('Load Track', self)
        self.track_button.setFixedWidth(150)
        self.track_button.setFont(QFont('Arial', 12))
        self.track_button.clicked.connect(self.load_track)
        self.track_button.move(325, 200)
        self.track_button.show()
        self.car_data = None

        self.visualization_button = QPushButton('Sim Visualization', self)
        self.visualization_button.setFixedWidth(220)
        self.visualization_button.setFont(QFont('Arial',12))
        self.visualization_button.clicked.connect(self.sim_visualization)
        self.visualization_button.move(500, 200)

        self.run_button = QPushButton('Run Sweep', self)
        self.run_button.setFixedWidth(150)
        self.run_button.setFont(QFont('Arial', 12))
        self.run_button.clicked.connect(self.run_sweep)
        self.run_button.move(745, 200)
        self.run_button.show()

        self.label = QLabel('Simulation Sweep Name:', self)
        self.label.setFont(QFont('Arial', 12))
        self.label.move(1156, 200)
        self.label.show()

        self.sweep_name = QLineEdit('', self)
        self.sweep_name.move(1392, 200)
        self.sweep_name.setStyleSheet('border-radius: 5px; padding: 3px; background-color: #2F2F2F; color: #ddd;')
        self.sweep_name.show()

        self.button_1 = []
        self.button_2 = []

        self.fixed_header = QLabel('Car Parameters', self)
        self.fixed_header.setFont(QFont('Arial', 16))
        self.fixed_header.move(190, 260)

        self.tree_widget = QTreeWidget(self)
        self.tree_widget.setHeaderHidden(True)
        self.tree_widget.move(200, 300)
        self.tree_widget.setFont(QFont('Arial', 12))
        self.tree_widget.setFixedWidth(700)
        self.tree_widget.setFixedHeight(500)
        self.tree_widget.setColumnCount(2)
        self.tree_widget.setColumnWidth(0, 350)
        self.tree_widget.setColumnWidth(1, 350)

        self.chassis_item = QTreeWidgetItem(self.tree_widget, ['Chassis'])
        self.carmass_item = QTreeWidgetItem(self.chassis_item, ['carRunningMass'])
        self.suspension_item = QTreeWidgetItem(self.tree_widget, ['Suspension'])
        self.suspension_front_item = QTreeWidgetItem(self.suspension_item, ['Front'])
        self.internal_front_suspension_item = QTreeWidgetItem(self.suspension_front_item, ['Internal'])
        self.spring_front_item = QTreeWidgetItem(self.internal_front_suspension_item, ['Spring'])
        self.bumpstop_front_item = QTreeWidgetItem(self.internal_front_suspension_item, ['BumpStop'])
        self.antiroll_front_item = QTreeWidgetItem(self.internal_front_suspension_item, ['antiRollBar'])
        self.suspension_rear_item = QTreeWidgetItem(self.suspension_item, ['Rear'])
        self.internal_rear_suspension_item = QTreeWidgetItem(self.suspension_rear_item, ['Internal'])
        self.spring_rear_item = QTreeWidgetItem(self.internal_rear_suspension_item, ['Spring'])
        self.bumpstop_rear_item = QTreeWidgetItem(self.internal_rear_suspension_item, ['BumpStop'])
        self.antiroll_rear_item = QTreeWidgetItem(self.internal_rear_suspension_item, ['antiRollBar'])
        self.aero_item = QTreeWidgetItem(self.tree_widget, ['Aero'])
        self.aero_front_lift_item = QTreeWidgetItem(self.aero_item, ['CLiftBodyF'])
        self.aero_rear_lift_item = QTreeWidgetItem(self.aero_item, ['CLiftBodyR'])
        self.aero_drag_item = QTreeWidgetItem(self.aero_item, ['CDragBody'])
        self.aero_flaps = QTreeWidgetItem(self.aero_item, ['Flaps'])
        self.tyre_item = QTreeWidgetItem(self.tree_widget, ['Tyres'])
        self.tyre_front_item = QTreeWidgetItem(self.tyre_item, ['Front'])
        self.tyre_rear_item = QTreeWidgetItem(self.tyre_item, ['Rear'])

        self.line_edits = {}
        self.add_editable_child(self.chassis_item, 'hRideFSetup')
        self.add_editable_child(self.chassis_item, 'hRideRSetup')
        self.add_editable_child(self.carmass_item, 'mCar')
        self.add_editable_child(self.carmass_item, 'rWeightBalF')
        self.add_editable_child(self.chassis_item, 'Icar')
        self.add_editable_child(self.chassis_item, 'mHubF')
        self.add_editable_child(self.chassis_item, 'mHubR')
        self.add_editable_child(self.chassis_item, 'kVerticalSuspensionComplianceF')
        self.add_editable_child(self.chassis_item, 'kVerticalSuspensionComplianceR')
        self.add_editable_child(self.chassis_item, 'zCoG')
        self.add_editable_child(self.chassis_item, 'Wheelbase')
        self.add_editable_child(self.chassis_item, 'Front Track')
        self.add_editable_child(self.chassis_item, 'Rear Track')
        self.add_editable_child(self.spring_front_item, 'kSpring')
        self.add_editable_child(self.bumpstop_front_item, 'Displacement')
        self.add_editable_child(self.bumpstop_front_item, 'Force')
        self.add_editable_child(self.bumpstop_front_item, 'xFreeGap')
        self.add_editable_child(self.antiroll_front_item, 'kAntiRollBar')
        self.add_editable_child(self.spring_rear_item, 'kSpring')
        self.add_editable_child(self.bumpstop_rear_item, 'Displacement')
        self.add_editable_child(self.bumpstop_rear_item, 'Force')
        self.add_editable_child(self.bumpstop_rear_item, 'xFreeGap')
        self.add_editable_child(self.antiroll_rear_item, 'kAntiRollBar')
        self.add_editable_child(self.aero_front_lift_item, 'Const')
        self.add_editable_child(self.aero_front_lift_item, 'hRideR')
        self.add_editable_child(self.aero_front_lift_item, 'hRideR*hRideR')
        self.add_editable_child(self.aero_front_lift_item, 'hRideR*hRideR*hRideR')
        self.add_editable_child(self.aero_front_lift_item, 'hRideF')
        self.add_editable_child(self.aero_front_lift_item, 'hRideF*hRideF')
        self.add_editable_child(self.aero_front_lift_item, 'hRideF*hRideF*hRideF')
        self.add_editable_child(self.aero_front_lift_item, 'hRideF*hRideR')
        self.add_editable_child(self.aero_front_lift_item, 'hRideF*hRideR*hRideR')
        self.add_editable_child(self.aero_front_lift_item, 'hRideF*hRideF*hRideR')
        self.add_editable_child(self.aero_front_lift_item, 'hRideF*hRideF*hRideR*hRideR')
        self.add_editable_child(self.aero_front_lift_item, 'aFlapF')
        self.add_editable_child(self.aero_front_lift_item, 'aFlapF*aFlapF')
        self.add_editable_child(self.aero_front_lift_item, 'aFlapR')
        self.add_editable_child(self.aero_front_lift_item, 'aFlapR*aFlapR')
        self.add_editable_child(self.aero_rear_lift_item, 'Const')
        self.add_editable_child(self.aero_rear_lift_item, 'hRideR')
        self.add_editable_child(self.aero_rear_lift_item, 'hRideR*hRideR')
        self.add_editable_child(self.aero_rear_lift_item, 'hRideR*hRideR*hRideR')
        self.add_editable_child(self.aero_rear_lift_item, 'hRideF')
        self.add_editable_child(self.aero_rear_lift_item, 'hRideF*hRideF')
        self.add_editable_child(self.aero_rear_lift_item, 'hRideF*hRideF*hRideF')
        self.add_editable_child(self.aero_rear_lift_item, 'hRideF*hRideR')
        self.add_editable_child(self.aero_rear_lift_item, 'hRideF*hRideR*hRideR')
        self.add_editable_child(self.aero_rear_lift_item, 'hRideF*hRideF*hRideR')
        self.add_editable_child(self.aero_rear_lift_item, 'hRideF*hRideF*hRideR*hRideR')
        self.add_editable_child(self.aero_rear_lift_item, 'aFlapF')
        self.add_editable_child(self.aero_rear_lift_item, 'aFlapF*aFlapF')
        self.add_editable_child(self.aero_rear_lift_item, 'aFlapR')
        self.add_editable_child(self.aero_rear_lift_item, 'aFlapR*aFlapR')
        self.add_editable_child(self.aero_drag_item, 'Const')
        self.add_editable_child(self.aero_drag_item, 'hRideR')
        self.add_editable_child(self.aero_drag_item, 'hRideR*hRideR')
        self.add_editable_child(self.aero_drag_item, 'hRideR*hRideR*hRideR')
        self.add_editable_child(self.aero_drag_item, 'hRideF')
        self.add_editable_child(self.aero_drag_item, 'hRideF*hRideF')
        self.add_editable_child(self.aero_drag_item, 'hRideF*hRideF*hRideF')
        self.add_editable_child(self.aero_drag_item, 'hRideF*hRideR')
        self.add_editable_child(self.aero_drag_item, 'hRideF*hRideR*hRideR')
        self.add_editable_child(self.aero_drag_item, 'hRideF*hRideF*hRideR')
        self.add_editable_child(self.aero_drag_item, 'hRideF*hRideF*hRideR*hRideR')
        self.add_editable_child(self.aero_drag_item, 'aFlapF')
        self.add_editable_child(self.aero_drag_item, 'aFlapF*aFlapF')
        self.add_editable_child(self.aero_drag_item, 'aFlapR')
        self.add_editable_child(self.aero_drag_item, 'aFlapR*aFlapR')
        self.add_editable_child(self.aero_flaps, 'aFlapF')
        self.add_editable_child(self.aero_flaps, 'aFlapR')
        self.add_editable_child(self.tyre_front_item, 'kVerticalTyre')
        self.add_editable_child(self.tyre_rear_item, 'kVerticalTyre')

        self.set_dark_mode()

        self.showMaximized()

    def add_editable_child(self, parent_item, name):

        category = parent_item.text(0)  

        full_category = category
        current_parent = parent_item.parent()
        while current_parent:
            full_category = current_parent.text(0) + '_' + full_category
            current_parent = current_parent.parent()

        unique_name = f'{full_category}_{name}'

        child_item = QTreeWidgetItem([name])
        parent_item.addChild(child_item)

        line_edit = QLineEdit()
        line_edit.setFixedWidth(200)
        line_edit.setStyleSheet('border-radius: 5px; padding: 3px; background-color: #2F2F2F; color: #ddd;')
        line_edit.setAlignment(Qt.AlignRight)
        self.line_edits[unique_name] = line_edit

        parent_item.treeWidget().setItemWidget(child_item, 1, line_edit)
        
        child_item.setText(0, name)

    def load_json(self):
        try:
            archive, _ = QFileDialog.getOpenFileName(self, 'Set-up JSON', '', 'JSON Files (*.json)')
            if archive:
                with open(archive, 'r') as f:
                    self.car_data = json.load(f)

                QMessageBox.information(self, 'Info', 'Car loaded successfully.')
            else:
                raise FileNotFoundError('No files were selected.')
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Error loading JSON: {str(e)}')

        self.set_line_edit_value('Chassis', 'hRideFSetup', self.car_data.get('config').get('chassis', {}).get('hRideFSetup', ''))
        self.set_line_edit_value('Chassis', 'hRideRSetup', self.car_data.get('config').get('chassis', {}).get('hRideRSetup', ''))
        self.set_line_edit_value('Chassis_carRunningMass', 'mCar', self.car_data.get('config').get('chassis', {}).get('carRunningMass').get('mCar', ''))
        self.set_line_edit_value('Chassis_carRunningMass', 'rWeightBalF', self.car_data.get('config').get('chassis', {}).get('carRunningMass').get('rWeightBalF', ''))
        self.set_line_edit_value('Chassis', 'Icar', self.car_data.get('config').get('chassis', {}).get('ICar', ''))
        self.set_line_edit_value('Chassis', 'mHubF', self.car_data.get('config').get('chassis', {}).get('mHubF', ''))
        self.set_line_edit_value('Chassis', 'mHubR', self.car_data.get('config').get('chassis', {}).get('mHubR', ''))
        self.set_line_edit_value('Chassis', 'kVerticalSuspensionComplianceF', self.car_data.get('config').get('chassis', {}).get('kVerticalSuspensionComplianceF', ''))
        self.set_line_edit_value('Chassis', 'kVerticalSuspensionComplianceR', self.car_data.get('config').get('chassis', {}).get('kVerticalSuspensionComplianceR', ''))
        self.set_line_edit_value('Chassis', 'zCoG', self.car_data.get('config').get('chassis', {}).get('zCoG', ''))
        self.set_line_edit_value('Chassis', 'Wheelbase', -self.car_data.get('config', {}).get('suspension', {}).get('rear', {}).get('external', {}).get('pickUpPts', {}).get('rAxleC', ['', '', ''])[0])
        self.set_line_edit_value('Chassis', 'Front Track', 2*self.car_data.get('config', {}).get('suspension', {}).get('front', {}).get('external', {}).get('pickUpPts', {}).get('rAxleC', ['', '', ''])[1])
        self.set_line_edit_value('Chassis', 'Rear Track', 2*self.car_data.get('config', {}).get('suspension', {}).get('rear', {}).get('external', {}).get('pickUpPts', {}).get('rAxleC', ['', '', ''])[1])
        self.set_line_edit_value('Suspension_Front_Internal_Spring', 'kSpring', self.car_data.get('config').get('suspension', {}).get('front', {}).get('internal', {}).get('spring', {}).get('kSpring', ''))
        self.set_line_edit_value('Suspension_Front_Internal_BumpStop', 'Force', self.car_data.get('config').get('suspension', {}).get('front', {}).get('internal', {}).get('bumpStop', {}).get('FData', ''))
        self.set_line_edit_value('Suspension_Front_Internal_BumpStop', 'Displacement', self.car_data.get('config').get('suspension', {}).get('front', {}).get('internal', {}).get('bumpStop', {}).get('xData', ''))
        self.set_line_edit_value('Suspension_Front_Internal_BumpStop', 'xFreeGap', self.car_data.get('config').get('suspension', {}).get('front', {}).get('internal', {}).get('bumpStop', {}).get('xFreeGap', ''))
        self.set_line_edit_value('Suspension_Front_Internal_antiRollBar', 'kAntiRollBar', self.car_data.get('config').get('suspension', {}).get('front', {}).get('internal', {}).get('antiRollBar', {}).get('kAntiRollBar', ''))
        self.set_line_edit_value('Suspension_Rear_Internal_Spring', 'kSpring', self.car_data.get('config').get('suspension', {}).get('rear', {}).get('internal', {}).get('spring', {}).get('kSpring', ''))
        self.set_line_edit_value('Suspension_Rear_Internal_BumpStop', 'Force', self.car_data.get('config').get('suspension', {}).get('rear', {}).get('internal', {}).get('bumpStop', {}).get('FData', ''))
        self.set_line_edit_value('Suspension_Rear_Internal_BumpStop', 'Displacement', self.car_data.get('config').get('suspension', {}).get('rear', {}).get('internal', {}).get('bumpStop', {}).get('xData', ''))
        self.set_line_edit_value('Suspension_Rear_Internal_BumpStop', 'xFreeGap', self.car_data.get('config').get('suspension', {}).get('rear', {}).get('internal', {}).get('bumpStop', {}).get('xFreeGap', ''))
        self.set_line_edit_value('Suspension_Rear_Internal_antiRollBar', 'kAntiRollBar', self.car_data.get('config').get('suspension', {}).get('rear', {}).get('internal', {}).get('antiRollBar', {}).get('kAntiRollBar', ''))
        self.set_line_edit_value('Aero_CLiftBodyF', 'Const', self.car_data['config']['aero']['PolynomialCLiftBodyFDefinition'][0]['coefficient'])
        self.set_line_edit_value('Aero_CLiftBodyF', 'hRideR', self.car_data['config']['aero']['PolynomialCLiftBodyFDefinition'][1]['coefficient'])
        self.set_line_edit_value('Aero_CLiftBodyF', 'hRideR*hRideR', self.car_data['config']['aero']['PolynomialCLiftBodyFDefinition'][2]['coefficient'])
        self.set_line_edit_value('Aero_CLiftBodyF', 'hRideR*hRideR*hRideR', self.car_data['config']['aero']['PolynomialCLiftBodyFDefinition'][3]['coefficient'])
        self.set_line_edit_value('Aero_CLiftBodyF', 'hRideF', self.car_data['config']['aero']['PolynomialCLiftBodyFDefinition'][4]['coefficient'])
        self.set_line_edit_value('Aero_CLiftBodyF', 'hRideF*hRideF', self.car_data['config']['aero']['PolynomialCLiftBodyFDefinition'][7]['coefficient'])
        self.set_line_edit_value('Aero_CLiftBodyF', 'hRideF*hRideF*hRideF', self.car_data['config']['aero']['PolynomialCLiftBodyFDefinition'][10]['coefficient'])
        self.set_line_edit_value('Aero_CLiftBodyF', 'hRideF*hRideR', self.car_data['config']['aero']['PolynomialCLiftBodyFDefinition'][5]['coefficient'])
        self.set_line_edit_value('Aero_CLiftBodyF', 'hRideF*hRideR*hRideR', self.car_data['config']['aero']['PolynomialCLiftBodyFDefinition'][6]['coefficient'])
        self.set_line_edit_value('Aero_CLiftBodyF', 'hRideF*hRideF*hRideR', self.car_data['config']['aero']['PolynomialCLiftBodyFDefinition'][8]['coefficient'])
        self.set_line_edit_value('Aero_CLiftBodyF', 'hRideF*hRideF*hRideR*hRideR', self.car_data['config']['aero']['PolynomialCLiftBodyFDefinition'][9]['coefficient'])
        self.set_line_edit_value('Aero_CLiftBodyF', 'aFlapF', self.car_data['config']['aero']['PolynomialCLiftBodyFDefinition'][11]['coefficient'])
        self.set_line_edit_value('Aero_CLiftBodyF', 'aFlapF*aFlapF', self.car_data['config']['aero']['PolynomialCLiftBodyFDefinition'][12]['coefficient'])
        self.set_line_edit_value('Aero_CLiftBodyF', 'aFlapR', self.car_data['config']['aero']['PolynomialCLiftBodyFDefinition'][13]['coefficient'])
        self.set_line_edit_value('Aero_CLiftBodyF', 'aFlapR*aFlapR', self.car_data['config']['aero']['PolynomialCLiftBodyFDefinition'][14]['coefficient'])
        self.set_line_edit_value('Aero_CLiftBodyR', 'Const', self.car_data['config']['aero']['PolynomialCLiftBodyFDefinition'][0]['coefficient'])
        self.set_line_edit_value('Aero_CLiftBodyR', 'hRideR', self.car_data['config']['aero']['PolynomialCLiftBodyFDefinition'][1]['coefficient'])
        self.set_line_edit_value('Aero_CLiftBodyR', 'hRideR*hRideR', self.car_data['config']['aero']['PolynomialCLiftBodyFDefinition'][2]['coefficient'])
        self.set_line_edit_value('Aero_CLiftBodyR', 'hRideR*hRideR*hRideR', self.car_data['config']['aero']['PolynomialCLiftBodyFDefinition'][3]['coefficient'])
        self.set_line_edit_value('Aero_CLiftBodyR', 'hRideF', self.car_data['config']['aero']['PolynomialCLiftBodyFDefinition'][4]['coefficient'])
        self.set_line_edit_value('Aero_CLiftBodyR', 'hRideF*hRideF', self.car_data['config']['aero']['PolynomialCLiftBodyFDefinition'][7]['coefficient'])
        self.set_line_edit_value('Aero_CLiftBodyR', 'hRideF*hRideF*hRideF', self.car_data['config']['aero']['PolynomialCLiftBodyFDefinition'][10]['coefficient'])
        self.set_line_edit_value('Aero_CLiftBodyR', 'hRideF*hRideR', self.car_data['config']['aero']['PolynomialCLiftBodyFDefinition'][5]['coefficient'])
        self.set_line_edit_value('Aero_CLiftBodyR', 'hRideF*hRideR*hRideR', self.car_data['config']['aero']['PolynomialCLiftBodyFDefinition'][6]['coefficient'])
        self.set_line_edit_value('Aero_CLiftBodyR', 'hRideF*hRideF*hRideR', self.car_data['config']['aero']['PolynomialCLiftBodyFDefinition'][8]['coefficient'])
        self.set_line_edit_value('Aero_CLiftBodyR', 'hRideF*hRideF*hRideR*hRideR', self.car_data['config']['aero']['PolynomialCLiftBodyFDefinition'][9]['coefficient'])
        self.set_line_edit_value('Aero_CLiftBodyR', 'aFlapF', self.car_data['config']['aero']['PolynomialCLiftBodyFDefinition'][11]['coefficient'])
        self.set_line_edit_value('Aero_CLiftBodyR', 'aFlapF*aFlapF', self.car_data['config']['aero']['PolynomialCLiftBodyFDefinition'][12]['coefficient'])
        self.set_line_edit_value('Aero_CLiftBodyR', 'aFlapR', self.car_data['config']['aero']['PolynomialCLiftBodyFDefinition'][13]['coefficient'])
        self.set_line_edit_value('Aero_CLiftBodyR', 'aFlapR*aFlapR', self.car_data['config']['aero']['PolynomialCLiftBodyFDefinition'][14]['coefficient'])
        self.set_line_edit_value('Aero_CDragBody', 'Const', self.car_data['config']['aero']['PolynomialCLiftBodyFDefinition'][0]['coefficient'])
        self.set_line_edit_value('Aero_CDragBody', 'hRideR', self.car_data['config']['aero']['PolynomialCLiftBodyFDefinition'][1]['coefficient'])
        self.set_line_edit_value('Aero_CDragBody', 'hRideR*hRideR', self.car_data['config']['aero']['PolynomialCLiftBodyFDefinition'][2]['coefficient'])
        self.set_line_edit_value('Aero_CDragBody', 'hRideR*hRideR*hRideR', self.car_data['config']['aero']['PolynomialCLiftBodyFDefinition'][3]['coefficient'])
        self.set_line_edit_value('Aero_CDragBody', 'hRideF', self.car_data['config']['aero']['PolynomialCLiftBodyFDefinition'][4]['coefficient'])
        self.set_line_edit_value('Aero_CDragBody', 'hRideF*hRideF', self.car_data['config']['aero']['PolynomialCLiftBodyFDefinition'][7]['coefficient'])
        self.set_line_edit_value('Aero_CDragBody', 'hRideF*hRideF*hRideF', self.car_data['config']['aero']['PolynomialCLiftBodyFDefinition'][10]['coefficient'])
        self.set_line_edit_value('Aero_CDragBody', 'hRideF*hRideR', self.car_data['config']['aero']['PolynomialCLiftBodyFDefinition'][5]['coefficient'])
        self.set_line_edit_value('Aero_CDragBody', 'hRideF*hRideR*hRideR', self.car_data['config']['aero']['PolynomialCLiftBodyFDefinition'][6]['coefficient'])
        self.set_line_edit_value('Aero_CDragBody', 'hRideF*hRideF*hRideR', self.car_data['config']['aero']['PolynomialCLiftBodyFDefinition'][8]['coefficient'])
        self.set_line_edit_value('Aero_CDragBody', 'hRideF*hRideF*hRideR*hRideR', self.car_data['config']['aero']['PolynomialCLiftBodyFDefinition'][9]['coefficient'])
        self.set_line_edit_value('Aero_CDragBody', 'aFlapF', self.car_data['config']['aero']['PolynomialCLiftBodyFDefinition'][11]['coefficient'])
        self.set_line_edit_value('Aero_CDragBody', 'aFlapF*aFlapF', self.car_data['config']['aero']['PolynomialCLiftBodyFDefinition'][12]['coefficient'])
        self.set_line_edit_value('Aero_CDragBody', 'aFlapR', self.car_data['config']['aero']['PolynomialCLiftBodyFDefinition'][13]['coefficient'])
        self.set_line_edit_value('Aero_CDragBody', 'aFlapR*aFlapR', self.car_data['config']['aero']['PolynomialCLiftBodyFDefinition'][14]['coefficient'])
        self.set_line_edit_value('Aero_Flaps', 'aFlapF', self.car_data.get('config', {}).get('aero', {}).get('flapAngles', {}).get('aFlapF', ''))
        self.set_line_edit_value('Aero_Flaps', 'aFlapR', self.car_data.get('config', {}).get('aero', {}).get('flapAngles', {}).get('aFlapR', ''))

    def load_track(self):
        archive, _ = QFileDialog.getOpenFileName(None, 'Load Track', '', 'CSV Files (*.csv)')
        if archive:
            try:
                with open(archive, mode='r') as file:
                    reader = csv.reader(file)
                    next(reader)
                    values = list(reader)

                    self.times = [float(row[0]) for row in values]
                    self.ztrackFL = [float(row[1])/1000 for row in values]
                    self.ztrackFR = [float(row[2])/1000 for row in values]
                    self.ztrackRL = [float(row[3])/1000 for row in values]
                    self.ztrackRR = [float(row[4])/1000 for row in values]
                    self.Ax = [float(row[5]) for row in values]
                    self.Ay = [float(row[6]) for row in values]
                    self.vCar = [float(row[7]) for row in values]
                    self.rPedal = [float(row[8]) for row in values]
                    self.pBrake = [float(row[9]) for row in values]
                    

            except Exception as e:
                print(f'Error loading Front Damper: {e}')

    def sim_visualization():
        archive, _ = QFileDialog.getOpenFileName(None, 'Load Simulation Results', '', 'CSV Files (*.csv)')
        if archive:
            try:
                with open(archive, mode='r') as file:
                    reader = csv.reader(file)
                    next(reader)
                    values = list(reader)



            except Exception as e:
                print(f'Error loading Front Damper: {e}')

    def set_line_edit_value(self, category, name, value):
        unique_name = f'{category}_{name}'

        if unique_name in self.line_edits:
                self.line_edits[unique_name].setText(str(value))

    def set_dark_mode(self):
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Background, QColor(65, 65, 65))
        dark_palette.setColor(QPalette.Window, QColor(65, 65, 65))
        dark_palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.Button, QColor(28, 28, 28))
        dark_palette.setColor(QPalette.ButtonText, QColor(0, 0, 0))
        dark_palette.setColor(QPalette.Highlight, QColor(0, 120, 215))
        dark_palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
        self.setPalette(dark_palette)

        tree_style = '''
            QTreeWidget {
                background-color: #414141;
                color: #ddd;
                border: none;
            }
            QTreeWidget::item {
                padding: 5px;
            }
        '''
        self.tree_widget.setStyleSheet(tree_style)
        self.tree_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)


    def refresh(self):
        for button in self.button_1:
            button.deleteLater()
        for button in self.button_2:
            button.deleteLater()

        self.button_1.clear()
        self.button_2.clear()

        num_simulations = self.spinbox.value()

        global parameters_damper_front, parameters_damper_rear

        parameters_damper_front = [None] * num_simulations
        parameters_damper_rear = [None] * num_simulations

        x_pos = 1127
        y_pos = 300
        button_width = 200

        for i in range(num_simulations):
            button_1 = QPushButton(f'Front Damper Sim {i+1}', self)
            button_1.setFixedWidth(button_width)
            button_1.setFont(QFont('Arial', 12))

            button_1.clicked.connect(self.create_loader(i+1, load_csv_1))
            button_1.move(x_pos, y_pos)
            button_1.show()

            button_2 = QPushButton(f'Rear Damper Sim {i+1}', self)
            button_2.setFixedWidth(button_width)
            button_2.setFont(QFont('Arial', 12))

            button_2.clicked.connect(self.create_loader(i+1, load_csv_2))
            button_2.move(x_pos + button_width + 10, y_pos)
            button_2.show()

            self.button_1.append(button_1)
            self.button_2.append(button_2)

            y_pos += 40

    def create_loader(self, sim_id, loader_function):
        def loader():
            loader_function(sim_id)
        return loader

    def run_sweep(self):
        run_sweep(self)

if __name__ == '__main__':
    multiprocessing.freeze_support()

    app = QApplication(sys.argv)
    window = RideSimApp()
    window.show()
    sys.exit(app.exec_())
