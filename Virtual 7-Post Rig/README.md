## 1. INTRODUCTION

The Ride Sim tool consists of a 7-degree-of-freedom model (D-O-F), meaning it includes 7 differential equations that describe the system’s motion. Consequently, the simulations capture transient states, enabling the analysis of the damping dynamics, which are velocity dependent.

<p align="center">
  <img src="https://github.com/user-attachments/assets/ef1f61ea-6413-4eea-968a-c4f7d7cf9dc2" alt="Descripción de la imagen"/>
  <br>
  <em>Figure 1: 7 D-O-F vehicle model.</em>
</p>

The 7 D-O-F are heave, pitch, roll, and the vertical movements of the four corners, as shown in Figure 1. The differential equations that describe these movements are shown within the equations below.

- Heave Equation

$$
\ddot{h} = \frac{1}{M_s} \left[ F_{FR} + F_{FL} + F_{RL} + F_{RR} + D_{FR} + D_{FL} + D_{RL} + D_{RR} - \frac{1}{2}(c_{l,f} + c_{l,r}) \rho v_x^2 \right]
$$

$$
\begin{aligned}
F_{FR} &= \frac{1}{\frac{1}{k_{FR} + \frac{B_{FR}}{\delta_{FR}}} + \frac{1}{k_{instf}}} \cdot \delta_{FR} \\
F_{FL} &= \frac{1}{\frac{1}{k_{FL} + \frac{B_{FL}}{\delta_{FL}}} + \frac{1}{k_{instf}}} \cdot \delta_{FL} \\
F_{RL} &= \frac{1}{\frac{1}{k_{RL} + \frac{B_{RL}}{\delta_{RL}}} + \frac{1}{k_{instr}}} \cdot \delta_{RL} \\
F_{RR} &= \frac{1}{\frac{1}{k_{RR} + \frac{B_{RR}}{\delta_{RR}}} + \frac{1}{k_{instr}}} \cdot \delta_{RR}
\end{aligned}
$$

$$
\begin{aligned}
D_{FR} = c_f (\dot{z_{FR}} - \dot\delta_{FR})
\end{aligned}
$$

$$
D_{FL} = c_f (\dot{z_{FL}} - \dot\delta_{FL})
$$

$$
D_{RL} = c_r (\dot{z_{RL}} - \dot\delta_{RL})
$$

$$
D_{RR} = c_r (\dot{z_{RR}} - \dot\delta_{RR})
$$

$$
\begin{aligned}
\delta_{FR} &= z_{FR} - \Delta_{FR}, \quad & \Delta_{FR} &= -l_f \phi - \frac{t_F}{2} \theta + h \\
\delta_{FL} &= z_{FL} - \Delta_{FL}, \quad & \Delta_{FL} &= -l_f \phi + \frac{t_F}{2} \theta + h \\
\delta_{RL} &= z_{RL} - \Delta_{RL}, \quad & \Delta_{RL} &= l_r \phi + \frac{t_R}{2} \theta + h \\
\delta_{RR} &= z_{RR} - \Delta_{RR}, \quad & \Delta_{RR} &= l_r \phi - \frac{t_R}{2} \theta + h \\
B_{ij} &= \text{bumpstop response for corner } ij
\end{aligned}
$$

$$
\begin{aligned}
\Delta_{FR} &= -l_f \phi - \frac{t_F}{2} \theta + h \\
\Delta_{FL} &= -l_f \phi + \frac{t_F}{2} \theta + h \\
\Delta_{RL} &= l_r \phi + \frac{t_R}{2} \theta + h \\
\Delta_{RR} &= l_r \phi - \frac{t_R}{2} \theta + h
\end{aligned}
$$

- Roll Equation

$$
\ddot{\theta} = \frac{1}{I_{xx}} \left[ \frac{t_F}{2} (F_{FL} + D_{FL}) - \frac{t_F}{2} (F_{FR} + D_{FR}) + \frac{t_R}{2} (F_{RL} + D_{RL}) - \frac{t_R}{2} (F_{RR} + D_{RR}) + a_y \cdot 9.81 \cdot M_s \cdot (h - z_{CoG})\right]
$$

Where each suspension force $$F_{ij}$$ and each damping force $$D_{ij}$$ is defined as:

$$
\begin{aligned}
F_{ij} =\frac{1}{\displaystyle \frac{1}{k_{ij} + \displaystyle \frac{bumpstop_{x}(\delta_{ij})}{\delta_{ij}} + K_{\text{arb}}} + \frac{1}{k_{\text{inst}}}}\cdot \delta_{ij}
\end{aligned}
$$

$$
\begin{aligned}
D_{ij} = c_{x}( \dot{z_{ij}} - \dot{\Delta}_{ij} )
\end{aligned}
$$

Where:

$$
\begin{aligned}
\delta_{ij} = z_{ij} - \Delta_{ij}
\Delta_{FL} = -l_f \phi + \frac{t_F}{2} \theta + h \\
\Delta_{FR} = -l_f \phi - \frac{t_F}{2} \theta + h \\
\Delta_{RL} = l_r \phi + \frac{t_R}{2} \theta + h \\
\Delta_{RR} = l_r \phi - \frac{t_R}{2} \theta + h
\end{aligned}
$$

- Pitch Equation

$$
\ddot{\phi} = \frac{1}{I_{yy}} \left[l_r (F_{RR} + D_{RR} + F_{RL} + D_{RL}) - l_f (F_{FR} + D_{FR} + F_{FL} + D_{FL}) - a_x \cdot 9.81 \cdot M_s \cdot (h - z_{CoG}) - l_f \cdot \tfrac{1}{2} c_{l,\text{front}} \rho \left( \tfrac{v_x}{3.6} \right)^2 + l_r \cdot \tfrac{1}{2} c_{l,\text{rear}} \rho \left( \tfrac{v_x}{3.6} \right)^2 - (h - z_{CoG}) \cdot \tfrac{1}{2} c_d \rho v_x^2\right]
$$

$$
\begin{aligned}
F_{ij} = \frac{1}{\displaystyle \frac{1}{k_{ij} + \displaystyle \frac{bumpstop_{x}(z_{ij} - \Delta_{ij} - z_{ij}^{\text{free}})}{z_{ij} - \Delta_{ij} - z_{ij}^{\text{free}}}} + \frac{1}{k_{\text{inst}}}}\cdot (z_{ij} - \Delta_{ij})
\end{aligned}
$$

$$
D_{ij} = c_x \cdot (\dot{z_{ij}} - \dot{\Delta}_{ij})
$$

$$
\begin{aligned}
\Delta_{FR} &= -l_f \phi - \frac{t_F}{2} \theta + h \\
\Delta_{FL} &= -l_f \phi + \frac{t_F}{2} \theta + h \\
\Delta_{RR} &= l_r \phi - \frac{t_R}{2} \theta + h \\
\Delta_{RL} &= l_r \phi + \frac{t_R}{2} \theta + h
\end{aligned}
$$

- FR Corner Equation

$$
\ddot{z_{FR}} = \frac{1}{m_{\text{hubF}}} \left[- F_{FR} - D_{FR} + k_{tf} (z_{track_{FR}} - z_{FR})\right]
$$

- FL Corner Equation

$$
\ddot{z_{FL}} = \frac{1}{m_{\text{hubF}}} \left[- F_{FL} - D_{FL} + k_{tf} (z_{track_{FL}} - z_{FL})\right]
$$

- RL Corner Equation

$$
\ddot{z_{RL}} = \frac{1}{m_{\text{hubR}}} \left[- F_{RL} - D_{RL} + k_{tr} (z_{track_{RL}} - z_{RL})\right]
$$

- RR Corner Equation

$$
\ddot{z_{RR}} = \frac{1}{m_{\text{hubR}}} \left[- F_{RR} - D_{RR} + k_{tr} (z_{track_{RR}} - z_{RR})\right]
$$

