import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.signal import welch
import random

def tope_fuerza(x, topout, bottomout, k_tope=1e6):
    if x < topout:
        return -k_tope * (topout - x)
    elif x > bottomout:
        return -k_tope * (bottomout - x)
    else:
        return 0.0

# --- MODELO VERTICAL PURO 7DOF (HEAVE, PITCH, ROLL + 4 MASAS NO SUSPENDIDAS) ---
def vehicle_model_simple(t, z, params, ztrack_funcs):
    # Estados: h, hdot, phi, phi_dot, theta, theta_dot,
    # zFR, zFRdot, zFL, zFLdot, zRL, zRLdot, zRR, zRRdot = z
    h, hdot, phi, phi_dot, theta, theta_dot,zFR, zFRdot, zFL, zFLdot, zRL, zRLdot, zRR, zRRdot = z

    # Interpolación de inputs de pista
    ztrack_FL, ztrack_FR, ztrack_RL, ztrack_RR = [f(t) for f in ztrack_funcs]

    # Parámetros globales
    Ms = params['ms']
    Ixx = params['Ixx']
    Iyy = params['Iyy']
    lf = params['lf']
    lr = params['lr']
    tF = params['tf']
    tR = params['tr']
    mHubF = params['mHubF']
    mHubR = params['mHubR']

    # Resortes
    kFL = params['kFL']
    kFR = params['kFR']
    kRL = params['kRL']
    kRR = params['kRR']
    # Resortes de instalación en serie
    kinstf = params['kinstf']
    kinstr = params['kinstr']

    # Bumpstop y amortiguador
    bump_front = params['bumpstop_front']
    bump_rear  = params['bumpstop_rear']
    damper_front = params['damper_front']
    damper_rear  = params['damper_rear']

    # Recorridos físicos (stroke) de cada amortiguador
    stroke_FL = params.get('stroke_FL', 0.059)
    stroke_FR = params.get('stroke_FR', 0.059)
    stroke_RL = params.get('stroke_RL', 0.059)
    stroke_RR = params.get('stroke_RR', 0.059)

    # Límites de travel (top y bottom) basados en posición estática
    z_topout_FL    = params['x_FL_static'] - stroke_FL/2
    z_bottomout_FL = params['x_FL_static'] + stroke_FL/2
    z_topout_FR    = params['x_FR_static'] - stroke_FR/2
    z_bottomout_FR = params['x_FR_static'] + stroke_FR/2
    z_topout_RL    = params['x_RL_static'] - stroke_RL/2
    z_bottomout_RL = params['x_RL_static'] + stroke_RL/2
    z_topout_RR    = params['x_RR_static'] - stroke_RR/2
    z_bottomout_RR = params['x_RR_static'] + stroke_RR/2

    g = 9.81

    # Helper para calcular la fuerza de rueda con clipping al recorrido físico
    def wheel_force(z_w, z_w_dot,
                    phi_off, theta_off,
                    phi_dot_off, theta_dot_off,
                    k_spring, k_inst,
                    bump, damper,
                    z_top, z_bot,
                    gap):
        # Desplazamiento crudo
        x_raw = z_w - (phi_off + theta_off + h)
        # Clipping al recorrido físico
        x_clipped = np.clip(x_raw, z_top, z_bot)
        # Bumpstop (compresión más allá del gap)
        compression = x_raw - gap
        f_bump = bump(np.maximum(0, compression))
        # Resortes en serie: resorte + instalación
        k_total = 1.0/(1.0/k_spring + 1.0/k_inst)
        f_spring = k_total * x_clipped
        # Amortiguador: velocidad relativa resorte-carrocería
        rel_vel = z_w_dot - (phi_dot_off + theta_dot_off + hdot)
        f_damper = damper(rel_vel)
        # Tope blando como respaldo
        f_stop = tope_fuerza(x_raw, z_top, z_bot)
        return f_spring + f_bump + f_damper + f_stop

    # Cálculo de offsets estáticos y dinámicos
    phi_off_front = -lf * phi
    phi_off_rear  =  lr * phi
    theta_off_front =  (tF/2) * theta
    theta_off_rear  =  (tR/2) * theta
    phi_dot_off_front = -lf * phi_dot
    phi_dot_off_rear  =  lr * phi_dot
    theta_dot_off_front =  (tF/2) * theta_dot
    theta_dot_off_rear  =  (tR/2) * theta_dot

    # --- Fuerzas por rueda aplicando clipping ---
    F_FL = wheel_force(zFL, zFLdot,
                       phi_off_front, theta_off_front,
                       phi_dot_off_front, theta_dot_off_front,
                       kFL, kinstf,
                       bump_front, damper_front,
                       z_topout_FL, z_bottomout_FL,
                       params['gap_bumpstop_FL'])

    F_FR = wheel_force(zFR, zFRdot,
                       phi_off_front, -theta_off_front,
                       phi_dot_off_front, -theta_dot_off_front,
                       kFR, kinstf,
                       bump_front, damper_front,
                       z_topout_FR, z_bottomout_FR,
                       params['gap_bumpstop_FR'])

    F_RL = wheel_force(zRL, zRLdot,
                       phi_off_rear, theta_off_rear,
                       phi_dot_off_rear, theta_dot_off_rear,
                       kRL, kinstr,
                       bump_rear, damper_rear,
                       z_topout_RL, z_bottomout_RL,
                       params['gap_bumpstop_RL'])

    F_RR = wheel_force(zRR, zRRdot,
                       phi_off_rear, -theta_off_rear,
                       phi_dot_off_rear, -theta_dot_off_rear,
                       kRR, kinstr,
                       bump_rear, damper_rear,
                       z_topout_RR, z_bottomout_RR,
                       params['gap_bumpstop_RR'])

    x_FL_raw = zFL   - (phi_off_front +  theta_off_front +  h)
    x_FR_raw = zFR   - (phi_off_front -  theta_off_front +  h)
    x_RL_raw = zRL   - (phi_off_rear  +  theta_off_rear  +  h)
    x_RR_raw = zRR   - (phi_off_rear  -  theta_off_rear  +  h)

    T_arb_f = params['k_arb_f'] * (x_FL_raw - x_FR_raw)
    T_arb_r = params['k_arb_r'] * (x_RL_raw - x_RR_raw)

    lever_f = tF / 2
    lever_r = tR / 2

    F_FL +=  T_arb_f / lever_f
    F_FR += -T_arb_f / lever_f
    F_RL +=  T_arb_r / lever_r
    F_RR += -T_arb_r / lever_r

    # --- Aerodinámica ---
    Fz_aero_front, Fz_aero_rear, F_drag = compute_aero_forces(
        vx=params.get('vx', 30.0),
        hRideF=params.get('hRideF', 0.02),
        hRideR=params.get('hRideR', 0.04),
        aero_poly=params.get('aero_polynomials', {})
    )
    F_FL += 0.5 * Fz_aero_front
    F_FR += 0.5 * Fz_aero_front
    F_RL += 0.5 * Fz_aero_rear
    F_RR += 0.5 * Fz_aero_rear

    # --- Transferencias de peso ---
    l = lf + lr
    h_cg = params.get('zCoG', 0.3)
    a_x = params.get('ax', 0.0)
    a_y = params.get('ay', 0.0)

    # Longitudinal
    dFz_long = Ms * a_x * h_cg / l
    F_FL -= 0.5 * dFz_long
    F_FR -= 0.5 * dFz_long
    F_RL += 0.5 * dFz_long
    F_RR += 0.5 * dFz_long

    # Lateral
    dFz_lat_f = Ms * a_y * h_cg / tF
    dFz_lat_r = Ms * a_y * h_cg / tR
    F_FL += 0.5 * dFz_lat_f
    F_FR -= 0.5 * dFz_lat_f
    F_RL += 0.5 * dFz_lat_r
    F_RR -= 0.5 * dFz_lat_r

    # === Ecuaciones de dinámica ===
    h_dd    = (F_FL + F_FR + F_RL + F_RR - Ms*g) / Ms
    theta_dd = ((tF/2)*(F_FL - F_FR) + (tR/2)*(F_RL - F_RR)) / Ixx
    phi_dd   = (lr*(F_RR + F_RL) - lf*(F_FR + F_FL)) / Iyy

    # Masas no suspendidas
    zFR_dd = (-F_FR + params['ktf']*(ztrack_FR - zFR)) / mHubF
    zFL_dd = (-F_FL + params['ktf']*(ztrack_FL - zFL)) / mHubF
    zRL_dd = (-F_RL + params['ktr']*(ztrack_RL - zRL)) / mHubR
    zRR_dd = (-F_RR + params['ktr']*(ztrack_RR - zRR)) / mHubR

    return [
        hdot, h_dd,
        phi_dot, phi_dd,
        theta_dot, theta_dd,
        zFRdot, zFR_dd,
        zFLdot, zFL_dd,
        zRLdot, zRL_dd,
        zRRdot, zRR_dd
    ]

from scipy.optimize import fsolve

def compute_aero_forces(vx, hRideF, hRideR, aero_poly, rho_air=1.225, area_ref=1):
    """
    Calcula las fuerzas aerodinámicas verticales y de arrastre (drag).
    """

    def evaluate_poly(poly_dict, hF, hR):
        variables = {
            "Const": 1.0,
            "hRideF": hF,
            "hRideR": hR,
            "hRideF*hRideF": hF**2,
            "hRideR*hRideR": hR**2,
            "hRideF*hRideF*hRideF": hF**3,
            "hRideR*hRideR*hRideR": hR**3,
            "hRideF*hRideR": hF * hR,
        }
        return sum(coef * variables.get(expr, 0.0) for expr, coef in poly_dict.items())

    # --- Coeficientes ---
    Clf = evaluate_poly(aero_poly.get("CLiftBodyF", {}), hRideF, hRideR)
    Clr = evaluate_poly(aero_poly.get("CLiftBodyR", {}), hRideF, hRideR)
    Cd  = evaluate_poly(aero_poly.get("CDragBody", {}), hRideF, hRideR)

    # --- Fuerzas ---
    q = 0.5 * rho_air * vx**2  # presión dinámica

    Fz_aero_front = -Clf * q * area_ref
    Fz_aero_rear  = -Clr * q * area_ref
    F_drag        = Cd  * q * area_ref

    return Fz_aero_front, Fz_aero_rear, F_drag

def compute_static_equilibrium(params, vx=0.0):
    """
    Resuelve el equilibrio estático: encuentra h, phi, theta y las posiciones de las masas no suspendidas
    de forma que las fuerzas verticales y los momentos se equilibren.
    """

    h0_est = estimate_h0_from_static_travel(params, phi=0.0, theta=0.0)
    # Constantes
    g = 9.81
    Ms = params['ms']
    lf = params['lf']
    lr = params['lr']
    hRideF = params.get('hRideF', 0.02)
    hRideR = params.get('hRideR', 0.04)
    tf = params['tf']
    tr = params['tr']
    
    mHubF = params['mHubF']
    mHubR = params['mHubR']

    kFL = params['kFL']
    kFR = params['kFR']
    kRL = params['kRL']
    kRR = params['kRR']
    kinstf = params['kinstf']
    kinstr = params['kinstr']
    ktf = params['ktf']
    ktr = params['ktr']

    bumpstop_front = params['bumpstop_front']
    bumpstop_rear = params['bumpstop_rear']

    z_FL_free = params['z_FL_free']
    z_FR_free = params['z_FR_free']
    z_RL_free = params['z_RL_free']
    z_RR_free = params['z_RR_free']

    # Antirrolbar si está presente
    k_arb_f = params.get('k_arb_f', 0.0)
    k_arb_r = params.get('k_arb_r', 0.0)


    def residual(x):
        # Variables: h, phi, theta, zFR, zFL, zRL, zRR
        h, phi, theta, zFR, zFL, zRL, zRR = x

        # Altura de la masa suspendida en cada esquina (según geometría del chasis rígido)
        zs_FR = h - lf*phi - (tf/2)*theta
        zs_FL = h - lf*phi + (tf/2)*theta
        zs_RR = h + lr*phi - (tr/2)*theta
        zs_RL = h + lr*phi + (tr/2)*theta

        # Travel de suspensión
        x_FR = zFR - zs_FR
        x_FL = zFL - zs_FL
        x_RR = zRR - zs_RR
        x_RL = zRL - zs_RL

        params['x_static_FL'] = x_FL
        params['x_static_FR'] = x_FR
        params['x_static_RL'] = x_RL
        params['x_static_RR'] = x_RR

        # Rigidez efectiva muelle + bumpstop + rigidez instalación
        kFR_eff = 1 / (1 / (kFR + bumpstop_front(x_FR - z_FR_free)) + 1 / kinstf)
        kFL_eff = 1 / (1 / (kFL + bumpstop_front(x_FL - z_FL_free)) + 1 / kinstf)
        kRR_eff = 1 / (1 / (kRR + bumpstop_rear(x_RR - z_RR_free)) + 1 / kinstr)
        kRL_eff = 1 / (1 / (kRL + bumpstop_rear(x_RL - z_RL_free)) + 1 / kinstr)

        # Fuerzas suspensión
        F_FR = kFR_eff * x_FR
        F_FL = kFL_eff * x_FL
        F_RR = kRR_eff * x_RR
        F_RL = kRL_eff * x_RL

        # --- Añadir fuerzas aerodinámicas si hay velocidad ---
        Fz_aero_front, Fz_aero_rear, _ = compute_aero_forces(
            vx=vx,
            hRideF=params.get("hRideF", 0.02),
            hRideR=params.get("hRideR", 0.04),
            aero_poly=params.get("aero_polynomials", {})
        )
        # Distribuir por rueda
        F_FL += 0.5 * Fz_aero_front
        F_FR += 0.5 * Fz_aero_front
        F_RL += 0.5 * Fz_aero_rear
        F_RR += 0.5 * Fz_aero_rear

        # Fuerza total vertical (suma fuerzas suspensión - peso)
        R1 = F_FR + F_FL + F_RR + F_RL - Ms * g

        # Momento de roll (respecto al eje longitudinal)
        R2 = (tf/2)*(F_FL - F_FR) + (tr/2)*(F_RL - F_RR)

        # Momento de pitch (respecto al eje transversal)
        R3 = lr*(F_RR + F_RL) - lf*(F_FR + F_FL)

        # Equilibrio masas no suspendidas: z_i tal que fuerza muelle + neumático = peso mu
        R4 = -F_FR + ktf*(0 - zFR) - mHubF * g
        R5 = -F_FL + ktf*(0 - zFL) - mHubF * g
        R6 = -F_RR + ktr*(0 - zRR) - mHubR * g
        R7 = -F_RL + ktr*(0 - zRL) - mHubR * g

        return [R1, R2, R3, R4, R5, R6, R7]

    # Heave inicial como media de ambos ride heights
    h_init = h0_est #(lr * hRideF + lf * hRideR) / (lf + lr)

    # Pitch inicial derivado de la diferencia entre ambos
    phi_init = (hRideR - hRideF) / (lf + lr)

    # Roll puede mantenerse en cero
    theta_init = 0.0

    # Travel estimado estático por rueda
    g = 9.81
    W = params['ms'] * g
    lf = params['lf']
    lr = params['lr']
    kf_eff = 1 / (1 / params['kFL'] + 1 / params['kinstf'])
    kr_eff = 1 / (1 / params['kRL'] + 1 / params['kinstr'])

    # Estimación inicial masas no suspendidas
    W   = Ms*g  
    wbal_f = params['rWeightBalF']
    Wf     = W * wbal_f
    Wr     = W * (1.0 - wbal_f)

    # Travel + z_free + altura sobre suelo (z_ui)
    zFR0 = Wf / (2 * kf_eff) + params['z_FR_free']
    zFL0 = Wf / (2 * kf_eff) + params['z_FL_free']
    zRR0 = Wr / (2 * kr_eff) + params['z_RR_free']
    zRL0 = Wr / (2 * kr_eff) + params['z_RL_free']

    x0 = [h_init, phi_init, theta_init, zFR0, zFL0, zRL0, zRR0]

    sol, info, ier, msg = fsolve(residual, x0, full_output=True)
    if ier != 1:
        print(f"[WARNING] fsolve no ha convergido: {msg}")
    # === Cálculo de márgenes al bumpstop tras resolver el equilibrio estático ===
    for corner in ['FL','FR','RL','RR']:
        x_static = params[f'x_static_{corner}']
        topout   = params[f'z_topout_{corner}']

        # margen respecto a la carrera del amortiguador
        margen_ext  = topout - x_static

        print(f"[INFO] Márgenes {corner}: "
            f"extensión = {margen_ext*1000:.2f} mm, ")

    return sol

def run_vehicle_model_simple(t_vec, z_tracks, vx, ax, ay, throttle, brake, params):
    ztrack_funcs = [interp1d(t_vec, z_tracks[i], bounds_error=False, fill_value="extrapolate") for i in range(4)]

    # --- Parámetros ---
    ms = params['ms']
    g = 9.81
    lf = params['lf']
    lr = params['lr']
    tf = params['tf']
    tr = params['tr']
    kFL = params['kFL']
    kFR = params['kFR']
    kRL = params['kRL']
    kRR = params['kRR']

    # Interpoladores de inputs
    vx_func = interp1d(t_vec, vx, bounds_error=False, fill_value="extrapolate")
    ax_func = interp1d(t_vec, ax, bounds_error=False, fill_value="extrapolate")
    ay_func = interp1d(t_vec, ay, bounds_error=False, fill_value="extrapolate")
    throttle_func = interp1d(t_vec, throttle, bounds_error=False, fill_value="extrapolate")
    brake_func = interp1d(t_vec, brake, bounds_error=False, fill_value="extrapolate")
    vx0 = float(vx_func(t_vec[0]))

    # --- Equilibrio estático ---
    h0, phi0, theta0, zFR0, zFL0, zRL0, zRR0 = compute_static_equilibrium(params, vx0)

    # Travel estático (referencia para topes reales)
    x_FL_static = zFL0 - (-lf * phi0 + (tf / 2) * theta0 + h0)
    x_FR_static = zFR0 - (-lf * phi0 - (tf / 2) * theta0 + h0)
    x_RL_static = zRL0 - ( lr * phi0 + (tr / 2) * theta0 + h0)
    x_RR_static = zRR0 - ( lr * phi0 - (tr / 2) * theta0 + h0)

    # Guardamos el punto estático para cálculo de topes reales
    params.update({
        'x_FL_static': x_FL_static,
        'x_FR_static': x_FR_static,
        'x_RL_static': x_RL_static,
        'x_RR_static': x_RR_static,
        'z_topout_FL': x_FL_static - params['stroke_FL'] / 2,
        'z_bottomout_FL': x_FL_static + params['stroke_FL'] / 2,
        'z_topout_FR': x_FR_static - params['stroke_FR'] / 2,
        'z_bottomout_FR': x_FR_static + params['stroke_FR'] / 2,
        'z_topout_RL': x_RL_static - params['stroke_RL'] / 2,
        'z_bottomout_RL': x_RL_static + params['stroke_RL'] / 2,
        'z_topout_RR': x_RR_static - params['stroke_RR'] / 2,
        'z_bottomout_RR': x_RR_static + params['stroke_RR'] / 2,
    })

    y0 = [h0, 0.0, phi0, 0.0, theta0, 0.0, zFR0, 0.0, zFL0, 0.0, zRL0, 0.0, zRR0, 0.0]

    def rhs(t, y):
        params['vx'] = float(vx_func(t))
        params['ax'] = float(ax_func(t))
        params['ay'] = float(ay_func(t))
        return vehicle_model_simple(t, y, params, ztrack_funcs)

    sol = solve_ivp(rhs, (t_vec[0], t_vec[-1]), y0, t_eval=t_vec, method='RK45', max_step=0.001)

    return sol

def estimate_h0_from_static_travel(params, phi=0.0, theta=0.0):
    """
    Calcula un h0 realista a partir del reparto de carga y rigideces, asegurando travel >= 0 en cada esquina.
    """
    g = 9.81
    Ms = params['ms']
    wbal_f = params['rWeightBalF']
    lf = params['lf']
    lr = params['lr']
    tf = params['tf']
    tr = params['tr']

    # Posición de cada esquina (x longitudinal, y lateral)
    pos = {
        'FL': (-lf, +tf / 2),
        'FR': (-lf, -tf / 2),
        'RL': (+lr, +tr / 2),
        'RR': (+lr, -tr / 2)
    }

    # Reparto estático por eje
    W = Ms * g
    Wf = W * wbal_f 
    Wr = W - Wf
    F_static = {
        'FL': Wf / 2,
        'FR': Wf / 2,
        'RL': Wr / 2,
        'RR': Wr / 2
    }

    # Rigideces efectivas muelle + rigidez de instalación
    k_eff = {
        'FL': 1 / (1 / params['kFL'] + 1 / params['kinstf']),
        'FR': 1 / (1 / params['kFR'] + 1 / params['kinstf']),
        'RL': 1 / (1 / params['kRL'] + 1 / params['kinstr']),
        'RR': 1 / (1 / params['kRR'] + 1 / params['kinstr']),
    }

    z_free = {
        'FL': params['z_FL_free'],
        'FR': params['z_FR_free'],
        'RL': params['z_RL_free'],
        'RR': params['z_RR_free'],
    }

    h_list = []
    for corner in ['FL', 'FR', 'RL', 'RR']:
        xi, yi = pos[corner]
        x_spring = F_static[corner] / k_eff[corner]  # compresión del muelle

        # Limitar travel al rango permitido
        topout = params[f'z_topout_{corner}']
        bottomout = params[f'z_bottomout_{corner}']
        x_spring_clipped = np.clip(x_spring, topout, bottomout)

        if not np.isclose(x_spring, x_spring_clipped, atol=1e-5):
            print(f"[WARN] {corner} travel limitado: {x_spring*1000:.2f} mm → {x_spring_clipped*1000:.2f} mm")

        z_ui = 0.0  # rueda apoyada en el suelo
        z_si = z_ui - x_spring_clipped - z_free[corner]      # posición del chasis en esa esquina
        h_i = z_si - (phi * xi + theta * yi)                 # despejamos h
        h_list.append(h_i)
    print(f"[DEBUG] Reparto de peso: Wf = {Wf:.2f} N, Wr = {Wr:.2f} N")
    return np.mean(h_list)

# --- POSTPROCESADO BÁSICO ---
def postprocess_7dof(sol, params, z_tracks, t_vec, throttle, brake, vx):
    """
    Calcula travel, fuerzas y outputs básicos para cada esquina a partir de la simulación 7-DOF.
    Todo el cálculo está vectorizado con NumPy para mayor eficiencia.
    """

    import numpy as np
    from scipy.signal import welch
    from model import compute_aero_forces

    # ──────────────────────────────────────────────────────────────────────────────
    # 1) Parámetros y configuraciones iniciales
    # ──────────────────────────────────────────────────────────────────────────────
    n_corners = 4
    t   = sol.t              # vector de tiempo (N muestras)
    N   = len(t)
    dt  = np.mean(np.diff(t))
    fs  = 1.0 / dt
    nperseg = 2048
    noverlap = nperseg // 2

    # Dimensiones del vehículo (en metros)
    wheelbase = params.get('wheelbase', 2.5)
    track_f   = params.get('tf', 1.6)
    track_r   = params.get('tr', 1.6)
    lf        = params.get('lf', wheelbase / 2)
    lr        = params.get('lr', wheelbase / 2)

    # Posiciones [x, y] de cada rueda en coordenadas del CG (FL, FR, RL, RR)
    pos = np.array([
        [-lf,  track_f / 2],   # FL
        [-lf, -track_f / 2],   # FR
        [ lr,  track_r / 2],   # RL
        [ lr, -track_r / 2],   # RR
    ])

    # Índices de estado en sol.y: desplazamientos "unsprung" (zu) y velocidades (zu_dot)
    zu_idx    = [8,  6, 10, 12]
    zudot_idx = [9,  7, 11, 13]

    # ──────────────────────────────────────────────────────────────────────────────
    # 2) Extraer estados principales de sol.y
    # ──────────────────────────────────────────────────────────────────────────────
    h      = sol.y[0, :]     # heave global del chasis [m]
    hdot   = sol.y[1, :]
    phi    = sol.y[2, :]     # pitch [rad]
    phi_dot= sol.y[3, :]
    theta  = sol.y[4, :]     # roll [rad]
    theta_dot = sol.y[5, :]

    # Desplazamientos "unsprung" (ruedas) y sus velocidades
    zu      = np.stack([sol.y[idx, :] for idx in zu_idx],    axis=0)  # shape: (4, N)
    zudot   = np.stack([sol.y[idx, :] for idx in zudot_idx], axis=0)  # shape: (4, N)

    # ──────────────────────────────────────────────────────────────────────────────
    # 3) Cinemática de suspensión: travel, posiciones de rueda (zs) y márgenes
    # ──────────────────────────────────────────────────────────────────────────────
    # zs: posición vertical del chasis en cada rueda (4, N)
    zs = (
        h[None, :] +
        pos[:, 0:1] * phi[None, :]   +   # aporte por pitch × distancia en x
        pos[:, 1:2] * theta[None, :]    # aporte por roll  × distancia en y
    )

    # Travel absoluto (“unsprung” – “chassis”) en cada esquina
    x_spring   = zu - zs                # shape: (4, N)

    # Límites física de recorrido (stop extension/compression)
    z_topout    = np.array([
        params['z_topout_FL'], params['z_topout_FR'],
        params['z_topout_RL'], params['z_topout_RR']
    ])
    z_bottomout = np.array([
        params['z_bottomout_FL'], params['z_bottomout_FR'],
        params['z_bottomout_RL'], params['z_bottomout_RR']
    ])
    stroke = np.array([
        params['stroke_FL'], params['stroke_FR'],
        params['stroke_RL'], params['stroke_RR']
    ])
    gap_bump = np.array([
        params['gap_bumpstop_FL'], params['gap_bumpstop_FR'],
        params['gap_bumpstop_RL'], params['gap_bumpstop_RR']
    ])
    hub_mass = np.array([
        params['mHubF'], params['mHubF'],
        params['mHubR'], params['mHubR']
    ])[:, None]  # shape (4,1)
    zt = np.stack(
        [np.interp(t, t_vec, z_tracks[i]) for i in range(n_corners)],
        axis=0
    )  # (4, N)

    ae_front = np.zeros_like(h)
    ae_rear  = np.zeros_like(h)
    for i, (v_i, h_i, phi_i) in enumerate(zip(vx, h, phi)):
        dyn_hF = h_i - lf * phi_i + params.get('hRideF', 0.0)
        dyn_hR = h_i + lr * phi_i + params.get('hRideR', 0.0)
        Ff, Fr, _ = compute_aero_forces(
            vx=v_i,
            hRideF=dyn_hF,
            hRideR=dyn_hR,
            aero_poly=params.get('aero_polynomials', {})
        )
        ae_front[i] = Ff
        ae_rear[i]  = Fr

    # peso estático
    g    = 9.81
    Wtot = params['ms'] * g
    Wf   = Wtot * params['rWeightBalF']
    Wr   = Wtot - Wf
    static = np.array([Wf/2, Wf/2, Wr/2, Wr/2])[:,None]  # (4,1)

    aero = np.vstack([
        -0.5 * ae_front,
        -0.5 * ae_front,
        -0.5 * ae_rear,
        -0.5 * ae_rear,
    ])  # (4, N)

    kt = np.array([
        params['ktf'], params['ktf'],
        params['ktr'], params['ktr']
    ])  # neumáticos

    f_tire     = kt[:,None] * (zt - zu)   # (4, N)
    wheel_load_N = static + aero + f_tire + hub_mass*g
    wheel_load    = wheel_load_N/9.81   
    # ────────────────────────────────────────────────────────────────────────────
    # Convertir wheel_load [N] → [kg] y sumar masa no suspendida por rueda
    # ────────────────────────────────────────────────────────────────────────────
    g = 9.81  # m/s²
    # vector de masas no suspendidas por rueda [kg], en el mismo orden FL,FR,RL,RR


    # Divide por g para pasar de N a “kgf”, y suma la masa de la llanta
    wheel_load = wheel_load / g + hub_mass
    # ──────────────────────────────────────────────────────────────────────────────
    # 3.3) Márgenes dinámicos y estáticos traseros
    # ──────────────────────────────────────────────────────────────────────────────
    margen_ext       = x_spring - z_topout[:,None]
    margen_comp      = z_bottomout[:,None] - x_spring
    xRL_static       = x_spring[2,0]
    xRR_static       = x_spring[3,0]
    z_free_RL        = params['z_RL_free']
    z_free_RR        = params['z_RR_free']
    travel_static   = x_spring[:,0][:,None]
    travel_rel      = x_spring - travel_static

    # ──────────────────────────────────────────────────────────────────────────────
    # 4) Fuerzas suspensión: muelle, bumpstop y damper
    # ──────────────────────────────────────────────────────────────────────────────
    k = np.array([
        params['kFL'], params['kFR'],
        params['kRL'], params['kRR']
    ])

    f_spring = k[:,None] * x_spring

    bump_funcs = [
        params['bumpstop_front'], params['bumpstop_front'],
        params['bumpstop_rear'],  params['bumpstop_rear']
    ]
    f_bump = np.zeros_like(x_spring)
    for i in range(n_corners):
        comp = np.maximum(0, x_spring[i] - gap_bump[i])
        f_bump[i] = bump_funcs[i](comp)

    v_chassis = (
        hdot[None,:]
        + pos[:,0:1] * phi_dot[None,:]
        + pos[:,1:2] * theta_dot[None,:]
    )
    v_damper = zudot - v_chassis

    damper_funcs = [
        params['damper_front'], params['damper_front'],
        params['damper_rear'],  params['damper_rear']
    ]
    f_damper = np.zeros_like(v_damper)
    for i in range(n_corners):
        f_damper[i] = damper_funcs[i](v_damper[i])


    # ──────────────────────────────────────────────────────────────────────────────
    # 5) Fuerza de neumático y track excitation (interpolación de z_tracks)
    # ──────────────────────────────────────────────────────────────────────────────
    f_tire_max    = np.max(f_tire,axis=1)
    f_tire_min    = np.min(f_tire,axis=1)
    f_tire_mean   = np.mean(f_tire,axis=1)
    f_tire_std    = np.std(f_tire,axis=1)
    f_tire_var    = np.var(f_tire,axis=1)

    # ──────────────────────────────────────────────────────────────────────────────
    # 6) Condiciones Grip-Limited (lateral, frenada y tracción)
    # ──────────────────────────────────────────────────────────────────────────────
    throttle_signal = np.interp(t, t_vec, throttle)
    brake_signal    = np.interp(t, t_vec, brake)
    threshold       = 0.5

    grip_lateral_mask    = (throttle_signal < threshold) & (brake_signal < threshold)
    grip_brake_mask      = (brake_signal > threshold)    & (throttle_signal < threshold)
    grip_traction_mask   = (throttle_signal > threshold) & (brake_signal < threshold)

    # Fuerza tire grip-limited (max y min) en condición lateral
    if np.any(grip_lateral_mask):
        f_tire_gl_max = np.max(wheel_load_N[:, grip_lateral_mask], axis=1)
        f_tire_gl_min = np.min(wheel_load_N[:, grip_lateral_mask], axis=1)
    else:
        f_tire_gl_max = np.zeros(4)
        f_tire_gl_min = np.zeros(4)

    # ── Grip-Limited en frenada ──
    if np.any(grip_brake_mask):
        fl_brake = np.mean(wheel_load_N[0:2, grip_brake_mask], axis=0)
        rl_brake = np.mean(wheel_load_N[2:4, grip_brake_mask], axis=0)
        front_load_rms_brake = np.sqrt(np.mean(fl_brake**2))
        rear_load_rms_brake  = np.sqrt(np.mean(rl_brake**2))
    else:
        front_load_rms_brake = 0.0
        rear_load_rms_brake  = 0.0

    # ── Grip-Limited en tracción ──
    if np.any(grip_traction_mask):
        fl_trac = np.mean(wheel_load_N[0:2, grip_traction_mask], axis=0)
        rl_trac = np.mean(wheel_load_N[2:4, grip_traction_mask], axis=0)
        front_load_rms_traction = np.sqrt(np.mean(fl_trac**2))
        rear_load_rms_traction  = np.sqrt(np.mean(rl_trac**2))
    else:
        front_load_rms_traction = 0.0
        rear_load_rms_traction  = 0.0

    # ──────────────────────────────────────────────────────────────────────────────
    # 7) Road-noise: wheel vertical-speed RMS [mm/s] para cada rueda
    # ──────────────────────────────────────────────────────────────────────────────
    # zt está en [m], convertimos a mm antes de derivar
    v_mm_s = [np.gradient(zt[i], dt) for i in range(n_corners)]  # lista de 4 arrays (mm/s)
    rms_per_wheel = [np.sqrt(np.mean(v**2)) for v in v_mm_s]

    # Promedio por eje (front / rear)
    front_noise_vals = 0.5 * (rms_per_wheel[0] + rms_per_wheel[1])
    rear_noise_vals  = 0.5 * (rms_per_wheel[2] + rms_per_wheel[3])

    # ──────────────────────────────────────────────────────────────────────────────
    # 8) RMS en Grip-Limited (heave y carga) + no-grip
    # ──────────────────────────────────────────────────────────────────────────────
    # Heave por eje en [m]:
    heave_front = h - lf * phi
    heave_rear  = h + lr * phi

    # RMS de heave en mask grip-lateral (m → convertir más tarde a mm)
    if np.any(grip_lateral_mask):
        frh_rms = np.sqrt(np.mean(heave_front[grip_lateral_mask]**2)) * 1000  # mm
        rrh_rms = np.sqrt(np.mean(heave_rear[grip_lateral_mask]**2)) * 1000  # mm
    else:
        frh_rms = 0.0
        rrh_rms = 0.0

    # Cargas front/rear  RMS en mask grip-lateral
    if np.any(grip_lateral_mask):
        fl_gl = np.mean(wheel_load_N[0:2, grip_lateral_mask], axis=0)
        rl_gl = np.mean(wheel_load_N[2:4, grip_lateral_mask], axis=0)
        front_load_rms = np.sqrt(np.mean(fl_gl**2))
        rear_load_rms  = np.sqrt(np.mean(rl_gl**2))
    else:
        front_load_rms = 0.0
        rear_load_rms  = 0.0

    # RMS no-grip (uso para comparación)
    non_grip_mask = ~grip_lateral_mask
    if np.any(non_grip_mask):
        frh_rms_nongrip = np.sqrt(np.mean(heave_front[non_grip_mask]**2))
        rrh_rms_nongrip = np.sqrt(np.mean(heave_rear[non_grip_mask]**2))

        fl_ng = np.mean(f_tire[0:2, non_grip_mask], axis=0)
        front_load_rms_nongrip = np.sqrt(np.mean(fl_ng**2))
    else:
        frh_rms_nongrip = 0.0
        rrh_rms_nongrip = 0.0
        front_load_rms_nongrip = 0.0

    # RMS de desplazamiento de pista (zt) en mask grip-lateral y no-grip
    if np.any(grip_lateral_mask):
        ztrack_rms_grip = np.sqrt(np.mean(zt[:, grip_lateral_mask]**2))
    else:
        ztrack_rms_grip = 0.0

    if np.any(non_grip_mask):
        ztrack_rms_nongrip = np.sqrt(np.mean(zt[:, non_grip_mask]**2))
    else:
        ztrack_rms_nongrip = 0.0

    # RMS de pitch (global) en grados
    pitch_rms = np.sqrt(np.mean(phi**2)) * 180 / np.pi

    # ──────────────────────────────────────────────────────────────────────────────
    # 9) PSD (Heave global, Heave por eje, Pitch global, Pitch por eje) y FFT de carga
    # ──────────────────────────────────────────────────────────────────────────────
    # --- PSD de heave global [m²/Hz] ---
    f_psd_heave, Pxx_heave = welch(h, fs=fs, nperseg=nperseg, noverlap=noverlap)
    # --- PSD de heave por eje [mm²/Hz] ---
    z_front_mm = heave_front * 1000
    z_rear_mm  = heave_rear  * 1000
    f_heave_f, Pxx_heave_f = welch(z_front_mm, fs=fs, nperseg=nperseg, noverlap=noverlap)
    f_heave_r, Pxx_heave_r = welch(z_rear_mm,  fs=fs, nperseg=nperseg, noverlap=noverlap)
    # en dB (10·log10 porque es PSD – potencia)
    Pxx_heave_f_dB = 10 * np.log10(Pxx_heave_f + 1e-30)
    Pxx_heave_r_dB = 10 * np.log10(Pxx_heave_r + 1e-30)

    # --- PSD de pitch global [rad²/Hz] ---
    f_psd_pitch, Pxx_pitch = welch(phi, fs=fs, nperseg=nperseg, noverlap=noverlap)
    # --- PSD de pitch inducido [mm²/Hz] por eje ---
    pitch_front_mm = (-lf * phi) * 1000
    pitch_rear_mm  = ( lr * phi) * 1000
    f_pitch_f, Pxx_pitch_f = welch(pitch_front_mm, fs=fs, nperseg=nperseg, noverlap=noverlap)
    f_pitch_r, Pxx_pitch_r = welch(pitch_rear_mm,  fs=fs, nperseg=nperseg, noverlap=noverlap)
    Pxx_pitch_f_dB = 10 * np.log10(Pxx_pitch_f + 1e-30)
    Pxx_pitch_r_dB = 10 * np.log10(Pxx_pitch_r + 1e-30)

    # --- PSD de carga total por eje [N²/Hz] ---
    front_load_sig = 0.5*(wheel_load_N[0] + wheel_load_N[1])
    rear_load_sig  = 0.5*(wheel_load_N[2] + wheel_load_N[3])
    f_load, Pxx_load_front = welch(front_load_sig, fs=fs, nperseg=nperseg, noverlap=noverlap)
    _,     Pxx_load_rear  = welch(rear_load_sig,  fs=fs, nperseg=nperseg, noverlap=noverlap)
    # PSD en dB
    Pxx_load_front_dB = 10 * np.log10(Pxx_load_front + 1e-30)
    Pxx_load_rear_dB  = 10 * np.log10(Pxx_load_rear  + 1e-30)

    # --- FFT de carga total por eje (magnitud) ---
    # Solo frecuencias positivas:
    F_front = np.fft.rfft(front_load_sig)
    F_rear  = np.fft.rfft(rear_load_sig)
    f_vals  = np.fft.rfftfreq(len(t), d=dt)
    # Magnitud lineal → dB (20·log10 porque es amplitud)
    mag_front_dB = 20 * np.log10(np.abs(F_front) + 1e-30)
    mag_rear_dB  = 20 * np.log10(np.abs(F_rear)  + 1e-30)

    # ──────────────────────────────────────────────────────────────────────────────
    # 10) Acumulated Track Noise Normalized by Lap Time (mm/s)
    # ──────────────────────────────────────────────────────────────────────────────
    lap_time = float(t[-1] - t[0])  # [s]
    acc_tracknoise_mm = []
    for w in range(n_corners):
        z_mm = zt[w] * 1000                  # [m] → [mm]
        suma_abs = np.sum(np.abs(z_mm))      # Σ |z(tᵢ)| en [mm]
        acc_norm = suma_abs / lap_time       # [mm/s]
        acc_tracknoise_mm.append(acc_norm)

    # ──────────────────────────────────────────────────────────────────────────────
    # 11) Resultado final: devolver todos los valores en un diccionario
    # ──────────────────────────────────────────────────────────────────────────────
    return {
        # --- Cinemática básica ---
        'travel':              x_spring,               # (4, N)
        'travel_static':       travel_static,          # (4, 1)
        'travel_rel':          travel_rel,             # (4, N)
        'damper_travel':       zu - zs,                # (4, N)
        'margen_ext':          margen_ext,             # (4, N)
        'margen_comp':         margen_comp,            # (4, N)
        'z_free':              np.array([
                                   params['z_FL_free'], params['z_FR_free'],
                                   params['z_RL_free'], params['z_RR_free']
                               ]),

        # --- Fuerzas de suspensión ---
        'f_spring':            f_spring,               # (4, N)
        'f_damper':            f_damper,               # (4, N)
        'f_bump':              f_bump,                 # (4, N)

        # --- Fuerza normal total por rueda ---
        'wheel_load_N':        wheel_load_N,           # (4, N) [N]
        'wheel_load_kg':       wheel_load,             # (4, N) [kgf]

        # --- Estadísticos de neumático ---
        'f_tire':              f_tire,                 # (4, N)
        'f_tire_max':          f_tire_max,             # (4,)
        'f_tire_min':          f_tire_min,             # (4,)
        'f_tire_mean':         f_tire_mean,            # (4,)
        'f_tire_std':          f_tire_std,             # (4,)
        'f_tire_variation':    f_tire_var,
        'Fz_aero_front': ae_front,   # (N,)
        'Fz_aero_rear':  ae_rear,    # (N,)

        # --- Grip-limited ---
        'grip_mask_lat':       grip_lateral_mask,      # (N,)
        'f_tire_gl_max':     f_tire_gl_max,        # (4,) [N]
        'f_tire_gl_min':     f_tire_gl_min,        # (4,) [N]

        # --- RMS en grip-lateral vs no-grip ---
        'heave_f_rms_mm':      frh_rms,                # scalar [mm]
        'heave_r_rms_mm':      rrh_rms,                # scalar [mm]
        'load_f_rms_N':        front_load_rms,         # scalar [N]
        'load_r_rms_N':        rear_load_rms,          # scalar [N]
        'heave_f_rms_ng_mm':   frh_rms_nongrip,        # scalar [mm]
        'heave_r_rms_ng_mm':   rrh_rms_nongrip,        # scalar [mm]
        'load_f_rms_ng':       front_load_rms_nongrip, # scalar [N]
        'ztrack_rms_lat':      ztrack_rms_grip,        # scalar [m]
        'ztrack_rms_ng':       ztrack_rms_nongrip,     # scalar [m]
        'front_load_rms_brake':    front_load_rms_brake,   # scalar [N]
        'rear_load_rms_brake':     rear_load_rms_brake,    # scalar [N]
        'front_load_rms_traction': front_load_rms_traction,# scalar [N]
        'rear_load_rms_traction':  rear_load_rms_traction, # scalar [N]

        # --- Pitch y aceleraciones ---
        'pitch_rms_deg':       pitch_rms,              # scalar [°]
        'pitch_acc':           np.gradient(sol.y[3], t),   # (N,) [rad/s²]
        'roll_acc':            np.gradient(sol.y[5], t),   # (N,) [rad/s²]

        # --- Road noise (wheel-speed RMS) ---
        'roadnoise_front':     front_noise_vals,       # scalar [mm/s]
        'roadnoise_rear':      rear_noise_vals,        # scalar [mm/s]

        # --- PSD de heave y pitch ---
        'f_psd_heave':         f_psd_heave,            # (M,)
        'psd_heave':           Pxx_heave,              # (M,)
        'f_psd_heave_f':       f_heave_f,              # (M,)
        'psd_heave_f':         Pxx_heave_f,            # (M,)
        'psd_heave_f_dB':      Pxx_heave_f_dB,         # (M,)
        'f_psd_heave_r':       f_heave_r,              # (M,)
        'psd_heave_r':         Pxx_heave_r,            # (M,)
        'psd_heave_r_dB':      Pxx_heave_r_dB,         # (M,)
        'f_psd_pitch':         f_psd_pitch,            # (M,)
        'psd_pitch':           Pxx_pitch,              # (M,)

        # --- PSD y FFT de carga total ---
        'f_psd_load':          f_load,                 # (M,)
        'psd_load_f_dB':       Pxx_load_front_dB,      # (M,)
        'psd_load_r_dB':       Pxx_load_rear_dB,       # (M,)
        'f_fft_load':          f_vals,                 # (N//2+1,)
        'fft_load_mag_f_dB':   mag_front_dB,           # (N//2+1,)
        'fft_load_mag_r_dB':   mag_rear_dB,            # (N//2+1,)
        'f_psd_pitch_f':       f_pitch_f,              # (M,)
        'psd_pitch_f_dB':      Pxx_pitch_f_dB,         # (M,)
        'f_psd_pitch_r':       f_pitch_r,              # (M,)
        'psd_pitch_r_dB':      Pxx_pitch_r_dB,         # (M,)
        # --- Auxiliares de visualización ---
        'zt':                  zt,                     # (4, N)
        'z_chassis':           h,                      # (N,)
        'vx':                  np.interp(t, t_vec, vx),# (N,)
        'distance':            np.cumsum(np.interp(t, t_vec, vx))*dt,  # (N,)
        'lap_time':            lap_time,               # scalar
        'track_name':          params.get('track_name','Unknown Track'),
        't':                   t                       # (N,)
    }

# --- FIN DEL MODELO ---
