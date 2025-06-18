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

    z_topout_FL    = params['z_topout_FL']
    z_bottomout_FL = params['z_bottomout_FL']
    z_topout_FR    = params['z_topout_FR']
    z_bottomout_FR = params['z_bottomout_FR']
    z_topout_RL    = params['z_topout_RL']
    z_bottomout_RL = params['z_bottomout_RL']
    z_topout_RR    = params['z_topout_RR']
    z_bottomout_RR = params['z_bottomout_RR']

    x_bump_FL = params['x_bump_FL']
    x_bump_FR = params['x_bump_FR']
    x_bump_RL = params['x_bump_RL']
    x_bump_RR = params['x_bump_RR']

    g = 9.81

    # Helper para calcular la fuerza de rueda con clipping al recorrido físico
    def wheel_force(z_w, z_w_dot,
                    phi_off, theta_off,
                    phi_dot_off, theta_dot_off,
                    k_spring, k_inst,
                    bump, damper,
                    z_top, z_bot,
                    x_bump):
        x_raw = z_w - (phi_off + theta_off + h)
        # Clipping al recorrido físico
        x_clipped = np.clip(x_raw, z_top, z_bot)
        # Bumpstop (compresión más allá del gap)
        comp_bump = np.maximum(0.0, x_raw - x_bump)
        f_bump    = bump(comp_bump)
        # Resortes en serie: resorte + instalación
        k_total = 1.0/(1.0/k_spring + 1.0/k_inst)
        f_spring = k_total * x_clipped
        # Amortiguador: velocidad relativa resorte-carrocería
        rel_vel = z_w_dot - (phi_dot_off + theta_dot_off + hdot)
        f_damper = damper(rel_vel)
        # Tope como respaldo
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
                       x_bump_FL)

    F_FR = wheel_force(zFR, zFRdot,
                       phi_off_front, -theta_off_front,
                       phi_dot_off_front, -theta_dot_off_front,
                       kFR, kinstf,
                       bump_front, damper_front,
                       z_topout_FR, z_bottomout_FR,
                       x_bump_FR)

    F_RL = wheel_force(zRL, zRLdot,
                       phi_off_rear, theta_off_rear,
                       phi_dot_off_rear, theta_dot_off_rear,
                       kRL, kinstr,
                       bump_rear, damper_rear,
                       z_topout_RL, z_bottomout_RL,
                       x_bump_RL)

    F_RR = wheel_force(zRR, zRRdot,
                       phi_off_rear, -theta_off_rear,
                       phi_dot_off_rear, -theta_dot_off_rear,
                       kRR, kinstr,
                       bump_rear, damper_rear,
                       z_topout_RR, z_bottomout_RR,
                       x_bump_RR)

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
    h_ddot     = (F_FL + F_FR + F_RL + F_RR - Ms*g) / Ms
    phi_ddot   = (lr*(F_RR + F_RL) - lf*(F_FR + F_FL)) / Iyy
    theta_ddot = ((tF/2)*(F_FL - F_FR) + (tR/2)*(F_RL - F_RR)) / Ixx

    # Masas no suspendidas
    zFR_ddot = (-F_FR + params['ktf']*(ztrack_FR - zFR)) / mHubF
    zFL_ddot = (-F_FL + params['ktf']*(ztrack_FL - zFL)) / mHubF
    zRL_ddot = (-F_RL + params['ktr']*(ztrack_RL - zRL)) / mHubR
    zRR_ddot = (-F_RR + params['ktr']*(ztrack_RR - zRR)) / mHubR

    return [
        hdot,     h_ddot,
        phi_dot,  phi_ddot,
        theta_dot,theta_ddot,
        zFRdot,   zFR_ddot,
        zFLdot,   zFL_ddot,
        zRLdot,   zRL_ddot,
        zRRdot,   zRR_ddot
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
        kFR_eff = 1 / (1 / kFR + 1 / kinstf)
        kFL_eff = 1 / (1 / kFL + 1 / kinstf)
        kRR_eff = 1 / (1 / kRR + 1 / kinstr)
        kRL_eff = 1 / (1 / kRL + 1 / kinstr)

        # Fuerzas suspensión: muelle + bump-stop por separado
        f_bump_FR = bumpstop_front(np.maximum(0, x_FR - z_FR_free))
        f_bump_FL = bumpstop_front(np.maximum(0, x_FL - z_FL_free))
        f_bump_RR = bumpstop_rear(np.maximum(0, x_RR - z_RR_free))
        f_bump_RL = bumpstop_rear(np.maximum(0, x_RL - z_RL_free))

        F_FR = kFR_eff * x_FR + f_bump_FR
        F_FL = kFL_eff * x_FL + f_bump_FL
        F_RR = kRR_eff * x_RR + f_bump_RR
        F_RL = kRL_eff * x_RL + f_bump_RL

        # --- Fuerzas de la antirollbar ---
        if k_arb_f != 0.0:
            T_arb_f = k_arb_f * (x_FL - x_FR)
            lever_f = tf / 2.0
            F_FL += T_arb_f / lever_f
            F_FR += -T_arb_f / lever_f
        if k_arb_r != 0.0:
            T_arb_r = k_arb_r * (x_RL - x_RR)
            lever_r = tr / 2.0
            F_RL += T_arb_r / lever_r
            F_RR += -T_arb_r / lever_r

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

    # tras calcular x_FL_static, x_FR_static, etc.:
    for corner, x_stat in zip(
        ['FL','FR','RL','RR'],
        [x_FL_static, x_FR_static, x_RL_static, x_RR_static]
    ):
        stroke = params[f'stroke_{corner}']           # carrera total
        # por defecto, asignamos todo el espacio por encima del punto estático
        # como stroke_top = compresión estática, y el resto a stroke_bottom
        stroke_top    = params.get(f'stroke_top_{corner}', x_stat)
        stroke_bottom = params.get(f'stroke_bottom_{corner}', stroke - stroke_top)
        # gap_bumpstop_corner = espacio libre desde la compresión estática
        gap0 = params[f'gap_bumpstop_{corner}']

        # x_bump = posición a partir de la cual empieza a actuar el bumpstop
        params[f'x_bump_{corner}'] = x_stat + gap0
        params[f'z_topout_{corner}']    = x_stat - stroke_top
        params[f'z_bottomout_{corner}'] = x_stat + stroke_bottom
    # Guardamos el punto estático para cálculo de topes reales
    params.update({
        'x_FL_static': x_FL_static,
        'x_FR_static': x_FR_static,
        'x_RL_static': x_RL_static,
        'x_RR_static': x_RR_static,
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
    nperseg = 1024
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
    travel_abs = x_spring.copy()

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
        0.5 * ae_front,
        0.5 * ae_front,
        0.5 * ae_rear,
        0.5 * ae_rear,
    ])  # (4, N)

    kt = np.array([
        params['ktf'], params['ktf'],
        params['ktr'], params['ktr']
    ])  # neumáticos

    f_tire     = kt[:,None] * (zt - zu)   # (4, N)
    wheel_load = aero + static + f_tire   #  + f_tire(4, N)

    # ──────────────────────────────────────────────────────────────────────────────
    # 3.3) Márgenes dinámicos y estáticos traseros
    # ──────────────────────────────────────────────────────────────────────────────
    margen_ext       = x_spring - z_topout[:,None]
    margen_comp      = z_bottomout[:,None] - x_spring
    xRL_static       = x_spring[2,0]
    xRR_static       = x_spring[3,0]
    z_free_RL        = params['z_RL_free']
    z_free_RR        = params['z_RR_free']
    margen_ext_s_RL  = xRL_static - params['z_topout_RL']
    margen_comp_s_RL = z_free_RL - (xRL_static - params['z_topout_RL'])
    margen_ext_s_RR  = xRR_static - params['z_topout_RR']
    margen_comp_s_RR = z_free_RR - (xRR_static - params['z_topout_RR'])

    print(f"[INFO] Travel estático trasero: RL = {xRL_static*1000:.2f} mm, RR = {xRR_static*1000:.2f} mm")
    print(f"[INFO] Margen RL (estático corregido): extensión = {margen_ext_s_RL*1000:.2f} mm, compresión = {margen_comp_s_RL*1000:.2f} mm")
    print(f"[INFO] Margen RR (estático corregido): extensión = {margen_ext_s_RR*1000:.2f} mm, compresión = {margen_comp_s_RR*1000:.2f} mm")

    travel_static   = x_spring[:,0][:,None]
    travel_rel      = x_spring - travel_static
    travel_max      = np.max(travel_rel,axis=1)
    travel_min      = np.min(travel_rel,axis=1)
    travel_range    = travel_max - travel_min
    travel_used_pct = 100 * travel_range / stroke

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

    # --- Fuerzas de la antirollbar ---
    k_arb_f = params.get('k_arb_f', 0.0)
    k_arb_r = params.get('k_arb_r', 0.0)
    f_arb = np.zeros_like(x_spring)
    arb_torque_front = np.zeros(N)
    arb_torque_rear  = np.zeros(N)
    if k_arb_f != 0.0:
        arb_torque_front = k_arb_f * (x_spring[0] - x_spring[1])
        lever_f = track_f / 2.0
        f_arb[0] += arb_torque_front / lever_f
        f_arb[1] += -arb_torque_front / lever_f
    if k_arb_r != 0.0:
        arb_torque_rear = k_arb_r * (x_spring[2] - x_spring[3])
        lever_r = track_r / 2.0
        f_arb[2] += arb_torque_rear / lever_r
        f_arb[3] += -arb_torque_rear / lever_r

        # 4d) Fuerza neta en rueda (nunca negativa)
    wheel_load = (aero + static + f_tire + f_arb) / 9.81   #  + f_tire(4, N)
    wheel_load_max = np.max(wheel_load, axis=1)   # máximo por rueda [N]
    wheel_load_min = np.min(wheel_load, axis=1)   # mínimo por rueda [N]
    f_wheel = (f_spring + f_bump + f_damper + f_arb)/ 9.81    # (4, N)
    f_wheel[f_wheel < 0] = 0                  # clamp por si acaso

    f_damp_FL, Pxx_damp_FL = welch(f_damper[0], fs=fs, nperseg=nperseg, noverlap=noverlap)
    f_damp_RL, Pxx_damp_RL = welch(f_damper[2], fs=fs, nperseg=nperseg, noverlap=noverlap)

    # ──────────────────────────────────────────────────────────────────────────────
    # 5) Fuerza de neumático y track excitation (interpolación de z_tracks)
    # ──────────────────────────────────────────────────────────────────────────────
    f_tire_max    = np.max(f_tire,axis=1)
    f_tire_min    = np.min(f_tire,axis=1)
    f_tire_mean   = np.mean(f_tire,axis=1)
    f_tire_std    = np.std(f_tire,axis=1)
    f_tire_var    = f_tire_std / f_tire_mean
    f_tire_var_f  = np.mean(f_tire_var[:2])
    f_tire_var_r  = np.mean(f_tire_var[2:])

    # ──────────────────────────────────────────────────────────────────────────────
    # 6) Condiciones Grip-Limited (lateral, frenada y tracción)
    # ──────────────────────────────────────────────────────────────────────────────
    throttle_signal = np.interp(t, t_vec, throttle)
    brake_signal    = np.interp(t, t_vec, brake)
    threshold       = 0.5

    grip_lateral_mask    = (throttle_signal < threshold) & (brake_signal < threshold)
    grip_brake_mask      = (brake_signal > threshold)    & (throttle_signal < threshold)
    grip_traction_mask   = (throttle_signal > threshold) & (brake_signal < threshold)
    grip_limited_pct     = 100 * np.sum(grip_lateral_mask) / N

    # Fuerza tire grip-limited (max y min) en condición lateral
    if np.any(grip_lateral_mask):
        f_tire_gl_max = np.max(f_tire[:, grip_lateral_mask], axis=1)
        f_tire_gl_min = np.min(f_tire[:, grip_lateral_mask], axis=1)
    else:
        f_tire_gl_max = np.zeros(4)
        f_tire_gl_min = np.zeros(4)

    # ── Grip-Limited en frenada ──
    if np.any(grip_brake_mask):
        fl_brake = np.mean(f_tire[0:2, grip_brake_mask], axis=0)
        rl_brake = np.mean(f_tire[2:4, grip_brake_mask], axis=0)
        front_load_rms_brake = np.sqrt(np.mean(fl_brake**2))
        rear_load_rms_brake  = np.sqrt(np.mean(rl_brake**2))
    else:
        front_load_rms_brake = 0.0
        rear_load_rms_brake  = 0.0

    # ── Grip-Limited en tracción ──
    if np.any(grip_traction_mask):
        fl_trac = np.mean(f_tire[0:2, grip_traction_mask], axis=0)
        rl_trac = np.mean(f_tire[2:4, grip_traction_mask], axis=0)
        front_load_rms_traction = np.sqrt(np.mean(fl_trac**2))
        rear_load_rms_traction  = np.sqrt(np.mean(rl_trac**2))
    else:
        front_load_rms_traction = 0.0
        rear_load_rms_traction  = 0.0

    # ──────────────────────────────────────────────────────────────────────────────
    # 7) Road-noise: wheel vertical-speed RMS [mm/s] para cada rueda
    # ──────────────────────────────────────────────────────────────────────────────
    # zt está en [m], convertimos a mm antes de derivar
    v_mm_s = [np.gradient(zt[i] * 1000, dt) for i in range(n_corners)]  # lista de 4 arrays (mm/s)
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
        frh_rms = np.sqrt(np.mean(heave_front[grip_lateral_mask]**2))
        rrh_rms = np.sqrt(np.mean(heave_rear[grip_lateral_mask]**2))
    else:
        frh_rms = 0.0
        rrh_rms = 0.0

    # Cargas front/rear  RMS en mask grip-lateral
    if np.any(grip_lateral_mask):
        fl_gl = np.mean(f_tire[0:2, grip_lateral_mask], axis=0)
        rl_gl = np.mean(f_tire[2:4, grip_lateral_mask], axis=0)
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
    # 9) PSD (Heave global, Heave por eje, Pitch global, Pitch por eje)
    # ──────────────────────────────────────────────────────────────────────────────
    # Heave del chasis [m]
    f_psd_heave, Pxx_heave   = welch(h,  fs=fs, nperseg=nperseg, noverlap=noverlap)
    # Heave por eje en mm
    z_front_mm   = heave_front * 1000
    z_rear_mm    = heave_rear  * 1000
    f_heave_f, Pxx_heave_f   = welch(z_front_mm, fs=fs, nperseg=nperseg, noverlap=noverlap)
    f_heave_r, Pxx_heave_r   = welch(z_rear_mm,  fs=fs, nperseg=nperseg, noverlap=noverlap)

    # Pitch global [rad]
    f_psd_pitch, Pxx_pitch   = welch(phi, fs=fs, nperseg=nperseg, noverlap=noverlap)
    # Pitch inducido en vertical por eje (m → mm)
    pitch_front_mm = (-lf * phi) * 1000
    pitch_rear_mm  = ( lr * phi) * 1000
    f_pitch_f, Pxx_pitch_f   = welch(pitch_front_mm, fs=fs, nperseg=nperseg, noverlap=noverlap)
    f_pitch_r, Pxx_pitch_r   = welch(pitch_rear_mm,  fs=fs, nperseg=nperseg, noverlap=noverlap)

    # Convertir a picos en dB (opcional)
    psd_heave_peak = 20 * np.log10(np.max(Pxx_heave) + 1e-30)
    psd_pitch_peak = 20 * np.log10(np.max(Pxx_pitch) + 1e-30)

    # ──────────────────────────────────────────────────────────────────────────────
    # 9bis) Calcular la FFT/PSD de la Fuerza de Neumático (“carga”) – Front vs Rear
    # ──────────────────────────────────────────────────────────────────────────────
    #   f_tire es ya un array (4, N) con [FL, FR, RL, RR].

    front_signal = (wheel_load[0, :]) /9.81  # Promedio de fuerza en las dos ruedas delanteras
    rear_signal  = (wheel_load[2, :]) /9.81  # Promedio de fuerza en las dos ruedas traseras

    # Usamos FFT unidireccional (solo frecuencias positivas):
    F_front = np.fft.rfft(front_signal)    # Componente compleja espectral frontal
    F_rear  = np.fft.rfft(rear_signal)     # Componente compleja espectral trasera
    f_vals  = np.fft.rfftfreq(N, d=dt)     # Vector de frecuencias (longitud N//2 + 1)

    # Magnitud lineal y a continuación convertimos a dB (normalizo por (N/2)):
    mag_front_linear = np.abs(F_front)                     
    mag_rear_linear  = np.abs(F_rear)
    mag_front_dB = 20.0 * np.log10(mag_front_linear / (N/2) + 1e-30)
    mag_rear_dB  = 20.0 * np.log10(mag_rear_linear  / (N/2) + 1e-30)

    # Fase en grados:
    phase_front_deg = np.angle(F_front, deg=True)
    phase_rear_deg  = np.angle(F_rear,  deg=True)

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
        'travel':             x_spring,               # (4, N)
        'travel_rel':         travel_rel,             # (4, N)
        'travel_static':      travel_static,          # (4, 1)
        'travel_max':         travel_max,             # (4,)
        'travel_min':         travel_min,             # (4,)
        'travel_range':       travel_range,           # (4,)
        'travel_used_pct':    travel_used_pct,        # (4,)
        'travel_abs':         travel_abs,             # (4, N)
        'damper_travel':      zu - zs,                # (4, N)
        'margen_ext':         margen_ext,             # (4, N)
        'margen_comp':        margen_comp,            # (4, N)
        'z_free':             np.array([
                                  params['z_FL_free'], params['z_FR_free'],
                                  params['z_RL_free'], params['z_RR_free']
                              ]),

        # --- Fuerzas de suspensión ---
        'f_spring':           f_spring,               # (4, N)
        'f_damper':           f_damper,               # (4, N)
        'f_bump':             f_bump,                 # (4, N)
        'f_arb':              f_arb,                  # (4, N)
        'arb_torque_front':   arb_torque_front,       # (N,)
        'arb_torque_rear':    arb_torque_rear,   

        # --- Fuerza de neumático ---
        'f_tire':             f_tire,                 # (4, N)
        'f_tire_max':         f_tire_max,             # (4,)
        'f_tire_min':         f_tire_min,             # (4,)
        'f_tire_mean':        f_tire_mean,            # (4,)
        'f_tire_std':         f_tire_std,             # (4,)
        'f_tire_variation':   f_tire_var,
        'f_tire_variation_front': f_tire_var_f,
        'f_tire_variation_rear':  f_tire_var_r,
        'Fz_aero_front':      ae_front,               # (N,) [N]
        'Fz_aero_rear':       ae_rear,                # (N,) [N]
        'f_tire_raw':  f_tire,      
        'wheel_load':  wheel_load, 
        'wheel_load_max':    wheel_load_max,        # (4,)
        'wheel_load_min':    wheel_load_min,        # (4,)
        'f_wheel':      f_wheel,


        # --- Grip-limited (lateral, brake, traction) ---
        'grip_limited_lateral_mask':  grip_lateral_mask,      # (N,)
        'grip_limited_brake_mask':    grip_brake_mask,        # (N,)
        'grip_limited_traction_mask': grip_traction_mask,     # (N,)
        'grip_limited_lateral_pct':   grip_limited_pct,       # scalar
        'f_tire_grip_limited_max':    f_tire_gl_max,          # (4,)
        'f_tire_grip_limited_min':    f_tire_gl_min,          # (4,)
        'front_load_rms_brake':       front_load_rms_brake,   # scalar
        'rear_load_rms_brake':        rear_load_rms_brake,    # scalar
        'front_load_rms_traction':    front_load_rms_traction,# scalar
        'rear_load_rms_traction':     rear_load_rms_traction, # scalar

        # --- RMS de heave / carga en grip-limited vs no-grip ---
        'frh_rms':            frh_rms,                # heave front RMS [m]
        'rrh_rms':            rrh_rms,                # heave rear RMS [m]
        'front_load_rms':     front_load_rms,         # tire load front RMS [N]
        'rear_load_rms':      rear_load_rms,          # tire load rear RMS [N]
        'frh_rms_nongrip':    frh_rms_nongrip,        # heave front RMS no-grip [m]
        'rrh_rms_nongrip':    rrh_rms_nongrip,        # heave rear RMS no-grip [m]
        'front_load_rms_nongrip': front_load_rms_nongrip,
        'ztrack_rms_grip':    ztrack_rms_grip,        # (scalar)
        'ztrack_rms_nongrip': ztrack_rms_nongrip,     # (scalar)

        # --- Pitch y roll ---
        'pitch_rms':          pitch_rms,              # RMS φ [°]
        'phi_ddot':           np.gradient(sol.y[3], t),   # pitch acc [rad/s²]
        'theta_ddot':         np.gradient(sol.y[5], t),   # roll acc [rad/s²]

        # --- Road noise (wheel-speed RMS) ---
        'tracknoise_accu_front': front_noise_vals,    # (scalar)
        'tracknoise_accu_rear':  rear_noise_vals,     # (scalar)

        # --- PSDs (global y por eje) ---
        # Heave global [m²/Hz]:
        'f_psd':              f_psd_heave,            # (M,)
        'psd_heave':          Pxx_heave,              # (M,)
        'psd_heave_peak':     psd_heave_peak,         # dB

        # Heave por eje [mm²/Hz]:
        'heave_front':        z_front_mm,             # (N,)
        'heave_rear':         z_rear_mm,              # (N,)
        'f_psd_front':        f_heave_f,              # (M,)
        'psd_heave_front':    Pxx_heave_f,            # (M,)
        'f_psd_rear':         f_heave_r,              # (M,)
        'psd_heave_rear':     Pxx_heave_r,            # (M,)

        # Pitch global [rad²/Hz]:
        'f_psd_pitch':        f_psd_pitch,            # (M,)
        'psd_pitch':          Pxx_pitch,              # (M,)
        'psd_pitch_peak':     psd_pitch_peak,         # dB

        # Pitch inducido → vertical por eje [mm²/Hz]:
        'f_psd_pitch_front':  f_pitch_f,              # (M,)
        'psd_pitch_front':    Pxx_pitch_f,            # (M,)
        'f_psd_pitch_rear':   f_pitch_r,              # (M,)
        'psd_pitch_rear':     Pxx_pitch_r,            # (M,)

        'f_psd_damper'   : f_damp_FL,                  # vector de frecuencias
        'psd_damper_mag_FL' : 10 * np.log10(Pxx_damp_FL),
        'psd_damper_mag_RL' : 10 * np.log10(Pxx_damp_RL),

        # --- Resultados auxiliares para Dash/HTML export ---
        'travel_static':      travel_static,          # (4,1)
        'zt':                 zt,                     # (4, N)
        'z_chassis':          h,                      # (N,)
        'h_ddot':             np.gradient(sol.y[1], t),   # (N,)
        'v_damper':           v_damper,               # (4, N)
        'x_spring':           x_spring,               # (4, N)
        'v_chassis':          v_chassis,              # (4, N)
        'lap_time':           lap_time,               # scalar
        'pitch_deg':          np.degrees(phi),        # (N,)
        'distance':           np.cumsum(vx_interp := np.interp(t, t_vec, vx)) * np.gradient(t),  # (N,)
        'track_name':         params.get('track_name', 'Unknown Track'),
        'vx': vx_interp,
        'z_topout_FL': params['z_topout_FL'],
        'z_bottomout_FL': params['z_bottomout_FL'],
        'z_topout_FR': params['z_topout_FR'],
        'z_bottomout_FR': params['z_bottomout_FR'],
        'z_topout_RL': params['z_topout_RL'],
        'z_bottomout_RL': params['z_bottomout_RL'],
        'z_topout_RR': params['z_topout_RR'],
        'z_bottomout_RR': params['z_bottomout_RR'],
        't_vec':              t,                     # (N,)
        'f_psd_load'           : f_vals,             # vector de frecuencias [Hz]
        'psd_load_mag_front'   : mag_front_dB,       # magnitud dB eje frontal
        'psd_load_phase_front' : phase_front_deg,    # fase [°] eje frontal
        'psd_load_mag_rear'    : mag_rear_dB,        # magnitud dB eje trasero
        'psd_load_phase_rear'  : phase_rear_deg,     # fase [°] eje trasero        
    }


# --- FIN DEL MODELO ---
