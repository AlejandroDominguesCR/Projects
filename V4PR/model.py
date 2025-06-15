import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.signal import welch
from scipy.optimize import fsolve

def tope_fuerza(x, topout, bottomout, k_tope=1e6):
    if x < topout:
        return -k_tope * (topout - x)
    elif x > bottomout:
        return -k_tope * (bottomout - x)
    else:
        return 0.0

def vehicle_model_simple(t, z, params, ztrack_funcs):
    # Desempaquetado de estados
    h, hdot, phi, phi_dot, theta, theta_dot, zFR, zFRdot, zFL, zFLdot, zRL, zRLdot, zRR, zRRdot = z

    # Interpola inputs de pista
    ztrack_FL, ztrack_FR, ztrack_RL, ztrack_RR = [f(t) for f in ztrack_funcs]

    # Parámetros globales
    Ms, Ixx, Iyy = params['ms'], params['Ixx'], params['Iyy']
    lf, lr, tF, tR = params['lf'], params['lr'], params['tf'], params['tr']
    mHubF, mHubR   = params['mHubF'], params['mHubR']

    # Resortes y amortiguadores
    kFL, kFR = params['kFL'], params['kFR']
    kRL, kRR = params['kRL'], params['kRR']
    kinstf, kinstr = params['kinstf'], params['kinstr']
    bump_front, bump_rear = params['bumpstop_front'], params['bumpstop_rear']
    damper_front, damper_rear = params['damper_front'], params['damper_rear']
    ktf, ktr = params['ktf'], params['ktr']

    # Stroke y top/bottom estático (coordenada rueda)
    stroke_FL = params.get('stroke_FL')
    stroke_FR = params.get('stroke_FR')
    stroke_RL = params.get('stroke_RL')
    stroke_RR = params.get('stroke_RR')

    # Coordenadas estáticas del muelle (rueda)
    x_FL_static = params['x_FL_static']
    x_FR_static = params['x_FR_static']
    x_RL_static = params['x_RL_static']
    x_RR_static = params['x_RR_static']

    z_top_FL    = (x_FL_static - stroke_FL/2)
    z_bot_FL    = (x_FL_static + stroke_FL/2)
    z_top_FR    = (x_FR_static - stroke_FR/2)
    z_bot_FR    = (x_FR_static + stroke_FR/2)
    z_top_RL    = (x_RL_static - stroke_RL/2)
    z_bot_RL    = (x_RL_static + stroke_RL/2)
    z_top_RR    = (x_RR_static - stroke_RR/2)
    z_bot_RR    = (x_RR_static + stroke_RR/2)
    
    g = 9.81

    # Helper para fuerza de rueda con MR en desplazamiento y velocidad
    def wheel_force(z_w, z_w_dot, phi_off, theta_off, phi_dot_off, theta_dot_off,
                    k_s, k_i, bump, damper, gap, z_top, z_bot):
        # 1) desplazamiento de rueda respecto al chasis
        x_raw = z_w - (phi_off + theta_off + h)
        # 2) bump-stop ya en coordenada rueda
        comp = x_raw - gap
        f_bump = bump(np.maximum(0, comp))
        # 3) rigidez equivalente muelle + compliance de instalación
        k_wheel = 1.0 / (1.0 / k_s + 1.0 / k_i)
        f_spring = k_wheel * x_raw
        # 4) amortiguador en coordenada rueda
        v_raw = z_w_dot - (phi_dot_off + theta_dot_off + hdot)
        f_damp = damper(v_raw)
        # 5) topes mecánicos en coordenada rueda
        f_stop = tope_fuerza(x_raw, z_top, z_bot)
        return f_spring + f_bump + f_damp + f_stop

    # Offsets estáticos y dinámicos
    phi_off_f = -lf * phi;  phi_off_r = lr * phi
    theta_off_f = (tF/2) * theta;  theta_off_r = (tR/2) * theta
    phi_dot_off_f = -lf * phi_dot;  phi_dot_off_r = lr * phi_dot
    theta_dot_off_f = (tF/2) * theta_dot;  theta_dot_off_r = (tR/2) * theta_dot

    x_FL = zFL - (phi_off_f + theta_off_f + h)
    x_FR = zFR - (phi_off_f - theta_off_f + h)
    x_RL = zRL - (phi_off_r + theta_off_r + h)
    x_RR = zRR - (phi_off_r - theta_off_r + h)

    # Fuerzas por rueda
    F_FL = wheel_force(
        zFL, zFLdot, phi_off_f, theta_off_f, phi_dot_off_f, theta_dot_off_f,
        kFL, kinstf, bump_front, damper_front,
        params['gap_bumpstop_FL'], z_top_FL, z_bot_FL
    )
    F_FR = wheel_force(
        zFR, zFRdot, phi_off_f, -theta_off_f, phi_dot_off_f, -theta_dot_off_f,
        kFR, kinstf, bump_front, damper_front,
        params['gap_bumpstop_FR'], z_top_FR, z_bot_FR
    )
    F_RL = wheel_force(
        zRL, zRLdot, phi_off_r, theta_off_r, phi_dot_off_r, theta_dot_off_r,
        kRL, kinstr, bump_rear, damper_rear,
        params['gap_bumpstop_RL'], z_top_RL, z_bot_RL
    )
    F_RR = wheel_force(
        zRR, zRRdot, phi_off_r, -theta_off_r, phi_dot_off_r, -theta_dot_off_r,
        kRR, kinstr, bump_rear, damper_rear,
        params['gap_bumpstop_RR'], z_top_RR, z_bot_RR
    )
    
    F_arb_front = 0.5 * params['k_arb_f'] * (x_FL - x_FR)
    F_arb_rear  = 0.5 * params['k_arb_r'] * (x_RL - x_RR)

    F_FL +=  F_arb_front
    F_FR += -F_arb_front
    F_RL +=  F_arb_rear
    F_RR += -F_arb_rear

    # Dinámica chasis
    h_dd = (F_FL + F_FR + F_RL + F_RR - Ms*g) / Ms
    theta_dd = ((tF/2)*(F_FL - F_FR) + (tR/2)*(F_RL - F_RR)) / Ixx
    phi_dd = (lr*(F_RR + F_RL) - lf*(F_FR + F_FL)) / Iyy

    # Masas no suspendidas
    zFR_dd = (-F_FR + ktf*(ztrack_FR - zFR) - mHubF*g) / mHubF
    zFL_dd = (-F_FL + ktf*(ztrack_FL - zFL) - mHubF*g) / mHubF
    zRL_dd = (-F_RL + ktr*(ztrack_RL - zRL) - mHubR*g) / mHubR
    zRR_dd = (-F_RR + ktr*(ztrack_RR - zRR) - mHubR*g) / mHubR

    return [
        hdot, h_dd,
        phi_dot, phi_dd,
        theta_dot, theta_dd,
        zFRdot, zFR_dd,
        zFLdot, zFL_dd,
        zRLdot, zRL_dd,
        zRRdot, zRR_dd
    ]

def compute_static_equilibrium(params):
    """
    Resuelve el equilibrio estático: encuentra h, phi, theta y las posiciones
    de las masas no suspendidas de forma que las fuerzas verticales y los momentos
    se equilibren. Además imprime una comparativa entre compresión teórica,
    compresión real y xFreeGap del JSON.
    """
    # Constantes y parámetros

    h0_est = estimate_h0_from_static_travel(params, phi=0.0, theta=0.0)

    g      = 9.81
    Ms     = params['ms']
    lf     = params['lf']
    lr     = params['lr']
    tf     = params['tf']
    tr     = params['tr']
    mHubF  = params['mHubF']
    mHubR  = params['mHubR']

    kFL    = params['kFL']
    kFR    = params['kFR']
    kRL    = params['kRL']
    kRR    = params['kRR']
    kinstf = params['kinstf']
    kinstr = params['kinstr']
    ktf    = params['ktf']
    ktr    = params['ktr']

    # Gaps tal como vienen del JSON (medidos con el coche en el suelo)
    gap_FL = params['gap_bumpstop_FL']
    gap_FR = params['gap_bumpstop_FR']
    gap_RL = params['gap_bumpstop_RL']
    gap_RR = params['gap_bumpstop_RR']

    # Antirrolbar si está presente
    k_arb_f = params.get('k_arb_f', 0.0)
    k_arb_r = params.get('k_arb_r', 0.0)

    def residual(x):
        # Variables: h, phi, theta, zFR, zFL, zRL, zRR
        h, phi, theta, zFR, zFL, zRL, zRR = x

        # Posiciones chassis en cada esquina
        zs_FR = h - lf*phi - (tf/2)*theta
        zs_FL = h - lf*phi + (tf/2)*theta
        zs_RR = h + lr*phi - (tr/2)*theta
        zs_RL = h + lr*phi + (tr/2)*theta

        # Travel de suspensión
        x_FR = zFR - zs_FR
        x_FL = zFL - zs_FL
        x_RR = zRR - zs_RR
        x_RL = zRL - zs_RL

        # Guardar compresión real
        params['x_static_FL'] = x_FL
        params['x_static_FR'] = x_FR
        params['x_static_RL'] = x_RL
        params['x_static_RR'] = x_RR

        # Wheel-rate (muelle trasladado a rueda)
        kFL_w = kFL
        kFR_w = kFR
        kRL_w = kRL
        kRR_w = kRR

        # Rigidez efectiva (muelle en serie con instalación)
        kFL_eff = 1.0/(1.0/kFL_w + 1.0/kinstf)
        kFR_eff = 1.0/(1.0/kFR_w + 1.0/kinstf)
        kRL_eff = 1.0/(1.0/kRL_w + 1.0/kinstr)
        kRR_eff = 1.0/(1.0/kRR_w + 1.0/kinstr)

        # Fuerzas de resorte
        F_spring_FL = kFL_eff * x_FL
        F_spring_FR = kFR_eff * x_FR
        F_spring_RL = kRL_eff * x_RL
        F_spring_RR = kRR_eff * x_RR

        # Anti-roll bar
        F_arb_front = (k_arb_f/2.0)*(x_FL - x_FR)
        F_arb_rear  = (k_arb_r/2.0)*(x_RL - x_RR)

        # Suma de fuerzas de suspensión (sin bump-stop)
        F_FL = F_spring_FL +  F_arb_front
        F_FR = F_spring_FR -  F_arb_front
        F_RL = F_spring_RL +  F_arb_rear
        F_RR = F_spring_RR -  F_arb_rear

        # 1) Equilibrio vertical chasis
        R1 = (F_FL + F_FR + F_RL + F_RR) - Ms*g
        # 2) Momento roll
        R2 = (tf/2)*(F_FL - F_FR) + (tr/2)*(F_RL - F_RR)
        # 3) Momento pitch
        R3 = lr*(F_RR + F_RL) - lf*(F_FR + F_FL)
        # 4-7) Equilibrio masas no suspendidas (neumático vs peso)
        R4 = -F_FR + ktf*(0.0 - zFR) - mHubF*g
        R5 = -F_FL + ktf*(0.0 - zFL) - mHubF*g
        R6 = -F_RR + ktr*(0.0 - zRR) - mHubR*g
        R7 = -F_RL + ktr*(0.0 - zRL) - mHubR*g

        return [R1, R2, R3, R4, R5, R6, R7]

    # Condiciones iniciales
    h_init   = h0_est # (lr*params.get('hRideF',0.02) + lf*params.get('hRideR',0.04)) / (lf+lr)
    phi_init = (params.get('hRideR',0.04) - params.get('hRideF',0.02)) / (lf+lr)
    theta_init = 0.0

    # Estimación inicial masas no suspendidas
    W   = Ms*g  
    wbal_f = params['rWeightBalF']
    Wf     = W * wbal_f
    Wr     = W * (1.0 - wbal_f)

    # Wheel-rates para inicial guess
    kFL_w = kFL
    kFR_w = kFR
    kRL_w = kRL
    kRR_w = kRR

    kFL_eff = 1.0/(1.0/kFL_w + 1.0/kinstf)
    kFR_eff = 1.0/(1.0/kFR_w + 1.0/kinstf)
    kRL_eff = 1.0/(1.0/kRL_w + 1.0/kinstr)
    kRR_eff = 1.0/(1.0/kRR_w + 1.0/kinstr)

    zFR0 = Wf/(2*kFR_eff) 
    zFL0 = Wf/(2*kFL_eff) 
    zRR0 = Wr/(2*kRR_eff) 
    zRL0 = Wr/(2*kRL_eff) 

    x0 = [h_init, phi_init, theta_init, zFR0, zFL0, zRL0, zRR0]

    # Resolver estático
    sol, info, ier, msg = fsolve(residual, x0, full_output=True)
    if ier != 1:
        print(f"[WARNING] fsolve no ha convergido: {msg}")

    # ——— Comparativa compresión teórica vs real vs xFreeGap ———
    # justo después de compute_static_equilibrium(params)
    corners = ['FL','FR','RL','RR']
    x_wheel   = {c: params[f'x_static_{c}']    for c in corners}
    k_spring  = {c: params[f'k{c}']            for c in corners}
    k_spring2 = {c: k_spring[c]                for c in corners}
    print("\n[CHECK SPRING]: compresión de muelle en estático")
    for c in corners:
        x_w     = x_wheel[c]
        x_s     = x_w                 # muelle en coordenada rueda
        f_s     = k_spring2[c] * x_w
        print(
            f"  {c}: travel rueda = {x_w*1e3:6.2f} mm, "
            f"compresión muelle = {x_s*1e3:6.2f} mm, "
            f"f_spring = {f_s:7.1f} N, "
        )


    return sol

def run_vehicle_model_simple(t_vec, z_tracks, params):
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

    # --- Equilibrio estático ---
    h0, phi0, theta0, zFR0, zFL0, zRL0, zRR0 = compute_static_equilibrium(params)

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
    })

    if params.get('override_bumpstop_positions', False):
        params.update({
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
     'FL': params['gap_bumpstop_FL'],
     'FR': params['gap_bumpstop_FR'],
     'RL': params['gap_bumpstop_RL'],
     'RR': params['gap_bumpstop_RR'],
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

        if corner in ('FL','FR'):
            z_ui = params.get('hRideF')
        else:
            z_ui = params.get('hRideR')
        z_si = z_ui - x_spring_clipped - z_free[corner]      # posición del chasis en esa esquina
        h_i = z_si - (phi * xi + theta * yi)                 # despejamos h
        h_list.append(h_i)
    print(f"[DEBUG] Reparto de peso: Wf = {Wf:.2f} N, Wr = {Wr:.2f} N")
    return np.mean(h_list)

def postprocess_7dof(sol, params, z_tracks, t_vec):
    """
    Calcula travel, fuerzas, PSD y diversos estadísticos (incluyendo RMS)
    para cada esquina a partir de la simulación 7-DOF.
    """
    import numpy as np
    from scipy.signal import welch

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

    # Geometría del vehículo
    wheelbase = params.get('wheelbase', 2.5)
    track_f   = params.get('tf', 1.6)
    track_r   = params.get('tr', 1.6)
    lf        = params.get('lf', wheelbase / 2)
    lr        = params.get('lr', wheelbase / 2)

    pos = np.array([
        [-lf,  track_f / 2],   # FL
        [-lf, -track_f / 2],   # FR
        [ lr,  track_r / 2],   # RL
        [ lr, -track_r / 2],   # RR
    ])

    # Índices de sol.y para los grados de libertad “unsprung”
    zu_idx    = [8,  6, 10, 12]
    zudot_idx = [9,  7, 11, 13]

    # ──────────────────────────────────────────────────────────────────────────────
    # 2) Extraer estados principales
    # ──────────────────────────────────────────────────────────────────────────────
    h       = sol.y[0, :]    # heave global del chasis [m]
    hdot    = sol.y[1, :]
    phi     = sol.y[2, :]    # pitch [rad]
    phi_dot = sol.y[3, :]
    theta   = sol.y[4, :]    # roll [rad]
    theta_dot = sol.y[5, :]

    # Desplazamientos unsprung
    heave_front = h - lf * phi   # posición vertical del chasis en ejes front [m]
    heave_rear  = h + lr * phi   # posición vertical del chasis en ejes rear  [m]
    zu    = np.stack([sol.y[idx, :] for idx in zu_idx],    axis=0)  # (4, N)
    zudot = np.stack([sol.y[idx, :] for idx in zudot_idx], axis=0)  # (4, N)

    # ──────────────────────────────────────────────────────────────────────────────
    # 3) Cinemática de suspensión: zs (chassis en ejes), travel y márgenes
    # ──────────────────────────────────────────────────────────────────────────────
    zs = (
        h[None, :] +
        pos[:, 0:1] * phi[None, :] +
        pos[:, 1:2] * theta[None, :]
    )  # (4, N)

    x_spring   = zu - zs              # travel absoluto (rueda – chasis)  (4, N)
    travel_abs = x_spring.copy()      # podemos guardarlo si hace falta

    # Limites de recorrido (stop out/in)
    z_topout    = np.array([
        params['z_topout_FL'], params['z_topout_FR'],
        params['z_topout_RL'], params['z_topout_RR']
    ])  # (4,)
    z_bottomout = np.array([
        params['z_bottomout_FL'], params['z_bottomout_FR'],
        params['z_bottomout_RL'], params['z_bottomout_RR']
    ])  # (4,)
    stroke = np.array([
        params['stroke_FL'], params['stroke_FR'],
        params['stroke_RL'], params['stroke_RR']
    ])  # (4,)
    gap_bump = np.array([
        params['gap_bumpstop_FL'], params['gap_bumpstop_FR'],
        params['gap_bumpstop_RL'], params['gap_bumpstop_RR']
    ])  # (4,)

    # Margen dinámico
    margen_ext  = x_spring - z_topout[:, None]  # (4, N)
    margen_comp = z_bottomout[:, None] - x_spring  # (4, N)

    # Travel estático (valor inicial en t=0)
    travel_static = x_spring[:, 0][:, None]    # (4,1)
    travel_rel    = x_spring - travel_static   # (4, N)
    travel_max    = np.max(travel_rel, axis=1) # (4,)
    travel_min    = np.min(travel_rel, axis=1) # (4,)
    travel_range  = travel_max - travel_min    # (4,)
    travel_used_pct = 100 * travel_range / stroke  # (% recorrido usado) (4,)

    # ──────────────────────────────────────────────────────────────────────────────
    # 4) Fuerzas de resortes, bumpstops y amortiguador
    # ──────────────────────────────────────────────────────────────────────────────
    k = np.array([
        params['kFL'], params['kFR'], params['kRL'], params['kRR']
    ])  # (4,) [N/m]

    f_spring = k[:, None] * x_spring  # (4, N)

    # Bump stops (solo si supera “gap” + tope superior/inferior)
    bumpstop_funcs = [
        params['bumpstop_front'], params['bumpstop_front'],
        params['bumpstop_rear'],  params['bumpstop_rear']
    ]
    f_bump = np.zeros_like(x_spring)  # (4, N)
    bump_disp  = np.zeros_like(x_spring)  # (4, N) ← aquí guardamos la compresión
    for i in range(n_corners):
        # compresión dinámica: travel_rel > gap → bump-stop
        comp_dyn = np.maximum(0, travel_rel[i, :] - gap_bump[i])
        bump_disp[i, :] = comp_dyn
        f_bump[i, :]    = bumpstop_funcs[i](comp_dyn)

    # Velocidad relativa del amortiguador: (ż_unsprung - ż_chassis)
    v_chassis = (
        hdot[None, :] +
        pos[:, 0:1] * phi_dot[None, :] +
        pos[:, 1:2] * theta_dot[None, :]
    )  # (4, N)
    v_damper = zudot - v_chassis  # (4, N)

    damper_funcs = [
        params['damper_front'], params['damper_front'],
        params['damper_rear'],  params['damper_rear']
    ]
    f_damper = np.zeros_like(v_damper)  # (4, N)
    for i in range(n_corners):
        f_damper[i, :] = damper_funcs[i](v_damper[i, :])

    # ──────────────────────────────────────────────────────────────────────────────
    # 5) Fuerza del neumático y estadísticas asociadas
    # ──────────────────────────────────────────────────────────────────────────────
    kt = np.array([
        params['ktf'], params['ktf'], params['ktr'], params['ktr']
    ])  # (4,) [N/m]

    # Interpolar pista en cada rueda
    zt = np.stack(
        [np.interp(t, t_vec, z_tracks[i]) for i in range(n_corners)],
        axis=0
    )  # (4, N) [m]

    # Fuerza del neumático: kt * (z_track - z_unsprung)
    f_tire = kt[:, None] * (zt - zu)  # (4, N) [N]

    f_tire_max = np.max(f_tire, axis=1)    # (4,)
    f_tire_min = np.min(f_tire, axis=1)    # (4,)
    f_tire_mean= np.mean(f_tire, axis=1)   # (4,)
    f_tire_std = np.std(f_tire, axis=1)    # (4,)
    f_tire_variation = f_tire_std / (f_tire_mean + 1e-12)  # (4,)

    f_tire_variation_front = np.mean(f_tire_variation[:2])  # scalar
    f_tire_variation_rear  = np.mean(f_tire_variation[2:])  # scalar

    # Calcular RMS de fuerzas de neumático (front/rear)
    front_load_rms = np.sqrt(np.mean(f_tire[:2, :]**2))
    rear_load_rms  = np.sqrt(np.mean(f_tire[2:, :]**2))

        # 7) Agregar gaps al post
    gaps = {
        'gap_bumpstop_FL': params['gap_bumpstop_FL'],
        'gap_bumpstop_FR': params['gap_bumpstop_FR'],
        'gap_bumpstop_RL': params['gap_bumpstop_RL'],
        'gap_bumpstop_RR': params['gap_bumpstop_RR']
    }

    # ──────────────────────────────────────────────────────────────────────────────
    # 6) Road‐noise: RMS de velocidad vertical de rueda [mm/s]
    # ──────────────────────────────────────────────────────────────────────────────
    # Convertir z_track a [mm] y derivar en el tiempo
    v_mm_s = [np.gradient(zt[i] * 1000, dt) for i in range(n_corners)]  # lista de 4 arrays (mm/s)
    rms_per_wheel = [np.sqrt(np.mean(v**2)) for v in v_mm_s]  # (4,)
    front_noise_vals = 0.5 * (rms_per_wheel[0] + rms_per_wheel[1])
    rear_noise_vals  = 0.5 * (rms_per_wheel[2] + rms_per_wheel[3])

    # ──────────────────────────────────────────────────────────────────────────────
    # 7) PSD (Heave global, Heave front/rear, Pitch global, Pitch front/rear)
    # ──────────────────────────────────────────────────────────────────────────────
    f_psd_heave, Pxx_heave = welch(h,  fs=fs, nperseg=nperseg, noverlap=noverlap)
    z_front_mm = heave_front * 1000
    z_rear_mm  = heave_rear  * 1000
    f_heave_f, Pxx_heave_f = welch(z_front_mm, fs=fs, nperseg=nperseg, noverlap=noverlap)
    f_heave_r, Pxx_heave_r = welch(z_rear_mm,  fs=fs, nperseg=nperseg, noverlap=noverlap)

    f_psd_pitch, Pxx_pitch = welch(phi, fs=fs, nperseg=nperseg, noverlap=noverlap)
    pitch_front_mm = (-lf * phi) * 1000
    pitch_rear_mm  = ( lr * phi) * 1000
    f_pitch_f, Pxx_pitch_f = welch(pitch_front_mm, fs=fs, nperseg=nperseg, noverlap=noverlap)
    f_pitch_r, Pxx_pitch_r = welch(pitch_rear_mm,  fs=fs, nperseg=nperseg, noverlap=noverlap)

    # Picos en dB (opcional)
    psd_heave_peak = 10 * np.log10(np.max(Pxx_heave) + 1e-30)
    psd_pitch_peak = 10 * np.log10(np.max(Pxx_pitch) + 1e-30)

    # ──────────────────────────────────────────────────────────────────────────────
    # 8) RMS de heave (front/rear) y RMS de pitch
    # ──────────────────────────────────────────────────────────────────────────────
    frh_rms = np.sqrt(np.mean(heave_front**2))  # [m]
    rrh_rms = np.sqrt(np.mean(heave_rear**2))   # [m]
    pitch_deg = np.degrees(phi)                 # [°]
    pitch_rms = np.sqrt(np.mean(pitch_deg**2))  # [°]

    # ──────────────────────────────────────────────────────────────────────────────
    # 9) Acumulated Track Noise Normalized by Lap Time (mm/s)
    # ──────────────────────────────────────────────────────────────────────────────
    lap_time = float(t[-1] - t[0])  # [s]
    acc_tracknoise_mm = []
    for w in range(n_corners):
        z_mm = zt[w] * 1000
        acc_norm = np.sum(np.abs(z_mm)) / lap_time
        acc_tracknoise_mm.append(acc_norm)
    tracknoise_front_accu = 0.5 * (acc_tracknoise_mm[0] + acc_tracknoise_mm[1])
    tracknoise_rear_accu  = 0.5 * (acc_tracknoise_mm[2] + acc_tracknoise_mm[3])

    # ──────────────────────────────────────────────────────────────────────────────
    # 9bis) Calcular la FFT/PSD de la Fuerza de Neumático (“carga”) – Front vs Rear
    # ──────────────────────────────────────────────────────────────────────────────
    #   f_tire es ya un array (4, N) con [FL, FR, RL, RR].
    #   Queremos la señal promediada por ejes: 
    #     • front_signal = (FL + FR)/2 
    #     • rear_signal  = (RL + RR)/2

    front_signal = (f_tire[0, :]) / 9.81 # 
    rear_signal  = (f_tire[2, :]) / 9.81  # 

    f_vals, Pxx_front = welch(front_signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    _,      Pxx_rear  = welch(rear_signal,  fs=fs, nperseg=nperseg, noverlap=noverlap)

    # Magnitud en dB
    mag_front_dB = 10.0 * np.log10(Pxx_front + 1e-30)
    mag_rear_dB  = 10.0 * np.log10(Pxx_rear  + 1e-30)


    # Fase en grados:
    #phase_front_deg = np.angle(F_front, deg=True)
    #phase_rear_deg  = np.angle(F_rear,  deg=True)

    # ──────────────────────────────────────────────────────────────────────────────
    # 10) Devolver todos los resultados en un diccionario
    # ──────────────────────────────────────────────────────────────────────────────
    return {
        # --- Cinemática básica ---
        'travel':             x_spring,               # (4, N)
        'travel_rel':         travel_rel,             # (4, N)
        'bump_disp':         bump_disp,              # (4, N)        
        'travel_static':      travel_static,          # (4, 1)
        'travel_max':         travel_max,             # (4,)
        'travel_min':         travel_min,             # (4,)
        'travel_range':       travel_range,           # (4,)
        'travel_used_pct':    travel_used_pct,        # (4,)
        'travel_abs':         travel_abs,             # (4, N)
        'damper_travel':      zu - zs,                # (4, N)
        'margen_ext':         margen_ext,             # (4, N)
        'margen_comp':        margen_comp,            # (4, N)
        'z_free': np.array([
        params['gap_bumpstop_FL'], params['gap_bumpstop_FR'],
        params['gap_bumpstop_RL'], params['gap_bumpstop_RR']
        ]),  # (4,)

        # --- Fuerzas de suspensión ---
        'f_spring':           f_spring,               # (4, N)
        'f_damper':           f_damper,               # (4, N)
        'f_bump':             f_bump,                 # (4, N)

        # --- Fuerza de neumático ---
        'f_tire':             f_tire,                 # (4, N)
        'f_tire_max':         f_tire_max,             # (4,)
        'f_tire_min':         f_tire_min,             # (4,)
        'f_tire_mean':        f_tire_mean,            # (4,)
        'f_tire_std':         f_tire_std,             # (4,)
        'f_tire_variation':   f_tire_variation,       # (4,)
        'f_tire_variation_front': f_tire_variation_front,  # scalar
        'f_tire_variation_rear':  f_tire_variation_rear,   # scalar
    
            # --- Bump-stop Gaps ---
        'gap_bumpstop_FL':    params['gap_bumpstop_FL'],
        'gap_bumpstop_FR':    params['gap_bumpstop_FR'],
        'gap_bumpstop_RL':    params['gap_bumpstop_RL'],
        'gap_bumpstop_RR':    params['gap_bumpstop_RR'],


        # --- RMS de fuerzas de neumático front/rear ---
        'front_load_rms':       front_load_rms,       # [N]
        'rear_load_rms':        rear_load_rms,        # [N]

        # --- RMS de heave front/rear ---
        'frh_rms':            frh_rms,                # [m]
        'rrh_rms':            rrh_rms,                # [m]

        # --- RMS de pitch ---
        'pitch_rms':          pitch_rms,              # [°]

        # --- Road noise (wheel-speed RMS) ---
        'tracknoise_accu_front': front_noise_vals,    # [mm/s]
        'tracknoise_accu_rear':  rear_noise_vals,     # [mm/s]

        # --- PSDs (global y por eje) ---
        'f_psd':              f_psd_heave,            # (M,)
        'psd_heave':          Pxx_heave,              # (M,)
        'psd_heave_peak':     psd_heave_peak,         # [dB]

        'heave_front':        z_front_mm,             # (N,)
        'heave_rear':         z_rear_mm,              # (N,)
        'f_psd_front':        f_heave_f,              # (M,)
        'psd_heave_front':    Pxx_heave_f,            # (M,)
        'f_psd_rear':         f_heave_r,              # (M,)
        'psd_heave_rear':     Pxx_heave_r,            # (M,)

        'f_psd_pitch':        f_psd_pitch,            # (M,)
        'psd_pitch':          Pxx_pitch,              # (M,)
        'psd_pitch_peak':     psd_pitch_peak,         # [dB]

        'f_psd_pitch_front':  f_pitch_f,              # (M,)
        'psd_pitch_front':    Pxx_pitch_f,            # (M,)
        'f_psd_pitch_rear':   f_pitch_r,              # (M,)
        'psd_pitch_rear':     Pxx_pitch_r,            # (M,)

        # --- Resultados auxiliares para Dash/HTML export ---
        'zt':                 zt,                     # (4, N)
        'z_chassis':          h,                      # (N,)
        'h_ddot':             np.gradient(hdot, t),   # (N,)
        'v_damper':           v_damper,               # (4, N)
        'x_spring':           x_spring,               # (4, N)
        'v_chassis':          v_chassis,              # (4, N)
        'lap_time':           float(t[-1] - t[0]),    # [s]
        'pitch_deg':          pitch_deg,              # (N,)
        'track_name':         params.get('track_name', 'Unknown Track'),
        't_vec':              t,                      # (N,)

        'f_psd_load'           : f_vals,             # vector de frecuencias [Hz]
        'psd_load_mag_front'   : mag_front_dB,       # magnitud dB eje frontal
        #'psd_load_phase_front' : phase_front_deg,    # fase [°] eje frontal
        'psd_load_mag_rear'    : mag_rear_dB,        # magnitud dB eje trasero
        #'psd_load_phase_rear'  : phase_rear_deg,     # fase [°] eje trasero

    
    }





