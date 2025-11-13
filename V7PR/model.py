import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.signal import welch
from scipy.optimize import fsolve
from scipy.signal import savgol_filter

def smooth_signal(data, window=50, polyorder=3):
    """
    Aplica un filtro Savitzky-Golay a una señal o matriz de señales.
    Si data tiene forma (4, N), suaviza cada rueda.
    """
    if len(data.shape) == 1:
        return savgol_filter(data, window_length=window, polyorder=polyorder)
    else:
        return np.array([savgol_filter(channel, window_length=window, polyorder=polyorder) for channel in data])
    
def delayed_signal(func_or_val, t_query, tau, t0=0.0):
    """Si recibo función, evalúo en t - tau; si recibo escalar, lo devuelvo."""
    if callable(func_or_val):
        td = max(t_query - float(tau), t0)
        return float(func_or_val(td))
    return float(func_or_val)
    
def camber_front_from_height(x_m):
    # x puede ser escalar o array (m)
    x = 1000.0 * np.asarray(x_m, dtype=float)   # mm
    return (-0.000028 * x**2) + (0.006688 * x) - 4.00012  # deg

def camber_rear_from_height(x_m):
    x = 1000.0 * np.asarray(x_m, dtype=float)   # mm
    return (-0.01861 * x) - 2.00275  # deg

def rc_front_from_heave(h_axle_m):
    x_mm = 1000.0 * np.asarray(h_axle_m, dtype=float)
    rc_mm = (-1.3055 * x_mm) + 11.147
    return rc_mm * 1e-3  # m

def rc_rear_from_heave(h_axle_m):
    x_mm = 1000.0 * np.asarray(h_axle_m, dtype=float)
    rc_mm = (1.2037 * x_mm) + 78.127
    return rc_mm * 1e-3  # m

def vehicle_model_simple(t, z, params, ztrack_funcs):
    # Estados: h, hdot, phi, phi_dot, theta, theta_dot,
    # zFR, zFRdot, zFL, zFLdot, zRL, zRLdot, zRR, zRRdot = z
    h, hdot, phi, phi_dot, theta, theta_dot, zFR, zFRdot, zFL, zFLdot, zRL, zRLdot, zRR, zRRdot = z
    drs = params['drs_func'](t) if params.get('drs_func') else 0.0
    # Interpolación de inputs de pista
    ztrack_FL, ztrack_FR, ztrack_RL, ztrack_RR = [f(t) for f in ztrack_funcs]

    # ===== helpers camber (x en m, polinomios esperan mm) =====
    def camber_front_from_height(x_m):
        x = 1000.0 * float(x_m)
        return (-0.000028 * x * x) + (0.006688 * x) - 4.00012  # deg

    def camber_rear_from_height(x_m):
        x = 1000.0 * float(x_m)
        return (-0.01861 * x) - 2.00275  # deg

    def rc_front_from_heave(h_axle_m):
    # h_axle_m en metros → x en mm
        x_mm = 1000.0 * float(h_axle_m)
        rc_mm = (-1.3055 * x_mm) + 11.147
        return rc_mm * 1e-3  # m

    def rc_rear_from_heave(h_axle_m):
        x_mm = 1000.0 * float(h_axle_m)
        rc_mm = (1.2037 * x_mm) + 78.127
        return rc_mm * 1e-3  # m


    # Parámetros globales
    Ms = params['ms']
    Ixx = params['Ixx']
    Iyy = params['Iyy']
    lf = params['lf']
    lr = params['lr']
    wb = params['lf'] + params['lr']
    tF = params['tf']
    tR = params['tr']
    mHubF = params['mHubF']
    mHubR = params['mHubR']

    # Resortes (en RUEDA)
    kFL = params['kFL']
    kFR = params['kFR']
    kRL = params['kRL']
    kRR = params['kRR']

    # Resortes de instalación en serie (en RUEDA)
    kinstf = params['kinstf']
    kinstr = params['kinstr']

    # Bumpstop y amortiguador
    bump_front = params['bumpstop_front']
    bump_rear  = params['bumpstop_rear']
    damper_front = params['damper_front']
    damper_rear  = params['damper_rear']
    
    z_topout_FL    = params.get('z_topout_FL')    # en PISTÓN
    z_bottomout_FL = params.get('z_bottomout_FL')
    z_topout_FR    = params.get('z_topout_FR')
    z_bottomout_FR = params.get('z_bottomout_FR')
    z_topout_RL    = params.get('z_topout_RL')
    z_bottomout_RL = params.get('z_bottomout_RL')
    z_topout_RR    = params.get('z_topout_RR')
    z_bottomout_RR = params.get('z_bottomout_RR')

    gap_FL = params['gap_bumpstop_FL']  # en PISTÓN
    gap_FR = params['gap_bumpstop_FR']
    gap_RL = params['gap_bumpstop_RL']
    gap_RR = params['gap_bumpstop_RR']

    R_tire = float(params.get('tire_radius', 0.33))  # m

    # Altura del eje por eje (centro del coche, sin ±track)
    h_axle_F = h - lf * phi + params.get('hRideF')  # 
    h_axle_R = h + lr * phi + params.get('hRideR')  # 

    # Camber cinemático por eje (grados) y aporte de roll ±theta
    camber_F_deg = camber_front_from_height(h_axle_F)
    camber_R_deg = camber_rear_from_height(h_axle_R)

    #camber total respecto al suelo = camber ± roll
    gamma_FL = np.deg2rad(camber_F_deg - theta)
    gamma_FR = np.deg2rad(camber_F_deg + theta)
    gamma_RL = np.deg2rad(camber_R_deg - theta)
    gamma_RR = np.deg2rad(camber_R_deg + theta)

    # Bajada del centro de rueda por camber (m)
    dz_cam_FL = R_tire * (1.0 - np.cos(np.abs(gamma_FL)))
    dz_cam_FR = R_tire * (1.0 - np.cos(np.abs(gamma_FR)))
    dz_cam_RL = R_tire * (1.0 - np.cos(np.abs(gamma_RL)))
    dz_cam_RR = R_tire * (1.0 - np.cos(np.abs(gamma_RR)))

    rc_f = rc_front_from_heave(h_axle_F)  # m
    rc_r = rc_rear_from_heave(h_axle_R)   # m

    # --- al inicio, tras leer ktf/ktr y mHubF/mHubR ---
    ctf = float(params.get('ctf', 2500.0))
    ctr = float(params.get('ctr', 2500.0))

    # helper derivada temporal del perfil de pista
    def ddt(f, t, h=1e-4):
        return (f(t + h) - f(t - h)) / (2.0*h)

    # velocidades del perfil de pista (en m/s)
    ztrack_FL_dot = ddt(ztrack_funcs[0], t)
    ztrack_FR_dot = ddt(ztrack_funcs[1], t)
    ztrack_RL_dot = ddt(ztrack_funcs[2], t)
    ztrack_RR_dot = ddt(ztrack_funcs[3], t)


    # Helper para calcular la fuerza de rueda
    def wheel_force(z_w, z_w_dot,
                    phi_off, theta_off,
                    phi_dot_off, theta_dot_off,
                    k_spring, k_inst,
                    spring_preload, bump, damper,
                    mr_dw, mr_spring,
                    z_top, z_bot,
                    free_gap,
                    x_static,
                    dz_cam=0.0):
        """
        Devuelve fuerza en RUEDA.
        k_spring, k_inst en RUEDA; bump/damper en PISTÓN; z_top/z_bot/gap en PISTÓN.
        mr_dw = MR wheel/damper (damper); mr_spring = MR wheel/damper (muelle).
        """
        MRd   = (mr_dw)        # MR del damper
        MRs   = (mr_spring)    # MR del muelle

        # Travel y velocidad relativos en RUEDA (corrigidos por camber)
        x_wheel = (z_w - dz_cam) - (phi_off + theta_off + h)            # RUEDA
        v_rel_w = z_w_dot - (phi_dot_off + theta_dot_off + hdot)        # RUEDA

        # Reparto serie (en RUEDA) -> travel muelle (RUEDA) y a PISTÓN con MR del muelle
        ratio       = k_inst / (k_spring + k_inst)
        x_spring_p  = (ratio * x_wheel) / MRs                            # PISTÓN
        v_damper_p  = (ratio * v_rel_w) / MRd                            # PISTÓN

        # Límites y baseline en PISTÓN (muelle)
        x_p_clip   = (np.clip((x_spring_p/MRs), z_top, z_bot))            # PISTÓN
        baseline_p = x_static / MRs                               # PISTÓN

        # Fuerza del muelle (PISTÓN → RUEDA con MR del muelle)
        k_spring_p = k_spring * (MRs**2)                          # PISTÓN
        F_spring_p = k_spring_p * x_spring_p + float(spring_preload)     # PISTÓN
        F_spring_w = F_spring_p / MRs                                    # RUEDA

        # Damper (PISTÓN → RUEDA con MR del damper)
        F_damper_w = ((damper(v_damper_p))) / MRd

        # Bump y top-out (PISTÓN → RUEDA con MR del damper)
        x_p_rel = x_p_clip - baseline_p

        F_bump_w = 0.0
        if bump is not None:
            eng_comp = max(0.0, x_p_rel - (free_gap))               # comp pistón
            if eng_comp > 0.0:
                F_bump_w = (bump(eng_comp)) / MRd                   # RUEDA

        F_topout_w = 0.0
        topout = params.get('topout_func', None)
        if topout is not None:
            eng_reb = max(0.0, (z_top) - x_p_clip)                  # extensión pistón
            if eng_reb > 0.0:
                F_topout_w = - (topout(eng_reb)) / MRd              # RUEDA

        return F_spring_w + F_damper_w + F_bump_w + F_topout_w

    # Cálculo de offsets estáticos y dinámicos
    phi_off_front       = -lf * phi
    phi_off_rear        =  lr * phi
    theta_off_front     =  (tF/2) * theta
    theta_off_rear      =  (tR/2) * theta
    phi_dot_off_front   = -lf * phi_dot
    phi_dot_off_rear    =  lr * phi_dot
    theta_dot_off_front =  (tF/2) * theta_dot
    theta_dot_off_rear  =  (tR/2) * theta_dot

    # --- Fuerzas por rueda aplicando límites SAG-aware ---
    F_FL = wheel_force(zFL, zFLdot,
        phi_off_front,  theta_off_front,
        phi_dot_off_front,  theta_dot_off_front,
        kFL, kinstf,
        params['FSpringPreload_FL'],
        bump_front, damper_front,
        params['MR_FL'], params.get('MR_spring_FL', params['MR_FL']),
        z_topout_FL, z_bottomout_FL,
        gap_FL,
        params['sag_wheel_FL'],
        dz_cam=dz_cam_FL,
    )

    F_FR = wheel_force(zFR, zFRdot,
        phi_off_front, -theta_off_front,
        phi_dot_off_front, -theta_dot_off_front,
        kFR, kinstf,
        params['FSpringPreload_FR'],
        bump_front, damper_front,
        params['MR_FR'], params.get('MR_spring_FR', params['MR_FR']),
        z_topout_FR, z_bottomout_FR,
        gap_FR,
        params['sag_wheel_FR'],
        dz_cam=dz_cam_FR,
    )

    F_RL = wheel_force(zRL, zRLdot,
        phi_off_rear,  theta_off_rear,
        phi_dot_off_rear,  theta_dot_off_rear,
        kRL, kinstr,
        params['FSpringPreload_RL'],
        bump_rear, damper_rear,
        params['MR_RL'], params.get('MR_spring_RL', params['MR_RL']),
        z_topout_RL, z_bottomout_RL,
        gap_RL,
        params['sag_wheel_RL'],
        dz_cam=dz_cam_RL,
    )

    F_RR = wheel_force(zRR, zRRdot,
        phi_off_rear, -theta_off_rear,
        phi_dot_off_rear, -theta_dot_off_rear,
        kRR, kinstr,
        params['FSpringPreload_RR'],
        bump_rear, damper_rear,
        params['MR_RR'], params.get('MR_spring_RR', params['MR_RR']),
        z_topout_RR, z_bottomout_RR,
        gap_RR,
        params['sag_wheel_RR'],
        dz_cam=dz_cam_RR,
    )

    # 1) travels crudos corregidos por camber (para ARB en nodo)
    x_FL_raw = (zFL + dz_cam_FL) - (phi_off_front +  theta_off_front + h)
    x_FR_raw = (zFR + dz_cam_FR) - (phi_off_front -  theta_off_front + h)
    x_RL_raw = (zRL + dz_cam_RL) - (phi_off_rear  +  theta_off_rear  + h)
    x_RR_raw = (zRR + dz_cam_RR) - (phi_off_rear  -  theta_off_rear  + h)

    # 2) ratios de serie por esquina (en RUEDA)
    ratio_FL = kinstf / (kinstf + kFL)
    ratio_FR = kinstf / (kinstf + kFR)
    ratio_RL = kinstr / (kinstr + kRL)
    ratio_RR = kinstr / (kinstr + kRR)

    # 3) nodo (rueda) tras instalación
    x_node_FL = ratio_FL * x_FL_raw
    x_node_FR = ratio_FR * x_FR_raw
    x_node_RL = ratio_RL * x_RL_raw
    x_node_RR = ratio_RR * x_RR_raw

    # 4) ARB con diferencial del nodo (no crudo)
    DeltaF = x_node_FL - x_node_FR
    DeltaR = x_node_RL - x_node_RR

    F_FL +=  +params['k_arb_f'] * DeltaF
    F_FR +=  -params['k_arb_f'] * DeltaF
    F_RL +=  +params['k_arb_r'] * DeltaR
    F_RR +=  -params['k_arb_r'] * DeltaR

    # --- Aerodinámica ---
    dyn_hF = h - lf * phi + params.get('hRideF')
    dyn_hR = h + lr * phi + params.get('hRideR')
    Fz_aero_front, Fz_aero_rear, F_drag = compute_aero_forces(
        vx=params.get('vx'),
        hRideF=dyn_hF,
        hRideR=dyn_hR,
        aero_poly=params.get('aero_polynomials', {}),
        drs_frac=drs
    )
    
    # -------------------- transferencias (sin delay por batalla) --------------------
    h_cg = abs(params.get('zCoG', 0.3))  # m
    ax_in = params.get('ax_func', params.get('ax', 0.0))
    ay_in = params.get('ay_func', params.get('ay', 0.0))

    ax_now = delayed_signal(ax_in, t, 0.0)
    ay_now = delayed_signal(ay_in, t, 0.0)

    # Parte elástica 
    M_long_el = (Ms * ax_now * h_cg)  # N·m

    # === LATERAL con ROLL CENTERS (geométrica + elástica) ===
    z_RA_cg = (rc_f * lr + rc_r * lf) / wb   # m
    h_eff   = h_cg - z_RA_cg                 # brazo elástico

    # Reparto de masa suspendida por eje
    wbal_f = float(params.get('rWeightBalF', 0.5))
    MsF = Ms * wbal_f
    MsR = Ms * (1.0 - wbal_f)

    # Parte GEOMÉTRICA por eje (no pasa por θ)
    dF_lat_geo_f = (MsF * ay_now * (h_eff)) / tF
    dF_lat_geo_r = (MsR * ay_now * (h_eff)) / tR

    # Parte ELÁSTICA (sí pasa por θ)
    M_lat_el = Ms * ay_now * h_eff

    # rigidez de rueda efectiva muelle+instalación
    kFL_eff = 1.0/(1.0/kFL + 1.0/kinstf)
    kFR_eff = 1.0/(1.0/kFR + 1.0/kinstf)
    kRL_eff = 1.0/(1.0/kRL + 1.0/kinstr)
    kRR_eff = 1.0/(1.0/kRR + 1.0/kinstr)

    # barras (N/m en rueda) → aporte a rigidez de roll
    k_arb_f = float(params.get('k_arb_f', 0.0))
    k_arb_r = float(params.get('k_arb_r', 0.0))

    Kphi_f = ((kFL_eff + kFR_eff) * (tF/2.0)**2) + (2*k_arb_f) * (tF/2.0)**2
    Kphi_r = ((kRL_eff + kRR_eff) * (tR/2.0)**2) + (2*k_arb_r) * (tR/2.0)**2
    Kphi_tot = Kphi_f + Kphi_r 

    # par asignado a cada eje y su ΔW elástico correspondiente
    M_lat_f = M_lat_el * (Kphi_f / Kphi_tot)
    M_lat_r = M_lat_el - M_lat_f

    dF_lat_el_f = M_lat_f / tF
    dF_lat_el_r = M_lat_r / tR

    dF_lat_u_f = ((mHubF*2)*ay_in*R_tire) / tF
    dF_lat_u_r = ((mHubR*2)*ay_in*R_tire) / tR

    F_FL += +dF_lat_geo_f + dF_lat_el_f + dF_lat_u_f;  F_FR += -dF_lat_geo_f - dF_lat_el_f - dF_lat_u_f
    F_RL += +dF_lat_geo_r + dF_lat_el_r + dF_lat_u_r;  F_RR += -dF_lat_geo_r - dF_lat_el_r - + dF_lat_u_r

    Fz_aero_tot = Fz_aero_front + Fz_aero_rear

    # Dinámica de heave/roll/pitch
    h_ddot = (F_FL + F_FR + F_RL + F_RR + Fz_aero_tot) / Ms

    phi_ddot = (
        lr * (F_RR + F_RL + Fz_aero_rear) - lf * (F_FR + F_FL + Fz_aero_front) - (h * F_drag) - M_long_el
    ) / Iyy

    theta_ddot = (
        (tF/2) * (F_FL - F_FR) + (tR/2) * (F_RL - F_RR )
    ) / Ixx
    
    # -------------------- masas no suspendidas --------------------
    zFR_ddot = (-(F_FR + 0.5*Fz_aero_front)
                + params['ktf']*(ztrack_FR - zFR)
                + ctf*(ztrack_FR_dot - zFRdot)) / mHubF

    zFL_ddot = (-(F_FL + 0.5*Fz_aero_front)
                + params['ktf']*(ztrack_FL - zFL)
                + ctf*(ztrack_FL_dot - zFLdot)) / mHubF

    zRL_ddot = (-(F_RL + 0.5*Fz_aero_rear)
                + params['ktr']*(ztrack_RL - zRL)
                + ctr*(ztrack_RL_dot - zRLdot)) / mHubR

    zRR_ddot = (-(F_RR + 0.5*Fz_aero_rear)
                + params['ktr']*(ztrack_RR - zRR)
                + ctr*(ztrack_RR_dot - zRRdot)) / mHubR
    
    return [
        hdot,      h_ddot,
        phi_dot,   phi_ddot,
        theta_dot, theta_ddot,
        zFRdot,    zFR_ddot,
        zFLdot,    zFL_ddot,
        zRLdot,    zRL_ddot,
        zRRdot,    zRR_ddot
    ]

def compute_aero_forces(vx, hRideF, hRideR, aero_poly, drs_frac=0.0, rho_air=1.225, area_ref=1):
    """
    Calcula las fuerzas aerodinámicas verticales y de arrastre (drag).
    """
    fa     = aero_poly.get('flapAngles', {})
    aFlapF = fa.get('aFlapF', 0.0)
    aFlapR = fa.get('aFlapR', 0.0)

    def evaluate_poly(poly_dict, hF, hR, aFlapF, aFlapR):
        vars = {
            "Const":                         1.0,
            "hRideF":                        hF,
            "hRideR":                        hR,
            "hRideF*hRideF":                 hF**2,
            "hRideR*hRideR":                 hR**2,
            "hRideF*hRideF*hRideF":          hF**3,
            "hRideR*hRideR*hRideR":          hR**3,
            "hRideF*hRideR":                 hF * hR,
            "hRideF*hRideR*hRideR":          hF * (hR**2),
            "hRideF*hRideF*hRideR":          (hF**2) * hR,
            "hRideF*hRideF*hRideR*hRideR":   (hF**2) * (hR**2),

            # monomios de flap
            "aFlapF":                        aFlapF,
            "aFlapF*aFlapF":                 aFlapF**2,
            "aFlapR":                        aFlapR,
            "aFlapR*aFlapR":                 aFlapR**2,
        }
        # suma coef * valor_monomio para cada término del polinomio
        return sum(k*vars.get(expr,0.0) for expr,k in poly_dict.items())


    # --- mapa base (flap cerrado) ---
    Clf0 = evaluate_poly(aero_poly['CLiftBodyF'], hRideF, hRideR, aFlapF, aFlapR)
    Clr0 = evaluate_poly(aero_poly['CLiftBodyR'], hRideF, hRideR, aFlapF, aFlapR)
    Cd0  = evaluate_poly(aero_poly['CDragBody' ], hRideF, hRideR, aFlapF, aFlapR)

    # --- ∆ mapa DRS --------------
    dClf = drs_frac * evaluate_poly(aero_poly['DRS_CLiftF'], hRideF, hRideR, aFlapF, aFlapR)
    dClr = drs_frac * evaluate_poly(aero_poly['DRS_CLiftR'], hRideF, hRideR, aFlapF, aFlapR)
    dCd  = drs_frac * evaluate_poly(aero_poly['DRS_CDrag' ], hRideF, hRideR, aFlapF, aFlapR)

    Clf = (Clf0 + (dClf)) * aero_poly['rCLiftBodyFFactor'] + \
          aero_poly['coefficientOffsets'].get('CLiftBodyFUserOffset',0)
    Clr = (Clr0 + dClr) * aero_poly['rCLiftBodyRFactor'] + \
          aero_poly['coefficientOffsets'].get('CLiftBodyRUserOffset',0)
    Cd  = (Cd0  + dCd ) * aero_poly['rCDragBodyFactor']
    # --- Fuerzas ---
    q = 0.5 * rho_air * vx**2  # presión dinámica

    Fz_aero_front = -Clf * q * area_ref
    Fz_aero_rear  = -Clr * q * area_ref
    F_drag        = Cd  * q * area_ref

    return Fz_aero_front, Fz_aero_rear, F_drag

def compute_static_equilibrium(params, vx=0):
    """
    Equilibrio estático con efecto de camber en el travel:
      x_corner = (z_sprung_corner) - (z_unsprung_corner + dz_cam_corner)
    y baseline x_static coherente con el solver.
    """
    import numpy as np
    from math import cos

    # ── helpers de camber (entrada altura eje en m; polinomios esperan mm) ──
    def camber_front_from_height(x_m):
        x = 1000.0 * float(x_m)
        return (-0.000028 * x * x) + (0.006688 * x) - 4.00012  # deg

    def camber_rear_from_height(x_m):
        x = 1000.0 * float(x_m)
        return (-0.01861 * x) - 2.00275  # deg

    def deg2rad(d):
        return d * np.pi / 180.0

    # ── constantes / params ──
    g   = 9.81
    Ms  = float(params['ms'])
    lf  = float(params['lf'])
    lr  = float(params['lr'])
    tf  = float(params['tf'])
    tr  = float(params['tr'])

    kFL = float(params['kFL']);  kFR = float(params['kFR'])
    kRL = float(params['kRL']);  kRR = float(params['kRR'])
    kinstf = float(params['kinstf']);  kinstr = float(params['kinstr'])
    ktf = float(params['ktf']);  ktr = float(params['ktr'])

    spring_preload_FL = float(params.get('FSpringPreload_FL', 0.0))
    spring_preload_FR = float(params.get('FSpringPreload_FR', 0.0))
    spring_preload_RL = float(params.get('FSpringPreload_RL', 0.0))
    spring_preload_RR = float(params.get('FSpringPreload_RR', 0.0))

    k_arb_f = float(params.get('k_arb_f', 0.0))
    k_arb_r = float(params.get('k_arb_r', 0.0))

    mr_wd_f = params.get('MR_FL', 1.0)
    mr_wd_r = params.get('MR_RL', 1.0)

    R_tire = float(params.get('tire_radius', 0.33))

    # ── residual de equilibrio ──
    def residual(x):
        # Variables: h, phi, theta, zFR, zFL, zRL, zRR  (z* son US respecto suelo)
        h, phi, theta, zFR, zFL, zRL, zRR = x

        # Altura de la masa suspendida en cada esquina (chasis rígido)
        zs_FR = h - lf*phi - (tf/2)*theta
        zs_FL = h - lf*phi + (tf/2)*theta
        zs_RR = h + lr*phi - (tr/2)*theta
        zs_RL = h + lr*phi + (tr/2)*theta

        # Camber cinemático por eje (en función de alturas de eje)
        h_axle_F = h - lf*phi
        h_axle_R = h + lr*phi
        camber_F_deg = camber_front_from_height(h_axle_F)
        camber_R_deg = camber_rear_from_height(h_axle_R)

        # Roll de carrocería suma/resta al camber por lado
        gamma_FL = deg2rad(camber_F_deg - theta)
        gamma_FR = deg2rad(camber_F_deg + theta)
        gamma_RL = deg2rad(camber_R_deg - theta)
        gamma_RR = deg2rad(camber_R_deg + theta)

        # Caída vertical aparente del centro de rueda por camber (aprox geométrica)
        dz_cam_FL = R_tire * (1.0 - cos(abs(gamma_FL)))
        dz_cam_FR = R_tire * (1.0 - cos(abs(gamma_FR)))
        dz_cam_RL = R_tire * (1.0 - cos(abs(gamma_RL)))
        dz_cam_RR = R_tire * (1.0 - cos(abs(gamma_RR)))

        # Travel RUEDA incluyendo camber (coherente con el solver dinámico)
        x_FR = (zs_FR - (zFR - dz_cam_FR))
        x_FL = (zs_FL - (zFL - dz_cam_FL))
        x_RR = (zs_RR - (zRR - dz_cam_RR))
        x_RL = (zs_RL - (zRL - dz_cam_RL))

        # Rigidez efectiva muelle + instalación (en RUEDA)
        kFR_eff = 1.0 / (1.0/kFR + 1.0/kinstf)
        kFL_eff = 1.0 / (1.0/kFL + 1.0/kinstf)
        kRR_eff = 1.0 / (1.0/kRR + 1.0/kinstr)
        kRL_eff = 1.0 / (1.0/kRL + 1.0/kinstr)

        # Fuerzas de muelle (signo: x>0 ⇒ RUEDA abajo respecto a sprung ⇒ muelle empuja hacia arriba)
        F_FR = -kFR_eff * x_FR + spring_preload_FR
        F_FL = -kFL_eff * x_FL + spring_preload_FL
        F_RR = -kRR_eff * x_RR + spring_preload_RR
        F_RL = -kRL_eff * x_RL + spring_preload_RL

        # ARB en nodo (tras instalación): r = k_inst/(k_inst+k_spring)
        r_FL = kinstf/(kinstf + kFL); r_FR = kinstf/(kinstf + kFR)
        r_RL = kinstr/(kinstr + kRL); r_RR = kinstr/(kinstr + kRR)
        if k_arb_f:
            DeltaF = (r_FL * x_FL) - (r_FR * x_FR)
            F_FL += +k_arb_f * DeltaF
            F_FR += -k_arb_f * DeltaF
        if k_arb_r:
            DeltaR = (r_RL * x_RL) - (r_RR * x_RR)
            F_RL += +k_arb_r * DeltaR
            F_RR += -k_arb_r * DeltaR

        # R1: equilibrio vertical de la masa suspendida
        R1 = (F_FR + F_FL + F_RR + F_RL) - Ms*g

        # R2: momento de roll (eje X) por palancas ±track/2
        R2 = (tf/2)*(F_FL - F_FR) + (tr/2)*(F_RL - F_RR)

        # R3: momento de pitch (eje Y) por palancas ±lf/lr
        R3 = lr*(F_RR + F_RL) - lf*(F_FR + F_FL)

        # R4–R7: equilibrio de cada masa no suspendida (sin peso US explícito)
        #        Tire fuerza = k_t*(z_track - z_wheel); z_track=0 en estático.
        R4 = F_FR + ktf*(0.0 - zFR)
        R5 = F_FL + ktf*(0.0 - zFL)
        R6 = F_RR + ktr*(0.0 - zRR)
        R7 = F_RL + ktr*(0.0 - zRL)

        return [R1, R2, R3, R4, R5, R6, R7]

    # ── inicialización ──
    # Heave/pitch/roll iniciales
    h_init   = -0.02
    phi_init = np.arctan2((params.get('hRideF', 0.02) - params.get('hRideR', 0.04)), (lf + lr))
    theta_init = 0.0

    # Cargas por eje para estimar zUS inicial
    W   = Ms*g
    wbal_f = float(params.get('rWeightBalF', 0.5))
    Wf = W * wbal_f; Wr = W - Wf
    zFR0 = (Wf/2.0) / ktf
    zFL0 = (Wf/2.0) / ktf
    zRR0 = (Wr/2.0) / ktr
    zRL0 = (Wr/2.0) / ktr

    x0 = [h_init, phi_init, theta_init, zFR0, zFL0, zRL0, zRR0]

    from scipy.optimize import fsolve
    sol, info, ier, msg = fsolve(residual, x0, full_output=True)
    h, phi, theta, zFR, zFL, zRL, zRR = sol

    # Recalcula camber y dz_cam con la solución estática para guardar x_static coherente
    h_axle_F = h - lf*phi
    h_axle_R = h + lr*phi
    camber_F_deg = camber_front_from_height(h_axle_F)
    camber_R_deg = camber_rear_from_height(h_axle_R)
    gamma_FL = deg2rad(camber_F_deg - theta)
    gamma_FR = deg2rad(camber_F_deg + theta)
    gamma_RL = deg2rad(camber_R_deg - theta)
    gamma_RR = deg2rad(camber_R_deg + theta)
    dz_cam_FL = R_tire * (1.0 - np.cos(abs(gamma_FL)))
    dz_cam_FR = R_tire * (1.0 - np.cos(abs(gamma_FR)))
    dz_cam_RL = R_tire * (1.0 - np.cos(abs(gamma_RL)))
    dz_cam_RR = R_tire * (1.0 - np.cos(abs(gamma_RR)))

    zs_FR = h - lf*phi - (tf/2)*theta
    zs_FL = h - lf*phi + (tf/2)*theta
    zs_RR = h + lr*phi - (tr/2)*theta
    zs_RL = h + lr*phi + (tr/2)*theta

    x_FR = (zs_FR - (zFR + dz_cam_FR))
    x_FL = (zs_FL - (zFL + dz_cam_FL))
    x_RR = (zs_RR - (zRR + dz_cam_RR))
    x_RL = (zs_RL - (zRL + dz_cam_RL))

    # Guarda baseline estático en RUEDA (coherente con solver: baseline_p = x_static/MR)
    params['sag_wheel_FR'] = zFR * mr_wd_f
    params['sag_wheel_FL'] = zFL * mr_wd_f
    params['sag_wheel_RR'] = zRR * mr_wd_r
    params['sag_wheel_RL'] = zRL * mr_wd_r

    # Logs opcionales
    R0 = np.array(residual(x0))
    R  = np.array(info['fvec'])
    print(f"[equil] ||R(x0)|| = {np.linalg.norm(R0):.2e}")
    print(f"[equil] ier={ier} (1=OK), nfev={info['nfev']}, ||R(sol)||={np.linalg.norm(R):.2e}")
    print(f"[equil] sol = h={h:.6f}, phi={phi:.6f}, theta={theta:.6f}, "
          f"zFR={zFR:.6f}, zFL={zFL:.6f}, zRL={zRL:.6f}, zRR={zRR:.6f}")
    print(f"[equil] sag_wheel [mm]: "
          f"FL={params['sag_wheel_FL']*1000:.2f}, FR={params['sag_wheel_FR']*1000:.2f}, "
          f"RL={params['sag_wheel_RL']*1000:.2f}, RR={params['sag_wheel_RR']*1000:.2f}")
    print(f"Debug Rigideces = kFL = {kFL}, kinstf = {kinstf}, kRL = {kRL}, kinstr = {kinstr}, kFL_eff = {1.0/(1.0/kFL + 1.0/kinstf)}, kRL_eff = {1.0/(1.0/kRL + 1.0/kinstr)}, ratio = {kinstf/(kinstf + kFL)}")

    return sol

def run_vehicle_model_simple(t_vec, z_tracks, vx, ax, ay, rpedal, pbrake, params, drs_func=None, pre_roll_s=0.0, baseline_tail_s=0.25):
    """
    Versión rápida con RK45:
      - Interpolación lineal (rápida) para entradas.
      - Paso máximo relativo a dt_min para no saltarse transitorios.
      - Tolerancias ajustadas (rtol=1e-3, atol=1e-6).
      - Mantiene límites SAG-aware y el *1000 en z_top/z_bot.
    """

    # --- asegurar tiempo estrictamente creciente ---
    t = np.asarray(t_vec, dtype=float).copy()
    for i in range(1, len(t)):
        if t[i] <= t[i-1]:
            t[i] = t[i-1] + 1e-9
    dt_min = np.min(np.diff(t)) if len(t) > 1 else 1e-3

    # Interpoladores que usa vehicle_model_simple internamente (lineal)
    ztrack_funcs = [interp1d(t, z_tracks[i], kind='linear', bounds_error=False, fill_value="extrapolate")
                    for i in range(4)]
    params['drs_func'] = drs_func

    # --- Parámetros geométricos ---
    lf = params['lf']; lr = params['lr']; tf = params['tf']; tr = params['tr']

    # Interpoladores de inputs (lineal → más barato que PCHIP)
    vx_fun = interp1d(t, vx, kind='linear', bounds_error=False, fill_value="extrapolate")
    ax_fun = interp1d(t, ax, kind='linear', bounds_error=False, fill_value="extrapolate")
    ay_fun = interp1d(t, ay, kind='linear', bounds_error=False, fill_value="extrapolate")

    # Velocidad inicial para el equilibrio estático
    vx0 = float(vx_fun(t[0]))

    # --- Equilibrio estático ---
    h0, phi0, theta0, zFR0, zFL0, zRL0, zRR0 = compute_static_equilibrium(params, vx0)

    # Límites SAG-aware (mantengo tu *1000 a propósito)
    for c in ("FL", "FR", "RL", "RR"):
        mr_wd  = params[f"MR_{c}"]                 # rueda->pistón
        stroke = params[f"stroke_{c}"]             # [m] carrera total de pistón
        sag_w  = params.get(f"sag_wheel_{c}", 0.0)  # [m] sag en rueda
        sag_p  = sag_w / mr_wd                   # baseline en PISTÓN (con signo)
        span   = stroke - abs(sag_p)             # margen disponible hacia top/bottom
        z_top  = -sag_p    # EXTENSIÓN (mínimo x_p)
        z_bot  =  span    # COMPRESIÓN (máximo x_p)             # max clip (bottom-out)
        params[f"z_topout_{c}"]    = z_top
        params[f"z_bottomout_{c}"] = z_bot


    # Estado inicial
    y0 = [h0, 0.0, phi0, 0.0, theta0, 0.0, zFR0, 0.0, zFL0, 0.0, zRL0, 0.0, zRR0, 0.0]

    def rhs(ti, y):
        params['vx'] = float(vx_fun(ti))
        params['ax'] = float(ax_fun(ti))
        params['ay'] = float(ay_fun(ti))
        return vehicle_model_simple(ti, y, params, ztrack_funcs)

        # --- Solver rápido: RK45 bien “tuneado” ---
    sol = solve_ivp(
        rhs,
        (t[0], t[-1]),
        y0,
        t_eval=t,
        method='RK45',
        rtol=3e-3,
        atol=1e-5,
        max_step=dt_min,     # antes: max(2e-3, 0.5*dt_min)
        first_step=dt_min    # antes: 0.5*dt_min
    )


    return sol

def postprocess_7dof(sol, params, z_tracks, t_vec, throttle, brake, vx, ax, ay):

    """
    Calcula travel, fuerzas y outputs básicos para cada esquina a partir de la simulación 7-DOF.
    Todo el cálculo está vectorizado con NumPy para mayor eficiencia.
    """
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
    # Interp a tiempo del solver
    zt = np.stack([np.interp(t, t_vec, z_tracks[i]) for i in range(4)], axis=0)  # (4,N)
    vx_i = np.interp(t, t_vec, vx).astype(float)

    # ──────────────────────────────────────────────────────────────────────────────
    # 2) Extraer estados principales de sol.y
    # ──────────────────────────────────────────────────────────────────────────────
    h      = sol.y[0, :]     # heave global del chasis [m]
    hdot   = sol.y[1, :]
    phi    = sol.y[2, :]    # pitch [rad]
    phi_dot= sol.y[3, :]
    theta  = sol.y[4, :]     # roll [rad]
    theta_dot = sol.y[5, :]

    # Dimensiones del vehículo (en metros)
    wheelbase = params.get('wheelbase')
    tF = params.get('tf')
    tR = params.get('tr')
    lf        = params.get('lf')
    lr        = params.get('lr')
    hRideF = params.get('hRideF', 0.0)     # [m]
    hRideR = params.get('hRideR', 0.0)     # [m]
    mr_wd = np.array([params['MR_FL'],params['MR_FR'],params['MR_RL'],params['MR_RR']])[:, None]  # (4,1) damper to wheel

    kin_f = float(params.get('kinstf', 0.0)); kin_r = float(params.get('kinstr', 0.0))
    kFL = float(params['kFL']); kFR = float(params['kFR'])
    kRL = float(params['kRL']); kRR = float(params['kRR'])

    k_inst = np.array([kin_f, kin_f, kin_r, kin_r])[:,None]
    k_spr  = np.array([kFL, kFR, kRL, kRR])[:,None]
    k_spr_p = k_spr * mr_wd**2  # (4,1) rigidez en coordenada rueda
    ratio = k_inst / (k_spr_p + k_inst)  # (4,1)

    k_total_w = np.where(k_inst>0.0, 1.0/(1.0/k_spr + 1.0/k_inst), k_spr)  # (4,1)

    phi_off_front       = -lf * phi
    phi_off_rear        =  lr * phi
    theta_off_front     =  (tF/2) * theta
    theta_off_rear      =  (tR/2) * theta

    zt = np.stack(
        [np.interp(t, t_vec, z_tracks[i]) for i in range(n_corners)],
        axis=0
    )  # (4, N)
    
    dt = np.gradient(t_vec)

    # Desplazamientos "unsprung" (ruedas) y sus velocidades
    zFL, zFR, zRL, zRR = (sol.y[8, :]), (sol.y[6, :]), (sol.y[10, :]), (sol.y[12, :])
    zdFL,zdFR,zdRL,zdRR= sol.y[9, :], sol.y[7, :], sol.y[11, :], sol.y[13, :]
    z_wheel = np.vstack([(zFL), (zFR), (zRL), (zRR)])              # (4,N)

    sag_wheel = np.array([
        float(params.get('sag_wheel_FL', 0.0)),
        float(params.get('sag_wheel_FR', 0.0)),
        float(params.get('sag_wheel_RL', 0.0)),
        float(params.get('sag_wheel_RR', 0.0)),
    ])[:,None]                            # (4,1) RUEDA

    sag_piston = sag_wheel / mr_wd             # (4,1) PISTÓN

    # ──────────────────────────────────────────────────────────────────────────────
    # 3) Cinemática de suspensión: travel, posiciones de rueda (zs) y márgenes
    # ──────────────────────────────────────────────────────────────────────────────
    ZS_FL = zFL - (h + phi_off_front + theta_off_front) 
    ZS_FR = zFR - (h + phi_off_front - theta_off_front)
    ZS_RL = zRL - (h + phi_off_rear  + theta_off_rear)  
    ZS_RR = zRR - (h + phi_off_rear  - theta_off_rear) 
    zs = np.vstack([ZS_FL, ZS_FR, ZS_RL, ZS_RR])  # (4,N)

    g    = 9.81
    Wtot = params['ms'] * g
    Wf   = Wtot * params['rWeightBalF']
    Wr   = Wtot - Wf
    static = np.array([Wf/2, Wf/2, Wr/2, Wr/2])[:,None]  # (4,1)
 #-------------------------------------------------------------------
    R_tire = float(params.get('tire_radius', 0.33))  # m

    # Altura del eje por eje (centro del coche, sin ±track)
    h_axle_F = h - lf * phi + params.get('hRideF')  # 
    h_axle_R = h + lr * phi + params.get('hRideR')  # 

    # Camber cinemático por eje (grados) y aporte de roll ±theta
    camber_F_deg = camber_front_from_height(h_axle_F)
    camber_R_deg = camber_rear_from_height(h_axle_R)

    #camber total respecto al suelo = camber ± roll
    gamma_FL = np.deg2rad(camber_F_deg - theta)
    gamma_FR = np.deg2rad(camber_F_deg + theta)
    gamma_RL = np.deg2rad(camber_R_deg - theta)
    gamma_RR = np.deg2rad(camber_R_deg + theta)

    # Bajada del centro de rueda por camber (m)
    dz_cam_FL = R_tire * (1.0 - np.cos(np.abs(gamma_FL)))
    dz_cam_FR = R_tire * (1.0 - np.cos(np.abs(gamma_FR)))
    dz_cam_RL = R_tire * (1.0 - np.cos(np.abs(gamma_RL)))
    dz_cam_RR = R_tire * (1.0 - np.cos(np.abs(gamma_RR)))

    x_FL_b = (zFL-dz_cam_FL) - (phi_off_front + theta_off_front + h)
    x_FR_b = (zFR-dz_cam_FR) - (phi_off_front - theta_off_front + h)
    x_RL_b = (zRL-dz_cam_RL) - (phi_off_rear  + theta_off_rear  + h)
    x_RR_b = (zRR-dz_cam_RR) - (phi_off_rear  - theta_off_rear  + h)
    x_wheel_b = np.vstack([x_FL_b, x_FR_b, x_RL_b, x_RR_b])      # (4,N)

    x_spring_p = (x_wheel_b * ratio) + sag_wheel
    #---------------------------------------------------------------------
    def _get_lim(name, default):
        return float(params.get(name, default))
    z_top = np.array([
        _get_lim('z_topout_FL',  -params.get('stroke_FL',  -0.10)),
        _get_lim('z_topout_FR',  -params.get('stroke_FR',  -0.10)),
        _get_lim('z_topout_RL',  -params.get('stroke_RL',  -0.10)),
        _get_lim('z_topout_RR',  -params.get('stroke_RR',  -0.10)),
    ])[:,None]
    z_bot = np.array([
        _get_lim('z_bottomout_FL', params.get('stroke_FL', 0.10)),
        _get_lim('z_bottomout_FR', params.get('stroke_FR', 0.10)),
        _get_lim('z_bottomout_RL', params.get('stroke_RL', 0.10)),
        _get_lim('z_bottomout_RR', params.get('stroke_RR', 0.10)),
    ])[:,None]

    # ---------- 7) Aerodinámica ----------
    drs_vec = params['drs_func'](t) if params.get('drs_func') else np.zeros_like(t)

    Fz_front = np.zeros_like(t)
    Fz_rear  = np.zeros_like(t)

    for i in range(N):
        dyn_hF = h[i] - lf * phi[i] + hRideF 
        dyn_hR = h[i] + lr * phi[i] + hRideR 
        Ff, Fr, _ = compute_aero_forces(
            vx=vx_i[i],
            hRideF=dyn_hF,
            hRideR=dyn_hR,
            aero_poly=params.get('aero_polynomials', {}),
            drs_frac=float(drs_vec[i])
        )
        Fz_front[i] = Ff
        Fz_rear[i]  = Fr

    # Reparto por rueda
    aero = np.vstack([
        0.5 * Fz_front,  # FL
        0.5 * Fz_front,  # FR
        0.5 * Fz_rear,   # RL
        0.5 * Fz_rear    # RR
    ])  # (4, N)

    dyn_hF = ((h*mr_wd[0]) - lf * phi) +  hRideF  # (N,)
    RH_front_c = dyn_hF

    dyn_hR = ((h*mr_wd[2]) + lr * phi)  + hRideR  # (N,)
    RH_rear_c  = dyn_hR

    # ---------- 8) Neumático ----------
    ktf = float(params['ktf']); ktr = float(params['ktr'])
    ctf = float(params.get('ctf', 2500.0)); ctr = float(params.get('ctr', 2500.0))

    kt = np.array([ktf, ktf, ktr, ktr])[:, None]
    ct = np.array([ctf, ctf, ctr, ctr])[:, None]

    # velocidades: ẋ_track (derivada respecto a t) y ẋ_wheel ya las tienes (zdFL, ...):
    zt_dot = np.vstack([np.gradient(zt[i], t) for i in range(4)])   # (4,N)

    v_wheel = np.vstack([zdFL, zdFR, zdRL, zdRR])                   # (4,N)

    # fuerza de “neumático” (modelo Kelvin-Voigt): F = k·Δz + c·Δẋ
    f_tire = kt*(- z_wheel) + ct*(zt_dot - v_wheel)
    
    preload = np.array([
        params['FSpringPreload_FL'],
        params['FSpringPreload_FR'],
        params['FSpringPreload_RL'],
        params['FSpringPreload_RR'],
    ])[:, None]
 
    # ──────────────────────────────────────────────────────────────────────────────
    # 3.3) Márgenes dinámicos y estáticos traseros
    # ──────────────────────────────────────────────────────────────────────────────
    travel_static   = x_wheel_b[:,0][:,None]
    travel_rel      = (x_wheel_b + travel_static)/mr_wd # (4,N) en RUEDA (negativo = se comprime)
    travel_max      = np.max(travel_rel,axis=1)
    travel_min      = np.min(travel_rel,axis=1)

    # Amortiguador: velocidad relativa (rueda) como en el solver
    rel_vel_FL = zdFL- (hdot + (-lf)*phi_dot + ( tF/2.0)*theta_dot)
    rel_vel_FR = zdFR- (hdot + (-lf)*phi_dot + (-tF/2.0)*theta_dot)
    rel_vel_RL = zdRL - (hdot + ( lr)*phi_dot + ( tR/2.0)*theta_dot)
    rel_vel_RR = zdRR - (hdot + ( lr)*phi_dot + (-tR/2.0)*theta_dot)
    rel_vel = np.vstack([rel_vel_FL, rel_vel_FR, rel_vel_RL, rel_vel_RR])

    damper_front = params['damper_front']
    damper_rear  = params['damper_rear']
    damper_funcs = [damper_front, damper_front, damper_rear, damper_rear]
    f_damper = np.zeros_like((rel_vel*ratio))/mr_wd
    for i in range(4):
        f_damper[i] = damper_funcs[i](rel_vel[i]*ratio[i])/mr_wd[i]

    f_damper = np.vstack([((f_damper[0,:]), (f_damper[1,:]),(f_damper[2,:]), (f_damper[3,:]))]) /mr_wd  # (4,N)

    damper_travel_x= -(x_spring_p/mr_wd) # (4,N) en DAMPER (negativo = se comprime)

    x_spring = x_wheel_b/mr_wd

    x_p = np.clip(((x_spring_p)/mr_wd), z_top, z_bot)  # (4,N)
    x_w_clip = x_p * mr_wd                # (4,N)

    # Fuerza muelle en RUEDA (con clipping)
    f_spring = k_total_w * (x_w_clip) + preload  # (4,N)

    # Gaps en pistón (vienen del JSON en unidades de pistón)
    gap_p = np.array([
        params['gap_bumpstop_FL'],
        params['gap_bumpstop_FR'],
        params['gap_bumpstop_RL'],
        params['gap_bumpstop_RR'],
    ], dtype=float)[:, None]                              # gap de setup en piston

    baseline_p = sag_piston #compresion inicial del piston en suspensión estática 

    # Compresión POSITIVA y penetración:
    comp_rel_p   = (x_p) - baseline_p                 # (4, N)  travel relativo
    bump_disp_p  = comp_rel_p - gap_p               # (4, N)  desplazamiento "bump" firmado (neg = sin contacto)
    penetration  = np.maximum(0.0, bump_disp_p)     # (4, N)  solo la parte en contacto (>= 0)

    bumpstop_mm_piston = bump_disp_p            # (4,N) en mm (FL,FR,RL,RR)

    # LUTs en pistón -> fuerza a RUEDA dividiendo por MR
    f_bump = np.zeros_like(x_wheel_b)
    f_bump[0,:] = params['bumpstop_front'](penetration[0]) 
    f_bump[1,:] = params['bumpstop_front'](penetration[1]) 
    f_bump[2,:] = params['bumpstop_rear' ](penetration[2]) 
    f_bump[3,:] = params['bumpstop_rear' ](penetration[3]) 

    heave_front = (damper_travel_x[0]+damper_travel_x[1])/2    # m 
    heave_rear  =((damper_travel_x[2]+damper_travel_x[3])/2)      # m             

    # --- Fuerzas de la anti-roll bar ─ versión coherente con el solver ----------
    x_FL_raw = x_wheel_b[0]
    x_FR_raw = x_wheel_b[1]
    x_RL_raw = x_wheel_b[2]
    x_RR_raw = x_wheel_b[3]
    k_arb_f = float(params.get('k_arb_f', 0.0))
    k_arb_r = float(params.get('k_arb_r', 0.0))
    f_arb = np.zeros_like(x_wheel_b)
    dx_front = x_FL_raw - x_FR_raw    # (N,)
    dx_rear  = x_RL_raw - x_RR_raw    # (N,)
    if k_arb_f:
        arb_force_front =  2*k_arb_f * dx_front
        f_arb[0] += +arb_force_front
        f_arb[1] += -arb_force_front
    if k_arb_r:
        arb_force_rear =  2*k_arb_r * dx_rear
        f_arb[2] += +arb_force_rear
        f_arb[3] += -arb_force_rear
   
    # ---  Load-transfer inercial ------------------------------------------
    Ms     = params['ms']                        # masa suspendida
    Mu_f   = 2*params['mHubF']                  # hub mass delantera (dos ruedas)
    Mu_r   = 2*params['mHubR']                  # hub mass trasera
    h_cg   = abs(params.get('zCoG', 0.30))           # altura CG
    wheelbase = lf + lr
    wb = wheelbase

    ay_in = params.get('ay_func', params.get('ay', 0.0))
    vx_in = params.get('vx_func', params.get('vx', 50.0))

    ay_now  = delayed_signal(ay_in, t, 0.0)

    # Heave por eje
    h_axle_F = h - lf * phi + hRideF   # (N,)
    h_axle_R = h + lr * phi + hRideR   # (N,)

    # Roll centers variables (arrays en m)
    rc_f = ((-1.3055 * (h_axle_F * 1000.0)) + 11.147) * 1e-3
    rc_r = (( 1.2037 * (h_axle_R * 1000.0)) + 78.127) * 1e-3

    ax = np.interp(t, t_vec, ax).astype(float)
    ay = np.interp(t, t_vec, ay).astype(float)
    
    # 1) longitudinal  ( +ax ⇒ descarga el eje delantero )
    dF_long = (Ms*h_cg - Mu_f*lf - Mu_r*lr) * ax / wheelbase      # array (N,)  :contentReference[oaicite:0]{index=0}

        # === LATERAL con ROLL CENTERS (geométrica + elástica) ===
    # rc_f, rc_r (en metros) ya definidos arriba en tu función (líneas ~91-92)
    # Altura del eje de balanceo a la vertical del CG (interpolación F/R)
    z_RA_cg = (rc_f * lr + rc_r * lf) / wb       # m
    h_eff   = h_cg - z_RA_cg                     # brazo elástico

    # Reparto de masa suspendida por eje (si no lo tienes, 50/50)
    wbal_f = float(params.get('rWeightBalF', 0.5))
    MsF = Ms * wbal_f
    MsR = Ms * (1.0 - wbal_f)

    # Parte GEOMÉTRICA por eje (no pasa por θ → no mueve damper)
    # Convención de signos: ay>0 carga la derecha (FR, RR)
    dF_lat_geo_f =((MsF * ay * rc_f) / tF)
    dF_lat_geo_r =((MsR * ay * rc_r) / tR)

    # Parte ELÁSTICA (sí pasa por θ → mueve damper)
    M_lat_el = Ms * ay_now * h_eff  # N·m
    f_lat_el = M_lat_el / (tF + tR)  # N
    dF_lat_el = np.vstack([+f_lat_el,-f_lat_el, +f_lat_el, -f_lat_el])  # (4,N)

    # 2) lateral geométrico 
    dF_lat_f = (((MsF * ay * (h_eff)) / tF))
    dF_lat_r = (((MsR * ay * (h_eff)) / tR)) 

    R_tire = float(params.get('tire_radius', 0.33))  # m

    dF_lat_u_f = ((Mu_f)*ay_in*R_tire) / tF
    dF_lat_u_r = ((Mu_r)*ay_in*R_tire) / tR                 

    # ---  Matrices 4×N idénticas a las del solver --------------------------
    dF_lat  = np.vstack([+dF_lat_geo_f + dF_lat_f + dF_lat_u_f, -dF_lat_geo_f - dF_lat_f - dF_lat_u_f,
                        +dF_lat_geo_r + dF_lat_r + dF_lat_u_r, -dF_lat_geo_r - dF_lat_r - dF_lat_u_r])  # (4,N)

    f_dyn = dF_lat + dF_long             #  (+ dF_el si procede)
    # 4d) Fuerza neta en rueda (nunca negativa)
    wheel_load = (-aero + f_tire - dF_lat + static - f_damper ) / 9.81
    wheel_load_std = np.std(wheel_load, axis=1)          # (4,)
    wheel_load_max = np.max(wheel_load, axis=1)   # máximo por rueda [N]
    wheel_load_min = np.min(wheel_load, axis=1)   # mínimo por rueda [N]
    f_wheel = ((-aero) + f_dyn - f_spring)    # (4, N)
    
    f_damp_FL, Pxx_damp_FL = welch(f_damper[0], fs=fs, nperseg=nperseg, noverlap=noverlap)

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
    throttle_signal = np.interp(t, t_vec, throttle)   # rPedal  (0‑100 %)
    brake_signal    = np.interp(t, t_vec, brake)      # pBrake  (0‑100 %)
    ax_signal       = np.interp(t, t_vec, ax)         # m/s²  (+ acel / – frena)
    ay_signal       = np.interp(t, t_vec, ay)         # m/s²  (+ izq  / – dcha)

    # — umbrales algo más laxos —
    thr_throttle   = 10.0         # gas < 10 %   (antes 5 %)
    thr_brake      = 10.0          # freno < 10 % (antes 5 %)
    ax_long_limit  = 2.0           # |Ax| < 2.0 m/s²  ≈ 0.20 g   (antes 1.5)
    ay_lat_limit   = 0.9           # |Ay| > 0.9 m/s²  ≈ 0.09 g   (antes 1.2)
    lat_over_long  = 1.1           # Ay al menos 10 % mayor que Ax (antes 20 %)

    # 1) Grip‑limited LATERAL
    mask_no_pedals = (throttle_signal < thr_throttle) & (brake_signal < thr_brake)
    mask_lat       = (
        (np.abs(ax_signal) < ax_long_limit) &
        (np.abs(ay_signal) > ay_lat_limit)  &
        (np.abs(ay_signal) > lat_over_long * np.abs(ax_signal))
    )
    grip_lateral_mask = mask_no_pedals & mask_lat

    # 2) Grip‑limited FRENO
    grip_brake_mask = (
        (brake_signal > 30.0) &
        (ax_signal    < -ax_long_limit)    # decel > 2.0 m/s²
    )

    # 3) Grip‑limited TRACCIÓN
    grip_traction_mask = (
        (throttle_signal > 30.0) &
        (ax_signal       >  ax_long_limit) # acel  > 2.0 m/s²
    )

    grip_limited_pct = 100.0 * np.sum(grip_lateral_mask) / N

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
        front_load_rms_brake_std = np.std(fl_brake)
        rear_load_rms_brake_std  = np.std(rl_brake)
    else:
        front_load_rms_brake = 0.0
        rear_load_rms_brake  = 0.0
        front_load_rms_brake_std = 0.0
        rear_load_rms_brake_std  = 0.0

    # ── Grip-Limited en tracción ──
    if np.any(grip_traction_mask):
        fl_trac = np.mean(f_tire[0:2, grip_traction_mask], axis=0)
        rl_trac = np.mean(f_tire[2:4, grip_traction_mask], axis=0)
        front_load_rms_traction = np.sqrt(np.mean(fl_trac**2))
        rear_load_rms_traction  = np.sqrt(np.mean(rl_trac**2))
        front_load_rms_traction_std = np.std(fl_trac)
        rear_load_rms_traction_std  = np.std(rl_trac)
    else:
        front_load_rms_traction = 0.0
        rear_load_rms_traction  = 0.0
        front_load_rms_traction_std = 0.0
        rear_load_rms_traction_std  = 0.0

    # ──────────────────────────────────────────────────────────────────────────────
    # 7) Road-noise: wheel vertical-speed RMS [mm/s] para cada rueda
    # ──────────────────────────────────────────────────────────────────────────────
    # zt está en [m], convertimos a mm antes de derivar
    v_mm_s = [np.gradient(zt[i] * 1000, dt) for i in range(n_corners)]  # lista de 4 arrays (mm/s)
    rms_per_wheel = [np.sqrt(np.mean(v**2)) for v in v_mm_s]

    # Promedio por eje (front / rear)
    front_noise_vals = 0.5 * (rms_per_wheel[0] + rms_per_wheel[1])
    rear_noise_vals  = 0.5 * (rms_per_wheel[2] + rms_per_wheel[3])

    frh_rms_std = (np.std(heave_front[grip_lateral_mask])
                   if np.any(grip_lateral_mask) else 0.0)
    rrh_rms_std = (np.std(heave_rear[grip_lateral_mask])
                   if np.any(grip_lateral_mask) else 0.0)

    # ──────────────────────────────────────────────────────────────────────────────
    # 8) RMS en Grip-Limited (heave y carga) + no-grip
    # ──────────────────────────────────────────────────────────────────────────────

    # RMS de heave en mask grip-lateral (m → convertir más tarde a mm)
    if np.any(grip_lateral_mask):
        frh_rms = np.sqrt(np.mean(RH_front_c[grip_lateral_mask]**2))
        rrh_rms = np.sqrt(np.mean(RH_rear_c[grip_lateral_mask]**2))
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
        frh_rms_nongrip = np.sqrt(np.mean(RH_front_c[non_grip_mask]**2))
        rrh_rms_nongrip = np.sqrt(np.mean(RH_rear_c[non_grip_mask]**2))
        fl_ng = np.mean(wheel_load[0:2, non_grip_mask], axis=0)
        front_load_rms_nongrip = np.sqrt(np.mean(fl_ng**2))
        frh_rms_nongrip_std = np.std(RH_front_c[non_grip_mask])
        rrh_rms_nongrip_std = np.std(RH_rear_c[non_grip_mask])
    else:
        frh_rms_nongrip = 0.0
        rrh_rms_nongrip = 0.0
        front_load_rms_nongrip = 0.0
        frh_rms_nongrip_std = 0.0
        rrh_rms_nongrip_std = 0.0

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
    z_front_mm   = heave_front 
    z_rear_mm    = heave_rear  
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

    # ————————————————————————————————————————————————————————————————
    # PSD de Contact Patch Load (Welch)
    # ————————————————————————————————————————————————————————————————
    # señal media por eje
    front_sig = np.mean(wheel_load[0:2, :], axis=0)
    rear_sig  = np.mean(wheel_load[2:4, :], axis=0)

    # cálculo unilateral
    f_load, Pxx_f = welch(
        front_sig, fs=fs,
        window='hann', nperseg=nperseg, noverlap=noverlap,
        scaling='density', average='mean'
    )
    _,       Pxx_r = welch(
        rear_sig,  fs=fs,
        window='hann', nperseg=nperseg, noverlap=noverlap,
        scaling='density', average='mean'
    )

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
        'travel' :            bumpstop_mm_piston,             # (4, N) [m]                      # (4, N) [m]
        'RH_front':           dyn_hF, #RH_front_c,               # (N,) [m]
        'RH_rear':            dyn_hR, #RH_rear_c,                # (N,) [m]
        'travel_rel':         zs,             # (4, N)
        'travel_static':      travel_static,          # (4, 1)
        'travel_max':         travel_max,             # (4,)
        'travel_min':         travel_min,             # (4,)
        'damper_travel':      damper_travel_x, #(4, N)
        'dx_front':           dx_front,               # (N,) [m]
        'dx_rear':            dx_rear,                # (N,) [m]
        'dyn_hF':             dyn_hF,                 # (N,) [m]
        'dyn_hR':             dyn_hR,                 # (N,) [m]
        # --- Fuerzas de suspensión ---
        'f_spring':           f_spring,               # (4, N)
        'f_dyn':              f_dyn,               # (4, N)
        'f_bump':             f_bump,                 # (4, N)
        'f_arb':              f_arb,                  # (4, N)
        'f_damper':           f_damper,

        # --- Fuerza de neumático ---
        'f_tire':             f_tire,                 # (4, N)
        'f_tire_max':         f_tire_max,             # (4,)
        'f_tire_min':         f_tire_min,             # (4,)
        'f_tire_mean':        f_tire_mean,            # (4,)
        'f_tire_std':         f_tire_std,             # (4,)
        'f_tire_variation':   f_tire_var,
        'f_tire_variation_front': f_tire_var_f,
        'f_tire_variation_rear':  f_tire_var_r,
        'Fz_aero_front':      Fz_front,               # (N,) [N]
        'Fz_aero_rear':       Fz_rear,                # (N,) [N]     
        'wheel_load':         wheel_load, 
        'wheel_load_max':     wheel_load_max,        # (4,)
        'wheel_load_min':     wheel_load_min,        # (4,)
        'wheel_load_std':     wheel_load_std,
        'f_wheel':            f_wheel,

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
        'front_load_rms_brake_std':    front_load_rms_brake_std,
        'rear_load_rms_brake_std':     rear_load_rms_brake_std,
        'front_load_rms_traction_std': front_load_rms_traction_std,
        'rear_load_rms_traction_std':  rear_load_rms_traction_std,

        # --- RMS de heave / carga en grip-limited vs no-grip ---
        'frh_rms':            frh_rms,                # heave front RMS [m]
        'rrh_rms':            rrh_rms,                # heave rear RMS [m]
        'frh_rms_std':        frh_rms_std,
        'rrh_rms_std':        rrh_rms_std,
        'front_load_rms':     front_load_rms,         # tire load front RMS [N]
        'rear_load_rms':      rear_load_rms,          # tire load rear RMS [N]
        'frh_rms_nongrip':    frh_rms_nongrip,        # heave front RMS no-grip [m]
        'rrh_rms_nongrip':    rrh_rms_nongrip,        # heave rear RMS no-grip [m]
        'frh_rms_nongrip_std': frh_rms_nongrip_std,
        'rrh_rms_nongrip_std': rrh_rms_nongrip_std,
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
        'f_psd':              f_psd_heave,            # (M,)
        'psd_heave':          Pxx_heave,              # (M,)
        'psd_heave_peak':     psd_heave_peak,         # dB

        # Heave por eje [mm²/Hz]:
        'heave_front':        heave_front,             # (N,)
        'heave_rear':         heave_rear, # (N,)
        'f_psd_front':        f_heave_f,              # (M,)
        'psd_heave_front':    Pxx_heave_f,            # (M,)
        'f_psd_rear':         f_heave_r,              # (M,)
        'psd_heave_rear':     Pxx_heave_r,            # (M,)

        # Pitch global [rad²/Hz]:
        'f_psd_pitch':        f_psd_pitch,            # (M,)
        'pitch_front_mm':     pitch_front_mm,        # (N,)
        'pitch_rear_mm':      pitch_rear_mm,         # (N,)
        'psd_pitch':          Pxx_pitch,              # (M,)
        'psd_pitch_peak':     psd_pitch_peak,         # dB

        # Pitch inducido → vertical por eje [mm²/Hz]:
        'f_psd_pitch_front':  f_pitch_f,              # (M,)
        'psd_pitch_front':    Pxx_pitch_f,            # (M,)
        'f_psd_pitch_rear':   f_pitch_r,              # (M,)
        'psd_pitch_rear':     Pxx_pitch_r,            # (M,)

        'f_psd_damper'   :    f_damp_FL,                  # vector de frecuencias
        'psd_damper_mag_FL' : 10 * np.log10(Pxx_damp_FL),

        # --- Resultados auxiliares para Dash/HTML export ---                    
        'z_wheel':            z_wheel,                # (4, N)
        'zs':                 zs,                     # (4, N)
        'p_plot':             bumpstop_mm_piston,                 # (4, N)
        'x_spring_p':         x_spring_p,             # (4, N)    
        'h':                  h[0]+h,                      # (N,)
        'h_ddot':             np.gradient(sol.y[1], t),   # (N,)
        'v_damper':           rel_vel,               # (4, N)
        'x_spring':           x_spring,               # (4, N)
        'lap_time':           lap_time,               # scalar
        'pitch_deg':          np.degrees(phi),        # (N,)
        'distance':           np.cumsum(vx_interp := np.interp(t, t_vec, vx)) * np.gradient(t),  # (N,)
        'track_name':         params.get('track_name', 'Unknown Track'),
        'vx':                 vx_interp,
        't_vec':              t,                     # (N,)
        'f_psd_load'          : f_load,             # vector de frecuencias [Hz]
        'psd_load_mag_front'  : 10 * np.log10(Pxx_f + 1e-30),       # magnitud dB eje frontal
        'psd_load_mag_rear'   : 10 * np.log10(Pxx_r + 1e-30),        # magnitud dB eje trasero
   
    }


