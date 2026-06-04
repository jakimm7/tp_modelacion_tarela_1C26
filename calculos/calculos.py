import numpy as np
from pathlib import Path
from utils.utils import integrar_rk2, acc_luna, integrar_nystrom, energia_mecanica, integrar_euler, acc_orion, momento_angular
from graph.graph import guardar_figura, crear_figura, graficar_trayectoria, graficar_serie_temporal, graficar_lineas_referencia, caja_texto

OUT_DIR = Path("./graficos")

G = 6.674e-20
M_T = 5.972e24 
GM_T = G * M_T
R_LUNA = 384_400.0
R_PERIGEO = 362_600.0
R_APOGEO = 405_400.0
A_LUNA = (R_PERIGEO + R_APOGEO) / 2.0
V_PERI = np.sqrt(GM_T * (2.0 / R_PERIGEO - 1.0 / A_LUNA))
T_LUNAR = 2.0 * np.pi * np.sqrt(A_LUNA**3 / GM_T)

R0_LUNA = np.array([R_PERIGEO, 0.0])
V0_LUNA = np.array([0.0, V_PERI])

BG = "#0a0e1a"
GC = "#1e2a40"
EC = "#2979ff"
MC = "#b0bec5"
OC = "#ff6d00"
EUC = "#ef5350"
RKC = "#66bb6a"
NYC = "#29b6f6"
TC = "#eceff1"

def calculo_orbita_lunar():
    N = 40_000
    h = T_LUNAR / N
    rl, vl = integrar_rk2(R0_LUNA, V0_LUNA, acc_luna, h, N)
 
    dl = np.linalg.norm(rl, axis=1)
    sl = np.linalg.norm(vl, axis=1)
    ti = np.linspace(0, T_LUNAR, N + 1) / 86400
    pi_ = dl.argmin()
    ai_ = dl.argmax()
 
    print(f"  Perigeo simulado : {dl.min():.0f} km   (esperado ~{R_PERIGEO:.0f} km)")
    print(f"  Apogeo  simulado : {dl.max():.0f} km   (esperado ~{R_APOGEO:.0f} km)")
    print(f"  Vel. máx (perigeo): {sl[pi_]:.4f} km/s")
    print(f"  Vel. mín (apogeo) : {sl[ai_]:.4f} km/s")
 
    fig, axes = crear_figura(3, titulo="Punto 1 – Órbita Lunar: Validación", ancho=16, alto=5)
 
    graficar_trayectoria(
        axes[0],
        trayectorias=[{"r": rl, "color": MC, "lw": 1.2, "label": "Órbita Lunar"}],
        titulo="Trayectoria (1 período)",
        puntos_extra=[
            {"xy": (rl[pi_, 0], rl[pi_, 1]), "color": "#ff1744", "s": 60,
             "label": f"Perigeo {dl.min():.0f} km"},
            {"xy": (rl[ai_, 0], rl[ai_, 1]), "color": "#69f0ae", "s": 60,
             "label": f"Apogeo {dl.max():.0f} km"},
        ],
    )
 
    graficar_serie_temporal(
        axes[1],
        series=[{"t": ti, "y": dl, "color": MC, "lw": 1.2, "label": "Distancia"}],
        titulo="Distancia Tierra-Luna",
        ylabel="distancia [km]",
    )
    graficar_lineas_referencia(axes[1], [
        {"y": R_PERIGEO, "color": "#ff1744", "label": f"{R_PERIGEO:.0f} km"},
        {"y": R_APOGEO,  "color": "#69f0ae", "label": f"{R_APOGEO:.0f} km"},
    ])
    axes[1].legend(fontsize=8)
 
    graficar_serie_temporal(
        axes[2],
        series=[{"t": ti, "y": sl, "color": EC, "lw": 1.2}],
        titulo="Velocidad orbital lunar",
        ylabel="|v| [km/s]",
    )
    axes[2].annotate("Vel. máx\n(perigeo)",
                     xy=(ti[pi_], sl[pi_]),
                     xytext=(ti[pi_] + 1.0, sl[pi_] + 0.003),
                     arrowprops=dict(arrowstyle="->", color=TC), fontsize=8)
    axes[2].annotate("Vel. mín\n(apogeo)",
                     xy=(ti[ai_], sl[ai_]),
                     xytext=(ti[ai_] + 1.0, sl[ai_] - 0.005),
                     arrowprops=dict(arrowstyle="->", color=TC), fontsize=8)
 
    fig.tight_layout()
    guardar_figura(fig, "Punto1_Orbita_Lunar.png", OUT_DIR)
    return rl, vl, dl, sl, ti, pi_, ai_

def calculo_posicion_velocidad_orion(r0_orion, v0_orion):
    T_FLY = 9.0 * 86400
    N = 30_000
    h = T_FLY / N
 
    r_luna_fix = np.array([
        R_LUNA * np.cos(np.radians(180)),
        R_LUNA * np.sin(np.radians(180)),
    ])
 
    v_mag  = np.linalg.norm(v0_orion)
    a_base = np.arctan2(v0_orion[1], v0_orion[0])
 
    print("  Shooting method — buscando ángulo de free-return...")
    mejor_score = np.inf
    mejor_ang   = 0.0
    mejor_tray  = None
 
    for delta_deg in np.linspace(-8.0, 8.0, 60):
        angulo  = a_base + np.radians(delta_deg)
        v_trial = v_mag * np.array([np.cos(angulo), np.sin(angulo)])
        r_t, v_t = integrar_rk2(r0_orion, v_trial, acc_orion, h, N, r_luna=r_luna_fix)
 
        dist_final = np.linalg.norm(r_t[-1])
        dist_luna  = np.linalg.norm(r_t - r_luna_fix, axis=1).min()
        score      = dist_final / 1e4 - (5.0 if dist_luna < 100_000 else 0.0)
 
        if score < mejor_score:
            mejor_score = score
            mejor_ang   = delta_deg
            mejor_tray  = (r_t.copy(), v_t.copy())
 
    r_orion, v_orion = mejor_tray
    print(f"Ángulo óptimo    : {mejor_ang:.2f}°")
    print(f"Dist. mín a Luna : {np.linalg.norm(r_orion - r_luna_fix, axis=1).min():.0f} km")
    print(f"Dist. final Tierra: {np.linalg.norm(r_orion[-1]):.0f} km")
 
    rl2, _ = integrar_rk2(R0_LUNA, V0_LUNA, acc_luna, T_FLY / 20_000, 20_000)
 
    t_arr = np.linspace(0, T_FLY, N + 1) / 86400
    d_T = np.linalg.norm(r_orion, axis=1)
    d_L = np.linalg.norm(r_orion - r_luna_fix, axis=1)
 
    fig, axes = crear_figura(
        2,
        titulo="Puntos 2 y 3 – Trayectoria Orion (Problema 3 Cuerpos / Free-Return)",
        ancho=15, alto=7,
    )
 
    graficar_trayectoria(
        axes[0],
        trayectorias=[
            {"r": r_orion, "color": OC,  "lw": 1.5, "label": "Orion (RK2)", "zorder": 4},
            {"r": rl2,     "color": MC,  "lw": 0.8, "label": "Órbita Lunar",
             "ls": "--", "alpha": 0.5},
        ],
        titulo=f"Free-Return Trajectory  (Δθ = {mejor_ang:.2f}°)",
        punto_luna=r_luna_fix,
        puntos_extra=[
            {"xy": r0_orion,    "color": "#ffeb3b", "marker": "*", "s": 90,
             "label": "Orion t₀ (CSV)"},
            {"xy": r_orion[-1], "color": "#e91e63", "marker": "v", "s": 90,
             "label": "Orion t_final"},
        ],
    )
 
    graficar_serie_temporal(
        axes[1],
        series=[
            {"t": t_arr, "y": d_T / 1e3, "color": EC, "lw": 1.2,
             "label": "Dist. a la Tierra"},
            {"t": t_arr, "y": d_L / 1e3, "color": MC, "lw": 1.2,
             "label": "Dist. a la Luna"},
        ],
        titulo="Distancias durante el vuelo",
        ylabel="distancia [10³ km]",
    )
    graficar_lineas_referencia(axes[1], [
        {"y": 1.737, "color": "#ff1744", "ls": ":", "label": "R_Luna = 1 737 km"},
    ])
    axes[1].legend(fontsize=8)
 
    fig.tight_layout()
    guardar_figura(fig, "Punto2_3_Orion_FreeReturn.png", OUT_DIR)
    return r_orion, v_orion, r_luna_fix, mejor_ang, T_FLY
 
 
def calculo_euler_rk2(r0_orion, v0_orion, r_luna_fix, mejor_ang, t_fly):
    N = 20_000
    h = t_fly / N
 
    v_mag  = np.linalg.norm(v0_orion)
    a_base = np.arctan2(v0_orion[1], v0_orion[0])
    v_opt  = v_mag * np.array([
        np.cos(a_base + np.radians(mejor_ang)),
        np.sin(a_base + np.radians(mejor_ang)),
    ])
 
    r_eu, _ = integrar_euler(r0_orion, v_opt, acc_orion, h, N, r_luna=r_luna_fix)
    r_rk, _ = integrar_rk2  (r0_orion, v_opt, acc_orion, h, N, r_luna=r_luna_fix)
 
    t4    = np.linspace(0, t_fly, N + 1) / 86400
    diff4 = np.linalg.norm(r_eu - r_rk, axis=1)
 
    fig, axes = crear_figura(
        2, titulo="Punto 4 – Euler vs RK2: Precisión Numérica", ancho=15, alto=6,
    )
 
    # Panel 0: trayectorias comparadas
    graficar_trayectoria(
        axes[0],
        trayectorias=[
            {"r": r_eu, "color": EUC, "lw": 1.2, "label": "Euler  O(h)",  "alpha": 0.9},
            {"r": r_rk, "color": RKC, "lw": 1.2, "label": "RK2    O(h²)", "alpha": 0.9},
        ],
        titulo="Trayectorias (mismo paso h)",
        punto_luna=r_luna_fix,
    )
 
    graficar_serie_temporal(
        axes[1],
        series=[
            {"t": t4, "y": diff4 + 1e-3, "color": EUC, "lw": 1.4,
             "label": "|r_Euler − r_RK2|"},
        ],
        titulo="Divergencia Euler vs RK2",
        ylabel="error posición [km]  (log)",
        escala_log=True,
    )

    caja_texto(axes[1],
        "Euler: O(h)  → error ∝ h\n"
        "RK2:  O(h²) → error ∝ h²\n"
        "→ Euler falla la inserción lunar\n"
        "→ RK2 cierra el ciclo de retorno"
    )
 
    fig.tight_layout()
    guardar_figura(fig, "Punto4_Euler_vs_RK2.png", OUT_DIR)
    return r_eu, r_rk, diff4, t4
 
def simulacion_orbita_lunar():
    T5 = 6.0 * T_LUNAR
    N5 = 60_000
    h5 = T5 / N5
 
    r_eu5, v_eu5 = integrar_euler(R0_LUNA, V0_LUNA, acc_luna, h5, N5)
    r_rk5, v_rk5 = integrar_rk2  (R0_LUNA, V0_LUNA, acc_luna, h5, N5)
 
    d_eu5 = np.linalg.norm(r_eu5, axis=1)
    d_rk5 = np.linalg.norm(r_rk5, axis=1)
    E_eu5 = energia_mecanica(v_eu5, r_eu5)
    E_rk5 = energia_mecanica(v_rk5, r_rk5)
    E0    = E_eu5[0]
    t5    = np.linspace(0, T5, N5 + 1) / 86400
 
    fig, axes = crear_figura(
        3, titulo="Punto 5 – Inestabilidad a Largo Plazo: |γ| > 1", ancho=17, alto=5,
    )
 
    graficar_trayectoria(
        axes[0],
        trayectorias=[
            {"r": r_eu5, "color": EUC, "lw": 0.5, "alpha": 0.7, "label": "Euler"},
            {"r": r_rk5, "color": RKC, "lw": 0.5, "alpha": 0.7, "label": "RK2"},
        ],
        titulo="Trayectorias (6 períodos lunares)",
    )
 
    graficar_serie_temporal(
        axes[1],
        series=[
            {"t": t5, "y": d_eu5 / 1e3, "color": EUC, "lw": 0.8, "label": "Euler"},
            {"t": t5, "y": d_rk5 / 1e3, "color": RKC, "lw": 0.8, "label": "RK2"},
        ],
        titulo="Radio orbital",
        ylabel="|r| [10³ km]",
    )

    graficar_lineas_referencia(axes[1], [
        {"y": (R_PERIGEO + R_APOGEO) / 2 / 1e3, "color": "white",
         "ls": "--", "lw": 0.6, "alpha": 0.4, "label": "Radio medio"},
    ])
    axes[1].legend(fontsize=8)
 
    graficar_serie_temporal(
        axes[2],
        series=[
            {"t": t5, "y": (E_eu5 - E0) / abs(E0) * 100, "color": EUC,
             "lw": 0.8, "label": "Euler"},
            {"t": t5, "y": (E_rk5 - E0) / abs(E0) * 100, "color": RKC,
             "lw": 0.8, "label": "RK2"},
        ],
        titulo="Deriva de energía mecánica",
        ylabel="ΔE/E₀ [%]",
    )
    caja_texto(axes[2],
        "|γ| > 1  →  inestabilidad\n"
        "'Energía numérica' espuria\n"
        "→ espiral hacia afuera"
    )
 
    fig.tight_layout()
    guardar_figura(fig, "Punto5_LargoPlazo_Inestabilidad.png", OUT_DIR)
    return r_eu5, v_eu5, r_rk5, v_rk5, E0, t5, N5, h5
 
def metodo_alternativo_nynstrom(r_eu5, v_eu5, r_rk5, v_rk5, E0, t5, N5, h5):
    r_ny5, v_ny5 = integrar_nystrom(R0_LUNA, V0_LUNA, acc_luna, h5, N5)
 
    E_eu5 = energia_mecanica(v_eu5, r_eu5)
    E_rk5 = energia_mecanica(v_rk5, r_rk5)
    E_ny5 = energia_mecanica(v_ny5, r_ny5)
    L_eu  = momento_angular(r_eu5, v_eu5)
    L_rk  = momento_angular(r_rk5, v_rk5)
    L_ny  = momento_angular(r_ny5, v_ny5)
    L0    = L_eu[0]
 
    fig, axes = crear_figura(
        3,
        titulo="Punto 6 – Nyström vs Euler/RK2: Conservación (|γ₁γ₂| = 1)",
        ancho=17, alto=5,
    )
 
    graficar_trayectoria(
        axes[0],
        trayectorias=[
            {"r": r_ny5, "color": NYC, "lw": 0.8, "alpha": 0.9, "label": "Nyström"},
            {"r": r_eu5, "color": EUC, "lw": 0.4, "alpha": 0.4,
             "label": "Euler (espiral)"},
        ],
        titulo="Trayectorias (6 períodos)",
    )
 
    graficar_serie_temporal(
        axes[1],
        series=[
            {"t": t5, "y": (E_eu5 - E0) / abs(E0) * 100, "color": EUC,
             "lw": 0.8, "label": "Euler"},
            {"t": t5, "y": (E_rk5 - E0) / abs(E0) * 100, "color": RKC,
             "lw": 0.8, "label": "RK2"},
            {"t": t5, "y": (E_ny5 - E0) / abs(E0) * 100, "color": NYC,
             "lw": 0.8, "label": "Nyström"},
        ],
        titulo="Deriva de energía mecánica",
        ylabel="ΔE/E₀ [%]",
    )
 
    graficar_serie_temporal(
        axes[2],
        series=[
            {"t": t5, "y": (L_eu - L0) / abs(L0) * 100, "color": EUC,
             "lw": 0.8, "label": "Euler"},
            {"t": t5, "y": (L_rk - L0) / abs(L0) * 100, "color": RKC,
             "lw": 0.8, "label": "RK2"},
            {"t": t5, "y": (L_ny - L0) / abs(L0) * 100, "color": NYC,
             "lw": 0.8, "label": "Nyström"},
        ],
        titulo="Conservación del momento angular",
        ylabel="ΔL/L₀ [%]",
    )
    caja_texto(axes[2],
        "Nyström: |γ₁·γ₂| = 1\n"
        "→ γ solo rota en ℂ\n"
        "→ sin amplificación\n"
        "→ órbita estable"
    )
 
    fig.tight_layout()
    guardar_figura(fig, "Punto6_Nystrom_Conservativo.png", OUT_DIR)
    return r_ny5, v_ny5, E_ny5
