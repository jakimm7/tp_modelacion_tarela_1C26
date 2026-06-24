import numpy as np
from pathlib import Path

from utils.utils import (
    integrar_rk2, integrar_euler, integrar_nystrom,
    acc_luna, acc_orion, energia_mecanica, momento_angular,
    estudio_convergencia_periodico, estudio_convergencia_referencia,
    ajustar_orden_loglog, orden_local,
    radio_min_max_por_periodo, indices_por_periodo,
    imprimir_resultados, imprimir_tabla,
)
from graph.graph import (
    guardar_figura, crear_figura, graficar_trayectoria, graficar_serie_temporal,
    graficar_lineas_referencia, caja_texto, graficar_convergencia,
)

OUT_DIR = Path("./graficos")

NS_APENDICE_PTO_1 = [200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200]
NS_APENDICE_PTO_4 = [800, 1600, 3200, 6400, 12800, 25600, 51200, 102400]
NS_APENDICE_PTO_5_6 = [3000, 6000, 12000, 24000, 48000, 96000, 192000]

EXTENSION_GRAFICO = ".png"

NOMBRE_GRAFICO_NYSTROM = "Punto6_Nystrom_Conservativo"
NOMBRE_GRAFICO_CONV_NYSTROM = "Punto6b_Convergencia_Nystrom"
NOMBRE_GRAFICO_CONV_LARGO_PLAZO = "Punto5b_Convergencia_LargoPlazo"
NOMBRE_GRAFICO_CONV_ORION = "Punto4b_Convergencia_Orion"
NOMBRE_GRAFICO_EULER_RK2 = "Punto4_Euler_vs_RK2"
NOMBRE_GRAFICO_CALIBRACION = "Punto2_3_Efecto_Calibracion"

ANCHO_FIG = 8.5
ALTO_FIG = 6.5

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

EC  = "#2979ff"
MC  = "#b0bec5"
OC  = "#ff6d00"
EUC = "#ef5350"
RKC = "#66bb6a"
NYC = "#29b6f6"
TC  = "#eceff1"
AM  = "#ffeb3b"
RC  = "#ab47bc"
SC  = "#90a4ae"

def punto1_orbita_lunar():
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
            {"xy": (rl[pi_, 0], rl[pi_, 1]), "color": EUC, "s": 60,
             "label": f"Perigeo {dl.min():.0f} km"},
            {"xy": (rl[ai_, 0], rl[ai_, 1]), "color": NYC, "s": 60,
             "label": f"Apogeo {dl.max():.0f} km"},
        ],
        loc_leyenda="upper right"
    )

    graficar_serie_temporal(
        axes[1],
        series=[{"t": ti, "y": dl, "color": MC, "lw": 1.2, "label": "Distancia"}],
        titulo="Distancia Tierra-Luna",
        ylabel="distancia [km]",
    )
    graficar_lineas_referencia(axes[1], [
        {"y": R_PERIGEO, "color": EUC, "label": f"{R_PERIGEO:.0f} km"},
        {"y": R_APOGEO,  "color": NYC, "label": f"{R_APOGEO:.0f} km"},
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

def apendice_punto1b_euler_vs_rk2(rl, vl, dl, ti):
    N = 40_000
    h = T_LUNAR / N
    r_eu, v_eu = integrar_euler(R0_LUNA, V0_LUNA, acc_luna, h, N)
    r_rk, v_rk = rl, vl

    d_eu = np.linalg.norm(r_eu, axis=1)
    s_eu = np.linalg.norm(v_eu, axis=1)
    s_rk = np.linalg.norm(v_rk, axis=1)

    E_eu = energia_mecanica(v_eu, r_eu)
    E_rk = energia_mecanica(v_rk, r_rk)
    E0 = E_eu[0]

    err_cierre_eu = np.linalg.norm(r_eu[-1] - R0_LUNA)
    err_cierre_rk = np.linalg.norm(r_rk[-1] - R0_LUNA)

    imprimir_resultados("1b (Apéndice)", "Euler vs RK2 al paso de producción", [
        ("seccion", "Geometría de la órbita"),
        ("dato", "Apogeo simulado Euler", f"{d_eu.max():.0f}", "km"),
        ("dato", "Apogeo simulado RK2", f"{dl.max():.0f}", "km"),
        ("dato", "Error de apogeo Euler", f"{abs(d_eu.max()-R_APOGEO):.1f}", "km"),
        ("dato", "Error de apogeo RK2", f"{abs(dl.max()-R_APOGEO):.1f}", "km"),
        ("seccion", "Velocidades extremas"),
        ("dato", "Vel. máxima Euler (perigeo)", f"{s_eu.max():.6f}", "km/s"),
        ("dato", "Vel. máxima RK2 (perigeo)", f"{s_rk.max():.6f}", "km/s"),
        ("dato", "Vel. mínima Euler (apogeo)", f"{s_eu.min():.6f}", "km/s"),
        ("dato", "Vel. mínima RK2 (apogeo)", f"{s_rk.min():.6f}", "km/s"),
        ("seccion", "Cierre de órbita tras 1 período (debe ser ~0)"),
        ("dato", "Error de cierre Euler", f"{err_cierre_eu:.3e}", "km"),
        ("dato", "Error de cierre RK2", f"{err_cierre_rk:.3e}", "km"),
        ("seccion", "Deriva de energía mecánica dE/E0 (final)"),
        ("dato", "Euler", f"{(E_eu[-1]-E0)/abs(E0)*100:+.4e}", "%"),
        ("dato", "RK2", f"{(E_rk[-1]-E0)/abs(E0)*100:+.4e}", "%"),
        ("texto", ""),
        ("texto", "Euler acumula un error de cierre del orden de 10^3 km en 1 período;"),
        ("texto", "RK2 cierra la órbita con un error de apenas centésimas de km."),
    ])

    t_dias = ti
    fig, axes = crear_figura(3, titulo="Punto 1b – Órbita Lunar: Euler vs RK2 (mismo paso h)",
                              ancho=16, alto=5)

    graficar_trayectoria(
        axes[0],
        trayectorias=[
            {"r": r_eu, "color": EUC, "lw": 1.1, "label": "Euler", "zorder": 4},
            {"r": r_rk, "color": RKC, "lw": 1.1, "label": "RK2", "zorder": 3},
        ],
        titulo="Trayectoria (1 período)",
    )

    graficar_serie_temporal(
        axes[1],
        series=[
            {"t": t_dias, "y": d_eu, "color": EUC, "lw": 1.0, "label": "Euler"},
            {"t": t_dias, "y": dl,   "color": RKC, "lw": 1.0, "label": "RK2"},
        ],
        titulo="Distancia Tierra-Luna",
        ylabel="distancia [km]",
    )
    graficar_lineas_referencia(axes[1], [
        {"y": R_PERIGEO, "color": MC, "ls": ":", "label": f"{R_PERIGEO:.0f} km"},
        {"y": R_APOGEO,  "color": MC, "ls": ":", "label": f"{R_APOGEO:.0f} km"},
    ])

    graficar_serie_temporal(
        axes[2],
        series=[
            {"t": t_dias, "y": (E_eu - E0) / abs(E0) * 100, "color": EUC, "label": "Euler"},
            {"t": t_dias, "y": (E_rk - E0) / abs(E0) * 100, "color": RKC, "label": "RK2"},
        ],
        titulo="Deriva de energía mecánica",
        ylabel="ΔE/E0 [%]",
    )

    fig.tight_layout()
    guardar_figura(fig, "Punto1b_Euler_vs_RK2_Produccion.png", OUT_DIR)

    fig2, axes2 = crear_figura(2, titulo="Punto 1 (corregido) - Efecto del paso h en Euler vs RK2",
                                ancho=14, alto=6)
    for ax, N_grueso in zip(axes2, (200, 3200)):
        h_g = T_LUNAR / N_grueso
        r_eu_g, _ = integrar_euler(R0_LUNA, V0_LUNA, acc_luna, h_g, N_grueso)
        r_rk_g, _ = integrar_rk2(R0_LUNA, V0_LUNA, acc_luna, h_g, N_grueso)
        graficar_trayectoria(
            ax,
            trayectorias=[
                {"r": r_eu_g, "color": EUC, "lw": 1.3, "label": "Euler"},
                {"r": r_rk_g, "color": RKC, "lw": 1.3, "label": "RK2"},
            ],
            titulo=f"N = {N_grueso}    (h = {h_g:.0f} s)",
        )
    fig2.tight_layout()
    guardar_figura(fig2, "Punto1_Efecto_Paso_h.png", OUT_DIR)

    return err_cierre_eu, err_cierre_rk


def apendice_punto1c_convergencia():
    hs_eu, err_eu = estudio_convergencia_periodico(R0_LUNA, V0_LUNA, acc_luna, integrar_euler, T_LUNAR, NS_APENDICE_PTO_1)
    hs_rk, err_rk = estudio_convergencia_periodico(R0_LUNA, V0_LUNA, acc_luna, integrar_rk2,   T_LUNAR, NS_APENDICE_PTO_1)

    orden_eu = ajustar_orden_loglog(hs_eu, err_eu)
    orden_rk = ajustar_orden_loglog(hs_rk, err_rk)
    razon_eu = err_eu[:-1] / err_eu[1:]
    razon_rk = err_rk[:-1] / err_rk[1:]

    filas = []
    for i, N in enumerate(NS_APENDICE_PTO_1):
        r_eu_str = f"{razon_eu[i-1]:.3f}" if i > 0 else "—"
        r_rk_str = f"{razon_rk[i-1]:.3f}" if i > 0 else "—"
        filas.append([N, f"{hs_eu[i]:.1f}", f"{err_eu[i]:.3e}", f"{err_rk[i]:.3e}", r_eu_str, r_rk_str])

    print("\n  Cuadro 2 — Error de cierre de órbita en función del paso h (Euler vs RK2)")
    imprimir_tabla(["N", "h [s]", "Error Euler [km]", "Error RK2 [km]", "Razón Euler", "Razón RK2"], filas)

    print("\n  Cuadro 3 — Orden de convergencia teórico vs. empírico")
    imprimir_tabla(["Método", "Orden teórico", "Orden empírico"], [
        ["Euler explícito", "1", f"{orden_eu:.4f}"],
        ["RK2 (Punto Medio)", "2", f"{orden_rk:.4f}"],
    ])

    fig, ax = crear_figura(1, titulo="Punto 1c – Convergencia: orden de Euler y RK2", ancho=8, alto=6.5)
    ax = ax[0]
    h_ref = np.array([hs_eu.min(), hs_eu.max()])
    ref_h1 = err_eu[-1] * (h_ref / hs_eu[-1]) ** 1
    ref_h2 = err_rk[-1] * (h_ref / hs_rk[-1]) ** 2
    graficar_convergencia(
        ax,
        series=[
            {"h": hs_eu, "err": err_eu, "color": EUC, "marker": "o",
             "label": f"Euler (orden ajustado = {orden_eu:.2f})"},
            {"h": hs_rk, "err": err_rk, "color": RKC, "marker": "s",
             "label": f"RK2 (orden ajustado = {orden_rk:.2f})"},
        ],
        referencias=[
            {"h": h_ref, "y": ref_h1, "ls": "--", "color": RC, "label": "Referencia O(h¹)"},
            {"h": h_ref, "y": ref_h2, "ls": ":",  "color": RC, "label": "Referencia O(h²)"},
        ],
        titulo="",
        xlabel="paso de tiempo h [s]",
        ylabel="error de cierre de posición tras 1 período [km]",
    )
    fig.tight_layout()
    guardar_figura(fig, "Punto1c_Convergencia.png", OUT_DIR)

    return hs_eu, err_eu, hs_rk, err_rk, orden_eu, orden_rk

def punto2_3_calibracion_pvi(r0_orion, v0_orion):
    T_FLY = 9.0 * 86400
    N = 20_000
    h = T_FLY / N

    r_luna_fix = np.array([R_LUNA * np.cos(np.radians(180)), R_LUNA * np.sin(np.radians(180)),])

    v_mag  = np.linalg.norm(v0_orion)
    a_base = np.arctan2(v0_orion[1], v0_orion[0])

    r0_mod = np.linalg.norm(r0_orion)
    eps_orb = v_mag**2 / 2.0 - GM_T / r0_mod
    if eps_orb < 0:
        a_kepler = -GM_T / (2.0 * eps_orb)
        apoapsis_sup = 2.0 * a_kepler
        print(f"  [Vis-viva] Con |r0|={r0_mod:.0f} km y |v0|={v_mag:.4f} km/s (fijos, no")
        print(f"  dependen de Δθ): a = {a_kepler:.0f} km  ->  distancia máxima posible")
        print(f"  desde la Tierra < {apoapsis_sup:.0f} km (cota 2a), muy por debajo de")
        print(f"  R_Luna = {R_LUNA:.0f} km. Esto es consistente con que esta porción de la")
        print(f"  telemetría corresponde aún a una órbita de elevación (previa a la")
        print(f"  inyección translunar real), y explica por qué NINGÚN Δθ puede, por sí")
        print(f"  solo, producir un sobrevuelo lunar cercano sin alterar |v0|.")

    deltas_diag = np.linspace(-180.0, 180.0, 121)
    N_diag = 3_000
    h_diag = T_FLY / N_diag
    dist_luna_diag = np.empty(len(deltas_diag))
    dist_final_diag = np.empty(len(deltas_diag))
    for i, dd in enumerate(deltas_diag):
        angulo = a_base + np.radians(dd)
        v_trial = v_mag * np.array([np.cos(angulo), np.sin(angulo)])
        r_t, _ = integrar_rk2(r0_orion, v_trial, acc_orion, h_diag, N_diag, r_luna=r_luna_fix)
        dist_luna_diag[i] = np.linalg.norm(r_t - r_luna_fix, axis=1).min()
        dist_final_diag[i] = np.linalg.norm(r_t[-1])

    print("  Calibración paramétrica del ángulo inicial (PVI) — Grid Search Δθ ∈ [-45°, 45°]...")
    deltas_cal = np.linspace(-45.0, 45.0, 91)
    UMBRAL_SOBREVUELO = 100_000.0

    mejor_score = np.inf
    mejor_ang   = 0.0
    mejor_tray  = None
    dist_luna_cal  = np.empty(len(deltas_cal))
    dist_final_cal = np.empty(len(deltas_cal))

    for i, delta_deg in enumerate(deltas_cal):
        angulo  = a_base + np.radians(delta_deg)
        v_trial = v_mag * np.array([np.cos(angulo), np.sin(angulo)])
        r_t, v_t = integrar_rk2(r0_orion, v_trial, acc_orion, h, N, r_luna=r_luna_fix)

        dist_final = np.linalg.norm(r_t[-1])
        dist_luna  = np.linalg.norm(r_t - r_luna_fix, axis=1).min()
        dist_luna_cal[i]  = dist_luna
        dist_final_cal[i] = dist_final
        score = dist_final / 1e4 - (5.0 if dist_luna < UMBRAL_SOBREVUELO else 0.0)

        if score < mejor_score:
            mejor_score = score
            mejor_ang   = delta_deg
            mejor_tray  = (r_t.copy(), v_t.copy())

    r_orion, v_orion = mejor_tray
    dist_luna_min_opt = np.linalg.norm(r_orion - r_luna_fix, axis=1).min()
    print(f"Δθ óptimo (calibrado) : {mejor_ang:.2f}°")
    print(f"Dist. mín a Luna      : {dist_luna_min_opt:.0f} km")
    print(f"Dist. final a Tierra  : {np.linalg.norm(r_orion[-1]):.0f} km")

    rl2, _ = integrar_rk2(R0_LUNA, V0_LUNA, acc_luna, T_FLY / 20_000, 20_000)

    t_arr = np.linspace(0, T_FLY, N + 1) / 86400
    d_T = np.linalg.norm(r_orion, axis=1)
    d_L = np.linalg.norm(r_orion - r_luna_fix, axis=1)

    fig, axes = crear_figura(
        2,
        titulo="Puntos 2 y 3 – Trayectoria Orion (PVI calibrado paramétricamente)",
        ancho=15, alto=7,
    )

    graficar_trayectoria(
        axes[0],
        trayectorias=[
            {"r": r_orion, "color": OC,  "lw": 1.5, "label": "Orion (RK2)", "zorder": 4},
            {"r": rl2,     "color": MC,  "lw": 0.8, "label": "Órbita Lunar",
             "ls": "--", "alpha": 0.5},
        ],
        titulo=f"Trayectoria calibrada  (Δθ = {mejor_ang:.2f}°)",
        punto_luna=r_luna_fix,
        puntos_extra=[
            {"xy": r0_orion,    "color": AM, "marker": "*", "s": 90,
             "label": "Orion t₀ (CSV)"},
            {"xy": r_orion[-1], "color": EUC, "marker": "v", "s": 90,
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
        {"y": 1.737, "color": EUC, "ls": ":", "label": "R_Luna = 1 737 km"},
    ])
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    guardar_figura(fig, "Punto2_3_Orion_FreeReturn.png", OUT_DIR)

    datos_diagnostico = dict(
        deltas_diag=deltas_diag, dist_luna_diag=dist_luna_diag, dist_final_diag=dist_final_diag,
        deltas_cal=deltas_cal, dist_luna_cal=dist_luna_cal, dist_final_cal=dist_final_cal,
        umbral=UMBRAL_SOBREVUELO,
    )
    return r_orion, v_orion, r_luna_fix, mejor_ang, T_FLY, datos_diagnostico

def apendice_punto2_3_efecto_calibracion(r0_orion, v0_orion, r_luna_fix, mejor_ang, T_FLY):
    N = 20_000
    h = T_FLY / N
    v_mag  = np.linalg.norm(v0_orion)
    a_base = np.arctan2(v0_orion[1], v0_orion[0])

    r_sin, _ = integrar_rk2(r0_orion, v0_orion, acc_orion, h, N, r_luna=r_luna_fix)

    v_opt = v_mag * np.array([
        np.cos(a_base + np.radians(mejor_ang)),
        np.sin(a_base + np.radians(mejor_ang)),
    ])
    r_cal, _ = integrar_rk2(r0_orion, v_opt, acc_orion, h, N, r_luna=r_luna_fix)

    t_arr = np.linspace(0, T_FLY, N + 1) / 86400
    dL_sin = np.linalg.norm(r_sin - r_luna_fix, axis=1)
    dL_cal = np.linalg.norm(r_cal - r_luna_fix, axis=1)

    imprimir_resultados("2-3 (Apéndice)", "Efecto de la calibración paramétrica del ángulo Δθ", [
        ("dato", "Distancia final a la Tierra (sin calibrar, Δθ=0)", f"{np.linalg.norm(r_sin[-1]):.0f}", "km"),
        ("dato", "Distancia final a la Tierra (calibrado)", f"{np.linalg.norm(r_cal[-1]):.0f}", "km"),
        ("dato", "Mejora relativa en distancia final",
         f"{(1 - np.linalg.norm(r_cal[-1])/np.linalg.norm(r_sin[-1]))*100:.1f}", "%"),
    ])

    fig, axes = crear_figura(2, titulo="Puntos 2 y 3 (corregido) – Efecto de la calibración del ángulo Δθ", ancho=14, alto=6)

    graficar_trayectoria(
        axes[0],
        trayectorias=[
            {"r": r_sin, "color": SC, "lw": 1.2, "ls": "--", "alpha": 0.8, "label": "Sin calibrar (Δθ=0)"},
            {"r": r_cal, "color": OC, "lw": 1.5, "label": f"Calibrado (Δθ={mejor_ang:.0f}°)"},
        ],
        titulo="Trayectoria: con y sin calibrar",
        punto_luna=r_luna_fix,
    )

    graficar_serie_temporal(
        axes[1],
        series=[
            {"t": t_arr, "y": dL_sin / 1e3, "color": SC, "lw": 1.2, "label": "Dist. Luna, sin calibrar"},
            {"t": t_arr, "y": dL_cal / 1e3, "color": OC, "lw": 1.2, "label": "Dist. Luna, calibrado"},
        ],
        titulo="Distancia a la Luna",
        ylabel="distancia [10³ km]",
    )

    fig.tight_layout()
    guardar_figura(fig, f"{NOMBRE_GRAFICO_CALIBRACION}{EXTENSION_GRAFICO}", OUT_DIR)

def punto4_euler_rk2(r0_orion, v0_orion, r_luna_fix, mejor_ang, t_fly):
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

    fig, axes = crear_figura(2, titulo="Punto 4 – Euler vs RK2: Precisión Numérica", ancho=15, alto=6,)

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
    guardar_figura(fig, f"{NOMBRE_GRAFICO_EULER_RK2}{EXTENSION_GRAFICO}", OUT_DIR)
    return r_eu, r_rk, diff4, t4

def apendice_punto4b_convergencia_orion(r0_orion, v0_orion, r_luna_fix, mejor_ang, t_fly):
    v_mag  = np.linalg.norm(v0_orion)
    a_base = np.arctan2(v0_orion[1], v0_orion[0])
    v_opt  = v_mag * np.array([
        np.cos(a_base + np.radians(mejor_ang)),
        np.sin(a_base + np.radians(mejor_ang)),
    ])

    N_ref = 400_000
    h_ref = t_fly / N_ref
    r_ref_arr, _ = integrar_rk2(r0_orion, v_opt, acc_orion, h_ref, N_ref, r_luna=r_luna_fix)
    r_ref = r_ref_arr[-1]

    hs_eu, err_eu = estudio_convergencia_referencia(r0_orion, v_opt, acc_orion, integrar_euler, t_fly, NS_APENDICE_PTO_4, r_ref, r_luna=r_luna_fix)
    hs_rk, err_rk = estudio_convergencia_referencia(r0_orion, v_opt, acc_orion, integrar_rk2, t_fly, NS_APENDICE_PTO_4, r_ref, r_luna=r_luna_fix)

    p_eu = orden_local(hs_eu, err_eu)
    p_rk = orden_local(hs_rk, err_rk)
    orden_eu_tail = ajustar_orden_loglog(hs_eu[-4:], err_eu[-4:])
    orden_rk_tail = ajustar_orden_loglog(hs_rk[-4:], err_rk[-4:])

    filas = []
    for i, N in enumerate(NS_APENDICE_PTO_4):
        p_eu_str = f"{p_eu[i-1]:.3f}" if i > 0 else "—"
        p_rk_str = f"{p_rk[i-1]:.3f}" if i > 0 else "—"
        filas.append([N, f"{hs_eu[i]:.2f}", f"{err_eu[i]:.3e}", p_eu_str,
                      f"{err_rk[i]:.3e}", p_rk_str])
    print(f"\n  Cuadro 4 — Error global al final del vuelo respecto a referencia RK2 (N={N_ref})")
    imprimir_tabla(["N", "h [s]", "Error Euler [km]", "orden p", "Error RK2 [km]", "orden p"], filas)

    print("\n  Cuadro 5 — Orden de convergencia: Orion (tramo más fino) vs. Luna (Punto 1c)")
    imprimir_tabla(["Método", "Orden teórico", "Orden empírico (Orion, asintótico)"], [
        ["Euler explícito", "1", f"{orden_eu_tail:.4f}"],
        ["RK2 (Punto Medio)", "2", f"{orden_rk_tail:.4f}"],
    ])

    fig, ax = crear_figura(1, titulo="Punto 4b – Convergencia: orden de Euler y RK2 (Orion)", ancho=8, alto=6.5)
    ax = ax[0]
    h_refl = np.array([hs_eu.min(), hs_eu.max()])
    ref_h1 = err_eu[-1] * (h_refl / hs_eu[-1]) ** 1
    ref_h2 = err_rk[-1] * (h_refl / hs_rk[-1]) ** 2
    graficar_convergencia(
        ax,
        series=[
            {"h": hs_eu, "err": err_eu, "color": EUC, "marker": "o",
             "label": f"Euler (orden asintótico ≈ {orden_eu_tail:.2f})"},
            {"h": hs_rk, "err": err_rk, "color": RKC, "marker": "s",
             "label": f"RK2 (orden asintótico ≈ {orden_rk_tail:.2f})"},
        ],
        referencias=[
            {"h": h_refl, "y": ref_h1, "ls": "--", "color": RC, "label": "Referencia O(h¹)"},
            {"h": h_refl, "y": ref_h2, "ls": ":",  "color": RC, "label": "Referencia O(h²)"},
        ],
        xlabel="paso de tiempo h [s]",
        ylabel=f"error final vs. referencia RK2 N={N_ref} [km]",
    )
    fig.tight_layout()
    guardar_figura(fig, f"{NOMBRE_GRAFICO_CONV_ORION}{EXTENSION_GRAFICO}", OUT_DIR)

    return hs_eu, err_eu, hs_rk, err_rk, orden_eu_tail, orden_rk_tail

def punto5_largo_plazo():
    T5 = 6.0 * T_LUNAR
    N5 = 60_000
    h5 = T5 / N5
    N_PERIODOS = 6

    r_eu5, v_eu5 = integrar_euler(R0_LUNA, V0_LUNA, acc_luna, h5, N5)
    r_rk5, v_rk5 = integrar_rk2  (R0_LUNA, V0_LUNA, acc_luna, h5, N5)

    d_eu5 = np.linalg.norm(r_eu5, axis=1)
    d_rk5 = np.linalg.norm(r_rk5, axis=1)
    E_eu5 = energia_mecanica(v_eu5, r_eu5)
    E_rk5 = energia_mecanica(v_rk5, r_rk5)
    E0    = E_eu5[0]
    t5    = np.linspace(0, T5, N5 + 1) / 86400

    rango_eu = radio_min_max_por_periodo(r_eu5, N_PERIODOS)
    rango_rk = radio_min_max_por_periodo(r_rk5, N_PERIODOS)
    idx_periodo = indices_por_periodo(N5, N_PERIODOS)
    filas = []
    for k in range(N_PERIODOS):
        idx = idx_periodo[k]
        t_fin = t5[idx]
        dE_eu = (E_eu5[idx] - E0) / abs(E0) * 100
        dE_rk = (E_rk5[idx] - E0) / abs(E0) * 100
        filas.append([
            k + 1, f"{t_fin:.2f}",
            f"{rango_eu[k][0]:.0f}", f"{rango_eu[k][1]:.0f}", f"{dE_eu:.4f}",
            f"{rango_rk[k][0]:.0f}", f"{rango_rk[k][1]:.0f}", f"{dE_rk:.6f}",
        ])
    print(f"\n  Cuadro 6 — Evolución periodo a periodo de la órbita lunar (Euler vs RK2, {N_PERIODOS} períodos)")
    imprimir_tabla(
        ["periodo", "t_fin [d]", "r_min Euler", "r_max Euler", "dE/E0 Euler [%]",
         "r_min RK2", "r_max RK2", "dE/E0 RK2 [%]"],
        filas,
    )

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
        loc_leyenda="upper right"
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
    guardar_figura(fig, F"{NOMBRE_GRAFICO_CONV_LARGO_PLAZO}{EXTENSION_GRAFICO}", OUT_DIR)
    return r_eu5, v_eu5, r_rk5, v_rk5, E0, t5, N5, h5

def apendice_punto5b_convergencia_largo_plazo():
    T6 = 6.0 * T_LUNAR

    hs_eu, err_eu = estudio_convergencia_periodico(R0_LUNA, V0_LUNA, acc_luna, integrar_euler, T6, NS_APENDICE_PTO_5_6)
    hs_rk, err_rk = estudio_convergencia_periodico(R0_LUNA, V0_LUNA, acc_luna, integrar_rk2,   T6, NS_APENDICE_PTO_5_6)

    p_eu = orden_local(hs_eu, err_eu)
    p_rk = orden_local(hs_rk, err_rk)

    filas = []
    for i, N in enumerate(NS_APENDICE_PTO_5_6):
        p_eu_str = f"{p_eu[i-1]:.3f}" if i > 0 else "—"
        p_rk_str = f"{p_rk[i-1]:.3f}" if i > 0 else "—"
        filas.append([N, f"{hs_eu[i]:.2f}", f"{err_eu[i]:.3e}", p_eu_str,
                      f"{err_rk[i]:.3e}", p_rk_str])
    print(f"\n  Cuadro 7 — Convergencia de Euler y RK2 integrando 6 períodos lunares (T6 ≈ {T6/86400:.2f} días)")
    imprimir_tabla(["N", "h [s]", "e_Euler [km]", "orden p", "e_RK2 [km]", "orden p"], filas)

    fig, ax = crear_figura(1, titulo="Convergencia a largo plazo (6 períodos lunares, ~5.5 meses)", ancho=8, alto=6.5)
    ax = ax[0]
    h_refl = np.array([hs_rk.min(), hs_rk.max()])
    ref_h2 = err_rk[-1] * (h_refl / hs_rk[-1]) ** 2
    graficar_convergencia(
        ax,
        series=[
            {"h": hs_eu, "err": err_eu, "color": EUC, "marker": "o", "label": "Euler (simulado)"},
            {"h": hs_rk, "err": err_rk, "color": RKC, "marker": "s", "label": "RK2 (simulado)"},
        ],
        referencias=[
            {"h": h_refl, "y": ref_h2, "ls": "--", "color": RC, "label": "Referencia O(h²)"},
        ],
        xlabel="paso h [s]",
        ylabel="error ||r_N − r0|| tras 6 períodos [km]",
    )
    fig.tight_layout()
    guardar_figura(fig, "Punto5b_Convergencia_LargoPlazo.png", OUT_DIR)

    return hs_eu, err_eu, hs_rk, err_rk

def punto6_nystrom(r_eu5, v_eu5, r_rk5, v_rk5, E0, t5, N5, h5):
    N_PERIODOS = 6
    r_ny5, v_ny5 = integrar_nystrom(R0_LUNA, V0_LUNA, acc_luna, h5, N5)

    E_eu5 = energia_mecanica(v_eu5, r_eu5)
    E_rk5 = energia_mecanica(v_rk5, r_rk5)
    E_ny5 = energia_mecanica(v_ny5, r_ny5)
    L_eu  = momento_angular(r_eu5, v_eu5)
    L_rk  = momento_angular(r_rk5, v_rk5)
    L_ny  = momento_angular(r_ny5, v_ny5)
    L0    = L_eu[0]

    rango_eu = radio_min_max_por_periodo(r_eu5, N_PERIODOS)
    rango_rk = radio_min_max_por_periodo(r_rk5, N_PERIODOS)
    rango_ny = radio_min_max_por_periodo(r_ny5, N_PERIODOS)
    idx_periodo = indices_por_periodo(N5, N_PERIODOS)
    t5_dias = t5
    filas = []
    for k in range(N_PERIODOS):
        idx = idx_periodo[k]
        filas.append([
            k + 1, f"{t5_dias[idx]:.2f}",
            f"{rango_eu[k][0]:.0f}", f"{rango_eu[k][1]:.0f}",
            f"{rango_rk[k][0]:.0f}", f"{rango_rk[k][1]:.0f}",
            f"{rango_ny[k][0]:.0f}", f"{rango_ny[k][1]:.0f}",
        ])
    print(f"\n  Cuadro 8 — Radio orbital mínimo y máximo por periodo (Euler, RK2 y Nyström)")
    imprimir_tabla(
        ["periodo", "t_fin [d]", "r_min Eu", "r_max Eu", "r_min RK2", "r_max RK2",
         "r_min Ny", "r_max Ny"],
        filas,
    )

    dE_eu = (E_eu5[-1] - E0) / abs(E0) * 100
    dE_rk = (E_rk5[-1] - E0) / abs(E0) * 100
    dE_ny = (E_ny5[-1] - E0) / abs(E0) * 100
    dL_eu = (L_eu[-1] - L0) / abs(L0) * 100
    dL_rk = (L_rk[-1] - L0) / abs(L0) * 100
    dL_ny = (L_ny[-1] - L0) / abs(L0) * 100
    print(f"\n  Cuadro 9 — Deriva relativa de energía y momento angular (precisión extendida, 6 períodos)")
    imprimir_tabla(["Método", "ΔE/E0 [%]", "ΔL/L0 [%]"], [
        ["Euler",   f"{dE_eu:.6e}", f"{dL_eu:.6e}"],
        ["RK2",     f"{dE_rk:.6e}", f"{dL_rk:.6e}"],
        ["Nystrom", f"{dE_ny:.6e}", f"{dL_ny:.6e}"],
    ])

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
        loc_leyenda="upper right"
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
    guardar_figura(fig, f"{NOMBRE_GRAFICO_NYSTROM}{EXTENSION_GRAFICO}", OUT_DIR)
    return r_ny5, v_ny5, E_ny5

def apendice_punto6b_convergencia_nystrom(hs_eu, err_eu, hs_rk, err_rk):
    T6 = 6.0 * T_LUNAR

    hs_ny, err_ny = estudio_convergencia_periodico(R0_LUNA, V0_LUNA, acc_luna, integrar_nystrom, T6, NS_APENDICE_PTO_5_6)
    p_ny = orden_local(hs_ny, err_ny)

    filas = []
    for i, N in enumerate(NS_APENDICE_PTO_5_6):
        p_str = f"{p_ny[i-1]:.3f}" if i > 0 else "—"
        filas.append([N, f"{hs_ny[i]:.2f}", f"{err_ny[i]:.3e}", p_str])
    print(f"\n  Cuadro 10 — Convergencia de Nyström integrando 6 períodos lunares (T6 ≈ {T6/86400:.2f} días)")
    imprimir_tabla(["N", "h [s]", "e_Nystrom [km]", "orden p"], filas)

    fig, ax = crear_figura(1, titulo="Convergencia a largo plazo: Euler, RK2 y Nyström\n(6 períodos, ~5.5 meses)", ancho=ANCHO_FIG, alto=ALTO_FIG)
    ax = ax[0]
    h_refl = np.array([hs_rk.min(), hs_rk.max()])
    ref_h2 = err_rk[-1] * (h_refl / hs_rk[-1]) ** 2
    graficar_convergencia(
        ax,
        series=[
            {"h": hs_eu, "err": err_eu, "color": EUC, "marker": "o", "label": "Euler"},
            {"h": hs_rk, "err": err_rk, "color": RKC, "marker": "s", "label": "RK2"},
            {"h": hs_ny, "err": err_ny, "color": NYC, "marker": "^", "label": "Nyström/Stoermer"},
        ],
        referencias=[
            {"h": h_refl, "y": ref_h2, "ls": "--", "color": RC, "label": "Referencia O(h²)"},
        ],
        xlabel="paso h [s]",
        ylabel="error ||r_N − r0|| tras 6 períodos [km]",
    )
    fig.tight_layout()
    guardar_figura(fig, f"{NOMBRE_GRAFICO_CONV_NYSTROM}{EXTENSION_GRAFICO}", OUT_DIR)
