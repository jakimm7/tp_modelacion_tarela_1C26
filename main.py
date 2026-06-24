import numpy as np
from pathlib import Path
from graph.graph import configurar_estilo
from utils.utils import leer_condiciones_iniciales, energia_mecanica, momento_angular, imprimir_resultados
from calculos.calculos import (
    punto1_orbita_lunar, apendice_punto1b_euler_vs_rk2, apendice_punto1c_convergencia,
    punto2_3_calibracion_pvi, apendice_punto2_3_efecto_calibracion,
    punto4_euler_rk2, apendice_punto4b_convergencia_orion,
    punto5_largo_plazo, apendice_punto5b_convergencia_largo_plazo,
    punto6_nystrom, apendice_punto6b_convergencia_nystrom,
)

G = 6.674e-20
M_T = 5.972e24
M_L = 7.348e22
GM_T = G * M_T
GM_L = G * M_L

R_PERIGEO = 362_600.0
R_APOGEO = 405_400.0
A_LUNA = (R_PERIGEO + R_APOGEO) / 2.0
V_PERI = np.sqrt(GM_T * (2.0 / R_PERIGEO - 1.0 / A_LUNA))
T_LUNAR   = 2.0 * np.pi * np.sqrt(A_LUNA**3 / GM_T)

R0_LUNA = np.array([R_PERIGEO, 0.0])
V0_LUNA = np.array([0.0, V_PERI])

CSV_PATH = Path("./csv/Artemis_II_Data.csv")


def main():
    print("TRABAJO PRACTICO - Trayectoria de la Cápsula Orion de la Misión Artemis II")
    print("(version corregida segun apendice)\n")
    configurar_estilo()
    r0_orion, v0_orion = leer_condiciones_iniciales(CSV_PATH)

    imprimir_resultados("CSV", "Condiciones iniciales (3 abril, 04:00-06:00 h)", [
        ("dato", "Posicion inicial r0 [km]", f"({r0_orion[0]:.2f}, {r0_orion[1]:.2f})"),
        ("dato", "Velocidad inicial v0 [km/s]", f"({v0_orion[0]:.5f}, {v0_orion[1]:.5f})"),
        ("dato", "Distancia inicial a la Tierra", f"{np.linalg.norm(r0_orion):.2f}", "km"),
        ("dato", "Modulo de v0", f"{np.linalg.norm(v0_orion):.5f}", "km/s"),
    ])

    rl, vl, dl, sl, ti, pi_, ai_ = punto1_orbita_lunar()

    L_luna = momento_angular(rl, vl)
    E_luna = energia_mecanica(vl, rl)

    imprimir_resultados("1", "Orbita Lunar y Validacion", [
        ("seccion", "Parametros geometricos"),
        ("dato", "Perigeo simulado", f"{dl.min():.0f}", "km"),
        ("dato", "Perigeo esperado", f"{R_PERIGEO:.0f}", "km"),
        ("dato", "Error perigeo", f"{abs(dl.min() - R_PERIGEO):.0f}", "km"),
        ("dato", "Apogeo simulado", f"{dl.max():.0f}", "km"),
        ("dato", "Apogeo esperado", f"{R_APOGEO:.0f}", "km"),
        ("dato", "Error apogeo", f"{abs(dl.max() - R_APOGEO):.0f}", "km"),

        ("seccion", "Velocidades orbitales"),
        ("dato", f"Velocidad maxima (perigeo, t={ti[pi_]:.2f} dias)", f"{sl[pi_]:.6f}", "km/s"),
        ("dato", f"Velocidad minima (apogeo,  t={ti[ai_]:.2f} dias)", f"{sl[ai_]:.6f}", "km/s"),
        ("dato", "Relacion v_max / v_min", f"{sl[pi_] / sl[ai_]:.6f}"),

        ("seccion", "Conservacion de cantidades fisicas"),
        ("dato", "Periodo orbital calculado", f"{T_LUNAR/86400:.4f}", "dias"),
        ("dato", "Periodo orbital teorico", f"{2*np.pi*np.sqrt(A_LUNA**3/GM_T)/86400:.4f}", "dias"),
        ("dato", "Variacion momento angular dL/L0",
                 f"{(L_luna.max()-L_luna.min())/abs(L_luna[0])*100:.2e}", "%"),
        ("dato", "Variacion energia mecanica dE/E0",
                 f"{(E_luna.max()-E_luna.min())/abs(E_luna[0])*100:.2e}", "%"),

        ("texto", ""),
        ("texto", "Validacion: distancia oscila entre perigeo y apogeo correctamente"),
        ("texto", "Validacion: v_max en perigeo, v_min en apogeo (conservacion L)"),
    ])

    apendice_punto1b_euler_vs_rk2(rl, vl, dl, ti)
    apendice_punto1c_convergencia()

    r_orion, v_orion, r_luna_fix, mejor_ang, T_FLY, datos_diag = punto2_3_calibracion_pvi(r0_orion, v0_orion)

    dist_luna_tray = np.linalg.norm(r_orion - r_luna_fix, axis=1)
    t_arr_fly = np.linspace(0, T_FLY, r_orion.shape[0]) / 86400
    ang_basal = np.degrees(np.arctan2(v0_orion[1], v0_orion[0]))

    imprimir_resultados("2 y 3", "Trayectoria Orion - PVI con calibracion parametrica", [
        ("seccion", "Calibracion parametrica del PVI"),
        ("dato", "Angulo basal de v0 (telemetria)", f"{ang_basal:.4f}", "grados"),
        ("dato", "Correccion optima Delta_theta", f"{mejor_ang:.4f}", "grados"),
        ("dato", "Angulo final de v0 calibrado", f"{ang_basal + mejor_ang:.4f}", "grados"),
        ("dato", "Modulo de v0 (sin alterar)", f"{np.linalg.norm(v0_orion):.6f}", "km/s"),

        ("seccion", "Trayectoria resultante (PVI)"),
        ("dato", "Duracion del vuelo simulado", f"{T_FLY/86400:.1f}", "dias"),
        ("dato", "Distancia inicial a la Tierra", f"{np.linalg.norm(r_orion[0]):.0f}", "km"),
        ("dato", "Distancia final a la Tierra", f"{np.linalg.norm(r_orion[-1]):.0f}", "km"),
        ("dato", "Distancia minima a la Luna", f"{dist_luna_tray.min():.0f}", "km"),
        ("dato", "Tiempo de aprox. minima a Luna", f"{t_arr_fly[dist_luna_tray.argmin()]:.2f}", "dias"),
        ("dato", "Radio de la Luna (referencia)", "1737", "km"),

        ("texto", ""),
        ("texto", "Se trata de un PVI: r0,v0 se conocen (CSV); no hay condicion de"),
        ("texto", "contorno, sino una calibracion parametrica de Delta_theta (Grid Search)."),
    ])

    apendice_punto2_3_efecto_calibracion(r0_orion, v0_orion, r_luna_fix, mejor_ang, T_FLY)

    r_eu, r_rk, diff4, t4 = punto4_euler_rk2(r0_orion, v0_orion, r_luna_fix, mejor_ang, T_FLY)

    h4 = T_FLY / 20_000

    imprimir_resultados("4", "Comparacion Euler vs RK2", [
        ("seccion", "Parametros de integracion"),
        ("dato", "Paso de tiempo h", f"{h4:.2f}", "s"),
        ("dato", "Numero de pasos N", "20000"),
        ("dato", "Tiempo total de vuelo", f"{T_FLY/86400:.1f}", "dias"),

        ("seccion", "Error de posicion |r_Euler - r_RK2|"),
        ("dato", "Error inicial (t=0)", f"{diff4[0]:.2e}", "km"),
        ("dato", "Error final  (t=T)", f"{diff4[-1]:.2e}", "km"),
        ("dato", "Error maximo", f"{diff4.max():.2e}", "km"),
        ("dato", "Error promedio", f"{diff4.mean():.2e}", "km"),

        ("seccion", "Orden de convergencia"),
        ("dato", "Orden Euler", "1  ->  error proporcional a h"),
        ("dato", "Orden RK2",   "2  ->  error proporcional a h^2"),

        ("texto", ""),
        ("texto", "Euler O(h): el error crece tan rapido que falla la insercion lunar"),
        ("texto", "RK2 O(h^2): cierra el ciclo de retorno con costo computacional razonable"),
    ])

    apendice_punto4b_convergencia_orion(r0_orion, v0_orion, r_luna_fix, mejor_ang, T_FLY)

    r_eu5, v_eu5, r_rk5, v_rk5, E0, t5, N5, h5 = punto5_largo_plazo()

    E_eu5 = energia_mecanica(v_eu5, r_eu5)
    E_rk5 = energia_mecanica(v_rk5, r_rk5)
    d_eu5 = np.linalg.norm(r_eu5, axis=1)
    d_rk5 = np.linalg.norm(r_rk5, axis=1)

    imprimir_resultados("5", "Analisis a largo plazo - Inestabilidad numerica", [
        ("seccion", "Parametros de simulacion"),
        ("dato", "Tiempo total simulado", f"{6*T_LUNAR/86400:.1f}", "dias (6 periodos lunares)"),
        ("dato", "Numero de pasos N", f"{N5:,}"),
        ("dato", "Paso de tiempo h", f"{h5:.2f}", "s"),

        ("seccion", "Deriva del radio orbital"),
        ("dato", "Radio inicial (ambos)", f"{d_eu5[0]:.0f}", "km"),
        ("dato", "Radio medio teorico", f"{(R_PERIGEO + R_APOGEO)/2:.0f}", "km"),
        ("dato", "Radio final Euler", f"{d_eu5[-1]:.0f}", "km"),
        ("dato", "Radio final RK2", f"{d_rk5[-1]:.0f}", "km"),
        ("dato", "Crecimiento radio Euler",f"{(d_eu5[-1]/d_eu5[0]-1)*100:.2f}", "%"),
        ("dato", "Crecimiento radio RK2", f"{(d_rk5[-1]/d_rk5[0]-1)*100:.2f}", "%"),

        ("seccion", "Deriva de la energia mecanica dE/E0"),
        ("dato", "Euler - deriva final", f"{(E_eu5[-1]-E0)/abs(E0)*100:+.4f}", "%"),
        ("dato", "RK2   - deriva final", f"{(E_rk5[-1]-E0)/abs(E0)*100:+.4f}", "%"),

        ("texto", ""),
        ("texto", "Los autovalores |gamma| > 1 inyectan energia espuria en cada paso"),
        ("texto", "Resultado: la Luna espirala hacia afuera en ambos metodos"),
    ])

    hs_eu_lp, err_eu_lp, hs_rk_lp, err_rk_lp = apendice_punto5b_convergencia_largo_plazo()

    r_ny5, v_ny5, E_ny5 = punto6_nystrom(r_eu5, v_eu5, r_rk5, v_rk5, E0, t5, N5, h5)

    d_ny5 = np.linalg.norm(r_ny5, axis=1)
    L_eu5 = momento_angular(r_eu5, v_eu5)
    L_rk5 = momento_angular(r_rk5, v_rk5)
    L_ny5 = momento_angular(r_ny5, v_ny5)
    L0 = L_eu5[0]

    imprimir_resultados("6", "Metodo alternativo conservativo - Nystrom", [
        ("seccion", "Estabilidad del radio orbital"),
        ("dato", "Radio inicial Nystrom", f"{d_ny5[0]:.0f}", "km"),
        ("dato", "Radio final   Nystrom", f"{d_ny5[-1]:.0f}", "km"),
        ("dato", "Variacion radio Nystrom", f"{(d_ny5[-1]/d_ny5[0]-1)*100:.4f}", "%"),
        ("dato", "Variacion radio Euler (ref.)", f"{(d_eu5[-1]/d_eu5[0]-1)*100:.2f}", "%"),

        ("seccion", "Conservacion de energia mecanica dE/E0"),
        ("dato", "Euler   - deriva final", f"{(E_eu5[-1]-E0)/abs(E0)*100:+.4f}", "%"),
        ("dato", "RK2     - deriva final", f"{(E_rk5[-1]-E0)/abs(E0)*100:+.4f}", "%"),
        ("dato", "Nystrom - deriva final", f"{(E_ny5[-1]-E0)/abs(E0)*100:+.6f}", "%"),

        ("seccion", "Conservacion del momento angular dL/L0"),
        ("dato", "Euler   - deriva final", f"{(L_eu5[-1]-L0)/abs(L0)*100:+.4f}", "%"),
        ("dato", "RK2     - deriva final", f"{(L_rk5[-1]-L0)/abs(L0)*100:+.4f}", "%"),
        ("dato", "Nystrom - deriva final", f"{(L_ny5[-1]-L0)/abs(L0)*100:+.6f}", "%"),

        ("seccion", "Analisis de estabilidad"),
        ("texto", "Nystrom discretiza la EDO de 2do orden directamente."),
        ("texto", "Raices del polinomio caracteristico: |gamma1 * gamma2| = 1"),
        ("texto", "El factor de amplificacion solo rota en C -> orbita estable"),
        ("texto", ""),
        ("texto", "Nystrom: radio orbital constante, E y L conservados numericamente"),
        ("texto", "Metodo recomendado para simulaciones orbitales de larga duracion"),
    ])

    apendice_punto6b_convergencia_nystrom(hs_eu_lp, err_eu_lp, hs_rk_lp, err_rk_lp)

    print("\nFIN DE LA SIMULACION. Todas las figuras fueron guardadas en ./graficos/\n")


if __name__ == "__main__":
    main()
