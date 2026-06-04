import numpy as np
from pathlib import Path
from utils.utils import acc_luna, leer_condiciones_iniciales, integrar_rk2, energia_mecanica
from graph.graph import configurar_estilo
from calculos.calculos import calculo_orbita_lunar, calculo_posicion_velocidad_orion, calculo_euler_rk2, simulacion_orbita_lunar, metodo_alternativo_nynstrom

G = 6.674e-20
M_T = 5.972e24
M_L = 7.348e22
GM_T = G * M_T
GM_L = G * M_L

R_PERIGEO = 362_600.0
R_APOGEO = 405_400.0
A_LUNA = (R_PERIGEO + R_APOGEO) / 2.0
V_PERI = np.sqrt(GM_T * (2.0 / R_PERIGEO - 1.0 / A_LUNA))

R0_LUNA = np.array([R_PERIGEO, 0.0])
V0_LUNA = np.array([0.0, V_PERI])

CSV_PATH = Path("./csv/Artemis_II_Data.csv")

def main():
    configurar_estilo()

    print("Leyendo telemetría del CSV...")
    r0_orion, v0_orion = leer_condiciones_iniciales(CSV_PATH)
    print(f"r₀ = {r0_orion}  km")
    print(f"v₀ = {v0_orion}  km/s")
    print(f"|r₀| = {np.linalg.norm(r0_orion):.0f} km desde la Tierra")
    print("=" * 58)

    rl, dl, pi_, ai_ = calculo_orbita_lunar()

    r_orion, r_luna_fix, mejor_ang, T_FLY = calculo_posicion_velocidad_orion(r0_orion, v0_orion)

    N_l2 = 20_000
    h_l2 = T_FLY / N_l2
    rl2, _ = integrar_rk2(R0_LUNA, V0_LUNA, acc_luna, h_l2, N_l2)

    r_eu, r_rk, diff4, t4 = calculo_euler_rk2(r0_orion, v0_orion, r_luna_fix, mejor_ang, T_FLY)

    r_eu5, v_eu5, r_rk5, v_rk5, E0, t5, N5, h5 = simulacion_orbita_lunar()

    r_ny5, E_ny5 = metodo_alternativo_nynstrom(r_eu5, v_eu5, r_rk5, v_rk5, E0, t5, N5, h5)

    E_eu5 = energia_mecanica(v_eu5, r_eu5)
    E_rk5 = energia_mecanica(v_rk5, r_rk5)

    print("Calculos finalizados!")

if __name__ == "__main__":
    main()
