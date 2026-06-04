"""
Trabajo Práctico: Trayectoria de la Cápsula Orion (Misión Artemis II)
Modelación Numérica — 1er Cuatrimestre 2026
=============================================================
Bibliotecas: numpy, matplotlib, csv (stdlib)
Estructura:
  - Constantes físicas (módulo de configuración)
  - Funciones de aceleración
  - Lector de CSV
  - Integradores numéricos (Euler, RK2, Nyström)
  - Utilidades de graficado
  - punto_1() ... punto_6() + figura_resumen()
  - main()  →  orquesta todo el TP
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import csv
from pathlib import Path


# ══════════════════════════════════════════════════════════════
# SECCIÓN 1: CONFIGURACIÓN — constantes físicas y orbitales
# ══════════════════════════════════════════════════════════════

# Constantes gravitacionales
G    = 6.674e-20        # km³ / (kg·s²)
M_T  = 5.972e24         # kg  – Tierra
M_L  = 7.348e22         # kg  – Luna
GM_T = G * M_T
GM_L = G * M_L

# Parámetros orbitales lunares
R_LUNA    = 384_400.0   # km  – distancia media Tierra-Luna
R_PERIGEO = 362_600.0   # km
R_APOGEO  = 405_400.0   # km
A_LUNA    = (R_PERIGEO + R_APOGEO) / 2.0
V_PERI    = np.sqrt(GM_T * (2.0 / R_PERIGEO - 1.0 / A_LUNA))
T_LUNAR   = 2.0 * np.pi * np.sqrt(A_LUNA**3 / GM_T)  # período orbital ≈ 27.4 días

# Condiciones iniciales de la Luna (en perigeo, plano XY)
R0_LUNA = np.array([R_PERIGEO, 0.0])
V0_LUNA = np.array([0.0, V_PERI])

# Rutas de archivos
CSV_PATH = Path("/mnt/user-data/uploads/Artemis_II_Data.csv")
OUT_DIR  = Path("/mnt/user-data/outputs")

# Paleta de colores (tema espacial oscuro)
BG  = "#0a0e1a"   # fondo
GC  = "#1e2a40"   # grilla
EC  = "#2979ff"   # Tierra
MC  = "#b0bec5"   # Luna
OC  = "#ff6d00"   # Orion
EUC = "#ef5350"   # Euler
RKC = "#66bb6a"   # RK2
NYC = "#29b6f6"   # Nyström
TC  = "#eceff1"   # texto


# ══════════════════════════════════════════════════════════════
# SECCIÓN 2: FUNCIONES DE ACELERACIÓN GRAVITATORIA
# ══════════════════════════════════════════════════════════════

def acc_luna(r):
    """
    Aceleración de la Luna debida a la Tierra.
    Ecuación (1) del TP: d²r/dt² = -GM_T * r / |r|³
    """
    d = np.linalg.norm(r)
    return -GM_T * r / d**3


def acc_orion(r, r_luna):
    """
    Aceleración de la cápsula Orion en el Problema Restringido de 3 Cuerpos.
    Ecuación (2) del TP: atracción simultánea de Tierra y Luna.
    """
    dT = np.linalg.norm(r)
    aT = -GM_T * r / dT**3

    delta = r - r_luna
    dL = np.linalg.norm(delta)
    aL = -GM_L * delta / dL**3

    return aT + aL


# ══════════════════════════════════════════════════════════════
# SECCIÓN 3: LECTURA DE TELEMETRÍA (CSV)
# ══════════════════════════════════════════════════════════════

def parsear_float_europeo(s):
    """Convierte un float con coma decimal al formato Python (punto decimal)."""
    return float(s.replace(",", "."))


def leer_condiciones_iniciales(csv_path, fecha="2026-04-03", hora_min=4, hora_max=6):
    """
    Lee el CSV de telemetría y devuelve (r0, v0) en 2D [km, km/s]
    del primer registro que coincida con la fecha y franja horaria dada.

    Formato de columnas: timestamp ; x ; y ; z ; vx ; vy ; vz
    """
    with open(csv_path, "r") as f:
        for linea in f:
            linea = linea.strip()
            if not linea:
                continue
            cols = linea.split(";")
            ts   = cols[0]
            if fecha not in ts:
                continue
            hora = int(ts.split("T")[1].split(":")[0])
            if hora_min <= hora < hora_max:
                x,  y  = parsear_float_europeo(cols[1]), parsear_float_europeo(cols[2])
                vx, vy = parsear_float_europeo(cols[4]), parsear_float_europeo(cols[5])
                return np.array([x, y]), np.array([vx, vy])

    raise RuntimeError(
        f"No se encontraron datos para {fecha} entre {hora_min}:00 y {hora_max}:00 en el CSV."
    )


# ══════════════════════════════════════════════════════════════
# SECCIÓN 4: INTEGRADORES NUMÉRICOS
# ══════════════════════════════════════════════════════════════

def integrar_euler(r0, v0, acc_fn, h, N, **kw):
    """
    Método de Euler explícito (orden 1, O(h)).
    Simple pero acumula error lineal → inestable a largo plazo.
    """
    rs = np.empty((N + 1, 2))
    vs = np.empty((N + 1, 2))
    rs[0] = r0
    vs[0] = v0
    for i in range(N):
        a        = acc_fn(rs[i], **kw)
        vs[i+1]  = vs[i] + h * a
        rs[i+1]  = rs[i] + h * vs[i]
    return rs, vs


def integrar_rk2(r0, v0, acc_fn, h, N, **kw):
    """
    Runge-Kutta de orden 2 (método del punto medio, O(h²)).
    Evalúa la pendiente en el punto intermedio del paso.
    """
    rs = np.empty((N + 1, 2))
    vs = np.empty((N + 1, 2))
    rs[0] = r0
    vs[0] = v0
    for i in range(N):
        a1       = acc_fn(rs[i], **kw)
        r_mid    = rs[i] + 0.5 * h * vs[i]
        v_mid    = vs[i] + 0.5 * h * a1
        a2       = acc_fn(r_mid, **kw)
        rs[i+1]  = rs[i] + h * v_mid
        vs[i+1]  = vs[i] + h * a2
    return rs, vs


def integrar_nystrom(r0, v0, acc_fn, h, N, **kw):
    """
    Método de Nyström (simpléctico, O(h²)).
    Discretiza la EDO de 2do orden directamente — Ecuación (3) del TP:
        r_{n+1} = 2*r_n - r_{n-1} + h²*f(t_n, r_n)
    En formulación velocidad-posición su factor de amplificación cumple
    |γ₁·γ₂| = 1, garantizando estabilidad conservativa a largo plazo.
    """
    rs = np.empty((N + 1, 2))
    vs = np.empty((N + 1, 2))
    rs[0] = r0
    vs[0] = v0
    a = acc_fn(rs[0], **kw)
    for i in range(N):
        rs[i+1]  = rs[i] + h * vs[i] + 0.5 * h * h * a
        a_next   = acc_fn(rs[i+1], **kw)
        vs[i+1]  = vs[i] + 0.5 * h * (a + a_next)
        a        = a_next
    return rs, vs


# ══════════════════════════════════════════════════════════════
# SECCIÓN 5: UTILIDADES DE GRAFICADO
# ══════════════════════════════════════════════════════════════

def configurar_estilo():
    """Aplica el tema oscuro espacial a todos los gráficos de matplotlib."""
    plt.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    BG,
        "axes.edgecolor":    GC,
        "axes.labelcolor":   TC,
        "xtick.color":       TC,
        "ytick.color":       TC,
        "text.color":        TC,
        "grid.color":        GC,
        "grid.linestyle":    "--",
        "grid.alpha":        0.5,
        "legend.facecolor":  "#111827",
        "legend.edgecolor":  GC,
        "font.family":       "monospace",
    })


def guardar_figura(fig, nombre_archivo):
    """Guarda la figura en el directorio de salida y la cierra."""
    ruta = OUT_DIR / nombre_archivo
    fig.savefig(ruta, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✓ Guardado: {ruta}")


def caja_texto(ax, texto):
    """Agrega un cuadro de anotación en la esquina superior izquierda del eje."""
    ax.text(
        0.02, 0.97, texto,
        transform=ax.transAxes, va="top", fontsize=8,
        bbox=dict(facecolor="#111827", edgecolor=GC, alpha=0.85),
    )


def energia_mecanica(vs, rs):
    """Calcula la energía mecánica específica E = v²/2 - GM_T/|r| para cada paso."""
    d = np.linalg.norm(rs, axis=1)
    return 0.5 * np.sum(vs**2, axis=1) - GM_T / d


def momento_angular(rs, vs):
    """Calcula la componente z del momento angular L = r × v para cada paso."""
    return rs[:, 0] * vs[:, 1] - rs[:, 1] * vs[:, 0]


# ══════════════════════════════════════════════════════════════
# SECCIÓN 6: FUNCIONES POR PUNTO DEL TP
# ══════════════════════════════════════════════════════════════

def punto_1():
    """
    Punto 1 — Órbita Lunar y Validación.
    Integra la trayectoria de la Luna durante un período completo con RK2
    y valida perigeo/apogeo y la relación velocidad-distancia.
    Retorna (rl, vl, dl, sl, ti, pi_, ai_) para reutilizar en el resumen.
    """
    print("\n[1] Órbita lunar...")

    N  = 40_000
    h  = T_LUNAR / N
    rl, vl = integrar_rk2(R0_LUNA, V0_LUNA, acc_luna, h, N)

    dl  = np.linalg.norm(rl, axis=1)
    sl  = np.linalg.norm(vl, axis=1)
    ti  = np.linspace(0, T_LUNAR, N + 1) / 86400  # días
    pi_ = dl.argmin()
    ai_ = dl.argmax()

    print(f"  Perigeo simulado : {dl.min():.0f} km   (esperado ~{R_PERIGEO:.0f} km)")
    print(f"  Apogeo  simulado : {dl.max():.0f} km   (esperado ~{R_APOGEO:.0f} km)")
    print(f"  Vel. máx (perigeo): {sl[pi_]:.4f} km/s")
    print(f"  Vel. mín (apogeo) : {sl[ai_]:.4f} km/s")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Punto 1 – Órbita Lunar: Validación", fontsize=13, color=TC)

    # Panel izquierdo: trayectoria en el plano XY
    ax = axes[0]
    ax.plot(rl[:, 0], rl[:, 1], color=MC, lw=1.2)
    ax.scatter([0], [0], s=200, color=EC, zorder=5, label="Tierra")
    ax.scatter(rl[pi_, 0], rl[pi_, 1], color="#ff1744", s=60, zorder=6,
               label=f"Perigeo {dl.min():.0f} km")
    ax.scatter(rl[ai_, 0], rl[ai_, 1], color="#69f0ae", s=60, zorder=6,
               label=f"Apogeo {dl.max():.0f} km")
    ax.set_aspect("equal"); ax.grid(True); ax.legend(fontsize=8)
    ax.set_title("Trayectoria (1 período)")
    ax.set_xlabel("x [km]"); ax.set_ylabel("y [km]")

    # Panel central: distancia vs tiempo
    ax = axes[1]
    ax.plot(ti, dl, color=MC, lw=1.2)
    ax.axhline(R_PERIGEO, ls="--", color="#ff1744", lw=0.8, label=f"{R_PERIGEO:.0f} km")
    ax.axhline(R_APOGEO,  ls="--", color="#69f0ae", lw=0.8, label=f"{R_APOGEO:.0f} km")
    ax.set_title("Distancia Tierra-Luna")
    ax.set_xlabel("tiempo [días]"); ax.set_ylabel("distancia [km]")
    ax.legend(fontsize=8); ax.grid(True)

    # Panel derecho: velocidad vs tiempo
    ax = axes[2]
    ax.plot(ti, sl, color=EC, lw=1.2)
    ax.annotate("Vel. máx\n(perigeo)",
                xy=(ti[pi_], sl[pi_]),
                xytext=(ti[pi_] + 1.0, sl[pi_] + 0.003),
                arrowprops=dict(arrowstyle="->", color=TC), fontsize=8)
    ax.annotate("Vel. mín\n(apogeo)",
                xy=(ti[ai_], sl[ai_]),
                xytext=(ti[ai_] + 1.0, sl[ai_] - 0.005),
                arrowprops=dict(arrowstyle="->", color=TC), fontsize=8)
    ax.set_title("Velocidad orbital lunar")
    ax.set_xlabel("tiempo [días]"); ax.set_ylabel("|v| [km/s]")
    ax.grid(True)

    fig.tight_layout()
    guardar_figura(fig, "Punto1_Orbita_Lunar.png")

    return rl, vl, dl, sl, ti, pi_, ai_


def punto_2_3(r0_orion, v0_orion):
    """
    Puntos 2 y 3 — Trayectoria de la Cápsula Orion (Free-Return).
    Resuelve el Problema Restringido de 3 Cuerpos con RK2 y aplica
    el Shooting Method para encontrar el ángulo que cierra el "ocho".
    Retorna (r_orion, v_orion, r_luna_fix, best_ang, T_FLY).
    """
    print("\n[2/3] Trayectoria Free-Return de Orion...")

    T_FLY = 9.0 * 86400   # 9 días de vuelo en segundos
    N     = 30_000
    h     = T_FLY / N

    # Posición de la Luna en t₀: opuesta al punto de partida de Orion
    r_luna_fix = np.array([
        R_LUNA * np.cos(np.radians(180)),
        R_LUNA * np.sin(np.radians(180)),
    ])

    v_mag  = np.linalg.norm(v0_orion)
    a_base = np.arctan2(v0_orion[1], v0_orion[0])

    # ── Shooting Method ──────────────────────────────────────
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
    print(f"  Ángulo óptimo    : {mejor_ang:.2f}°")
    print(f"  Dist. mín a Luna : {np.linalg.norm(r_orion - r_luna_fix, axis=1).min():.0f} km")
    print(f"  Dist. final Tierra: {np.linalg.norm(r_orion[-1]):.0f} km")

    # Órbita lunar durante el vuelo (para superposición)
    N_l2   = 20_000
    h_l2   = T_FLY / N_l2
    rl2, _ = integrar_rk2(R0_LUNA, V0_LUNA, acc_luna, h_l2, N_l2)

    t_arr = np.linspace(0, T_FLY, N + 1) / 86400
    d_T   = np.linalg.norm(r_orion, axis=1)
    d_L   = np.linalg.norm(r_orion - r_luna_fix, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle(
        "Puntos 2 y 3 – Trayectoria Orion (Problema 3 Cuerpos / Free-Return)",
        fontsize=12, color=TC,
    )

    # Panel izquierdo: trayectoria 2D
    ax = axes[0]
    ax.plot(r_orion[:, 0], r_orion[:, 1], color=OC,  lw=1.5, label="Orion (RK2)", zorder=4)
    ax.plot(rl2[:, 0],     rl2[:, 1],     color=MC,  lw=0.8, ls="--", alpha=0.5,
            label="Órbita Lunar")
    ax.scatter([0], [0],    s=250, color=EC, zorder=6, label="Tierra",
               edgecolors="white", lw=0.5)
    ax.scatter(*r_luna_fix, s=130, color=MC, zorder=6, label="Luna (t₀)",
               edgecolors="white", lw=0.5)
    ax.scatter(*r0_orion,   s=90,  color="#ffeb3b", marker="*", zorder=7, label="Orion t₀ (CSV)")
    ax.scatter(*r_orion[-1],s=90,  color="#e91e63", marker="v", zorder=7, label="Orion t_final")
    ax.set_aspect("equal"); ax.grid(True); ax.legend(fontsize=8)
    ax.set_xlabel("x [km]"); ax.set_ylabel("y [km]")
    ax.set_title(f"Free-Return Trajectory  (Δθ = {mejor_ang:.2f}°)")

    # Panel derecho: distancias vs tiempo
    ax = axes[1]
    ax.plot(t_arr, d_T / 1e3, color=EC, lw=1.2, label="Dist. a la Tierra")
    ax.plot(t_arr, d_L / 1e3, color=MC, lw=1.2, label="Dist. a la Luna")
    ax.axhline(1.737, ls=":", color="#ff1744", lw=0.8, label="R_Luna = 1 737 km")
    ax.set_xlabel("tiempo [días]"); ax.set_ylabel("distancia [10³ km]")
    ax.set_title("Distancias durante el vuelo")
    ax.legend(fontsize=8); ax.grid(True)

    fig.tight_layout()
    guardar_figura(fig, "Punto2_3_Orion_FreeReturn.png")

    return r_orion, v_orion, r_luna_fix, mejor_ang, T_FLY


def punto_4(r0_orion, v0_orion, r_luna_fix, mejor_ang, T_FLY):
    """
    Punto 4 — Comparación Euler vs RK2.
    Integra la misma trayectoria de Orion con ambos métodos y grafica
    la divergencia de posición para evidenciar la diferencia de orden.
    Retorna (r_eu, r_rk, diff4, t4).
    """
    print("\n[4] Euler vs RK2...")

    N  = 20_000
    h  = T_FLY / N

    v_mag   = np.linalg.norm(v0_orion)
    a_base  = np.arctan2(v0_orion[1], v0_orion[0])
    v_opt   = v_mag * np.array([
        np.cos(a_base + np.radians(mejor_ang)),
        np.sin(a_base + np.radians(mejor_ang)),
    ])

    r_eu, _ = integrar_euler(r0_orion, v_opt, acc_orion, h, N, r_luna=r_luna_fix)
    r_rk, _ = integrar_rk2  (r0_orion, v_opt, acc_orion, h, N, r_luna=r_luna_fix)

    t4    = np.linspace(0, T_FLY, N + 1) / 86400
    diff4 = np.linalg.norm(r_eu - r_rk, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Punto 4 – Euler vs RK2: Precisión Numérica", fontsize=13, color=TC)

    ax = axes[0]
    ax.plot(r_eu[:, 0], r_eu[:, 1], color=EUC, lw=1.2, label="Euler  O(h)",  alpha=0.9)
    ax.plot(r_rk[:, 0], r_rk[:, 1], color=RKC, lw=1.2, label="RK2    O(h²)", alpha=0.9)
    ax.scatter([0], [0],    s=200, color=EC, zorder=5, label="Tierra")
    ax.scatter(*r_luna_fix, s=80,  color=MC, zorder=5, label="Luna")
    ax.set_aspect("equal"); ax.grid(True); ax.legend(fontsize=8)
    ax.set_xlabel("x [km]"); ax.set_ylabel("y [km]")
    ax.set_title("Trayectorias (mismo paso h)")

    ax = axes[1]
    ax.semilogy(t4, diff4 + 1e-3, color=EUC, lw=1.4, label="|r_Euler − r_RK2|")
    ax.set_xlabel("tiempo [días]"); ax.set_ylabel("error posición [km]  (log)")
    ax.set_title("Divergencia Euler vs RK2")
    ax.legend(fontsize=9); ax.grid(True)
    caja_texto(ax,
        "Euler: O(h)  → error ∝ h\n"
        "RK2:  O(h²) → error ∝ h²\n"
        "→ Euler falla la inserción lunar\n"
        "→ RK2 cierra el ciclo de retorno"
    )

    fig.tight_layout()
    guardar_figura(fig, "Punto4_Euler_vs_RK2.png")

    return r_eu, r_rk, diff4, t4


def punto_5():
    """
    Punto 5 — Análisis a largo plazo de la órbita lunar.
    Simula 6 períodos lunares con Euler y RK2 para evidenciar la
    inestabilidad numérica (|γ| > 1): la luna espirala hacia afuera.
    Retorna (r_eu5, v_eu5, r_rk5, v_rk5, E0, t5, N5, h5).
    """
    print("\n[5] Largo plazo — inestabilidad numérica...")

    T5 = 6.0 * T_LUNAR
    N5 = 60_000
    h5 = T5 / N5

    r_eu5, v_eu5 = integrar_euler (R0_LUNA, V0_LUNA, acc_luna, h5, N5)
    r_rk5, v_rk5 = integrar_rk2   (R0_LUNA, V0_LUNA, acc_luna, h5, N5)

    d_eu5 = np.linalg.norm(r_eu5, axis=1)
    d_rk5 = np.linalg.norm(r_rk5, axis=1)
    E_eu5 = energia_mecanica(v_eu5, r_eu5)
    E_rk5 = energia_mecanica(v_rk5, r_rk5)
    E0    = E_eu5[0]
    t5    = np.linspace(0, T5, N5 + 1) / 86400

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("Punto 5 – Inestabilidad a Largo Plazo: |γ| > 1", fontsize=13, color=TC)

    ax = axes[0]
    ax.plot(r_eu5[:, 0], r_eu5[:, 1], color=EUC, lw=0.5, alpha=0.7, label="Euler")
    ax.plot(r_rk5[:, 0], r_rk5[:, 1], color=RKC, lw=0.5, alpha=0.7, label="RK2")
    ax.scatter([0], [0], s=150, color=EC, zorder=5)
    ax.set_aspect("equal"); ax.grid(True); ax.legend(fontsize=8)
    ax.set_title("Trayectorias (6 períodos lunares)")
    ax.set_xlabel("x [km]"); ax.set_ylabel("y [km]")

    ax = axes[1]
    ax.plot(t5, d_eu5 / 1e3, color=EUC, lw=0.8, label="Euler")
    ax.plot(t5, d_rk5 / 1e3, color=RKC, lw=0.8, label="RK2")
    ax.axhline((R_PERIGEO + R_APOGEO) / 2 / 1e3,
               ls="--", color="white", lw=0.6, alpha=0.4, label="Radio medio")
    ax.set_title("Radio orbital"); ax.set_xlabel("tiempo [días]"); ax.set_ylabel("|r| [10³ km]")
    ax.legend(fontsize=8); ax.grid(True)

    ax = axes[2]
    ax.plot(t5, (E_eu5 - E0) / abs(E0) * 100, color=EUC, lw=0.8, label="Euler")
    ax.plot(t5, (E_rk5 - E0) / abs(E0) * 100, color=RKC, lw=0.8, label="RK2")
    ax.set_title("Deriva de energía mecánica")
    ax.set_xlabel("tiempo [días]"); ax.set_ylabel("ΔE/E₀ [%]")
    ax.legend(fontsize=8); ax.grid(True)
    caja_texto(ax,
        "|γ| > 1  →  inestabilidad\n"
        "'Energía numérica' espuria\n"
        "→ espiral hacia afuera"
    )

    fig.tight_layout()
    guardar_figura(fig, "Punto5_LargoPlazo_Inestabilidad.png")

    return r_eu5, v_eu5, r_rk5, v_rk5, E0, t5, N5, h5


def punto_6(r_eu5, v_eu5, r_rk5, v_rk5, E0, t5, N5, h5):
    """
    Punto 6 — Método alternativo conservativo: Nyström.
    Demuestra que |γ₁·γ₂| = 1 garantiza estabilidad: la energía
    y el momento angular se conservan a largo plazo.
    Retorna (r_ny5, v_ny5, E_ny5).
    """
    print("\n[6] Método de Nyström (simpléctico)...")

    r_ny5, v_ny5 = integrar_nystrom(R0_LUNA, V0_LUNA, acc_luna, h5, N5)

    E_eu5 = energia_mecanica(v_eu5, r_eu5)
    E_rk5 = energia_mecanica(v_rk5, r_rk5)
    E_ny5 = energia_mecanica(v_ny5, r_ny5)

    L_eu  = momento_angular(r_eu5, v_eu5)
    L_rk  = momento_angular(r_rk5, v_rk5)
    L_ny  = momento_angular(r_ny5, v_ny5)
    L0    = L_eu[0]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        "Punto 6 – Nyström vs Euler/RK2: Conservación (|γ₁γ₂| = 1)",
        fontsize=12, color=TC,
    )

    ax = axes[0]
    ax.plot(r_ny5[:, 0], r_ny5[:, 1], color=NYC, lw=0.8, alpha=0.9, label="Nyström")
    ax.plot(r_eu5[:, 0], r_eu5[:, 1], color=EUC, lw=0.4, alpha=0.4, label="Euler (espiral)")
    ax.scatter([0], [0], s=150, color=EC, zorder=5)
    ax.set_aspect("equal"); ax.grid(True); ax.legend(fontsize=8)
    ax.set_title("Trayectorias (6 períodos)")
    ax.set_xlabel("x [km]"); ax.set_ylabel("y [km]")

    ax = axes[1]
    ax.plot(t5, (E_eu5 - E0) / abs(E0) * 100, color=EUC, lw=0.8, label="Euler")
    ax.plot(t5, (E_rk5 - E0) / abs(E0) * 100, color=RKC, lw=0.8, label="RK2")
    ax.plot(t5, (E_ny5 - E0) / abs(E0) * 100, color=NYC, lw=0.8, label="Nyström")
    ax.set_title("Deriva de energía mecánica")
    ax.set_xlabel("tiempo [días]"); ax.set_ylabel("ΔE/E₀ [%]")
    ax.legend(fontsize=8); ax.grid(True)

    ax = axes[2]
    ax.plot(t5, (L_eu - L0) / abs(L0) * 100, color=EUC, lw=0.8, label="Euler")
    ax.plot(t5, (L_rk - L0) / abs(L0) * 100, color=RKC, lw=0.8, label="RK2")
    ax.plot(t5, (L_ny - L0) / abs(L0) * 100, color=NYC, lw=0.8, label="Nyström")
    ax.set_title("Conservación del momento angular")
    ax.set_xlabel("tiempo [días]"); ax.set_ylabel("ΔL/L₀ [%]")
    ax.legend(fontsize=8); ax.grid(True)
    caja_texto(ax,
        "Nyström: |γ₁·γ₂| = 1\n"
        "→ γ solo rota en ℂ\n"
        "→ sin amplificación\n"
        "→ órbita estable"
    )

    fig.tight_layout()
    guardar_figura(fig, "Punto6_Nystrom_Conservativo.png")

    return r_ny5, v_ny5, E_ny5


def figura_resumen(rl, dl, pi_, ai_,
                   r_orion, r_luna_fix, r0_orion, rl2,
                   r_eu, r_rk, diff4, t4,
                   r_eu5, r_rk5, r_ny5,
                   E_eu5, E_rk5, E_ny5,
                   E0, t5):
    """
    Panel 2×3 con un subgráfico por punto del TP para presentación general.
    """
    print("\n[*] Figura resumen...")

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        "Misión Artemis II — Resumen TP Modelación Numérica",
        fontsize=15, color=TC, fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)

    # ── 1: Órbita lunar ──────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(rl[:, 0], rl[:, 1], color=MC, lw=1.1)
    ax.scatter([0], [0], s=150, color=EC, zorder=5, label="Tierra")
    ax.scatter(rl[pi_, 0], rl[pi_, 1], color="#ff1744", s=50, zorder=6,
               label=f"Per. {dl.min():.0f} km")
    ax.scatter(rl[ai_, 0], rl[ai_, 1], color="#69f0ae", s=50, zorder=6,
               label=f"Ap. {dl.max():.0f} km")
    ax.set_aspect("equal"); ax.grid(True); ax.legend(fontsize=7)
    ax.set_title("1 – Órbita Lunar"); ax.set_xlabel("x [km]"); ax.set_ylabel("y [km]")

    # ── 2/3: Orion free-return ────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(r_orion[:, 0], r_orion[:, 1], color=OC, lw=1.4, label="Orion")
    ax.plot(rl2[:, 0],     rl2[:, 1],     color=MC, lw=0.7, ls="--", alpha=0.5,
            label="Órbita Lunar")
    ax.scatter([0], [0],    s=200, color=EC, zorder=5, label="Tierra")
    ax.scatter(*r_luna_fix, s=80,  color=MC, zorder=5)
    ax.scatter(*r0_orion,   s=80,  color="#ffeb3b", marker="*", zorder=7, label="t₀ CSV")
    ax.set_aspect("equal"); ax.grid(True); ax.legend(fontsize=7)
    ax.set_title("2/3 – Orion Free-Return"); ax.set_xlabel("x [km]"); ax.set_ylabel("y [km]")

    # ── 4: Euler vs RK2 (trayectorias) ───────────────────────
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(r_eu[:, 0], r_eu[:, 1], color=EUC, lw=1.1, label="Euler", alpha=0.85)
    ax.plot(r_rk[:, 0], r_rk[:, 1], color=RKC, lw=1.1, label="RK2",   alpha=0.85)
    ax.scatter([0], [0], s=150, color=EC, zorder=5)
    ax.set_aspect("equal"); ax.grid(True); ax.legend(fontsize=7)
    ax.set_title("4 – Euler vs RK2"); ax.set_xlabel("x [km]"); ax.set_ylabel("y [km]")

    # ── 4: Error de posición ──────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.semilogy(t4, diff4 + 1e-3, color=EUC, lw=1.2, label="|Δr| Euler−RK2")
    ax.set_title("4 – Error posición (log)")
    ax.set_xlabel("tiempo [días]"); ax.set_ylabel("[km]")
    ax.legend(fontsize=7); ax.grid(True)

    # ── 5/6: Deriva de energía ────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t5, (E_eu5 - E0) / abs(E0) * 100, color=EUC, lw=0.8, label="Euler")
    ax.plot(t5, (E_rk5 - E0) / abs(E0) * 100, color=RKC, lw=0.8, label="RK2")
    ax.plot(t5, (E_ny5 - E0) / abs(E0) * 100, color=NYC, lw=0.8, label="Nyström")
    ax.set_title("5/6 – Deriva de Energía")
    ax.set_xlabel("tiempo [días]"); ax.set_ylabel("ΔE/E₀ [%]")
    ax.legend(fontsize=7); ax.grid(True)

    # ── 6: Trayectorias Nyström vs Euler ─────────────────────
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(r_ny5[:, 0], r_ny5[:, 1], color=NYC, lw=0.8, alpha=0.9, label="Nyström (estable)")
    ax.plot(r_eu5[:, 0], r_eu5[:, 1], color=EUC, lw=0.4, alpha=0.4, label="Euler (espiral)")
    ax.scatter([0], [0], s=150, color=EC, zorder=5)
    ax.set_aspect("equal"); ax.grid(True); ax.legend(fontsize=7)
    ax.set_title("6 – Nyström (Simpléctico)"); ax.set_xlabel("x [km]"); ax.set_ylabel("y [km]")

    guardar_figura(fig, "Resumen_TP_ArtemisII.png")


# ══════════════════════════════════════════════════════════════
# SECCIÓN 7: MAIN — orquesta el TP completo
# ══════════════════════════════════════════════════════════════

def main():
    """
    Punto de entrada principal.
    Llama a cada función del TP en orden y pasa los resultados
    intermedios necesarios entre ellas.
    """
    configurar_estilo()

    # Lectura del CSV (condiciones iniciales de Orion)
    print("=" * 58)
    print("Leyendo telemetría del CSV (3 abril, 04–06h)...")
    r0_orion, v0_orion = leer_condiciones_iniciales(CSV_PATH)
    print(f"  r₀ = {r0_orion}  km")
    print(f"  v₀ = {v0_orion}  km/s")
    print(f"  |r₀| = {np.linalg.norm(r0_orion):.0f} km desde la Tierra")
    print("=" * 58)

    # ── Puntos del TP ─────────────────────────────────────────
    rl, vl, dl, sl, ti, pi_, ai_ = punto_1()

    r_orion, v_orion, r_luna_fix, mejor_ang, T_FLY = punto_2_3(r0_orion, v0_orion)

    # Órbita lunar durante el vuelo de Orion (para la figura resumen)
    N_l2 = 20_000
    h_l2 = T_FLY / N_l2
    rl2, _ = integrar_rk2(R0_LUNA, V0_LUNA, acc_luna, h_l2, N_l2)

    r_eu, r_rk, diff4, t4 = punto_4(r0_orion, v0_orion, r_luna_fix, mejor_ang, T_FLY)

    r_eu5, v_eu5, r_rk5, v_rk5, E0, t5, N5, h5 = punto_5()

    r_ny5, v_ny5, E_ny5 = punto_6(r_eu5, v_eu5, r_rk5, v_rk5, E0, t5, N5, h5)

    # Energías para el resumen (recalculadas para consistencia)
    E_eu5 = energia_mecanica(v_eu5, r_eu5)
    E_rk5 = energia_mecanica(v_rk5, r_rk5)

    figura_resumen(
        rl, dl, pi_, ai_,
        r_orion, r_luna_fix, r0_orion, rl2,
        r_eu, r_rk, diff4, t4,
        r_eu5, r_rk5, r_ny5,
        E_eu5, E_rk5, E_ny5,
        E0, t5,
    )

    print("\n" + "=" * 58)
    print("TP completado. Archivos en:", OUT_DIR)
    print("=" * 58)


if __name__ == "__main__":
    main()
