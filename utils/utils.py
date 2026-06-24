import numpy as np

G = 6.674e-20
M_T = 5.972e24
M_L = 7.348e22
GM_T = G * M_T
GM_L = G * M_L

def acc_luna(r):
    d = np.linalg.norm(r)
    return -GM_T * r / d**3

def acc_orion(r, r_luna):
    dT = np.linalg.norm(r)
    aT = -GM_T * r / dT**3

    delta = r - r_luna
    dL = np.linalg.norm(delta)
    aL = -GM_L * delta / dL**3

    return aT + aL

def energia_mecanica(vs, rs):
    d = np.linalg.norm(rs, axis=1)
    return 0.5 * np.sum(vs**2, axis=1) - GM_T / d

def momento_angular(rs, vs):
    return rs[:, 0] * vs[:, 1] - rs[:, 1] * vs[:, 0]

def integrar_euler(r0, v0, acc_fn, h, N, **kw):
    rs = np.empty((N + 1, 2))
    vs = np.empty((N + 1, 2))
    rs[0] = r0
    vs[0] = v0
    for i in range(N):
        a = acc_fn(rs[i], **kw)
        vs[i+1] = vs[i] + h * a
        rs[i+1] = rs[i] + h * vs[i]
    return rs, vs

def integrar_rk2(r0, v0, acc_fn, h, N, **kw):
    rs = np.empty((N + 1, 2))
    vs = np.empty((N + 1, 2))
    rs[0] = r0
    vs[0] = v0
    for i in range(N):
        a1 = acc_fn(rs[i], **kw)
        r_mid = rs[i] + 0.5 * h * vs[i]
        v_mid = vs[i] + 0.5 * h * a1
        a2 = acc_fn(r_mid, **kw)
        rs[i+1] = rs[i] + h * v_mid
        vs[i+1] = vs[i] + h * a2
    return rs, vs

def integrar_nystrom(r0, v0, acc_fn, h, N, **kw):
    rs = np.empty((N + 1, 2))
    vs = np.empty((N + 1, 2))
    rs[0] = r0
    vs[0] = v0
    a = acc_fn(rs[0], **kw)
    for i in range(N):
        rs[i+1] = rs[i] + h * vs[i] + 0.5 * h * h * a
        a_next = acc_fn(rs[i+1], **kw)
        vs[i+1] = vs[i] + 0.5 * h * (a + a_next)
        a = a_next
    return rs, vs

INTEGRADORES = {
    "euler": integrar_euler,
    "rk2": integrar_rk2,
    "nystrom": integrar_nystrom,
}

def parsear_float_europeo(s):
    return float(s.replace(",", "."))

def leer_condiciones_iniciales(csv_path, fecha="2026-04-03", hora_min=4, hora_max=6):
    with open(csv_path, "r") as f:
        for linea in f:
            linea = linea.strip()
            if not linea:
                continue
            cols = linea.split(";")
            ts = cols[0]
            if fecha not in ts:
                continue
            hora = int(ts.split("T")[1].split(":")[0])
            if hora_min <= hora < hora_max:
                x, y = parsear_float_europeo(cols[1]), parsear_float_europeo(cols[2])
                vx, vy = parsear_float_europeo(cols[4]), parsear_float_europeo(cols[5])
                return np.array([x, y]), np.array([vx, vy])

def imprimir_resultados(punto, descripcion, entradas):
    SEP = "=" * 62
    W = 36

    print(SEP)
    print(f"  PUNTO {punto}  -  {descripcion}")
    print(SEP)

    for item in entradas:
        tipo = item[0]

        if tipo == "seccion":
            print(f"\n  -- {item[1]}")

        elif tipo == "dato":
            etiqueta = item[1]
            valor = item[2]
            unidad = item[3] if len(item) > 3 else ""
            linea = f"{etiqueta:<{W}} {valor}"
            if unidad:
                linea += f"{unidad}"
            print(linea)

        elif tipo == "texto":
            print(f"  {item[1]}")

    print()

def imprimir_tabla(encabezados, filas, anchos=None):
    if anchos is None:
        anchos = [max(len(str(h)), *(len(str(f[i])) for f in filas)) + 2
                  for i, h in enumerate(encabezados)]
    linea_sep = "-" * (sum(anchos) + len(anchos) - 1)
    print(linea_sep)
    print("|".join(f"{str(h):^{a}}" for h, a in zip(encabezados, anchos)))
    print(linea_sep)
    for f in filas:
        print("|".join(f"{str(v):^{a}}" for v, a in zip(f, anchos)))
    print(linea_sep)

def estudio_convergencia_periodico(r0, v0, acc_fn, integrador, T, Ns, **kw):
    hs = np.empty(len(Ns))
    errores = np.empty(len(Ns))
    for i, N in enumerate(Ns):
        h = T / N
        r, _ = integrador(r0, v0, acc_fn, h, N, **kw)
        hs[i] = h
        errores[i] = np.linalg.norm(r[-1] - r0)
    return hs, errores

def estudio_convergencia_referencia(r0, v0, acc_fn, integrador, T, Ns, r_ref, **kw):
    hs = np.empty(len(Ns))
    errores = np.empty(len(Ns))
    for i, N in enumerate(Ns):
        h = T / N
        r, _ = integrador(r0, v0, acc_fn, h, N, **kw)
        hs[i] = h
        errores[i] = np.linalg.norm(r[-1] - r_ref)
    return hs, errores

def ajustar_orden_loglog(hs, errores):
    hs = np.asarray(hs, dtype=float)
    errores = np.asarray(errores, dtype=float)
    mask = (errores > 0) & np.isfinite(errores) & (hs > 0)
    pendiente, _ = np.polyfit(np.log(hs[mask]), np.log(errores[mask]), 1)
    return pendiente

def orden_local(hs, errores):
    errores = np.asarray(errores, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        razon = errores[:-1] / errores[1:]
        p = np.log2(razon)
    return p

def radio_min_max_por_periodo(r, n_periodos):
    N = r.shape[0] - 1
    paso = N // n_periodos
    resultados = []
    for k in range(1, n_periodos + 1):
        ini = (k - 1) * paso
        fin = k * paso + 1
        d = np.linalg.norm(r[ini:fin], axis=1)
        resultados.append((d.min(), d.max()))
    return resultados

def indices_por_periodo(N_total, n_periodos):
    paso = N_total // n_periodos
    return [k * paso for k in range(1, n_periodos + 1)]
