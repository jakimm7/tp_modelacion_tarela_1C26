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