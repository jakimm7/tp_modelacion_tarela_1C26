import matplotlib.pyplot as plt
import numpy as np

BG  = "#ffffff"
GC  = "#cccccc"
EC  = "#1565c0"
MC  = "#546e7a"
TC  = "#212121"

def configurar_estilo():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor":   BG,
        "axes.edgecolor":   GC, "axes.labelcolor":  TC,
        "xtick.color":      TC, "ytick.color":       TC,
        "text.color":       TC, "grid.color":        GC,
        "grid.linestyle":  "--", "grid.alpha":       0.6,
        "legend.facecolor": "#f5f5f5", "legend.edgecolor": GC,
        "font.family":      "monospace",
    })
 
def crear_figura(n_cols, n_rows=1, titulo="", ancho=16, alto=5):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(ancho, alto))
    if titulo:
        fig.suptitle(titulo, fontsize=13, color=TC)
    return fig, np.atleast_1d(axes).flatten()
 
def configurar_ejes(ax, titulo="", xlabel="", ylabel="", fontsize_legend=8,
                    loc_leyenda="best"):
    if titulo:
        ax.set_title(titulo)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(fontsize=fontsize_legend, loc=loc_leyenda)
 
def graficar_trayectoria(ax, trayectorias, titulo="", mostrar_tierra=True,
                          punto_luna=None, puntos_extra=None,
                          loc_leyenda="best"):
    for t in trayectorias:
        ax.plot(
            t["r"][:, 0], t["r"][:, 1],
            color=t["color"],
            lw=t.get("lw", 1.2),
            ls=t.get("ls", "-"),
            alpha=t.get("alpha", 1.0),
            label=t.get("label", ""),
            zorder=t.get("zorder", 3),
        )
 
    if mostrar_tierra:
        ax.scatter([0], [0], s=200, color=EC, zorder=6, label="Tierra",
                   edgecolors="black", linewidths=0.5)
 
    if punto_luna is not None:
        ax.scatter(*punto_luna, s=120, color=MC, zorder=6, label="Luna (t\u2080)",
                   edgecolors="black", linewidths=0.5)
 
    if puntos_extra:
        for p in puntos_extra:
            ax.scatter(
                *p["xy"],
                color=p.get("color", "white"),
                marker=p.get("marker", "o"),
                s=p.get("s", 60),
                zorder=p.get("zorder", 7),
                label=p.get("label", ""),
            )
 
    ax.set_aspect("equal")
    configurar_ejes(ax, titulo=titulo, xlabel="x [km]", ylabel="y [km]",
                    loc_leyenda=loc_leyenda)
 
def graficar_serie_temporal(ax, series, titulo="", xlabel="tiempo [días]",
                             ylabel="", escala_log=False):
    plot_fn = ax.semilogy if escala_log else ax.plot
 
    for s in series:
        plot_fn(
            s["t"], s["y"],
            color=s["color"],
            lw=s.get("lw", 1.0),
            label=s.get("label", ""),
            alpha=s.get("alpha", 1.0),
        )
 
    configurar_ejes(ax, titulo=titulo, xlabel=xlabel, ylabel=ylabel)
 
def graficar_lineas_referencia(ax, lineas):
    for lin in lineas:
        ax.axhline(
            lin["y"],
            color=lin["color"],
            ls=lin.get("ls", "--"),
            lw=lin.get("lw", 0.8),
            alpha=lin.get("alpha", 1.0),
            label=lin.get("label", ""),
        )
 
def caja_texto(ax, texto):
    """Agrega un cuadro de anotación informativa en la esquina superior izquierda."""
    ax.text(
        0.02, 0.97, texto,
        transform=ax.transAxes, va="top", fontsize=8,
        bbox=dict(facecolor="#f5f5f5", edgecolor=GC, alpha=0.9),
    )
 
def guardar_figura(fig, nombre_archivo, out_dir):
    """Guarda la figura en el directorio de salida, la cierra y confirma por consola."""
    ruta = out_dir / nombre_archivo
    fig.savefig(ruta, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Guardado: {ruta}")
