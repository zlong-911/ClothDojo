"""Horizontal (Flatten | Fold) ClothDojo benchmark layout."""
from stacked_benchmark_common import *


def build_horizontal(spec: Spec, scale: float) -> str:
    W = 3.40
    PT = TEX_PT_PER_IN
    safe_x, safe_y, center_safe = 7 / PT, 7 / PT, 10 / PT
    title_em = FINAL_FONT["title"][0] / PT
    legend_em = FINAL_FONT["legend"][0] / PT
    xlabel_em = FINAL_FONT["xlabel"][0] / PT
    legend_y = safe_y + 0.70 * legend_em
    xlabel_y = legend_y + 2.35 * xlabel_em
    axis_y = xlabel_y + 0.72 * xlabel_em
    plot_h = 5.80 * title_em
    title_y = axis_y + plot_h + 1.05 * title_em
    H = title_y + 0.62 * title_em + safe_y
    content_x0, content_x1 = safe_x, W - safe_x
    panel_w = (content_x1 - content_x0 - center_safe) / 2.0
    panel_x = [content_x0, content_x0 + panel_w + center_safe]
    axis_x = [panel_x[0] + 0.285, panel_x[1] + 0.225]
    axis_w = [panel_x[0] + panel_w - axis_x[0] - 0.020, panel_x[1] + panel_w - axis_x[1] - 0.020]
    geometry = [bar_geometry(len(spec.methods), axis_x[i], axis_w[i]) for i in range(2)]

    lines = common_preamble(spec, scale)
    add = lines.append
    add(rf"\noindent\begin{{picture}}({W:.4f},{H:.4f})(0,0)")
    add(rf"\linethickness{{{FINAL_STROKE_PT / scale:.3f}pt}}")
    for i, panel in enumerate(spec.panels):
        bar_w, _, _, seen_x, unseen_x = geometry[i]
        add(rf"\put({panel_x[i]+0.010:.4f},{title_y:.4f}){{\makebox(0,0)[l]{{\textcolor{{black}}{{\titlefont\bfseries ({'ab'[i]}) {panel.title}}}}}}}")
        add(rf"\put({axis_x[i]:.4f},{axis_y:.4f}){{{{\color{{black}}\line(0,1){{{plot_h:.4f}}}}}}}")
        add(rf"\put({axis_x[i]:.4f},{axis_y:.4f}){{{{\color{{black}}\line(1,0){{{axis_w[i]:.4f}}}}}}}")
        for tick in panel.ticks:
            y = axis_y + plot_h * tick / panel.ymax
            add(rf"\put({axis_x[i]:.4f},{y:.4f}){{{{\color{{black}}\line(-1,0){{0.026}}}}}}")
            add(rf"\put({axis_x[i]-0.035:.4f},{y:.4f}){{\makebox(0,0)[r]{{\textcolor{{black}}{{\tickfont {tick}}}}}}}")
        if i == 0:
            add(rf"\put({panel_x[i]+0.050:.4f},{axis_y+plot_h/2:.4f}){{\makebox(0,0){{\rotatebox{{90}}{{\textcolor{{black}}{{\axisfont SR (\%)}}}}}}}}")
        seen_c = (seen_x[0] + seen_x[-1] + bar_w) / 2
        unseen_c = (unseen_x[0] + unseen_x[-1] + bar_w) / 2
        add(rf"\put({seen_c:.4f},{xlabel_y:.4f}){{\makebox(0,0)[t]{{\textcolor{{black}}{{\xlabelfont Seen garment}}}}}}")
        add(rf"\put({unseen_c:.4f},{xlabel_y:.4f}){{\makebox(0,0)[t]{{\textcolor{{black}}{{\xlabelfont Unseen garment}}}}}}")
        for garment, xs in (("Seen", seen_x), ("Unseen", unseen_x)):
            for method, x in zip(spec.methods, xs):
                seen_app, unseen_app = panel.data[method.key][garment]
                lower, upper = unseen_app / 2.0, seen_app / 2.0
                hb, ht = plot_h * lower / panel.ymax, plot_h * upper / panel.ymax
                add(rf"\put({x:.4f},{axis_y:.4f}){{\color{{{method.key}Dark}}\rule{{{bar_w:.4f}in}}{{{hb:.4f}in}}}}")
                add(rf"\put({x:.4f},{axis_y+hb:.4f}){{\color{{{method.key}Light}}\rule{{{bar_w:.4f}in}}{{{ht:.4f}in}}}}")
                add(rf"\put({x:.4f},{axis_y:.4f}){{{{\color{{black}}\framebox({bar_w:.4f},{hb+ht:.4f}){{}}}}}}")
                add(rf"\put({x:.4f},{axis_y+hb:.4f}){{{{\color{{black}}\line(1,0){{{bar_w:.4f}}}}}}}")
                gap = (4.0 / PT) / scale
                add(rf"\put({x+bar_w/2:.4f},{axis_y+hb+ht+gap:.4f}){{\makebox(0,0)[b]{{\textcolor{{black}}{{\valuefont {half_up(lower+upper)}}}}}}}")
    add(legend_tex(spec.methods, W, legend_y))
    add(r"\end{picture}\end{document}")
    return "\n".join(lines) + "\n"
