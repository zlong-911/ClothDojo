"""Vertical (Flatten above Fold) ClothDojo benchmark layout."""
from stacked_benchmark_common import *


def build_vertical(spec: Spec, scale: float) -> str:
    W = 3.40
    PT = TEX_PT_PER_IN
    safe_x, safe_y = 7 / PT, 7 / PT
    title_em = FINAL_FONT["title"][0] / PT
    legend_em = FINAL_FONT["legend"][0] / PT
    xlabel_em = FINAL_FONT["xlabel"][0] / PT
    legend_y = safe_y + 0.72 * legend_em
    bottom_xlabel_y = legend_y + 2.35 * xlabel_em
    plot_to_xlabel = 0.70 * xlabel_em
    bottom_axis_y = bottom_xlabel_y + plot_to_xlabel
    bottom_plot_h = 5.90 * title_em
    bottom_title_y = bottom_axis_y + bottom_plot_h + 1.00 * title_em
    panel_gap = 2.10 * title_em
    top_xlabel_y = bottom_title_y + panel_gap
    top_axis_y = top_xlabel_y + plot_to_xlabel
    top_plot_h = 5.45 * title_em
    top_title_y = top_axis_y + top_plot_h + 1.00 * title_em
    H = top_title_y + 0.70 * title_em + safe_y
    axis_x, axis_w = 0.43, 2.62
    bar_w, _, _, seen_x, unseen_x = bar_geometry(len(spec.methods), axis_x, axis_w)

    lines = common_preamble(spec, scale)
    add = lines.append
    add(rf"\noindent\begin{{picture}}({W:.4f},{H:.4f})(0,0)")
    add(rf"\linethickness{{{FINAL_STROKE_PT / scale:.3f}pt}}")

    def panel(panel: Panel, title_y: float, axis_y: float, plot_h: float, xlabel_y: float) -> None:
        label = f"({'ab'[spec.panels.index(panel)]}) {panel.title}"
        add(rf"\put({safe_x:.4f},{title_y:.4f}){{\makebox(0,0)[l]{{\textcolor{{black}}{{\titlefont\bfseries {label}}}}}}}")
        add(rf"\put({axis_x:.4f},{axis_y:.4f}){{{{\color{{black}}\line(0,1){{{plot_h:.4f}}}}}}}")
        add(rf"\put({axis_x:.4f},{axis_y:.4f}){{{{\color{{black}}\line(1,0){{{axis_w:.4f}}}}}}}")
        for tick in panel.ticks:
            y = axis_y + plot_h * tick / panel.ymax
            add(rf"\put({axis_x:.4f},{y:.4f}){{{{\color{{black}}\line(-1,0){{0.027}}}}}}")
            add(rf"\put({axis_x-0.040:.4f},{y:.4f}){{\makebox(0,0)[r]{{\textcolor{{black}}{{\tickfont {tick}}}}}}}")
        add(rf"\put(0.145,{axis_y+plot_h/2:.4f}){{\makebox(0,0){{\rotatebox{{90}}{{\textcolor{{black}}{{\axisfont SR (\%)}}}}}}}}")
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

    panel(spec.panels[0], top_title_y, top_axis_y, top_plot_h, top_xlabel_y)
    panel(spec.panels[1], bottom_title_y, bottom_axis_y, bottom_plot_h, bottom_xlabel_y)
    add(legend_tex(spec.methods, W, legend_y))
    add(r"\end{picture}\end{document}")
    return "\n".join(lines) + "\n"
