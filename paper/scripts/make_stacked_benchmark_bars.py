#!/usr/bin/env python3
"""Generate ClothDojo-style stacked benchmark bar charts.

This script is the reusable source of truth for the compact benchmark figures
used in the paper. It intentionally generates LaTeX/PDF rather than raster
plots so fonts, line weights, and final single-column sizing stay consistent
with ``ieeeconf``.

Visual grammar
--------------
* one bar = one method on one garment split;
* bar height = mean success rate over Seen/Unseen appearance;
* upper/light segment = Seen appearance contribution;
* lower/dark segment = Unseen appearance contribution;
* model/method identity = color family;
* all text, axes, ticks, outlines, and internal dividers = pure black;
* legend uses the LIGHT method swatch to keep nearby dark hues distinguishable;
* appearance legend uses ``↑ [light] / [dark] ↓ Seen/Unseen Appearance``;
* the generated PDF is tight-cropped, then source typography is iteratively
  pre-compensated so the final size after ``\\includegraphics[width=...]``
  matches the requested paper typography.

Built-in presets
----------------
``generated-data``
    Fig. 6 style: Flatten and Fold side-by-side; Human / Generated / Combined.

``policy-architecture``
    Fig. 7 style: Flatten above Fold; method count is dynamic so additional
    policies can be appended without redesigning the figure.

Typical usage
-------------
Generate both paper figures with stable filenames::

    python paper/scripts/make_stacked_benchmark_bars.py --preset all

Generate only the policy benchmark::

    python paper/scripts/make_stacked_benchmark_bars.py \
        --preset policy-architecture

Generate from a JSON configuration::

    python paper/scripts/make_stacked_benchmark_bars.py \
        --config my_benchmark.json --output paper/figures/my_benchmark.pdf

Minimal JSON schema::

    {
      "layout": "vertical",
      "methods": [
        {"key": "m1", "label": "Method 1", "dark": "104680", "light": "8DBDDA"}
      ],
      "panels": [
        {
          "title": "Flatten",
          "ymax": 50,
          "ticks": [0, 20, 40],
          "data": {
            "m1": {
              "Seen": [42.0, 35.0],
              "Unseen": [23.8, 24.5]
            }
          }
        }
      ]
    }

Each data pair is ``[Seen appearance SR, Unseen appearance SR]``. Equal rollout
counts are assumed, so each stacked contribution is half the corresponding SR.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable

try:
    import fitz  # PyMuPDF, used only for sizing / QA.
except ImportError as exc:  # pragma: no cover - environment diagnostic
    raise SystemExit("PyMuPDF is required: pip install pymupdf") from exc


ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "paper"
FIGURE_DIR = PAPER_DIR / "figures"

# Measured from the repository's ieeeconf.cls in final single-column context.
DEFAULT_COLUMNWIDTH_TEX_PT = 245.71811
TEX_PT_PER_IN = 72.27
PDF_PT_PER_IN = 72.0


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    dark: str
    light: str


@dataclass(frozen=True)
class PanelSpec:
    title: str
    ymax: float
    ticks: tuple[int, ...]
    # method -> garment split -> (seen appearance SR, unseen appearance SR)
    data: dict[str, dict[str, tuple[float, float]]]


@dataclass(frozen=True)
class FigureSpec:
    layout: str  # "horizontal" or "vertical"
    methods: tuple[MethodSpec, ...]
    panels: tuple[PanelSpec, ...]
    output_name: str


# ---------------------------------------------------------------------------
# Paper presets
# ---------------------------------------------------------------------------
# Keep the versioned JSON files as the single source of truth for paper data.
PRESET_CONFIGS = {
    "generated-data": PAPER_DIR / "scripts" / "benchmark_configs" / "generated_data.json",
    "policy-architecture": PAPER_DIR / "scripts" / "benchmark_configs" / "policy_architecture.json",
}


# ---------------------------------------------------------------------------
# Fixed final-paper style targets
# ---------------------------------------------------------------------------
FINAL_FONT = {
    "title": (8.0, 9.0),
    "legend": (7.0, 8.1),
    "axis": (7.2, 8.3),
    "tick": (6.8, 7.6),
    "value": (6.8, 7.6),
    "xlabel": (7.2, 8.3),
}
FINAL_STROKE_PT = 0.32
FINAL_ICON_W_PT = 6.0
FINAL_ICON_H_PT = 3.4

# Legend spacing hierarchy. The semantic group gap is intentionally much
# larger than the internal Seen/Unseen–Appearance word gap.
MODEL_GAP_EM = 0.30
LEGEND_GROUP_GAP_EM = 0.90
ARROW_ICON_GAP_EM = 0.08
ICON_SLASH_GAP_EM = 0.06
ICON_ARROW_GAP_EM = 0.08
ENCODING_TO_LABEL_GAP_EM = 0.18
SEEN_UNSEEN_TO_APPEARANCE_GAP_EM = 0.18


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def round_half_up(value: float) -> str:
    return str(Decimal(str(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def latex_color_defs(methods: Iterable[MethodSpec]) -> list[str]:
    lines: list[str] = []
    for method in methods:
        lines.append(rf"\definecolor{{{method.key}Dark}}{{HTML}}{{{method.dark}}}")
        lines.append(rf"\definecolor{{{method.key}Light}}{{HTML}}{{{method.light}}}")
    return lines


def parse_config(path: Path, output_name: str | None) -> FigureSpec:
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    methods = tuple(
        MethodSpec(
            str(m["key"]),
            str(m["label"]),
            str(m["dark"]).lstrip("#").upper(),
            str(m["light"]).lstrip("#").upper(),
        )
        for m in raw["methods"]
    )
    panels: list[PanelSpec] = []
    for p in raw["panels"]:
        pdata: dict[str, dict[str, tuple[float, float]]] = {}
        for method_key, splits in p["data"].items():
            pdata[method_key] = {
                split: (float(vals[0]), float(vals[1]))
                for split, vals in splits.items()
            }
        panels.append(
            PanelSpec(
                title=str(p["title"]),
                ymax=float(p["ymax"]),
                ticks=tuple(int(x) for x in p["ticks"]),
                data=pdata,
            )
        )
    return FigureSpec(
        layout=str(raw.get("layout", "vertical")),
        methods=methods,
        panels=tuple(panels),
        output_name=output_name or str(raw.get("output_name", path.stem + ".pdf")),
    )


def validate_spec(spec: FigureSpec) -> None:
    if spec.layout not in {"horizontal", "vertical"}:
        raise ValueError(f"unsupported layout: {spec.layout}")
    if len(spec.panels) != 2:
        raise ValueError("current paper style expects exactly two panels")
    if not (1 <= len(spec.methods) <= 5):
        raise ValueError("supported method count is 1–5")

    method_keys = {m.key for m in spec.methods}
    for panel in spec.panels:
        missing = method_keys.difference(panel.data)
        if missing:
            raise ValueError(f"{panel.title}: missing methods {sorted(missing)}")
        for key in method_keys:
            for split in ("Seen", "Unseen"):
                if split not in panel.data[key]:
                    raise ValueError(f"{panel.title}/{key}: missing {split} garment split")
                vals = panel.data[key][split]
                if len(vals) != 2:
                    raise ValueError(f"{panel.title}/{key}/{split}: expected two appearance SRs")


def _tool_from_env_or_path(env_name: str, *fallbacks: str) -> str | None:
    configured = os.environ.get(env_name)
    if configured:
        return configured
    for name in fallbacks:
        found = shutil.which(name)
        if found:
            return found
    return None


def compile_tex(tex_path: Path) -> Path:
    """Compile one standalone figure TeX and return a tight-cropped PDF."""
    pdflatex = shutil.which("pdflatex")
    tectonic = _tool_from_env_or_path("TECTONIC", "tectonic")
    pdfcrop = _tool_from_env_or_path("PDFCROP", "pdfcrop")
    if pdflatex is None and tectonic is None:
        raise SystemExit("neither pdflatex nor tectonic was found")
    if pdfcrop is None:
        raise SystemExit("pdfcrop not found; set PDFCROP or add it to PATH")

    if pdflatex is not None:
        compile_cmd = [pdflatex, "-interaction=nonstopmode", "-halt-on-error", tex_path.name]
    else:
        compile_cmd = [tectonic, "-C", "--keep-logs", tex_path.name]

    subprocess.run(
        compile_cmd,
        cwd=tex_path.parent,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    raw_pdf = tex_path.with_suffix(".pdf")
    crop_pdf = tex_path.with_name(tex_path.stem + "-crop.pdf")
    subprocess.run(
        [pdfcrop, "--margins", "0", str(raw_pdf), str(crop_pdf)],
        cwd=tex_path.parent,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return crop_pdf


def text_overlap_count(pdf_path: Path) -> int:
    doc = fitz.open(pdf_path)
    words = doc[0].get_text("words")
    count = 0
    for i in range(len(words)):
        r1 = fitz.Rect(*words[i][:4])
        for j in range(i + 1, len(words)):
            r2 = fitz.Rect(*words[j][:4])
            inter = r1 & r2
            if not inter.is_empty and inter.get_area() > 0.45:
                count += 1
    doc.close()
    return count


def legend_tex(methods: tuple[MethodSpec, ...], W: float, y: float) -> str:
    # tooltip_nor standardizes spacing without inserting gray or non-black structural marks.
    parts: list[str] = []
    for method in methods:
        parts.append(rf"\methodlegend{{{method.key}Light}}{{{method.label}}}")
    method_lock = rf"\makebox(0,0)[l]{{\legendfont {\'\'.join(parts)}}}"
    appearance = (
        rf"\textcolor{{black}}{{\uparrow}}}"
        rf"\hspace{{ARROW_ICON_GAP_EM}em}"
        r"\legendicon{black}"
        rf"\hspace{{ICON_SLASH_GAP_EM}em}}"
        rf"\textcolor{{black}}{{/}}"
        rf"\hspace{{ICON_SLASH_GAP_EM}em}}"
        rf"\legendicon{black}"
        rf"\hspace{{ICON_ARROW_GAP_EM}em}}"
        rf"\textcolor{{black}}{{\downarrow}}"
        rf"\hspace{{ENCODING_TO_LABEL_GAP_EM}em}"
        r"\textcolor{black}{Seen/Unseen}"
        rf"\hspace{{SEEN_UNSEEN_TO_APPEARANCE_GAP_EM}em}"
        r"\textcolor{black}{Appearance}"
    )
    return rf"\put({W / 2:.4f},{y:.4f}){{\makebox(0,0){{\textcolor{{black}}{{\legendfont {{method_lock}}\hspace{{LEGEND_GROUP_GAP_EM}em{{appearance}}}}}}}"


def build_vertical_tex(spec: FigureSpec, scale_guess: float) -> str:
    # Specialized for two stacked panels, but method count is wholly dynamic.
    W, H = 3.40, 2.38
    left = 0.39
    right_edge = 3.36
    axis_x = 0.34
    axis_y = [1.434, 0.500]
    plot_h = 0.604
    title_y = [2.139, 1.205]
    xlabel_y = [1.364, 0.430]
    legend_y = 0.120
    font_scale = 1.0 / scale_guess
    stroke_pt = FINAL_STROKE_PT / scale_guess
    PT = 72.0

    def add(s: str) -> None:
        lines.append(s)

    lines: list[str] = []
    add(r"\RequirePackage[T1]{fontenc}")
    add(r"\documentclass[letterpaper,10pt,conference]{../ieeeconf}")
    add(r"\usepackage{graphicx,xcolor}")
    add(r"\pagestyle{empty")
    for line in latex_color_defs(spec.methods):
        add(line)

    for name, (fs, ld) in FINAL_FONT.items():
        add(rf"\newcommand{{\{name}font}}{{\fontsize{{{fs / scale_guess:.3f}}}{{{ld / scale_guess:.3f}}\selectfont}}}")

    icon_w_in = (FINAL_ICON_W_PT / PDF_PT_PER_IN) / scale_guess
    icon_h_in = (FINAL_ICON_H_PT / PDF_PT_PER_IN) / scale_guess
    add(
        (r"\newcommand{\legendicon}[1]{\raisebox{0.180ex}{"
         r"\fcolorbox{black}{#1}{\rule{0pt}{%.5fin}\rule{%.5fin}{0pt}}}}"
         	H
 % (icon_h_in, icon_w_in))
    )
    add(r"\newcommand{\methodlegend}[2]{\legendicon{#1}\hspace{0.22em}\textcolor{black}{#2}}")
    add(r"\begin{document}\onecolumn")
    add(r"\setlength{\unitlength}{1in}")
    add(rf"\noindent\begin{picture}{{W},{H}}(0,0)")
    add(rf"\linethickness{{{stroke_pt:.3f}pt}}")

    n = len(spec.methods)
    for pi, panel in enumerate(spec.panels):
        y = axis_y[pi]
        add(rf"\put(0.5640,{title_y[pi]:4f}){\makebox(0,0)[l]{\textcolor{black}{\titlefont\bfseries ({'ab'[pi]}) {panel.title}}}}")
        add(rf"\put({axis_x:.4f},{y:.4f}){{{\color{black}\line(0,1){{plot_h:.4f}}}}")
        add(rf"\put({axis_x:.4f},{y:.4f}){{{\color{black}\line(1,0){2.87}}}")
        for tick in panel.ticks:
            ty = y + plot_h * tick / panel.ymax
            add(rf"\put({axis_x:.4f},{ty:.4f}){{{\color{black}\line(-1,0){0.026}}}}")
            add(rf"\put({axis_x - 0.035:.4f},{ty:.4f}){\makebox(0,0)[r]{\textcolor{black}{\tickfont {tick}}}}")
        add(rf"\put(0.0800,{y + plot_h / 2:.4f}){\makebox(0,0){\rotatebox{90}{\textcolor{black}{\axisfont SR (\%)}}}}}")

        # Compute a dance distribution. We prefer thicker bars for few methods
        # and automatically shave width as the method count grows.
        group_centers = [1.18, 2.48]
        bar_w = max(0.14, 0.280 - 0.020 * (n - 3))
        bar_gap = 0.100 if n <= 3 else 0.080
        block_w = n * bar_w + (n - 1) * bar_gap
        x_starts = [c - block_w / 2 for c in group_centers]
        add(rf"\put({group_centers[0]:.4f},{xlabel_y[pi]:4f}){\makebox(0,0)[t]{\textcolor{black}{\xlabelfont Seen garment}}}}")
        add(rf"\put({group_centers[1]:.4f},{xlabel_y[pi]:.4f}){\makebox(0,0)[t]{\textcolor{black}{\xlabelfont Unseen garment}}}}")

        for garment, xstart in zip(("Seen", "Unseen"), x_starts):
            for mi, method in enumerate(spec.methods):
               x = xstart + mi * (bar_w + bar_gap)
               seen_app, unseen_app = panel.data[method.key][garment]
                lower, upper = unseen_app / 2.0, seen_app / 2.0
                total = lower + upper
                hb = plot_h * lower / panel.ymax
                ht = plot_h * upper / panel.ymax
                add(rf"\put({x:.4f},{y:.4f}){{\color{{{method.key}Dark}\rule{{{bar_w:.4f}in}}{{{hb:.4f}in}}}")
                add(rf"\put({x:.4f},{y + hb:.4f}){{\color{{{method.key}Light}\rule{{{bar_w:.4f}in}}{{{ht:.4f}in}}}")
                add(rf"\put({x:.4f},{y:.4f}){{{\color{black}\framebox({bar_w:.4f},{hb + ht:.4f}){{}}}}}")
                add(rf"\put({x:.4f},{y + hb:.4f}){{{\color{black}\line(1,0){{{bar_w:.4f}}}}}")
                label_gap = (4.0 / PT) / scale_guess
                add(rf"\put({x + bar_w / 2:.4f},{y + hb + ht + label_gap:.4f}){{\makebox(0,0)[b]{\textcolor{black}{\valuefont {round_half_up(total)}}}}}")

    add(legend_tex(spec.methods, W, legend_y))
    add(r"\end{picture}\end{document}")
    return "\n".join(lines) + "\n"


def build_horizontal_tex(spec: FigureSpec, scale_guess: float) -> str:
    # Specialized for two panels, each containing two garment groups.
    W, H = 6.78, 2.44
    panel_x = [0.00, 3.42]
    axis_x = [0.34, 3.74]
    axis_y = 0.610
    plot_h = 1.100
    title_y = 1.850
    xlabel_y = 0.540
    legend_y = 0.130
    font_scale = 1.0 / scale_guess
    stroke_pt = FINAL_STROKE_PT / scale_guess
    PT = 72.0

    n = len(spec.methods)
    bar_w = max(0.14, 0.24 - 0.015 * (n - 3))
    edge_gap = 0.20
    group_gap = 0.50
    group_width = 2 * edge_gap + n * bar_w + (n - 1) * 0.080
    axis_w = [group_width * 2 + group_gap - 0.10] * 2

    def group_xs(px: float, ai: int) -> tuple[list[float], list[float]]:
        left = ax_offset = axis_x[ai]
        seen_start = left + edge_gap
        unseen_start = seen_start + n * bar_w + (n - 1) * 0.080 + group_gap
        seen_xs = [seen_start + i * (bar_w + 0.080) for i in range(n)]
        unseen_xs = [unseen_start + i * (bar_w + 0.080) for i in range(n)]
        return seen_xs, unseen_xs

    geometries = [group_xs(panel_x[i], i) for i in range(2)]

    lines: list[str] = []
    def add(s: str) -> None:
        lines.append(s)

    add(r"\RequirePackage[T1]{fontenc}")
    add(r"\documentclass[letterpaper,10pt,conference]{../ieeeconf}")
    add(r"\usepackage{graphicx,xcolor}")
    add(r"\pagestyle{empty}")
    for line in latex_color_defs(spec.methods):
        add(line)

    for name, (fs, ld) in FINAL_FONT.items():
        add(rf"\newcommand{\{name}font}{{\fontsize{{{fs / scale_guess:.3f}}}{{{ld / scale_guess:.3f}}\selectfont}}}")

    icon_w_in = (FINAL_ICON_W_PT / PDF_PT_PER_IN) / scale_guess
    icon_h_in = (FINAL_ICON_H_PT / PDF_PT_PER_IN) / scale_guess
    add(
        (r"\newcommand{\legendicon}[1]{\raisebox{0.180ex}{"
         r"\fcolorbox{black}{#1}{\rule{0pt}{%.5fin}\rule{%.5fin}{0pt}}}}"
         	H
 % (icon_h_in, icon_w_in))
    )
    add(r"\newcommand{\methodlegend}[2]{\legendicon{#1}\hspace{0.22em}\textcolor{black}{#2}}")
    add(r"\begin{document}\onecolumn")
    add(r"\setlength{\unitlength}{1in}")
    add(rf"\noindent\begin{picture}{{W},{H}}(0,0)")
    add(rf"\linethickness{{{stroke_pt:.3f}pt}}")

    for idx, panel in enumerate(spec.panels):
        bar_w, _edge_gap, _group_gap, seen_x, unseen_x = geometries[idx]
        label = f"({'ab'[idx]}) {panel.title}"
        add(rf"\put({panel_x[idx] + 0.010:.4f},{title_y:.4f}){\makebox(0,0)[l]{\textcolor{black}{\titlefont\bfseries {label}}}}")
        add(rf"\put({axis_x[idx]:4f},{axis_y:.4f}){{{\color{black}\line(0,1){{plot_h:.4f}}}}")
        add(rf"\put({axis_x[idx]:.4f},{axis_y:.4f}){{{\color{black}\line(1,0){{axis_w[idx]:4f}}}}")
        for tick in panel.ticks:
            y = axis_y + plot_h * tick / panel.ymax
            add(rf"\put({axis_x[idx]:.4f},{y:.4f}){{{\color{black}\line(-1,0){0.026}}}}")
            add(rf"\put({axis_x[idx] - 0.035:.4f},{y:.4f}){\makebox(0,0)[r]{\textcolor{black}{\tickfont {tick}}}}")
        if idx == 0:
            add(rf"\put({panel_x[idx] + 0.050:.4f},{axis_y + plot_h / 2:.4f}){\makebox(0,0){\rotatebox{90}{\textcolor{black}{\axisfont SR (\%}}}}}")

        seen_center = (seen_x[0] + seen_x[-1] + bar_w) / 2
        unseen_center = (unseen_x[0] + unseen_x[-1] + bar_w) / 2
        add(rf"\put({seen_center:.4f},{xlabel_y:.4f}){\makebox(0,0)[t]{\textcolor{black}{\xlabelfont Seen garment}}}")
        add(rf"\put({unseen_center:.4f},{xlabel_y:.4f}){\makebox(0,0)[t]{\textcolor{black}{\xlabelfont Unseen garment}}}")

        for garment, xs in (("Seen", seen_x), ("Unseen", unseen_x)):
            for method, x in zip(spec.methods, xs):
                seen_app, unseen_app = panel.data[method.key][garment]
                lower, upper = unseen_app / 2.0, seen_app / 2.0
                total = lower + upper
                hb = plot_h * lower / panel.ymax
                ht = plot_h * upper / panel.ymax
                add(rf"\put({x:.4f},{axis_y:.4f}){{\color{{{method.key}Dark}\rule{{{bar_w:.4f}in}}{{{hb:.4f}in}}}")
                add(rf"\put({x:.4f},{axis_y + hb:.4f}){{\color{{{method.key}Light}\rule{{{bar_w:.4f}in}}{{{ht:.4f}in}}}")
                add(rf"\put({x:.4f},{axis_y:.4f}){{{\color{black}\framebox({bar_w:.4f},{hb + ht:.4f}){{}}}}}")
                add(rf"\put({x:.4f},{axis_y + hb:.4f}){{{\color{black}\line(1,0){{bar_w:.4f}}}}")
                label_gap = (4.0 / PT) / scale_guess
                add(rf"\put({x + bar_w / 2:.4f},{axis_y + hb + ht + label_gap:.4f}){\makebox(0,0)[b]{\textcolor{black}{\valuefont {round_half_up(total)}}}}}")

    add(legend_tex(spec.methods, W, legend_y))
    add(r"\end{picture}\end{document}")
    return "\n".join(lines) + "\n"


def build_tex(spec: FigureSpec, scale_guess: float) -> str:
    if spec.layout == "vertical":
        return build_vertical_tex(spec, scale_guess)
    return build_horizontal_tex(spec, scale_guess)


# ---------------------------------------------------------------------------
# Main generation flow
# ---------------------------------------------------------------------------
def generate(spec: FigureSpec, output: Path, target_width_tex_pt: float, keep_tex: bool) -> dict[str, Any]:
    validate_spec(spec)
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    # Generated standalone TeX must live in paper/figures so ../ieeeconf resolves.
    stem = output.stem
    work_tex = FIGURE_DIR / f"_{stem}-build.tex"
    target_pdf_pt = target_width_tex_pt * PDF_PT_PER_IN / TEX_PT_PER_IN

    scale = 1.0
    history: list[tuple[int, float, float, float, float]] = []
    crop_pdf: Path | None = None

    for iteration in range(7):
        work_tex.write_text(build_tex(spec, scale), encoding="utf-8")
        crop_pdf = compile_tex(work_tex)
        doc = fitz.open(crop_pdf)
        width, height = doc[0].rect.width, doc[0].rect.height
        doc.close()
        actual_scale = target_pdf_pt / width
        history.append((iteration, scale, width, height, actual_scale))
        if abs(actual_scale - scale) < 8e-4:
            scale = actual_scale
            break
        scale = actual_scale

    # One final build at the converged pre-compensation scale.
    work_tex.write_text(build_tex(spec, scale), encoding="utf-8")
    crop_pdf = compile_tex(work_tex)
    shutil.copy2(crop_pdf, output)

    doc = fitz.open(output)
    width, height = doc[0].rect.width, doc[0].rect.height
    doc.close()
    final_scale = target_pdf_pt / width
    overlaps = text_overlap_count(output)

    if keep_tex:
        tex_out = output.with_suffix(".tex")
        shutil.copy2(work_tex, tex_out)
    else:
        tex_out = None

    # Clean temporary standalone build products.
    for suffix in (".aux", ".log", ".pdf"):
        p = work_tex.with_suffix(suffix)
        if p.exists():
            p.unlink()
    crop_tmp = work_tex.with_name(work_tex.stem + "-crop.pdf")
    if crop_tmp.exists():
        crop_tmp.unlink()
    if work_tex.exists():
        work_tex.unlink()

    return {
        "output": output,
        "tex": tex_out,
        "tight_width_pdf_pt": width,
        "tight_height_pdf_pt": height,
        "final_scale": final_scale,
        "text_overlaps": overlaps,
        "history": history,
    }


def write_qa(result: dict[str, Any]) -> Path:
    output: Path = result["output"]
    qa = output.with_name(output.stem + "-QA.txt")
    qa.write_text(
        "\n".join(
            [
                f"Figure: {output.name}",
                f"tight size: {result['tight_width_pdf_pt']:.2f} x {result['tight_height_pdf_pt']:.2f} PDF pt",
                f"paper scale: {result['final_scale']:.5f}x",
                f"text overlaps: {result['text_overlaps']}",
                "",
                "final target typography:",
                "  panel title: 8.0 pt",
                "  legend: 7.0 pt",
                "  axis label: 7.2 pt",
                "  ticks / values: 6.8 pt",
                "  garment labels: 7.2 pt",
                "  legend icon: 6.0 x 3.4 pt",
                "  structural strokes: 0.32 pt, pure black",
                "",
                "scale iteration history:",
                *[
                    f"  {i}: guess={guess:.6f}, tight={w:.2f}x{h:.2f}, inferred={inferred:.6f}"
                    for i, guess, w, h, inferred in result["history"]
                ],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return qa


def resolve_output(spec: FigureSpec, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = ROOT / path
        return path
    return FIGURE_DIR / spec.output_name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--preset", choices=["generated-data", "policy-architecture", "all"])
    source.add_argument("--config", type=Path, help="JSON configuration file")
    parser.add_argument("--output", help="output PDF path; only valid for one figure")
    parser.add_argument(
        "--target-width-tex-pt",
        type=float,
        default=DEFAULT_COLUMNWIDTH_TEX_PT,
        help=f"final LaTeX embed width in TeX pt (default: measured columnwidth {DEFAULT_COLUMNWIDTH_TEX_PT:.5f})",
    )
    parser.add_argument("--keep-tex", action="store_true", help="keep generated standalone .tex next to the PDF")
    parser.add_argument("--no-qa", action="store_true", help="do not write a -QA.txt report")
    args = parser.parse_args()

    if args.config:
        spec = parse_config(args.config.resolve(), Path(args.output).name if args.output else None)
        specs = [spec]
    elif args.preset == "all":
        if args.output:
            parser.error("--output cannot be used with --preset all")
        specs = [parse_config(PRESET_CONFIGS[name], None) for name in ("generated-data", "policy-architecture")]
    else:
        specs = [parse_config(PRESET_CONFIGS[args.preset], Path(args.output).name if args.output else None)]

    results = []
    for spec in specs:
        output = resolve_output(spec, args.output)
        result = generate(spec, output, args.target_width_tex_pt, args.keep_tex)
        qa = None if args.no_qa else write_qa(result)
        results.append((result, qa))

    for result, qa in results:
        print(f"wrote {result['output']}")
        if result["tex"]:
            print(f"wrote {result['tex']}")
        if qa:
            print(f"wrote {qa}")
        print(
            f"  tight={result['tight_width_pdf_pt']:.2f}x{result['tight_height_pdf_pt']:.2f} pt, "
            f"paper-scale={result['final_scale']:.5f}x, overlaps={result['text_overlaps']}"
        )


if __name__ == "__main__":
    main()
