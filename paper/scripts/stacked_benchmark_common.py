"""Shared data model and LaTeX helpers for ClothDojo benchmark figures."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
FIGURES = PAPER / "figures"
CONFIGS = PAPER / "scripts" / "benchmark_configs"
PRESETS = {
    "generated-data": CONFIGS / "generated_data.json",
    "policy-architecture": CONFIGS / "policy_architecture.json",
}

COLUMNWIDTH_TEX_PT = 245.71811
TEX_PT_PER_IN = 72.27
PDF_PT_PER_IN = 72.0
FINAL_FONT = {
    "title": (8.0, 9.0), "legend": (7.0, 8.1), "axis": (7.2, 8.3),
    "tick": (6.8, 7.6), "value": (6.8, 7.6), "xlabel": (7.2, 8.3),
}
FINAL_STROKE_PT = 0.32
FINAL_ICON_W_PT, FINAL_ICON_H_PT = 6.0, 3.4
MODEL_GAP_EM = 0.30
LEGEND_GROUP_GAP_EM = 0.90
ARROW_ICON_GAP_EM = 0.08
ICON_SLASH_GAP_EM = 0.06
ICON_ARROW_GAP_EM = 0.08
ENCODING_TO_LABEL_GAP_EM = 0.18
SEEN_UNSEEN_TO_APPEARANCE_GAP_EM = 0.18


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    dark: str
    light: str


@dataclass(frozen=True)
class Panel:
    title: str
    ymax: float
    ticks: tuple[int, ...]
    data: dict[str, dict[str, tuple[float, float]]]


@dataclass(frozen=True)
class Spec:
    layout: str
    methods: tuple[Method, ...]
    panels: tuple[Panel, ...]
    output_name: str


def load_spec(path: Path, output_name: str | None = None) -> Spec:
    raw = json.loads(path.read_text(encoding="utf-8"))
    methods = tuple(Method(m["key"], m["label"], m["dark"].lstrip("#"), m["light"].lstrip("#")) for m in raw["methods"])
    panels = []
    for p in raw["panels"]:
        data = {
            mk: {split: (float(v[0]), float(v[1])) for split, v in splits.items()}
            for mk, splits in p["data"].items()
        }
        panels.append(Panel(p["title"], float(p["ymax"]), tuple(map(int, p["ticks"])), data))
    spec = Spec(raw.get("layout", "vertical"), methods, tuple(panels), output_name or raw.get("output_name", path.stem + ".pdf"))
    if spec.layout not in {"horizontal", "vertical"} or len(spec.panels) != 2 or not 1 <= len(spec.methods) <= 5:
        raise ValueError("expected horizontal/vertical layout, two panels, and 1--5 methods")
    for p in spec.panels:
        for m in spec.methods:
            for split in ("Seen", "Unseen"):
                if split not in p.data[m.key] or len(p.data[m.key][split]) != 2:
                    raise ValueError(f"bad data for {p.title}/{m.key}/{split}")
    return spec


def half_up(x: float) -> str:
    return str(Decimal(str(x)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def compiler(name: str) -> str | None:
    return os.environ.get(name.upper()) or shutil.which(name)


def compile_crop(tex: Path) -> Path:
    pdflatex, tectonic, pdfcrop = shutil.which("pdflatex"), compiler("tectonic"), compiler("pdfcrop")
    if pdflatex:
        cmd = [pdflatex, "-interaction=nonstopmode", "-halt-on-error", tex.name]
    elif tectonic:
        cmd = [tectonic, "-C", "--keep-logs", tex.name]
    else:
        raise SystemExit("pdflatex or tectonic is required")
    if not pdfcrop:
        raise SystemExit("pdfcrop is required (or set PDFCROP)")
    subprocess.run(cmd, cwd=tex.parent, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    raw = tex.with_suffix(".pdf")
    crop = tex.with_name(tex.stem + "-crop.pdf")
    subprocess.run([pdfcrop, "--margins", "0", str(raw), str(crop)], cwd=tex.parent, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return crop


def common_preamble(spec: Spec, scale: float) -> list[str]:
    out = [
        r"\RequirePackage[T1]{fontenc}",
        r"\documentclass[letterpaper,10pt,conference]{../ieeeconf}",
        r"\usepackage{graphicx,xcolor}", r"\pagestyle{empty}", r"\setlength{\fboxsep}{0pt}",
        rf"\setlength{{\fboxrule}}{{{FINAL_STROKE_PT / scale:.3f}pt}}",
    ]
    for m in spec.methods:
        out += [rf"\definecolor{{{m.key}Dark}}{{HTML}}{{{m.dark}}}", rf"\definecolor{{{m.key}Light}}{{HTML}}{{{m.light}}}"]
    for name, (size, leading) in FINAL_FONT.items():
        out.append(rf"\newcommand{{\{name}font}}{{\fontsize{{{size / scale:.3f}}}{{{leading / scale:.3f}}}\selectfont}}")
    iw = (FINAL_ICON_W_PT / PDF_PT_PER_IN) / scale
    ih = (FINAL_ICON_H_PT / PDF_PT_PER_IN) / scale
    out += [
        (r"\newcommand{\legendicon}[1]{\raisebox{0.180ex}{"
         r"\fcolorbox{black}{#1}{\rule{0pt}{%.5fin}\rule{%.5fin}{0pt}}}}" % (ih, iw)),
        r"\newcommand{\methodlegend}[2]{\legendicon{#1}\hspace{0.22em}\textcolor{black}{#2}}",
        r"\begin{document}\onecolumn", r"\setlength{\unitlength}{1in}",
    ]
    return out


def bar_geometry(n: int, axis_x: float, axis_w: float) -> tuple[float, float, float, list[float], list[float]]:
    garment_gap = 0.29
    edge_gap = 0.055 if n <= 3 else (0.040 if n == 4 else 0.030)
    cluster_w = (axis_w - garment_gap) / 2.0
    bar_w = min(0.22, (cluster_w - (n - 1) * edge_gap) / n)
    used = n * bar_w + (n - 1) * edge_gap
    side_pad = (cluster_w - used) / 2.0
    seen_start = axis_x + side_pad
    seen = [seen_start + i * (bar_w + edge_gap) for i in range(n)]
    unseen_start = axis_x + cluster_w + garment_gap + side_pad
    unseen = [unseen_start + i * (bar_w + edge_gap) for i in range(n)]
    return bar_w, edge_gap, garment_gap, seen, unseen


def legend_tex(methods: tuple[Method, ...], width: float, y: float) -> str:
    parts: list[str] = []
    for i, method in enumerate(methods):
        parts.append(rf"\methodlegend{{{method.key}Light}}{{{method.label}}}")
        if i < len(methods) - 1:
            parts.append(rf"\hspace{{{MODEL_GAP_EM:.2f}em}}")
    parts.extend([
        rf"\hspace{{{LEGEND_GROUP_GAP_EM:.2f}em}}", r"\textcolor{black}{\ensuremath{\uparrow}}",
        rf"\hspace{{{ARROW_ICON_GAP_EM:.2f}em}}\legendicon{{white}}",
        rf"\hspace{{{ICON_SLASH_GAP_EM:.2f}em}}\textcolor{{black}}{{/}}",
        rf"\hspace{{{ICON_SLASH_GAP_EM:.2f}em}}\legendicon{{black}}",
        rf"\hspace{{{ICON_ARROW_GAP_EM:.2f}em}}\textcolor{{black}}{{\ensuremath{{\downarrow}}}}",
        rf"\hspace{{{ENCODING_TO_LABEL_GAP_EM:.2f}em}}",
        r"\textcolor{black}{Seen/Unseen\kern" + f"{SEEN_UNSEEN_TO_APPEARANCE_GAP_EM:.2f}em" + r" Appearance}",
    ])
    return rf"\put(0,{y:.4f}){{\makebox({width:.4f},0){{\legendfont {''.join(parts)}}}}}"
