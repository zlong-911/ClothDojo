#!/usr/bin/env python3
"""Generate ClothDojo stacked benchmark figures from versioned JSON configs."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import fitz

from stacked_benchmark_common import (
    COLUMNWIDTH_TEX_PT, FIGURES, PDF_PT_PER_IN, PRESETS, ROOT, TEX_PT_PER_IN,
    Spec, compile_crop, load_spec,
)
from stacked_benchmark_horizontal import build_horizontal
from stacked_benchmark_vertical import build_vertical


def build(spec: Spec, scale: float) -> str:
    return build_horizontal(spec, scale) if spec.layout == "horizontal" else build_vertical(spec, scale)


def overlap_count(pdf: Path) -> int:
    doc = fitz.open(pdf)
    words = doc[0].get_text("words")
    count = 0
    for i, a in enumerate(words):
        ra = fitz.Rect(*a[:4])
        for b in words[i + 1:]:
            inter = ra & fitz.Rect(*b[:4])
            if not inter.is_empty and inter.get_area() > 0.45:
                count += 1
    doc.close()
    return count


def generate(spec: Spec, output: Path, target_tex_pt: float, keep_tex: bool = False) -> dict:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    work = FIGURES / f"_{output.stem}-build.tex"
    target_pdf_pt = target_tex_pt * PDF_PT_PER_IN / TEX_PT_PER_IN
    scale, history = 1.0, []
    for i in range(7):
        work.write_text(build(spec, scale), encoding="utf-8")
        crop = compile_crop(work)
        doc = fitz.open(crop)
        w, h = doc[0].rect.width, doc[0].rect.height
        doc.close()
        inferred = target_pdf_pt / w
        history.append((i, scale, w, h, inferred))
        if abs(inferred - scale) < 8e-4:
            scale = inferred
            break
        scale = inferred
    work.write_text(build(spec, scale), encoding="utf-8")
    crop = compile_crop(work)
    shutil.copy2(crop, output)
    doc = fitz.open(output)
    w, h = doc[0].rect.width, doc[0].rect.height
    doc.close()
    if keep_tex:
        shutil.copy2(work, output.with_suffix(".tex"))
    for p in [work, work.with_suffix(".aux"), work.with_suffix(".log"), work.with_suffix(".pdf"), work.with_name(work.stem + "-crop.pdf")]:
        if p.exists():
            p.unlink()
    return {"output": output, "width": w, "height": h, "scale": target_pdf_pt / w,
            "overlaps": overlap_count(output), "history": history}


def write_qa(r: dict) -> None:
    p = r["output"].with_name(r["output"].stem + "-QA.txt")
    p.write_text(
        f"Figure: {r['output'].name}\n"
        f"tight size: {r['width']:.2f} x {r['height']:.2f} PDF pt\n"
        f"paper scale: {r['scale']:.5f}x\n"
        f"text overlaps: {r['overlaps']}\n",
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--preset", choices=["generated-data", "policy-architecture", "all"])
    src.add_argument("--config", type=Path)
    ap.add_argument("--output")
    ap.add_argument("--target-width-tex-pt", type=float, default=COLUMNWIDTH_TEX_PT)
    ap.add_argument("--keep-tex", action="store_true")
    ap.add_argument("--no-qa", action="store_true")
    args = ap.parse_args()
    if args.config:
        specs = [load_spec(args.config.resolve(), Path(args.output).name if args.output else None)]
    elif args.preset == "all":
        if args.output:
            ap.error("--output cannot be used with --preset all")
        specs = [load_spec(PRESETS[k]) for k in ("generated-data", "policy-architecture")]
    else:
        specs = [load_spec(PRESETS[args.preset], Path(args.output).name if args.output else None)]
    for spec in specs:
        out = Path(args.output) if args.output else FIGURES / spec.output_name
        if not out.is_absolute():
            out = ROOT / out
        result = generate(spec, out, args.target_width_tex_pt, args.keep_tex)
        if not args.no_qa:
            write_qa(result)
        print(f"wrote {result['output']}\n  tight={result['width']:.2f}x{result['height']:.2f} pt, "
              f"paper-scale={result['scale']:.5f}x, overlaps={result['overlaps']}")


if __name__ == "__main__":
    main()
