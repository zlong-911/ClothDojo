# ClothDojo paper

`draft_zh.md` is the current authoritative content draft for an ICRA paper. It
establishes the paper's reasoning in Chinese before results are frozen.
`main.tex` contains the translated abstract; the remaining English sections
retain the submission skeleton and will be synchronized as the Chinese draft
converges, avoiding two competing prose versions.

`docs/benchmark.md` remains the detailed experiment-planning document. The
paper draft explains the scientific argument; the planning document owns
operational choices that may still change.

## Build

The draft uses the IEEE Robotics conference `ieeeconf` format. The unmodified
`ieeeconf.cls` from PaperCept's official template archive is included in this
directory. Template and compiler references are listed in
[`docs/reference_index.md`](../docs/reference_index.md#论文排版).

With a full LaTeX installation providing `latexmk` and `IEEEtran.bst`, run from
this directory:

```sh
make paper
```

Alternatively, Tectonic downloads the required standard packages automatically:

```sh
tectonic --keep-logs main.tex
```

The shared workspace has Tectonic 0.17.0 installed at
`/cfs/mlp/training/docker/user/hadoop-aipnlp/huangzilong09/tools/latex-runtime/bin/tectonic`.
Both workflows write `main.pdf` in this directory.
After the first successful Tectonic build, `tectonic --only-cached --keep-logs
main.tex` rebuilds without network access. Bibliography output is currently
commented out in `main.tex` because `references.bib` has no entries; enable it
when verified entries and citations are added.

The target venue is ICRA. The exact page limit, anonymity rules and template
must be checked against the author instructions for the selected ICRA year.

## Figures

`figures/clothdojo-teaser.png` is the full-width first-page teaser assembled
from selected lossless RTX renders. The standalone garment views and task grid
use separate render passes. Garments are shown as metric-scale PBR flat lays
under a shared camera instead of being normalized independently. The task grid
uses action-revealing contact keyframes, complete robot workcells, and separate
third-person cameras sized for each embodiment. It references the RoboDojo
studio, GenieSimAssets laboratory/home/factory scenes, and the Isaac Sim 5.0
Small Warehouse Digital Twin. The renderer validates every known floor surface
against workcell `z=0`; ARX X5 additionally requires 0.30 m of room-geometry
clearance around the complete workstation. The workcell feet contact the room
floor directly; no extra safety mat or generic ground plane is rendered.
Tabletops deterministically sample room-compatible GenieSim PBR Material prims
while preserving one material across each robot's flatten/fold pair. Paper
cameras are fixed per embodiment, whereas exposure and room-aware practical
light temperature use bounded, seeded variation. Cache GenieSimAssets at revision
`a0813aad7c16165daffbe6c2737754e0344809ee` under
`data/visual-assets/geniesim-3.2` and Isaac assets under
`data/visual-assets/isaac-5.0`. These raw assets are not committed or
redistributed. GenieSimAssets is CC BY-NC-SA 4.0, so publication and downstream
reuse must retain attribution and satisfy its non-commercial/share-alike terms;
the official source and license links are recorded in `docs/reference_index.md`.

Regenerate the formal 1024-square garment views and 1280 x 960 task views from
the repository root with the scripts' default 256 path-tracing samples per
pixel:

```sh
.venv/bin/python scripts/paper/render_teaser_assets.py \
  --plan paper/figure1_render_plan.json \
  --output outputs/paper/figure1_renders/final_geniesim_table \
  --gpu 0
.venv/bin/python scripts/paper/render_teaser_frames.py \
  --plan paper/figure1_render_plan.json \
  --output outputs/paper/figure1_renders/final_geniesim_table \
  --gpu 0
```

Then assemble the figure:

```sh
.venv/bin/python scripts/paper/compose_teaser.py \
  --renders outputs/paper/figure1_renders/final_geniesim_table \
  --plan paper/figure1_render_plan.json \
  --output paper/figures/clothdojo-teaser.png
```

`figures/framework-overview-legacy.png` is preserved from the old project as a
design reference. It contains outdated Newton/folding terminology and is not
included by `main.tex`; revise it against the current ClothDojo pipeline before
using it in the paper.
