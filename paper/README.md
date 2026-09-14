# ClothDojo paper

[`draft_zh.md`](draft_zh.md) is the authoritative Chinese content draft for
ICRA 2027. It includes verified preliminary results and explicitly marks
unfinished implementation and experiments.
[The EVAL-800 v2 experiment record](experiments/eval800_v2_large_support_20260908.md)
contains the four-checkpoint comparison after large-support target correction,
expert threshold recalibration, full metrics, and source links; its 90-second
terminal-success results remain historical evidence. Section 4 of the Chinese
draft was revised on 2026-09-10 around the current PDF's 60-second flattening
and 40-second folding tables and quality figure. It provides explicitly
unfinished drafts for downstream RGB generalization, data-source comparisons,
and failure analysis. Fold DAgger is marked as an abnormal training run under
investigation; its observations are not interpreted as a method trade-off.
This experiment revision awaits English translation.
[`submission_plan_zh.md`](submission_plan_zh.md) records the evidence boundaries,
ten-day completion priorities, and eight-page content budget.
The Chinese draft is the current content source. Its Method was reorganized
on 2026-09-10 around task definitions, the NOCS policy interface, learning from
demonstrations and corrections, and RGB data generation. Method explains the
model base, mirror transformation, and collection of folding demonstrations
from restored flattening demonstration terminal states. Asset coverage,
collected trajectories, exported training segments, and training budgets are
collected in Section 4.2. The [demonstration inventory audit](experiments/demonstration_inventory_20260910.md)
distinguishes 889 flattening source trajectories from their 919 training
segments, and verifies coverage of 182 flattening and 156 folding assets.
The [experiment priorities](experiments/experiment_priorities_zh.md) record the
agreed scope with fewer than five days remaining: randomized-appearance
expert-only versus expert-plus-rollout training, flattening before folding,
independent closed-loop evaluation, and an optional per-asset trajectory-scale
comparison. Fixed-appearance controls and multi-model benchmarking are deferred.
This revision and the updated data-completion claims await synchronization
with the English LaTeX and PDF.

[`method_draft_zh.md`](method_draft_zh.md) preserves the earlier Method discussion
and implementation notes; use Section 3 of `draft_zh.md` for current prose.
Evaluation metrics and protocols open Section 4. The
folding protocol uses the repository's current five-feature learned quality
score and fixed three-second success rule; the retired geometric two-threshold
prototype is not a paper method. The unified English Method is implemented in
`sections/method.tex` and included by `main.tex`; it predates the latest Chinese
Method rewrite, as do the English experiments and conclusion.
The [method narrative review and figure plan](experiments/method_narrative_review_20260908.md)
records the final review of this discussion draft, the roles of Figures 1–3,
and a proposed compact task-definition figure.

[`writing_translation_guide_zh.md`](writing_translation_guide_zh.md) records the
ICRA 2027 submission constraints, conference-style editing principles, and the
Chinese-to-English translation checklist and terminology decisions for this
paper.

`docs/benchmark.md` remains the detailed experiment-planning document. The
paper draft explains the scientific argument; the planning document owns
operational choices that may still change.

## Build

The draft uses the IEEE Robotics conference `ieeeconf` format. The unmodified
`ieeeconf.cls` from PaperCept's official template archive is included in this
directory. Template and compiler references are listed in
[`docs/reference_index.md`](../docs/reference_index.md#论文排版).

Run from this directory:

```sh
make paper
```

The Makefile automatically uses the persistent Tectonic runtime at
`../../../tools/latex-runtime/bin/tectonic` when it is available, and otherwise
falls back to `tectonic` on `PATH`. Builds use the local package cache and do
not require network access.

Alternatively, Tectonic downloads the required standard packages automatically:

```sh
tectonic --keep-logs main.tex
```

The shared workspace has Tectonic 0.17.0 installed at
`/cfs/mlp/training/docker/user/hadoop-aipnlp/huangzilong09/tools/latex-runtime/bin/tectonic`.
Both workflows write `main.pdf` in this directory.
After the first successful Tectonic build, `tectonic --only-cached --keep-logs
main.tex` rebuilds without network access. `references.bib` contains the works
cited throughout the current draft, with metadata checked against primary
sources. Bibliography output is enabled in `main.tex`; the unmodified CTAN
`IEEEtran.bst` v1.14 is included for offline builds.

The [ICRA 2027 call for papers](https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/)
sets an eight-page limit for the complete PDF, including references, and
requires double-anonymous review. Separate written supplementary material is
not accepted. The author memo records the submission-specific constraints.

## Figures

`figures/clothdojo-framework.pdf` is the full-width first-page framework figure.
It is a tightly cropped hybrid-vector PDF: rendered robot and garment frames
and illustrative evaluation charts remain raster images, while typography and
the evaluation border are vector content. Regenerate it with
`scripts/paper/compose_figure1_homepage.py` and its `--pdf-output` option.
Typography uses embedded Arial: module titles are 9 pt bold, the title-case
Flatten/Fold row labels are 9 pt regular, header explanations are 7 pt,
and evaluation labels are 8 pt. Module titles and explanations share one header line.
The right column contains Data Library and Evaluation; the latter shows three
illustrative charts, with its right edge aligned to the data grid and its bottom
edge aligned to module 02. The PDF page is cropped to the visible content with
zero added padding. Its native page is 503×197.3 pt; inclusion at the manuscript's
177.8 mm width scales the nominal 7/8/9 pt fonts by 504/503 (about 0.2%).
The final file is processed with `pdfcrop` 1.42 and Ghostscript from the persistent
`tools/latex-runtime/bin` directory. Because the full-page raster background is
included in Ghostscript's automatic bounds, pass visible-content bounds through
`--bbox` and use `--margins 0`. The complete rebuild commands and measured bounds
are recorded in `outputs/paper/figure1_arial_20260908/typography.md`; intermediate
files are generated in a temporary directory and removed after the build.

`figures/clothdojo-manipulation-sequences.pdf` is Figure 2 in `main.tex`.
It contains two flattening excerpts and two folding sequences, each with eight
chronological frames. Behavior groups use black labels and 0.7 pt rectangular
borders. Each title sits in a gap in the top edge, with 0.8 mm clearance on
each side of the text; the frame extends into the white title band.
It uses Arial Regular for all labels: Flatten/Fold are 9 pt in the manuscript,
while behavior titles and Sample/Time are 8 pt. The nominal 178 mm figure uses
9.0101 pt and 8.0090 pt respectively, and is included at the template's
177.8 mm text width. The title band is 4.8 mm high; axis arrows and title-edge
gaps follow the measured text bounds.
The font is loaded explicitly from the persistent file
`tools/fonts/msttcorefonts/Arial.TTF`, relative to the directory containing
`repos/`. It comes from Microsoft's Core Fonts for the Web
[Arial package](https://sourceforge.net/projects/corefonts/files/the%20fonts/final/arial32.exe/download).
The font file SHA-256 is
`35c0f3559d8db569e36c31095b8a60d441643d95f59139de40e23fada819b833`.
Black axes use 0.7 pt strokes and sit 2.2 mm outside the image grid. The
flattening/folding separator is dashed with the same 0.7 pt stroke. Text and
lines are vector content with the font embedded in the PDF.
Regenerate the figure, PNG, source manifest, and page preview from the repository
root using the existing rendered frames:

```sh
python paper/experiments/build_event_four_by_seven.py --eight
```

Then run `make -C paper paper` to rebuild the manuscript.
The eight-frame render plan is `experiments/event_four_by_eight_20260908.json`;
frame choices and behavior boundaries are recorded in
`experiments/event_four_by_eight_selection_20260908.md`.

`figures/clothdojo-method-framework.pdf` is Figure 3 in `main.tex`, placed
across both columns at the top of page 3 in the current draft. Its native width
after `pdfcrop` is 503.998 bp, matching the 504 bp `\textwidth` to within
0.002 bp. Its embedded Arial text therefore retains the calibrated
8.5/7.5/7/6.3 pt hierarchy. The caption is read from the adjacent
`.caption.tex` file. This figure presents human supervision, NOCS-guided trajectory generation
with a correction-aggregation and supervised fine-tuning loop, and visual
augmentation for RGB policy training. The bottom return arrow links selected
rollout states to VR correction; autonomous actions do not enter this update.
Regenerate the PDF, SVG, PNG, and source manifest with
`python paper/experiments/draw_method_framework.py` from the repository root
using a Python environment with Matplotlib and fontTools installed.

`figures/clothdojo-teaser.png` is the earlier full-width teaser assembled from
selected lossless RTX renders. It remains available as a design and rendering
reference but is no longer included by `main.tex`.

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
