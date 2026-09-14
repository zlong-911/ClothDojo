"""Prepare and compose curated replay frames; --eight selects the revised 4x8 layout."""
import json
import re
import runpy
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EIGHT = '--eight' in sys.argv
STEM = 'event_four_by_eight_20260908' if EIGHT else 'event_four_by_seven_actions_v5_20260907'
PLAN = ROOT / 'paper/experiments' / (STEM + '.json')
SELECTIONS = [
    ('flatten_event_nine_frames_20260907', 0, [75, 150, 325, 450, 660, 705, 720]),
    ('flatten_event_nine_frames_20260907', 1, [650, 865, 1670, 1800, 3350, 3400, 3435]),
    ('fold_nine_frames_20260907', 0, [125, 275, 550, 725, 810, 925, 975]),
    ('fold_nine_frames_20260907', 2, [100, 325, 695, 865, 1005, 1050, 1100]),
]
if EIGHT:
    SELECTIONS = [
        ('flatten_event_nine_frames_20260907', 0, [75, 150, 325, 450, 660, 705, 720, 975]),
        ('flatten_event_nine_frames_20260907', 1, [650, 865, 1670, 1800, 2200, 3350, 3400, 3435]),
        ('fold_nine_frames_20260907', 0, [125, 200, 275, 550, 725, 810, 925, 975]),
        ('fold_nine_frames_20260907', 2, [100, 325, 695, 750, 865, 1005, 1050, 1100]),
    ]

if '--prepare' in sys.argv:
    jobs = []
    for stem, index, frames in SELECTIONS:
        source = json.loads((ROOT/'paper/experiments'/f'{stem}.json').read_text())
        job = source['jobs'][index]
        job['frames'] = frames
        job['paper_backface_scale'] = 0.25
        jobs.append(job)
    PLAN.write_text(json.dumps({
        'schema': 'clothdojo_paper_teaser_render_v1', 'seed': 20260907,
        'source_batch_status': 'Flatten batch remains provisional, matching previously inspected videos.',
        'jobs': jobs,
    }, indent=2) + '\n')
else:
    unified = '--unified' in sys.argv or EIGHT
    details = '--details' in sys.argv or unified
    phases = '--phases' in sys.argv or details
    sys.argv = ['compare_event_layouts.py', '--rows', '4', '--cols', '8' if EIGHT else '7', '--plan-stem', STEM]
    if phases:
        sys.argv.append('--phases')
    if details:
        sys.argv.append('--details')
    if unified:
        sys.argv.append('--unified')
    runpy.run_path(str(ROOT/'paper/experiments/compare_event_layouts.py'), run_name='__main__')
    figure_stem = STEM + ('-unified-blue-red' if unified else '-details' if details else '-phases' if phases else '')
    if unified:
        figure_stem = 'clothdojo-manipulation-sequences'
    paper = ROOT/'paper'
    main = (paper/'main.tex').read_text()
    label = main.index('\\label{fig:folding-trajectories}')
    start = main.rindex('\\begin{figure*}', 0, label)
    block = re.sub(r'(\\includegraphics\[width=\\textwidth\]\{)figures/[^}]+',
                   lambda match: match[1] + f'figures/{figure_stem}.pdf',
                   main[start:label], count=1)
    main = main[:start] + block + main[label:]
    source = paper/f'{figure_stem}_preview.tex'
    source.write_text(main)
    build = paper/'build-layout-options'
    build.mkdir(exist_ok=True)
    binary = ROOT.parents[1]/'tools/latex-runtime/bin'
    subprocess.run([str(binary/'tectonic'), '-C', '--outdir', str(build), str(source)],
                   cwd=paper, check=True, capture_output=True)
    pdf = build/source.with_suffix('.pdf').name
    subprocess.run([str(binary/'pdftoppm'), '-f', '2', '-l', '2', '-scale-to', '1600',
                    '-png', '-singlefile', str(pdf), str(paper/'figures'/f'{figure_stem}-page2')], check=True)
