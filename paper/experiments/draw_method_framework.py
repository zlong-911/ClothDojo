"""Generate the publication method framework with audited typography."""
from pathlib import Path
import json
import math
import base64
import re
import subprocess
import tempfile
from io import BytesIO
from xml.etree import ElementTree
from fontTools import subset
from fontTools.ttLib import TTFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, Polygon

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/figures/clothdojo-method-framework"
FONT_DIR = ROOT.parents[1] / "tools/fonts/msttcorefonts"
LATEX_BIN = ROOT.parents[1] / "tools/latex-runtime/bin"
PDFCROP = LATEX_BIN / "pdfcrop"
PDFTEX = LATEX_BIN / "pdftex"
GHOSTSCRIPT = LATEX_BIN / "gs"
FONT_FILES = {"Regular":"Arial.TTF", "Bold":"Arialbd.TTF",
              "Italic":"Ariali.TTF", "BoldItalic":"Arialbi.TTF"}
for filename in FONT_FILES.values():
    font_manager.fontManager.addfont(str(FONT_DIR / filename))
plt.rcParams.update({
    "font.family":"Arial", "svg.fonttype":"none", "pdf.fonttype":42,
    "hatch.linewidth":.35, "mathtext.fontset":"custom",
    "mathtext.rm":"Arial", "mathtext.it":"Arial:italic",
    "mathtext.bf":"Arial:bold", "mathtext.bfit":"Arial:bold:italic", "mathtext.cal":"Arial",
    "mathtext.sf":"Arial", "mathtext.tt":"Arial",
})
# PDF uses 72 bp/in. The template's full-width insertion is 504 bp.
PAPER_WIDTH_MM = 177.8
W, H = PAPER_WIDTH_MM / 25.4 * 72, 210.
TITLE_SIZE, BODY_SIZE, SECONDARY_SIZE, SCRIPT_SIZE = 8.5, 7.5, 7., 6.3
FLOW_LABEL_SIZE = SECONDARY_SIZE
# Matplotlib 3.10.9 scales scripts through SHRINK_FACTOR. Preserve 6.3 pt
# scripts while recalculating display and module headings for the compact grid.
from matplotlib import _mathtext
_mathtext.SHRINK_FACTOR = SCRIPT_SIZE / BODY_SIZE
_mathtext.GROW_FACTOR = 1. / _mathtext.SHRINK_FACTOR
FRAME_LW, DETAIL_LW, FLOW_LW, ARROW_HEAD = .85, .60, .90, 3.6
INK, MUTED, EDGE, FLOW = "#1B2128", "#414B55", "#525D64", "#34434E"
WHITE, BAND, CONTROL = "#FFFFFF", "#ECE4D8", "#FFFFFF"
RAILS = ["#4D718C", "#88683F", "#955D51"]
STAGE_BACKGROUNDS = ["#F0F5F8", "#F8F4EC", "#F8F0EE"]
fig = plt.figure(figsize=(W/72,H/72), dpi=180, facecolor="white")
ax = fig.add_axes([0,0,1,1])
ax.set(xlim=(0,W),ylim=(H,0))
ax.axis("off")
labels, connectors, blocks, panels = [], [], [], []
script_labels = []
label_roles = {}

def text(x,y,value,size=BODY_SIZE,color=INK,bold=False,ha="left",role=None,**kwargs):
    a=ax.text(x,y,value,fontsize=size,color=color,va="center",ha=ha,
              weight="bold" if bold else "normal",zorder=20,**kwargs)
    labels.append(a)
    if role:
        label_roles[a] = role
    return a

def flow_text(x,y,value,ha="left",**kwargs):
    """Apply the single typography style used by every connector annotation."""
    return text(x,y,value,size=FLOW_LABEL_SIZE,color=FLOW,ha=ha,**kwargs)

def tau_label(x,y,sign,size=BODY_SIZE,color=INK):
    """Place the 6.3 pt sign close to an italic tau at the requested size."""
    tau = text(x,y,"τ",size=size,color=color,ha="center",fontstyle="italic")
    script = text(x+4.5,y-3.2,sign,size=SCRIPT_SIZE,color=color,ha="center")
    script_labels.append(script)
    return tau,script

def rect(x,y,w,h,fill=WHITE,edge=EDGE,lw=DETAIL_LW,z=1):
    ax.add_patch(Rectangle((x,y),w,h,facecolor=fill,edgecolor=edge,
                          linewidth=lw,joinstyle="miter",zorder=z))

def line(points,color=FLOW,lw=FLOW_LW,z=7):
    x,y=zip(*points)
    ax.plot(x,y,color=color,lw=lw,solid_capstyle="butt",solid_joinstyle="miter",zorder=z)

def arrow(points,color=FLOW):
    connectors.append(points)
    x0,y0=points[-2]; x1,y1=points[-1]
    length=math.hypot(x1-x0,y1-y0)
    ux,uy=(x1-x0)/length,(y1-y0)/length
    head=ARROW_HEAD
    base=(x1-head*.75*ux,y1-head*.75*uy)
    line([*points[:-1],base],color)
    ax.add_patch(Polygon([(x1,y1),(base[0]-uy*head/2,base[1]+ux*head/2),
                         (base[0]+uy*head/2,base[1]-ux*head/2)],
                         facecolor=color,edgecolor="none",zorder=10))

def module(x,w,title,rail,background):
    panels.append((x,4,w,178))
    # A borderless tint groups the stage without consuming layout space.
    rect(x,4,w,178,background,edge="none",z=-10)
    line([(x,23),(x+w,23)],rail,.65)
    text(x+4,13.5,title[:2],TITLE_SIZE,bold=True,color=rail)
    text(x+19,13.5,title[3:],TITLE_SIZE,bold=True)

def block(x,y,w,h,heading=None,heading_ha="left"):
    blocks.append((x,y,w,h))
    rect(x,y,w,h)
    if heading:
        text(x+w/2,y+7.5,heading,bold=True,ha="center",role="module_heading")

# Three aligned rows: conditions, execution, and learning.
# Preserve the center stage's width for the execution and correction loop.
module(4,120,"01 Human supervision",RAILS[0],STAGE_BACKGROUNDS[0])
module(132,250,"02 NOCS-guided trajectory generation",RAILS[1],STAGE_BACKGROUNDS[1])
module(390,110,"03 Visual augmentation",RAILS[2],STAGE_BACKGROUNDS[2])

block(26,29,76,38,"VR teleoperation",heading_ha="center")
for y,label in [(46.5,"Flatten / Fold"),(56.5,"Record demo")]:
    rect(32,y-.5,64,9,CONTROL,edge="none")
    text(64,y+4.5,label,size=SECONDARY_SIZE,ha="center",role="module_body")
expert_connector=[(64,67),(64,85)]
arrow(expert_connector)
expert_label=flow_text(57,76,r"$\mathcal{D}_E$",ha="right")
block(14,85,100,27)
text(64,93,"Supervised fine-tuning",bold=True,ha="center",role="module_heading")
text(64,105,"Pretrained VLA",size=SECONDARY_SIZE,color=MUTED,ha="center",role="module_body")
init_connector=[(114,98.5),(136,98.5)]
arrow(init_connector)
theta_init_label=flow_text(125,90,r"$\theta_{\mathrm{init}}$",ha="center")

block(136,29,152,38,"Policy inputs")
text(212,51,"NOCS images",size=SECONDARY_SIZE,ha="center",role="module_body")
text(212,61,"Robot state · instruction",size=SECONDARY_SIZE,ha="center",role="module_body")
arrow([(171,67),(171,85)])
observation_connector=[(256,85),(256,67)]
arrow(observation_connector)
observation_label=flow_text(263,76,"Observations")
block(136,85,70,27)
text(171,98.5,r"VLA policy $\pi_\theta$",bold=True,ha="center",role="module_heading")
block(224,85,64,27)
text(256,98.5,r"Simulator $s_t$",bold=True,ha="center",role="module_heading")
action_connector=[(206,98.5),(224,98.5)]
arrow(action_connector)
action_label=flow_text(215,90,r"$A_t$",ha="center")

block(295,29,83,38,"Task metrics")
text(336.5,51,"Shape · orientation",size=SECONDARY_SIZE,ha="center",role="module_body")
text(336.5,61,"Fold quality",size=SECONDARY_SIZE,ha="center",role="module_body")
arrow([(336.5,67),(336.5,85)])
block(295,85,83,27)
text(336.5,93,"Rollout screening",bold=True,ha="center",role="module_heading")
text(336.5,105,"Success / failure",size=SECONDARY_SIZE,ha="center",role="module_body")
arrow([(288,98.5),(295,98.5)])
success_connector=[(378,98.5),(394,98.5)]
arrow(success_connector)

block(14,133,100,45,"Rollout correction",heading_ha="center")
text(64,157,"Restore rollout state",size=SECONDARY_SIZE,ha="center",role="module_body")
text(64,168.5,"Record VR corrections",size=SECONDARY_SIZE,ha="center",role="module_body")
correction_connector=[(114,166.5),(136,166.5)]
arrow(correction_connector)
correction_label=flow_text(125,157.5,r"$\mathcal{D}_H$",ha="center")

# A pale, borderless region groups learning without competing with its arrows.
update_background=(132,133,250,49)
rect(*update_background,BAND,edge="none",z=0)
text(257,140.5,"Policy update with human corrections",bold=True,ha="center")
for x,w,heading in [
    (136,152,"Supervised fine-tuning"),
    (295,83,"State selection"),
]:
    block(x,151,w,27)
    text(x+w/2,157.5,heading,size=SECONDARY_SIZE,bold=True,ha="center",
         role="submodule_heading")
# Explicit baselines keep every child module centered with stable 8 bp leading.
text(212,170,r"$\mathcal{D}_E\,\cup\,\mathcal{D}_H$",size=SECONDARY_SIZE,color=MUTED,ha="center",role="module_body")
text(336.5,170,"Review failures",size=SECONDARY_SIZE,color=MUTED,ha="center",role="module_body")
state_connector=[(336.5,178),(336.5,188),(64,188),(64,178)]
arrow(state_connector)
update_connector=[(171,151),(171,112)]
arrow(update_connector)
theta_update_label=flow_text(
    178,123,"Updated policy")
rollout_connector=[(336.5,112),(336.5,151)]
arrow(rollout_connector)
rollout_label=flow_text(329.5,123,"Selected rollouts",ha="right")

block(394,29,102,38,"Domain randomization")
text(445,56,"Materials · scene · lighting",size=SECONDARY_SIZE,ha="center",role="module_body")
arrow([(445,67),(445,85)])
block(394,85,102,27)
text(445,93,"Trajectory re-rendering",bold=True,ha="center",role="module_heading")
text(445,105,"Selected successes",size=SECONDARY_SIZE,color=MUTED,ha="center",role="module_body")
arrow([(445,112),(445,133)])
block(394,133,102,45,"RGB policy training")
text(445,157,"RGB observations",size=SECONDARY_SIZE,color=MUTED,ha="center",role="module_body")
text(445,168.5,"Recorded actions",size=SECONDARY_SIZE,ha="center",role="module_body")


# Compact notation key: one baseline, consistent colon separators, no extra rule.
key_start = len(labels)
text(78,202,r"$\mathcal{D}_E$",ha="left")
text(90,202,": Expert demos",size=SECONDARY_SIZE)
text(150,202,r"$\mathcal{D}_H$",ha="left")
text(162,202,": Human corrections",size=SECONDARY_SIZE)

# Center the complete notation group using measured text widths.
fig.canvas.draw()
key_bounds = [label.get_window_extent(fig.canvas.get_renderer())
              for label in labels[key_start:]]
key_left = min(b.x0 for b in key_bounds)
key_right = max(b.x1 for b in key_bounds)
key_center = ax.transData.inverted().transform(((key_left+key_right)/2, 0))[0]
for label in labels[key_start:]:
    x, y = label.get_position()
    label.set_position((x + W/2 - key_center, y))

# As in v36, stage frames and three content rows share exact baselines.
assert len(panels)==3 and len({(y,h) for _,y,_,h in panels})==1

# Stage 01 centers its cards in the panel; state restoration has one return.
left_blocks = [b for b in blocks if b[0] < 112]
assert len(left_blocks) == 3
assert {x + w / 2 for x, _, w, _ in left_blocks} == {64}
assert [w for _, _, w, _ in left_blocks] == [76, 100, 100]
assert "Flatten / Fold" in {label.get_text() for label in labels}
assert "Flatten / fold" not in {label.get_text() for label in labels}

# Shared grid: both stage gaps, flow-facing insets, and bottom clearances match.
assert panels[1][0]-(panels[0][0]+panels[0][2]) == 8
assert panels[2][0]-(panels[1][0]+panels[1][2]) == 8
assert panels[0][0] == 4
assert abs(W-(panels[2][0]+panels[2][2])-4)<.01
assert (4+120)-(14+100) == 14-4 == 10
assert 136-132 == (132+250)-(295+83) == 4
assert 136-132 == (132+250)-(310+68) == 4
assert 394-390 == (390+110)-(394+102) == 4
assert update_background == (132,133,250,49)
for bottom_block in [(14,133,100,45),(136,151,152,27),(295,151,83,27),
                     (394,133,102,45)]:
    assert 182-(bottom_block[1]+bottom_block[3]) == 4

# Measure at final width. All sizes are physical PDF points at a 7-inch width.
fig.canvas.draw()
renderer=fig.canvas.get_renderer()
bounds=[]
for label in labels:
    b=label.get_window_extent(renderer)
    (x0,y0),(x1,y1)=ax.transData.inverted().transform([[b.x0,b.y1],[b.x1,b.y0]])
    assert x0>=0 and y0>=0 and x1<=W and y1<=H,(label.get_text(),(x0,y0,x1,y1))
    assert label.get_fontsize() >= (SCRIPT_SIZE if label in script_labels else SECONDARY_SIZE)
    bounds.append((label.get_text(),x0,y0,x1,y1))
for i,(a,x0,y0,x1,y1) in enumerate(bounds):
    for b,u0,v0,u1,v1 in bounds[i+1:]:
        assert min(x1,u1)-max(x0,u0)<.1 or min(y1,v1)-max(y0,v0)<.1,(a,b)

def on_block_border(point):
    px,py=point
    for x,y,w,h in blocks:
        on_vertical=(abs(px-x)<.01 or abs(px-(x+w))<.01) and y<=py<=y+h
        on_horizontal=(abs(py-y)<.01 or abs(py-(y+h))<.01) and x<=px<=x+w
        if on_vertical or on_horizontal:
            return True
    return False

for points in connectors:
    assert sum(math.dist(a,b) for a,b in zip(points,points[1:]))>=7,points
    assert len(points)==2 or points==state_connector,("unnecessary bent connector",points)
    assert on_block_border(points[0]),("detached source",points)
    assert on_block_border(points[-1]),("detached target",points)
    for (x0,y0),(x1,y1) in zip(points,points[1:]):
        for label,u0,v0,u1,v1 in bounds:
            if x0==x1:
                assert not(u0<x0<u1 and min(max(y0,y1),v1)>max(min(y0,y1),v0)),label
            elif y0==y1:
                assert not(v0<y0<v1 and min(max(x0,x1),u1)>max(min(x0,x1),u0)),label
for artist,(label,x0,y0,x1,y1) in zip(labels,bounds):
    cx,cy=(x0+x1)/2,(y0+y1)/2
    containing=[]
    for x,y,w,h in blocks:
        if x<cx<x+w and y<cy<y+h:
            containing.append((w*h,x,y,w,h))
            clearance=min(x0-x,x+w-x1,y0-y,y+h-y1)
            assert clearance>=1.5,(label,clearance)
    if label_roles.get(artist) in {"module_heading","submodule_heading","module_body"}:
        assert containing,label
        _,x,y,w,h=min(containing)
        horizontal_padding=min(x0-x,x+w-x1)
        assert horizontal_padding>=6,(label,horizontal_padding)
        assert artist.get_ha()=="center",(label,artist.get_ha())
        assert abs((x0+x1)/2-(x+w/2))<.1,(label,(x0+x1)/2,x+w/2)
for label,x0,y0,x1,y1 in bounds:
    if y0 < 13.5 < y1:
        panel = next(p for p in panels if p[0] <= (x0+x1)/2 <= p[0]+p[2])
        assert x0 >= panel[0]+3.5 and x1 <= panel[0]+panel[2]-1.5, label

# Audit every explicit connector annotation after math layout is resolved.
# The initialization label must also stay entirely inside its connector gap.
bound_by_artist={artist:bound for artist,bound in zip(labels,bounds)}
_,init_x0,init_y0,init_x1,init_y1=bound_by_artist[theta_init_label]
_,update_x0,update_y0,update_x1,update_y1=bound_by_artist[theta_update_label]
assert init_x0>=114 and init_x1<=136,("theta_init gap",init_x0,init_x1)
assert update_y0>=112 and update_y1<=133,("theta_update band",update_y0,update_y1)

def annotation_gap(artists,points):
    """Shortest distance between a label's union bbox and an axis-aligned edge."""
    if not isinstance(artists,(tuple,list)):
        artists=(artists,)
    boxes=[bound_by_artist[artist][1:] for artist in artists]
    bx0=min(box[0] for box in boxes); by0=min(box[1] for box in boxes)
    bx1=max(box[2] for box in boxes); by1=max(box[3] for box in boxes)
    distances=[]
    for (x0,y0),(x1,y1) in zip(points,points[1:]):
        if x0==x1:
            dx=0 if bx0<=x0<=bx1 else min(abs(x0-bx0),abs(x0-bx1))
            lo,hi=sorted((y0,y1))
            dy=0 if by0<=hi and by1>=lo else min(abs(lo-by1),abs(by0-hi))
        else:
            dy=0 if by0<=y0<=by1 else min(abs(y0-by0),abs(y0-by1))
            lo,hi=sorted((x0,x1))
            dx=0 if bx0<=hi and bx1>=lo else min(abs(lo-bx1),abs(bx0-hi))
        distances.append(math.hypot(dx,dy))
    return min(distances)

flow_annotations=[
    (expert_label,expert_connector,"side"),
    (theta_init_label,init_connector,"parallel"),
    (observation_label,observation_connector,"side"),
    (action_label,action_connector,"parallel"),
    (correction_label,correction_connector,"parallel"),
    (theta_update_label,update_connector,"side"),
    (rollout_label,rollout_connector,"side"),
]
gap_ranges={"parallel":(3.5,4.6),"side":(6.5,7.5)}
for artists,points,placement in flow_annotations:
    gap=annotation_gap(artists,points)
    low,high=gap_ranges[placement]
    assert low<=gap<=high,("connector annotation distance",gap,placement,points)
    for artist in artists if isinstance(artists,(tuple,list)) else (artists,):
        expected_size=SCRIPT_SIZE if artist in script_labels else FLOW_LABEL_SIZE
        assert artist.get_fontsize()==expected_size
        assert artist.get_color()==FLOW
assert correction_connector in connectors
assert update_connector in connectors
assert state_connector in connectors
assert rollout_connector in connectors
raw_pdf = OUT.with_name(OUT.name + "-uncropped.pdf")
fig.savefig(raw_pdf,dpi=450,facecolor="white")
for ext in ("svg","png"):
    fig.savefig(OUT.with_suffix("."+ext),dpi=450,facecolor="white")

# Crop the vertical canvas while retaining the deliberate 4 bp side insets.
# The minimal TeX Live runtime has no prebuilt pdfTeX format, so the temporary
# wrapper invokes pdfTeX in ini mode; pdfcrop's generated file uses primitives
# only and therefore needs no format file.
with tempfile.TemporaryDirectory(prefix=".method-framework-pdfcrop-",
                                 dir=OUT.parent) as temp_dir:
    temp_dir = Path(temp_dir)
    pdftex_wrapper = temp_dir / "pdftex-ini"
    pdftex_wrapper.write_text(
        "#!/bin/sh\nexec \"" + str(PDFTEX) + "\" -ini \"$@\"\n")
    pdftex_wrapper.chmod(0o755)
    cropped_pdf = temp_dir / "cropped.pdf"
    subprocess.run([
        str(PDFCROP), "--quiet", "--hires", "--margins", "3.99 0 3.99 0",
        "--gscmd", str(GHOSTSCRIPT), "--pdftexcmd", str(pdftex_wrapper),
        str(raw_pdf), str(cropped_pdf),
    ], cwd=ROOT, check=True)
    cropped_pdf.replace(OUT.with_suffix(".pdf"))
raw_pdf.unlink()

pdf_bytes = OUT.with_suffix(".pdf").read_bytes()
media_box = re.search(
    rb"/MediaBox\s*\[\s*0\s+0\s+([0-9.]+)\s+([0-9.]+)\s*\]", pdf_bytes)
assert media_box, "cropped PDF MediaBox not found"
cropped_width, cropped_height = map(float, media_box.groups())
insertion_scale = W / cropped_width
assert abs(insertion_scale - 1) < 1e-4

# Keep SVG text editable while preventing viewer-dependent font substitution.
svg_path = OUT.with_suffix(".svg")
svg_text = svg_path.read_text()
svg_tree = ElementTree.fromstring(svg_text)
used_text = "".join("".join(e.itertext()) for e in svg_tree.iter()
                    if e.tag.endswith("text"))
css = []
for weight, style, filename in [(400,"normal","Regular"),
                                (700,"normal","Bold"),
                                (400,"italic","Italic"),
                                (700,"italic","BoldItalic")]:
    font = TTFont(FONT_DIR / FONT_FILES[filename])
    missing = {ord(c) for c in used_text if ord(c)>32} - set(font.getBestCmap())
    assert not missing, (filename, missing)
    options = subset.Options()
    options.flavor = "woff"
    cutter = subset.Subsetter(options=options)
    cutter.populate(text=used_text)
    cutter.subset(font)
    font.flavor = "woff"
    stream = BytesIO()
    font.save(stream)
    data = base64.b64encode(stream.getvalue()).decode("ascii")
    css.append("@font-face{font-family:'Arial';font-style:"+style+
               ";font-weight:"+str(weight)+";src:url(data:font/woff;base64,"+
               data+") format('woff');}")
svg_text = svg_text.replace("<defs>", '<defs><style type="text/css">'+
                            "".join(css)+"</style>", 1)
svg_path.write_text(svg_text)


OUT.with_suffix(".sources.json").write_text(json.dumps({
    "content_source":["paper/draft_zh.md, sections 3–4","src/clothdojo/training/hil_sft.py","configs/training/pi05_aloha_openpi_rlinf_hil800_aggregate_fullonly_wd001_b256_strict_40k.yaml"],
    "code_audit":"Embedded post-layout assertions in paper/experiments/draw_method_framework.py",
    "authoring_canvas_pt":[W,H],
    "cropped_pdf_size_bp":[cropped_width,cropped_height],
    "cropped_pdf_width_mm":cropped_width / 72 * 25.4,
    "pdfcrop":{"version":"1.42","hires":True,"margins_bp":[3.99,0,3.99,0]},
    "font_sizes_pt":{"main_heading":round(TITLE_SIZE*insertion_scale,3),"module_heading":round(BODY_SIZE*insertion_scale,3),"secondary_module_text":round(SECONDARY_SIZE*insertion_scale,3),"connector_annotation":round(FLOW_LABEL_SIZE*insertion_scale,3),"math_scripts":round(SCRIPT_SIZE*insertion_scale,3)},
    "font_note":"Every connector annotation uses 7 pt regular Arial in FLOW color. Horizontal-shaft annotations keep 3.5–4.6 pt bbox clearance; vertical-shaft annotations keep 6.5–7.5 pt.",
    "font_conversion":f"Final font size = source font size * 504 / {cropped_width:.3f}. The pdfcrop side margins preserve the 504 bp design width; scale={insertion_scale:.8f}, so all sizes round to their authored values at 0.001 pt.",
    "font_family":"Arial",
    "font_directory":str(FONT_DIR),
    "svg_font_embedding":"Subset Arial WOFF Regular, Bold, Italic, BoldItalic",
    "design_references":["paper/figures/method-style-reference-1.png",
        "paper/figures/method-style-reference-2.png"],
    "caption":"paper/figures/clothdojo-method-framework.caption.tex",
    "publication_pdf":"paper/main.pdf",
    "typography_spec":"paper/experiments/method_framework_typography.md",
    "design":"Correction aggregation and supervised fine-tuning replace the former advantage-conditioned update. A bottom return connects selected rollout states to VR correction. The original three-stage palette, Arial typography, and 504 bp width are preserved; the canvas is 210 pt tall.",
    "palette":{"background":WHITE,"stage_backgrounds":STAGE_BACKGROUNDS,"update_group":BAND,"edge":EDGE,"side_bars":RAILS},
    "terminology_audit":"paper/experiments/method_framework_terminology_audit.md",
    "terminology_sources":["https://arxiv.org/html/2504.16054v1",
        "https://arxiv.org/html/1703.06907v1",
        "https://arxiv.org/html/1810.02890v1"],
    "formal_figure_replaced":True,
},ensure_ascii=False,indent=2)+"\n")
print(OUT)
