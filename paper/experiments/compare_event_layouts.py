"""Two flatten event rows and four folding rows on shared sample/time axes."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import font_manager
from matplotlib.patches import Rectangle, FancyBboxPatch

parser=argparse.ArgumentParser()
parser.add_argument('--rows',type=int,choices=[3,4,5,6],required=True)
parser.add_argument('--cols',type=int,choices=[7,8,9],required=True)
parser.add_argument('--plan-stem')
parser.add_argument('--phases',action='store_true')
parser.add_argument('--details',action='store_true')
parser.add_argument('--unified',action='store_true')
args=parser.parse_args()
if args.unified:
    args.details=True
if args.details:
    args.phases=True
ROOT=Path(__file__).resolve().parents[2]
rows=[]
sources=[]
for stem in ([args.plan_stem] if args.plan_stem else ['flatten_event_nine_frames_20260907','fold_nine_frames_20260907']):
    plan_path=ROOT/'paper/experiments'/f'{stem}.json'
    render_root=ROOT/'outputs/paper'/stem
    plan=json.loads(plan_path.read_text())
    manifest=json.loads((render_root/'manifest.json').read_text())
    records={(r['job'],r['frame_index']):r for r in manifest['records']}
    sources.append({'plan':str(plan_path.relative_to(ROOT)),
                    'manifest':str((render_root/'manifest.json').relative_to(ROOT))})
    for job in plan['jobs']:
        assert len(job['frames'])==(args.cols if args.plan_stem else 9)
        assert job['frames']==sorted(set(job['frames']))
        frames=[records[job['name'],f] for f in job['frames']]
        rows.append((job,render_root,frames))
assert len(rows)==(args.rows if args.plan_stem else 6)
reference=rows[0][2][0]
for job,root,frames in rows:
    for frame in frames:
        assert frame['camera_spec']==reference['camera_spec']
        assert frame['appearance']['lighting']==reference['appearance']['lighting']
        assert frame['appearance']['room']==reference['appearance']['room']
        assert frame['paper_table_material']==reference['paper_table_material']

row_ids=list(range(args.rows)) if args.plan_stem else {3:[0,1,2],4:[0,1,2,4],5:[0,1,2,3,4],6:[0,1,2,3,4,5]}[args.rows]
frame_ids=list(range(args.cols)) if args.plan_stem else {7:[0,1,3,4,6,7,8],8:[0,1,3,4,5,6,7,8],9:list(range(9))}[args.cols]
rows=[(rows[i][0],rows[i][1],[rows[i][2][k] for k in frame_ids]) for i in row_ids]

if args.unified:
    font_path=ROOT.parents[1]/'tools/fonts/msttcorefonts/Arial.TTF'
    font_manager.fontManager.addfont(str(font_path))
    font_family='Arial'
    assert font_manager.FontProperties(fname=str(font_path)).get_name()==font_family
else:
    for weight in ['Regular','Bold']:
        font_manager.fontManager.addfont(f'/usr/share/fonts/truetype/lato/Lato-{weight}.ttf')
    font_family='Lato'
plt.rcParams.update({'font.family':font_family,'svg.fonttype':'none','pdf.fonttype':42})
width=178.0
# The current ieeeconf paper includes this figure at its 7-inch text width.
paper_width_mm=177.8
font_scale=width/paper_width_mm
title_size_pt=8.0*font_scale if args.unified else 7
axis_text_pt=8.0*font_scale if args.unified else 6.5
task_text_pt=9.0*font_scale if args.unified else 7.5
axis_lw=0.7 if args.unified else 0.55
axis_x,axis_gap=1.5,1.3
if args.unified:
    axis_x,axis_gap=1.9,2.2
left=axis_x+axis_gap
gap,rowgap,groupgap=0.2,1.1,2.2
if args.details:
    gap,rowgap,groupgap=0.65,1.5,2.4
if args.unified:
    gap=0.9
    groupgap=4.0
right=0.35 if args.unified else 0
cw=(width-left-right-(args.cols-1)*gap)/args.cols
ch=cw*3/4
title_h=3.8 if args.phases else 0
if args.unified:
    title_h=4.8
ys=[title_h+i*(ch+title_h+rowgap)+(groupgap-rowgap if i>=2 else 0) for i in range(args.rows)]
bottom=ys[-1]+ch
height=bottom+axis_gap+1.8
fig=plt.figure(figsize=(width/25.4,height/25.4),facecolor='white',
               dpi=600 if args.unified else 100)
ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,width),ylim=(height,0));ax.axis('off')
if args.phases:
    ax.set_zorder(10)
    ax.patch.set_visible(False)
phases=[
    [(0,5,'Regrasp'),(5,7,'Fling')],
    [(0,2,'Lift & fling'),(2,4,'Unfold right sleeve'),(4,7,'Unfold left sleeve')],
    [(0,3,'Fold sides'),(3,4,'Fold in half'),(4,7,'Refine fold')],
    [(0,2,'Fold hem'),(2,3,'Fold hood'),(3,4,'Fold in half'),(4,7,'Refine fold')],
]
if args.details:
    phases=[
        [(0,5,'Regrasp in midair'),(5,7,'Fling open')],
        [(2,4,'Unroll & stretch'),(4,7,'Reveal, regrasp & toss')],
        [(3,5,'Switch hands to adjust'),(5,7,'Adjust after folding')],
        [(4,7,'Adjust after folding')],
    ]
if args.unified:
    phases=[
        [(0,5,'Regrasp in midair'),(5,7,'Fling open')],
        [(0,2,'Lift & fling'),(2,4,'Unroll & stretch'),(4,7,'Reveal, regrasp & toss')],
        [(0,3,'Fold sleeves & sides'),(3,5,'Switch hands to adjust'),(5,7,'Adjust after folding')],
        [(0,2,'Bring hem upward'),(2,4,'Fold hood, then body'),(4,7,'Adjust after folding')],
    ]
    if args.cols==8:
        phases=[
            [(0,5,'Regrasp in midair'),(5,7,'Fling open'),(7,8,'Adjust sleeve')],
            [(0,2,'Lift & fling'),(2,4,'Unroll & stretch'),(4,5,'Reposition'),(5,8,'Reveal, regrasp & toss')],
            [(0,2,'Fold hood'),(2,4,'Fold sleeves & sides'),(4,6,'Switch hands to adjust'),(6,8,'Adjust after folding')],
            [(0,2,'Bring hem upward'),(2,4,'Fold hood'),(4,5,'Rotate'),(5,6,'Fold body'),(6,8,'Adjust after folding')],
        ]
group_borders=[]
for row,(_,root,frames) in enumerate(rows):
    for col,record in enumerate(frames):
        im=mpimg.imread(root/record['image']);assert im.shape[:2]==(960,1280)
        x=left+col*(cw+gap)
        a=fig.add_axes([x/width,1-(ys[row]+ch)/height,cw/width,ch/height])
        a.imshow(im);a.axis('off')
    if args.phases:
        for group_index,(start,end,title) in enumerate(phases[row]):
            x=left+start*(cw+gap)
            w=(end-start)*cw+(end-start-1)*gap
            color=['#176F79','#A54B28'][group_index%2] if args.details else 'black'
            if args.unified:
                color='black'
            elif args.details:
                ax.add_patch(FancyBboxPatch((x,ys[row]),w,ch,
                    boxstyle='round,pad=0.10,rounding_size=0.65',fill=False,
                    edgecolor=color,linewidth=0.95,clip_on=False))
            else:
                ax.add_patch(Rectangle((x,ys[row]),w,ch,fill=False,edgecolor=color,linewidth=0.45))
            title_artist=ax.text(x+w/2,ys[row]-title_h/2,title,ha='center',va='center',
                    fontsize=title_size_pt,
                    fontweight='normal' if args.unified else 'bold',color=color)
            if args.unified:
                group_borders.append((x,ys[row]-title_h/2,w,ys[row]+ch,title_artist))
    if row<args.rows-1 and (not args.phases or row==1):
        sy=(ys[row]+ch+ys[row+1]-title_h)/2
        if args.unified:
            # Center the separator in the visible gap, accounting for the
            # space above the next row's label within its title band.
            label_height_mm=title_size_pt*25.4/72
            sy+=(title_h-label_height_mm)/4
        ax.plot([left,width],[sy,sy],color='black',
                lw=axis_lw if args.unified else 0.65 if row==1 else 0.25,
                linestyle=(0,(3,3)) if args.unified or row!=1 else '-',
                solid_capstyle='butt',dash_capstyle='butt')

arrow={'arrowstyle':'->','color':'black','lw':axis_lw,'mutation_scale':7 if args.unified else 6,'shrinkA':0,'shrinkB':0}
origin=(axis_x,bottom+axis_gap)
time_label=ax.text(width-max(right,0.35),origin[1],'Time',fontsize=axis_text_pt,color='black',ha='right',va='center')
sample_label=ax.text(axis_x,4.1,'Sample',fontsize=axis_text_pt,color='black',rotation=90,ha='center',va='center')
labels=[ax.text(axis_x,(ys[0]+ys[1]+ch)/2,'Flatten',fontsize=task_text_pt,color='black',
                rotation=90,ha='center',va='center'),
        ax.text(axis_x,(ys[2]+bottom)/2,'Fold',fontsize=task_text_pt,color='black',
                rotation=90,ha='center',va='center')]
# Break the axis around the actual text bounds, without covering it with a box.
fig.canvas.draw()
renderer=fig.canvas.get_renderer()
time_arrow_x=width-6.5
sample_arrow_y=8.4
if args.unified:
    time_box=time_label.get_window_extent(renderer).transformed(ax.transData.inverted())
    time_arrow_x=time_box.x0-0.9
    sample_box=sample_label.get_window_extent(renderer).transformed(ax.transData.inverted())
    sample_top,sample_bottom=sorted([sample_box.y0,sample_box.y1])
    sample_shift=max(0,0.45-sample_top)
    sample_label.set_y(sample_label.get_position()[1]+sample_shift)
    sample_arrow_y=sample_bottom+sample_shift+0.65
ax.annotate('',xy=(time_arrow_x,origin[1]),xytext=origin,arrowprops=arrow)
# Extend the frame into the white title band and break its top edge around
# the measured text, using the same open-line treatment as the sample axis.
for x,top,w,frame_bottom,label in group_borders:
    box=label.get_window_extent(renderer).transformed(ax.transData.inverted())
    text_left,text_right=box.x0-0.8,box.x1+0.8
    if not x<text_left<text_right<x+w:
        raise ValueError(f'Group label does not fit its frame: {label.get_text()}')
    ax.plot([text_left,x,x,x+w,x+w,text_right],
            [top,top,frame_bottom,frame_bottom,top,top],
            color='black',lw=axis_lw,solid_capstyle='butt',
            solid_joinstyle='miter',clip_on=False)
breaks=[]
for label in labels:
    box=label.get_window_extent(renderer).transformed(ax.transData.inverted())
    lo,hi=sorted([box.y0,box.y1]);breaks.append((lo-0.65,hi+0.65))
cursor=sample_arrow_y+0.4
for lo,hi in sorted(breaks):
    ax.plot([axis_x,axis_x],[cursor,lo],color='black',lw=axis_lw,solid_capstyle='butt')
    cursor=hi
ax.plot([axis_x,axis_x],[cursor,origin[1]],color='black',lw=axis_lw,solid_capstyle='butt')
ax.annotate('',xy=(axis_x,sample_arrow_y),xytext=(axis_x,sample_arrow_y+1.0),arrowprops=arrow)

dest=ROOT/'paper/figures/layout-options'/f'layout-{args.rows}x{args.cols}'
if args.plan_stem:
    dest=ROOT/'paper/figures'/args.plan_stem
if args.phases:
    dest=dest.with_name(dest.name+('-unified-blue-red' if args.unified else '-details' if args.details else '-phases'))
if args.unified:
    dest=ROOT/'paper/figures/clothdojo-manipulation-sequences'
dest.parent.mkdir(parents=True,exist_ok=True)
for ext in ['png','pdf']:
    fig.savefig(dest.with_suffix('.'+ext),dpi=600,facecolor='white')
dest.with_suffix('.sources.json').write_text(json.dumps({
    'sources':sources,'phases':phases if args.phases else None,'row_indices':row_ids,'frame_indices_within_nine':frame_ids,'physical_size_mm':[width,height],'frame_width_mm':cw,'crop':None,
    'rows':[{'name':j['name'],'trajectory':j['trajectory'],'frames':[f['frame_index'] for f in frames]} for j,_,frames in rows],
    'layout':{'font':font_family,'column_gap_mm':gap,'row_gap_mm':rowgap,'group_gap_mm':groupgap,
              'right_margin_mm':right,
              'group_border_colors':['#000000'] if args.unified else None,
              'group_color_order':'uniform' if args.unified else None,
              'group_border_pt':axis_lw if args.unified else None,
              'group_top_edge':'broken_around_title' if args.unified else 'continuous',
              'title_edge_gap_mm':0.8 if args.unified else None,
              'title_color':'matches_group_border',
              'title_size_pt':title_size_pt,
              'title_weight':'normal' if args.unified else 'bold',
              'axis_text_color':'#000000','axis_text_size_pt':axis_text_pt,
              'task_text_size_pt':task_text_pt,'axis_text_weight':'normal',
              'paper_width_mm':paper_width_mm if args.unified else None,
              'paper_text_size_pt':{'title':8.0,'axis':8.0,'task':9.0} if args.unified else None,
              'title_band_height_mm':title_h,
              'axis_linewidth_pt':axis_lw,
              'axis_to_image_gap_mm':axis_gap,'separator_pt':0.25,
              'task_separator_pt':axis_lw if args.unified else 0.65,
              'task_separator_style':'dashed' if args.unified else 'solid'},
    'note':'Top two rows show flattening event excerpts; remaining rows show folding sequences. Each row is independently chronological; rows are not synchronized. Original cam_high pose, focal length 6.5, uncropped 4:3 Isaac renders.'},indent=2))
print(f'{dest}\nPhysical size: {width:.1f} x {height:.1f} mm; frame width: {cw:.2f} mm')
