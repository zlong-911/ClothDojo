"""A hundred distinct real garments receding across a dark wood surface."""
import json
import random
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
base=ROOT/'outputs/visual_assets'
plan=json.loads((base/'garment_floor_options_20260911/white/plan.json').read_text())
front=sorted(plan['assets'],key=lambda a:a['size_m'][1])
used={a['asset_id'] for a in front}
refs=list((ROOT/'data/benchmarks/highest_pin_500_large_support_v1_20260908/templates').glob('*.npz'))
refs.sort();random.Random(42).shuffle(refs)
extra=[]
for path in refs:
    if path.stem in used:continue
    q=np.load(path)['particle_q'];size=np.ptp(q[:,:2],axis=0)
    m=front[len(extra)%34]
    extra.append(dict(asset_id=path.stem,reference=str(path.relative_to(ROOT)),
        size_m=size.tolist(),material=m['material'],color_channel=m['color_channel']))
    if len(extra)==66:break
assets=front+extra;counts=[8,8,9,9,11,11,11,11,11,11]
start=0;y=0;widths=[]
for count in counts:
    row=assets[start:start+count];start+=count
    h=max(a['size_m'][1] for a in row);gap=.16
    w=sum(a['size_m'][0] for a in row)+gap*(count-1);widths.append(w);x=-w/2
    for a in row:
        a['center']=[x+a['size_m'][0]/2,y+h/2];x+=a['size_m'][0]+gap
    y+=h+.22
plan.update(assets=assets,rows=counts,width_m=max(widths),height_m=y,
    floor_material='wood_table_001',lighting_scale=.55,
    camera_eye=[0,-7,4.4],camera_target=[0,3,0],
    selection='100 distinct real garments. The closest 34 use all 34 cloth material IDs exactly once; distant garments reuse materials.',
    layout='Ten receding rows, with 34 foreground/midground assets followed by 66 additional unique garments.')
out=base/'garment_depth_gallery_20260911';out.mkdir(parents=True,exist_ok=True)
(out/'plan.json').write_text(json.dumps(plan,indent=2))
print({'assets':len(assets),'width':max(widths),'depth':y})
