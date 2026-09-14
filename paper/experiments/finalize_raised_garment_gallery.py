"""Export the raised, edge-filled white gallery at 1.8:1."""
import json
from pathlib import Path
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/visual_assets/garment_depth_gallery_20260911/matte_wood/final/white_wide_depth/white_wide_depth_raised'
source=Image.open(OUT/'garment_gallery.png')
assert source.size==(5400,3000)
box=(270,0,5130,2700)
crop=source.crop(box)
assert crop.size==(4860,2700)
crop.save(OUT/'garment_gallery_cropped.png')
for target in [OUT/'garment_gallery_cropped.pdf',ROOT/'paper/figures/clothdojo-garment-assets.pdf']:
    crop.save(target,resolution=crop.width/3.4,quality=100,subsampling=0)
crop.resize((1260,700),Image.Resampling.LANCZOS).save(OUT/'preview.png')
metadata=dict(source=str((OUT/'garment_gallery.png').relative_to(ROOT)),
    plan=str((OUT/'plan.json').relative_to(ROOT)),crop_box_pixels=box,
    output_pixels=crop.size,aspect_ratio=1.8,assets=436,unique_assets=436,
    asset_pool='highest_pin_500_large_support_v1_20260908',render_spp=512,
    note='Bottom content ends at y=2591; crop ends at y=2700, leaving 109 px safety margin.')
(OUT/'crop.json').write_text(json.dumps(metadata,indent=2))
(ROOT/'paper/figures/clothdojo-garment-assets.sources.json').write_text(json.dumps(metadata,indent=2))
