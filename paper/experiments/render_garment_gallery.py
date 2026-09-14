"""One real 3-D gallery of naturally settled garments on a white floor."""
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(os.environ.get('GARMENT_GALLERY_OUT', str(ROOT/'outputs/visual_assets/garment_gallery_20260911')))


def main():
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    os.environ['PXR_WORK_THREAD_LIMIT']='1'
    os.environ['OMNI_KIT_ACCEPT_EULA']='yes'
    from clothdojo.data.isaac_render import (
        IsaacRandomizedCamera,MaterialAppearance,sample_visual_layout,
        _preload_python_libstdcpp,_require_isaac5_runtime,_start_isaac_app,
        _bind_polyhaven_material,
    )
    _require_isaac5_runtime();_preload_python_libstdcpp()
    from clothdojo.teleop.isaac_compat import configure_replicator_warp
    configure_replicator_warp()
    app=_start_isaac_app(0,ROOT/'outputs/visual_assets/diversity_20260911/runtime')
    from pxr import UsdGeom,UsdLux,UsdShade,Gf,Vt
    from clothdojo.robots import robot_profile
    from clothdojo.rendering import CameraSpec
    from clothdojo.trajectory import TrajectoryReader
    from clothdojo.sim.cloth import load_garment_mesh
    from clothdojo.teleop.isaac_stage import _write_st_primvar,_preview_material
    from clothdojo.assets.materials import load_material_catalog
    plan=json.loads((OUT/'plan.json').read_text())
    reader=TrajectoryReader(ROOT/'data/collections/shirt_fold/trajectories/20260825T090358.144233Z')
    state=reader.state_at(0);aid=reader.manifest.metadata['asset_id']
    eye=np.array(plan.get('camera_eye',[0.,-19.,12.]));target=np.array(plan.get('camera_target',[0.,0.,0.]))
    z=(eye-target)/np.linalg.norm(eye-target);x=np.cross([0,0,1],z);x/=np.linalg.norm(x);y=np.cross(z,x)
    matrix=np.eye(4);matrix[:3,:3]=np.stack([x,y,z],axis=1);matrix[:3,3]=eye
    frame_width=plan['width_m']+.8
    frame_height=plan['height_m']*z[2]+.5
    image_height=int(round(1800*frame_height/frame_width))
    focal=float(36*np.linalg.norm(eye)/frame_width)
    if 'camera_eye' in plan:
        image_height=int(plan.get('camera_image_height',800))
        corners=[]
        for item in plan['assets'][:plan.get('camera_frame_assets',len(plan['assets']))]:
            cx,cy=item['center'];w,h=item['size_m']
            for dx in [-w/2,w/2]:
                for dy in [-h/2,h/2]:corners.append([cx+dx,cy+dy,0])
        local=(np.asarray(corners)-eye)@matrix[:3,:3]
        ratios=local[:,:2]/(-local[:,2:])
        focal=min(18/(abs(ratios[:,0]).max()*1.08),
                  (18*image_height/1800)/(abs(ratios[:,1]).max()*1.1))
    spec=CameraSpec('garment_gallery',None,tuple(map(tuple,matrix)),focal,36.,36*image_height/1800-.01)
    render_width=int(plan.get('render_width',1800))
    render_height=round(image_height*render_width/1800)
    camera=IsaacRandomizedCamera(app,state,
        asset_path=ROOT/f'assets/garments/converted/{aid}/{aid}.usdc',
        assets_root=ROOT/'assets',visual_assets_root=ROOT/'data/visual-assets',
        device='cuda:0',base_seed=20260911,include_room=False,randomize=(),layouts=[],
        robot_profile=robot_profile('piper',assets_root=ROOT/'assets'),cameras=(spec,),
        render_resolution=(render_width,render_height),pathtrace_spp=int(plan.get('pathtrace_spp',128)),use_recorded_body_q=True,
        visual_profile='train')
    catalog=load_material_catalog()
    try:
        for stage in camera._layout_stages:
            for path in ['/World/Robot','/World/Table','/World/Ground','/World/Cloth']:
                UsdGeom.Imageable(stage.GetPrimAtPath(path)).MakeInvisible()
            floor=UsdGeom.Cube.Define(stage,'/World/GalleryFloor');floor.CreateSizeAttr(1)
            floor.AddTranslateOp().Set(Gf.Vec3d(0,0,-.025));floor.AddScaleOp().Set(Gf.Vec3f(200,200,.05))
            material=_preview_material(stage,'/World/Looks/GalleryWhite',tuple(plan.get('floor_color',[1.,1.,1.])),.9)
            UsdShade.MaterialBindingAPI.Apply(floor.GetPrim()).Bind(material)
            if 'floor_material' in plan:
                floor.MakeInvisible()
                surface=UsdGeom.Mesh.Define(stage,'/World/TexturedGalleryFloor')
                surface.CreatePointsAttr([(-100,-100,0),(100,-100,0),(100,100,0),(-100,100,0)])
                surface.CreateFaceVertexCountsAttr([4])
                surface.CreateFaceVertexIndicesAttr([0,1,2,3])
                surface.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
                _write_st_primvar(surface,np.array([[0,0],[200,0],[200,200],[0,200]],dtype=np.float32))
                _bind_polyhaven_material(stage,look='/World/Looks/GalleryFloorTexture',
                    prim=str(surface.GetPath()),material_root=ROOT/'data/visual-assets/polyhaven',
                    catalog=catalog,appearance=MaterialAppearance(plan['floor_material'],'Diffuse'))
            for i,item in enumerate(plan['assets']):
                asset=ROOT/f'assets/garments/converted/{item["asset_id"]}/{item["asset_id"]}.usdc'
                vertices,triangles,panel_vertices,panel_indices=load_garment_mesh(asset)
                with np.load(ROOT/item['reference']) as data:q=data['particle_q'].copy()
                assert len(q)==len(vertices)
                q[:,:2]-=.5*(q[:,:2].min(0)+q[:,:2].max(0))
                q[:,:2]+=np.array(item['center']);q[:,2]-=q[:,2].min();q[:,2]+=.002
                mesh=UsdGeom.Mesh.Define(stage,f'/World/Garment{i:02d}')
                mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(q))
                idx=np.asarray(triangles,dtype=np.int32).reshape(-1)
                mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3]*(len(idx)//3)))
                mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(idx))
                mesh.CreateDoubleSidedAttr(True);mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
                _write_st_primvar(mesh,panel_vertices[panel_indices])
                _bind_polyhaven_material(stage,look=f'/World/Looks/Garment{i:02d}',prim=str(mesh.GetPath()),
                    material_root=ROOT/'data/visual-assets/polyhaven',catalog=catalog,
                    appearance=MaterialAppearance(item['material'],item['color_channel']))
            for name,pos,radius,intensity in [('Fill',(4,3,8),2.5,300)]:
                light=UsdLux.SphereLight.Define(stage,f'/World/GalleryLights/{name}')
                light.CreateRadiusAttr(radius);light.CreateIntensityAttr(intensity)
                light.CreateColorAttr(Gf.Vec3f(1.,.97,.92))
                light.AddTranslateOp().Set(Gf.Vec3d(*pos))
            dome=UsdLux.DomeLight.Define(stage,'/World/GalleryLights/Ambient')
            dome.CreateIntensityAttr(1600*plan.get('lighting_scale',1.));dome.CreateColorAttr(Gf.Vec3f(1,1,1))
            sun=UsdLux.DistantLight.Define(stage,'/World/GalleryLights/SideSun')
            sun.CreateIntensityAttr(2500*plan.get('lighting_scale',1.));sun.CreateAngleAttr(22)
            direction=Gf.Vec3d(-.8,-.58,.62).GetNormalized()
            sun.AddOrientOp().Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0,0,1),direction).GetQuat()))
        from dataclasses import replace
        layout=replace(sample_visual_layout(20260911,0,visual_profile='train'),background=None)
        camera.begin_episode(0,source_episode_id='garment-gallery',layout=layout)
        # Turn off task illumination after the layout has authored it.
        for prim in camera._stage.Traverse():
            if str(prim.GetPath()).startswith('/World/Lighting/'):
                attr=prim.GetAttribute('inputs:intensity')
                if attr:attr.Set(0)
        camera._stage.Export(str(OUT/'gallery_scene.usda'))
        for _ in range(20):app.update()
        for _ in range(8):images=camera(state)
        pixels=images['observation.images.garment_gallery'][0]
        if pixels.std()<2:raise RuntimeError('Blank gallery render')
        Image.fromarray(pixels).save(OUT/'garment_gallery.png')
        Image.fromarray(pixels).save(OUT/'garment_gallery.pdf',resolution=render_width/3.4)
        # This preview has the pixel width of a roughly 3.4-inch column at 150 dpi.
        Image.fromarray(pixels).resize((510,round(510*image_height/1800)),Image.Resampling.LANCZOS).save(OUT/'single_column_preview.png')
        print('GALLERY_RENDERED',pixels.shape,float(pixels.mean()),flush=True)
    except BaseException:
        import traceback;traceback.print_exc();raise
    finally:
        camera.close();app.close(wait_for_replicator=False)


if __name__=='__main__':main()
