"""Optional Isaac RTX rendering for canonical Newton trajectory replay."""

from __future__ import annotations

import ctypes
import hashlib
import importlib.metadata
import os
import re
import socket
import sys
import tempfile
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np

from clothdojo.assets.materials import (
    MaterialAsset,
    MaterialCatalog,
    load_material_catalog,
    prepare_polyhaven_assets,
    require_polyhaven_assets,
)
from clothdojo.robots import RobotProfile

ROBODOJO_REVISION = "c037c1b3183a030724e694d69a08cb62369ed285"
_ISAAC_MAJOR_MINOR = (5, 0)
_DEVICE = re.compile(r"cuda:(\d+)\Z")
VisualComponent = Literal["cloth", "table", "ground", "lighting"]
VISUAL_COMPONENTS: tuple[VisualComponent, ...] = (
    "cloth",
    "table",
    "ground",
    "lighting",
)
_ROOMS = tuple(
    f"Simple_Room_{color}"
    for color in (
        "azure",
        "blue",
        "deep_blue",
        "green",
        "lime",
        "magenta",
        "orange",
        "pink",
        "purple",
        "red",
        "teal",
        "yellow",
    )
)


def prepare_isaac_assets(
    output: str | Path = "data/visual-assets", *, include_room: bool = False
) -> dict[str, Any]:
    """Prepare the curated Poly Haven set and optional RoboDojo rooms."""

    target = Path(output).expanduser().resolve()
    catalog = load_material_catalog()
    polyhaven = prepare_polyhaven_assets(target / "polyhaven", catalog)
    if include_room:
        from huggingface_hub import snapshot_download

        patterns = []
        for room in _ROOMS:
            patterns.extend(
                (
                    f"Assets/Room/{room}/base0.usd",
                    f"Assets/Room/{room}/SubUSDs/textures/DefaultMaterial_*",
                    f"Assets/Room/{room}/SubUSDs/textures/Parquet_*",
                    f"Assets/Room/{room}/SubUSDs/textures/WallWood_*",
                    f"Assets/Room/{room}/SubUSDs/textures/WhitePlastic_*",
                )
            )
        snapshot_download(
            repo_id="RoboDojo-Benchmark/RoboDojo",
            repo_type="dataset",
            revision=ROBODOJO_REVISION,
            allow_patterns=patterns,
            local_dir=target / "robodojo",
            max_workers=8,
        )
        revision = target / "robodojo/clothdojo_revision.txt"
        revision.parent.mkdir(parents=True, exist_ok=True)
        temporary = revision.with_name(f".{revision.name}.tmp")
        temporary.write_text(f"{ROBODOJO_REVISION}\n", encoding="utf-8")
        os.replace(temporary, revision)
    _require_visual_assets(target, catalog, include_room=include_room)
    return {
        "compatible": True,
        "output": str(target),
        "polyhaven": polyhaven,
        "robodojo_revision": ROBODOJO_REVISION if include_room else None,
        "rooms": "included" if include_room else "excluded",
    }


@dataclass(frozen=True, slots=True)
class MaterialAppearance:
    material: str
    color_channel: str


@dataclass(frozen=True, slots=True)
class LightingAppearance:
    environment: str
    exposure: float
    rotation_z: float


@dataclass(frozen=True, slots=True)
class VisualLayout:
    episode_index: int
    source_episode_id: str | None
    seed: int
    cloth: MaterialAppearance
    table: MaterialAppearance
    ground: MaterialAppearance
    lighting: LightingAppearance
    room: str | None


def sample_visual_layout(
    base_seed: int,
    episode_index: int,
    *,
    source_episode_id: str | None = None,
    include_room: bool = False,
    randomize: tuple[VisualComponent, ...] = VISUAL_COMPONENTS,
) -> VisualLayout:
    """Sample one deterministic, semantically valid appearance."""

    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise TypeError("visual randomization seed must be an integer")
    if base_seed < 0:
        raise ValueError("visual randomization seed must be non-negative")
    if isinstance(episode_index, bool) or not isinstance(episode_index, int):
        raise TypeError("episode index must be an integer")
    if episode_index < 0:
        raise ValueError("episode index must be non-negative")
    if not isinstance(include_room, bool):
        raise TypeError("include_room must be boolean")
    randomized = _validate_visual_components(randomize)
    if source_episode_id is not None:
        if (
            not isinstance(source_episode_id, str)
            or not source_episode_id
            or source_episode_id != source_episode_id.strip()
        ):
            raise ValueError("source_episode_id must be a non-empty string")
        episode_seed = int.from_bytes(
            hashlib.blake2s(source_episode_id.encode(), digest_size=4).digest(),
            "little",
        )
    else:
        episode_seed = episode_index
    sequence = np.random.SeedSequence([base_seed, episode_seed])
    catalog = load_material_catalog()
    table_rng = _component_rng(
        base_seed, episode_seed if "table" in randomized else 0, "table"
    )
    ground_rng = _component_rng(
        base_seed, episode_seed if "ground" in randomized else 0, "ground"
    )
    cloth_rng = _component_rng(
        base_seed, episode_seed if "cloth" in randomized else 0, "cloth"
    )
    lighting_rng = _component_rng(
        base_seed, episode_seed if "lighting" in randomized else 0, "lighting"
    )
    room_rng = _component_rng(base_seed, episode_seed, "room")
    lighting_asset = catalog.lighting[int(lighting_rng.integers(len(catalog.lighting)))]
    return VisualLayout(
        episode_index=episode_index,
        source_episode_id=source_episode_id,
        seed=int(sequence.generate_state(1, dtype=np.uint32)[0]),
        cloth=_sample_material(cloth_rng, catalog.pool("cloth")),
        table=_sample_material(table_rng, catalog.pool("table")),
        ground=_sample_material(ground_rng, catalog.pool("ground")),
        lighting=LightingAppearance(
            environment=lighting_asset.id,
            exposure=round(
                lighting_asset.exposure_offset
                + _rounded_uniform(lighting_rng, -0.75, 0.75),
                6,
            ),
            rotation_z=_rounded_uniform(lighting_rng, 0.0, 360.0),
        ),
        room=(
            _ROOMS[int(room_rng.integers(len(_ROOMS)))] if include_room else None
        ),
    )


def parse_visual_randomization(value: str) -> tuple[VisualComponent, ...]:
    """Parse the single user-facing visual randomization selector."""

    if value == "all":
        return VISUAL_COMPONENTS
    if value == "none":
        return ()
    names = value.split(",")
    if not names or any(not name or name != name.strip() for name in names):
        raise ValueError("expected none, all, or comma-separated visual components")
    if len(names) != len(set(names)):
        raise ValueError("visual randomization components must not be repeated")
    unknown = set(names).difference(VISUAL_COMPONENTS)
    if unknown:
        raise ValueError(f"unknown visual randomization components: {sorted(unknown)}")
    return tuple(name for name in VISUAL_COMPONENTS if name in names)


def _validate_visual_components(
    value: tuple[VisualComponent, ...],
) -> frozenset[VisualComponent]:
    if not isinstance(value, tuple):
        raise TypeError("visual randomization components must be a tuple")
    if len(value) != len(set(value)) or any(item not in VISUAL_COMPONENTS for item in value):
        raise ValueError("invalid visual randomization components")
    return frozenset(value)


def _sample_material(
    rng: np.random.Generator,
    materials: tuple[MaterialAsset, ...],
) -> MaterialAppearance:
    material = materials[int(rng.integers(len(materials)))]
    return MaterialAppearance(
        material=material.id,
        color_channel=material.color_channels[
            int(rng.integers(len(material.color_channels)))
        ],
    )


def _component_rng(base_seed: int, episode_seed: int, component: str) -> np.random.Generator:
    component_seed = int.from_bytes(
        hashlib.blake2s(component.encode(), digest_size=4).digest(), "little"
    )
    return np.random.default_rng(
        np.random.SeedSequence([base_seed, episode_seed, component_seed])
    )


def _rounded_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return round(float(rng.uniform(low, high)), 6)


def inspect_isaac_rollout_export(
    source: str | Path,
    *,
    seed: int,
    include_room: bool,
    randomize: tuple[VisualComponent, ...],
) -> dict:
    """Inspect the Isaac export plan without checking its rendering runtime."""

    sample_visual_layout(seed, 0, include_room=include_room, randomize=randomize)
    from clothdojo.data.rollout import inspect_rollout_collection_export

    result = inspect_rollout_collection_export(source)
    result.update(
        {
            "renderer": "isaac_rtx",
            "visual_randomization_seed": seed,
            "randomized_components": list(randomize),
            "room_mode": "robodojo_per_episode" if include_room else "disabled",
            "required_python": "3.11",
            "required_isaac_sim": "5.0.x",
            "required_gpu": "RTX GPU with RT Cores and Vulkan",
        }
    )
    return result


def export_isaac_rollout_collection(
    source: str | Path,
    output: str | Path,
    *,
    assets_root: str | Path,
    visual_assets_root: str | Path,
    device: str,
    seed: int,
    include_room: bool,
    randomize: tuple[VisualComponent, ...],
    repo_id: str | None,
    image_writer_threads: int,
) -> dict[str, Any]:
    """Render a canonical trajectory collection with Isaac 5 RTX."""

    gpu_id = _physical_gpu(device)
    runtime_device = "cuda:0"
    _require_isaac5_runtime()
    root = Path(source).expanduser().resolve()
    asset_root = Path(assets_root).expanduser().resolve()
    visual_assets = Path(visual_assets_root).expanduser().resolve()
    catalog = load_material_catalog()
    _validate_visual_components(randomize)
    _require_visual_assets(visual_assets, catalog, include_room=include_room)

    from clothdojo.data.rollout import _export_canonical_collection

    rendering: dict[str, Any] = {
        "backend": "isaac_rtx",
        "granularity": "episode",
        "sampling": "uniform_material_then_colorway",
        "appearance_source": "Poly Haven CC0 curated materials",
        "robodojo_revision": ROBODOJO_REVISION if include_room else None,
        "base_seed": seed,
        "randomized_components": list(randomize),
        "room_mode": "robodojo_per_episode" if include_room else "disabled",
        "layouts": [],
    }

    # Isaac 5's USDRT Fabric only supports logical cuda:0. Isolate the selected
    # physical GPU, then keep Isaac, Warp and Fabric on that single device.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "yes")
    os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")
    _preload_python_libstdcpp()
    from clothdojo.teleop.isaac_compat import configure_replicator_warp

    configure_replicator_warp()
    portable = (
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "clothdojo"
        / "isaac5-export"
        / f"{socket.gethostname()}-gpu-{gpu_id}"
    )
    portable.mkdir(parents=True, exist_ok=True)
    app = _start_isaac_app(gpu_id, portable)

    def camera_factory(asset_id: str, initial_state, profile):
        return IsaacRandomizedCamera(
            app,
            initial_state,
            asset_path=(
                asset_root
                / "garments"
                / "converted"
                / asset_id
                / f"{asset_id}.usdc"
            ),
            assets_root=asset_root,
            visual_assets_root=visual_assets,
            device=runtime_device,
            base_seed=seed,
            include_room=include_room,
            randomize=randomize,
            layouts=rendering["layouts"],
            robot_profile=profile,
        )

    try:
        return _export_canonical_collection(
            root,
            output,
            camera_factory=camera_factory,
            rendering=rendering,
            repo_id=repo_id,
            image_writer_threads=image_writer_threads,
        )
    except BaseException:
        # Kit may terminate during shutdown before Python's top-level exception
        # hook runs, so preserve the actual export failure first.
        traceback.print_exc()
        raise
    finally:
        if app is not None and app.is_running():
            app.close(wait_for_replicator=False)


class IsaacRandomizedCamera:
    """Project recorded Newton state into Isaac Fabric and RTX."""

    def __init__(
        self,
        app,
        initial_state,
        *,
        asset_path: Path,
        assets_root: Path,
        visual_assets_root: Path,
        device: str,
        base_seed: int,
        include_room: bool,
        randomize: tuple[VisualComponent, ...],
        layouts: list[dict[str, Any]],
        robot_profile: RobotProfile,
    ) -> None:
        import omni.usd
        import warp as wp

        body_q = np.asarray(initial_state.body_q, dtype=np.float32)
        particle_q = np.asarray(initial_state.particle_q, dtype=np.float32)
        if body_q.shape != (len(robot_profile.body_names), 7):
            raise ValueError(
                f"Isaac randomized export requires {robot_profile.robot_id} body_q shape "
                f"{(len(robot_profile.body_names), 7)}, got {body_q.shape}"
            )
        _validate_robot_installations(body_q, robot_profile)
        if particle_q.ndim != 2 or particle_q.shape[1] != 3:
            raise ValueError(
                "Isaac randomized export requires particle_q shape (particles, 3), "
                f"got {particle_q.shape}"
            )
        self._app = app
        self._base_seed = base_seed
        self._include_room = include_room
        self._randomize = randomize
        self._visual_assets_root = visual_assets_root
        self._layouts = layouts
        self._cameras = robot_profile.cameras
        self._closed = False
        self._episode_warmup = False
        self._temporary = tempfile.TemporaryDirectory(prefix="clothdojo-isaac-export-")
        plan = _RenderPlan(
            session_root=Path(self._temporary.name),
            assets_root=assets_root,
            asset_path=asset_path,
        )
        from clothdojo.teleop.isaac_stage import build_replay_render_stage

        stage_path = build_replay_render_stage(
            particle_q,
            plan,
            cloth_visual="normal",
            robot_profile=robot_profile,
        )
        context = omni.usd.get_context()
        if not context.open_stage(str(stage_path)):
            raise RuntimeError(f"Isaac could not open render stage: {stage_path}")
        for _ in range(30):
            app.update()
        self._stage = context.get_stage()
        if self._stage is None:
            raise RuntimeError("Isaac USD context has no render stage")
        from clothdojo.teleop.isaac_compat import configure_replicator_warp

        configure_replicator_warp()
        import omni.replicator.core as rep

        self._rep = rep
        self._model = SimpleNamespace(
            body_label=robot_profile.body_names,
            body_count=len(robot_profile.body_names),
            particle_count=int(particle_q.shape[0]),
            joint_child=_ArrayMetadata(np.arange(len(robot_profile.body_names))),
            joint_parent=_ArrayMetadata(np.asarray(robot_profile.body_parents)),
            body_parent_world_q=_body_parent_world_poses(robot_profile),
        )
        self._state = SimpleNamespace(
            body_q=wp.array(body_q, dtype=wp.transformf, device=device),
            particle_q=wp.array(particle_q, dtype=wp.vec3f, device=device),
        )
        self._usd_context = context
        self._device = device
        self._bridge = None
        self._camera_paths = _author_cameras(
            self._stage,
            self._model.body_label,
            self._cameras,
        )
        rep.settings.set_render_rtx_realtime(antialiasing="DLAA")
        self._render_product = rep.create.render_product_tiled(
            cameras=list(self._camera_paths),
            tile_resolution=(640, 480),
            force_new=True,
            name="ClothDojoCameras",
        )
        self._annotator = rep.AnnotatorRegistry.get_annotator(
            "rgb", device="cpu", do_array_copy=True
        )
        self._annotator.attach(self._render_product)
        for _ in range(4):
            app.update()

    def begin_episode(self, episode_index: int, *, source_episode_id: str | None) -> None:
        layout = sample_visual_layout(
            self._base_seed,
            episode_index,
            source_episode_id=source_episode_id,
            include_room=self._include_room,
            randomize=self._randomize,
        )
        _apply_layout(self._stage, layout, self._visual_assets_root)
        self._layouts.append(asdict(layout))
        self._episode_warmup = True
        for _ in range(3):
            self._app.update()
        # Layout edits can rebuild Fabric and discard custom attributes, so bind
        # only after the episode's USD appearance has settled.
        from clothdojo.teleop.isaac_fabric import NewtonFabricBridge

        self._bridge = NewtonFabricBridge(
            self._model,
            self._usd_context,
            device=self._device,
        )

    def __call__(self, state) -> dict[str, np.ndarray]:
        if self._bridge is None:
            raise RuntimeError("Isaac camera episode has not been initialized")
        self._state.body_q.assign(np.asarray(state.body_q, dtype=np.float32))
        self._state.particle_q.assign(np.asarray(state.particle_q, dtype=np.float32))
        self._bridge.update(self._state)
        for _ in range(4):
            self._app.update()
        if self._episode_warmup:
            self._capture()
            self._episode_warmup = False
        images = self._capture()
        return {
            f"observation.images.{camera.name}": image[None, ...]
            for camera, image in zip(self._cameras, images, strict=True)
        }

    def _capture(self) -> list[np.ndarray]:
        images = self._read_tiled_rgb()
        if any(_is_blank_frame(image) for image in images):
            for _ in range(20):
                self._app.update()
            images = self._read_tiled_rgb()
        for path, image in zip(self._camera_paths, images, strict=True):
            if _is_blank_frame(image):
                raise RuntimeError(f"Isaac returned a blank RGB frame: {path}")
        return images

    def _read_tiled_rgb(self) -> list[np.ndarray]:
        self._rep.orchestrator.step(
            rt_subframes=3,
            pause_timeline=True,
            delta_time=0.0,
            wait_for_render=True,
        )
        data = self._annotator.get_data(device="cpu", do_array_copy=True)
        if isinstance(data, dict):
            data = data.get("data")
        return _split_tiled_rgb(data, camera_count=len(self._camera_paths))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._annotator.detach([self._render_product.path])
        finally:
            self._render_product.destroy()
            self._temporary.cleanup()


@dataclass(frozen=True, slots=True)
class _RenderPlan:
    session_root: Path
    assets_root: Path
    asset_path: Path


class _ArrayMetadata:
    def __init__(self, values) -> None:
        self._values = np.asarray(values, dtype=np.int32)

    def numpy(self) -> np.ndarray:
        return self._values


def _validate_robot_installations(
    body_q: np.ndarray, profile: RobotProfile
) -> None:
    """Fail before rendering if Newton roots disagree with the robot profile."""

    roots = np.flatnonzero(np.asarray(profile.body_parents) == -1)
    expected_positions = np.asarray(
        [frame.position for frame in profile.root_body_frames], dtype=np.float32
    )
    expected_rotations = np.asarray(
        [frame.rotation_xyzw for frame in profile.root_body_frames], dtype=np.float32
    )
    actual = np.asarray(body_q, dtype=np.float32)[roots]
    position_error = np.linalg.norm(actual[:, :3] - expected_positions, axis=1)
    actual_rotation = actual[:, 3:7]
    actual_norms = np.linalg.norm(actual_rotation, axis=1, keepdims=True)
    expected_norms = np.linalg.norm(expected_rotations, axis=1, keepdims=True)
    if (
        not np.isfinite(actual).all()
        or np.any(actual_norms <= 1.0e-8)
        or np.any(expected_norms <= 1.0e-8)
    ):
        raise RuntimeError(
            f"Newton {profile.robot_id} root body poses disagree with its "
            "installation frames"
        )
    actual_rotation /= actual_norms
    expected_rotations /= expected_norms
    rotation_alignment = np.abs(np.sum(actual_rotation * expected_rotations, axis=1))
    if (
        np.any(position_error > 1.0e-5)
        or np.any(rotation_alignment < 1.0 - 1.0e-6)
    ):
        raise RuntimeError(
            f"Newton {profile.robot_id} root body poses disagree with its "
            "installation frames"
        )


def _body_parent_world_poses(profile: RobotProfile) -> np.ndarray:
    poses = np.zeros((len(profile.body_names), 7), dtype=np.float32)
    poses[:, 6] = 1.0
    roots = np.flatnonzero(np.asarray(profile.body_parents) == -1)
    for root, frame in zip(roots, profile.root_body_frames, strict=True):
        poses[root, :3] = frame.position
        poses[root, 3:7] = frame.rotation_xyzw
    return poses


def _author_cameras(stage, body_labels, cameras) -> tuple[str, ...]:
    from pxr import Gf, UsdGeom

    from clothdojo.teleop.isaac_fabric import _find_render_body_paths

    requested_names = tuple(view.name for view in cameras)
    embedded = {
        str(prim.GetAttribute("clothdojo:cameraName").Get() or prim.GetName()): str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.IsA(UsdGeom.Camera)
        and str(prim.GetAttribute("clothdojo:cameraName").Get() or prim.GetName())
        in requested_names
    }
    if embedded:
        if set(embedded) != set(requested_names):
            raise RuntimeError("robot USD contains only part of the requested camera set")
        return tuple(embedded[name] for name in requested_names)

    body_paths = _find_render_body_paths(stage, body_labels)
    label_to_path = {
        str(body_labels[index]).strip("/"): path
        for index, path in body_paths.items()
    }
    paths = []
    for view in cameras:
        parent = "/World/Cameras"
        if view.parent_body is not None:
            candidates = [
                path
                for label, path in label_to_path.items()
                if _matches_semantic_body(label, view.parent_body)
            ]
            if len(candidates) != 1:
                raise RuntimeError(
                    f"Isaac camera mount {view.parent_body!r} matched "
                    f"{len(candidates)} render bodies"
                )
            parent = f"{candidates[0]}/ClothDojoCameras"
        UsdGeom.Xform.Define(stage, parent)
        path = f"{parent}/{view.name}"
        camera = UsdGeom.Camera.Define(stage, path)
        matrix = np.asarray(view.parent_from_camera, dtype=np.float64)
        transform = Gf.Matrix4d(*[float(value) for value in matrix.T.reshape(-1)])
        UsdGeom.Xformable(camera).AddTransformOp().Set(transform)
        camera.CreateFocalLengthAttr(view.focal_length)
        camera.CreateHorizontalApertureAttr(view.horizontal_aperture)
        camera.CreateVerticalApertureAttr(view.vertical_aperture)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
        paths.append(path)
    return tuple(paths)


def _matches_semantic_body(label: str, semantic_name: str) -> bool:
    """Match a compact name such as ``left/camera`` to a full body path."""

    label_parts = label.strip("/").split("/")
    semantic_parts = semantic_name.strip("/").split("/")
    if not label_parts or not semantic_parts or label_parts[-1] != semantic_parts[-1]:
        return False
    positions = iter(label_parts[:-1])
    return all(part in positions for part in semantic_parts[:-1])


def _split_tiled_rgb(data, *, camera_count: int) -> list[np.ndarray]:
    tiles_per_side = int(np.ceil(np.sqrt(camera_count)))
    tiled = np.asarray(data)
    expected = (tiles_per_side * 480, tiles_per_side * 640)
    if tiled.ndim != 3 or tiled.shape[:2] != expected or tiled.shape[2] < 3:
        raise RuntimeError(
            f"Isaac tiled RGB shape {tiled.shape} does not match "
            f"{(*expected, '3 or 4 channels')}"
        )
    return [
        np.ascontiguousarray(
            tiled[row * 480 : (row + 1) * 480, column * 640 : (column + 1) * 640, :3]
        )
        for row, column in (
            divmod(index, tiles_per_side) for index in range(camera_count)
        )
    ]


def _apply_layout(stage, layout: VisualLayout, root: Path) -> None:
    from pxr import Gf, Sdf, UsdGeom

    catalog = load_material_catalog()
    material_root = root / "polyhaven"
    _bind_polyhaven_material(
        stage,
        look="/World/Looks/RandomizedTable",
        prim="/World/Table",
        material_root=material_root,
        catalog=catalog,
        appearance=layout.table,
    )
    _bind_polyhaven_material(
        stage,
        look="/World/Looks/RandomizedGround",
        prim="/World/Ground",
        material_root=material_root,
        catalog=catalog,
        appearance=layout.ground,
    )
    _bind_polyhaven_material(
        stage,
        look="/World/Looks/RandomizedCloth",
        prim="/World/Cloth",
        material_root=material_root,
        catalog=catalog,
        appearance=layout.cloth,
    )
    room = stage.GetPrimAtPath("/World/RoboDojoRoom")
    if layout.room is None:
        if room:
            stage.RemovePrim(room.GetPath())
    else:
        _reference_room(stage, root / "robodojo/Assets/Room", layout.room)
    dome = stage.GetPrimAtPath("/World/Lighting/Dome")
    dome.GetAttribute("inputs:color").Set(Gf.Vec3f(1.0, 1.0, 1.0))
    dome.GetAttribute("inputs:intensity").Set(1000.0)
    dome.CreateAttribute("inputs:exposure", Sdf.ValueTypeNames.Float).Set(
        layout.lighting.exposure
    )
    dome.GetAttribute("inputs:texture:file").Set(
        Sdf.AssetPath(
            str(
                (
                    material_root
                    / "lighting"
                    / f"{layout.lighting.environment}.hdr"
                ).resolve()
            )
        )
    )
    dome.CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool).Set(True)
    rotation = dome.GetAttribute("xformOp:rotateZ")
    if not rotation:
        rotation = UsdGeom.Xformable(dome).AddRotateZOp().GetAttr()
    rotation.Set(layout.lighting.rotation_z)
    key = stage.GetPrimAtPath("/World/Lighting/Key")
    key.GetAttribute("inputs:intensity").Set(0.0)


def _reference_room(stage, room_root: Path, room_name: str) -> None:
    from pxr import Gf, UsdGeom

    source = room_root / room_name / "base0.usd"
    room = UsdGeom.Xform.Define(stage, "/World/RoboDojoRoom")
    references = room.GetPrim().GetReferences()
    references.ClearReferences()
    references.AddReference(str(source.resolve()))
    xform = UsdGeom.Xformable(room)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(0.782, 0.0, 0.0))
    xform.AddRotateZOp().Set(-90.0)
    xform.AddScaleOp().Set(Gf.Vec3f(0.5, 0.5, 0.5))


def _bind_polyhaven_material(
    stage,
    *,
    look: str,
    prim: str,
    material_root: Path,
    catalog: MaterialCatalog,
    appearance: MaterialAppearance,
) -> None:
    existing = stage.GetPrimAtPath(look)
    if existing:
        stage.RemovePrim(existing.GetPath())
    from pxr import Gf, Sdf, UsdShade

    source = catalog.material(appearance.material)
    directory = source.directory(material_root)
    material = UsdShade.Material.Define(stage, look)
    shader = UsdShade.Shader.Define(stage, f"{look}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.5)

    reader = UsdShade.Shader.Define(stage, f"{look}/PrimvarReader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    transform = UsdShade.Shader.Define(stage, f"{look}/UvTransform")
    transform.CreateIdAttr("UsdTransform2d")
    transform.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.ConnectableAPI(), "result"
    )
    transform.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(
        Gf.Vec2f(*source.uv_scale())
    )
    transform.CreateInput("rotation", Sdf.ValueTypeNames.Float).Set(0.0)
    transform.CreateInput("translation", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(0.0, 0.0))
    transform.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    base_color = _uv_texture(
        stage,
        f"{look}/BaseColor",
        directory / source.base_color_name(appearance.color_channel),
        transform,
        source_color_space="sRGB",
    )
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        base_color.ConnectableAPI(), "rgb"
    )
    roughness = _uv_texture(
        stage,
        f"{look}/Roughness",
        directory / "roughness.jpg",
        transform,
        source_color_space="raw",
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(
        roughness.ConnectableAPI(), "r"
    )
    normal = _uv_texture(
        stage,
        f"{look}/Normal",
        directory / "normal_gl.jpg",
        transform,
        source_color_space="raw",
    )
    normal.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(
        Gf.Vec4f(2.0, 2.0, 2.0, 1.0)
    )
    normal.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(
        Gf.Vec4f(-1.0, -1.0, -1.0, 0.0)
    )
    normal.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
        normal.ConnectableAPI(), "rgb"
    )
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    _bind(stage, prim, material)


def _uv_texture(stage, path: str, source: Path, transform, *, source_color_space: str):
    from pxr import Sdf, UsdShade

    shader = UsdShade.Shader.Define(stage, path)
    shader.CreateIdAttr("UsdUVTexture")
    shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(str(source.resolve()))
    )
    shader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        transform.ConnectableAPI(), "result"
    )
    shader.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(source_color_space)
    shader.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    shader.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    shader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    shader.CreateOutput("r", Sdf.ValueTypeNames.Float)
    return shader


def _bind(stage, prim_path: str, material) -> None:
    from pxr import UsdShade

    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise RuntimeError(f"render prim is missing: {prim_path}")
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def _require_visual_assets(
    root: Path,
    catalog: MaterialCatalog,
    *,
    include_room: bool,
) -> None:
    try:
        require_polyhaven_assets(root / "polyhaven", catalog)
    except FileNotFoundError as exc:
        room_flag = " --include-room" if include_room else ""
        raise FileNotFoundError(
            f"{exc}; run `clothdojo data prepare-isaac-assets{room_flag}` first"
        ) from None
    if include_room:
        _require_robodojo_rooms(root / "robodojo")


def _require_robodojo_rooms(root: Path) -> None:
    revision_path = root / "clothdojo_revision.txt"
    try:
        revision = revision_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"RoboDojo rooms are not prepared at {root}; run "
            "`clothdojo data prepare-isaac-assets --include-room` first"
        ) from None
    if revision != ROBODOJO_REVISION:
        raise RuntimeError(
            f"RoboDojo assets use revision {revision!r}, expected {ROBODOJO_REVISION!r}"
        )
    missing = [
        room
        for room in _ROOMS
        if not (root / "Assets/Room" / room / "base0.usd").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"RoboDojo room assets are incomplete: {sorted(missing)}")


def _require_isaac5_runtime() -> None:
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(
            f"Isaac RTX export requires Python 3.11, got {sys.version_info.major}."
            f"{sys.version_info.minor}"
        )
    try:
        version = importlib.metadata.version("isaacsim")
    except importlib.metadata.PackageNotFoundError:
        raise RuntimeError(
            "Isaac RTX export requires the optional Isaac Sim 5 runtime"
        ) from None
    numbers = tuple(int(value) for value in re.findall(r"\d+", version)[:2])
    if numbers != _ISAAC_MAJOR_MINOR:
        raise RuntimeError(f"Isaac RTX export requires Isaac Sim 5.0.x, got {version}")


def _start_isaac_app(gpu_id: int, portable: Path):
    from isaacsim import SimulationApp

    original_argv = sys.argv
    sys.argv = [sys.argv[0], "--portable-root", str(portable)]
    try:
        return SimulationApp(
            {
                "headless": True,
                "renderer": "RaytracedLighting",
                "width": 640,
                "height": 480,
                "multi_gpu": False,
                # Vulkan keeps host GPU ordinals when CUDA is isolated.
                "active_gpu": gpu_id,
                "physics_gpu": 0,
                "enable_crashreporter": False,
                "limit_cpu_threads": 4,
                "extra_args": [
                    "--/rtx/hydra/supportMultiTickRate=false",
                    "--/rtx/post/motionBlur/enabled=false",
                ],
            },
            experience=str(Path(__file__).with_name("isaac_export.kit")),
        )
    finally:
        sys.argv = original_argv


def _preload_python_libstdcpp() -> None:
    """Keep Kit compatible with Python distributions that bundle a newer C++ ABI."""

    library = Path(sys.base_prefix) / "lib" / "libstdc++.so.6"
    if sys.platform.startswith("linux") and library.is_file():
        ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)


def _physical_gpu(device: str) -> int:
    match = _DEVICE.fullmatch(device)
    if match is None:
        raise ValueError("Isaac RTX export device must be a physical CUDA index such as cuda:0")
    return int(match.group(1))


def _is_blank_frame(image: np.ndarray) -> bool:
    return not np.any(image) or float(np.std(image)) < 1.0
