"""FastAPI control plane for the collection workspace."""

import asyncio
import gzip
import hashlib
import json
import subprocess
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import urlencode

from pydantic import BaseModel, Field

from clothdojo.collection.assets import AssetCatalog, AssetStatus
from clothdojo.collection.config import CollectionConfig, load_config
from clothdojo.collection.hil import HILCatalog
from clothdojo.collection.review import AnnotationConflictError, ReviewCatalog
from clothdojo.collection.review_export import (
    ReviewExportBusyError,
    ReviewExportManager,
)
from clothdojo.collection.supervisor import NoAvailableSlotError, TeleopSupervisor
from clothdojo.collection.workspace import CollectionWorkspace

VRDevice = Literal["pico4ultra", "quest3", "quest3s", "quest2"]

_VR_STREAM_BASE = {
    "codec": "h264",
    "immersiveMode": "vr",
    "deviceFrameRate": "72",
    "maxStreamingBitrateMbps": "80",
    "perEyeWidth": "1280",
    "perEyeHeight": "1152",
    "enablePoseSmoothing": "true",
    "posePredictionFactor": "1",
    "panelHiddenAtStart": "true",
    "cloudxrAutoRefreshMode": "any",
}
_VR_DEVICE_PARAMS = {
    "pico4ultra": {
        **_VR_STREAM_BASE,
        "deviceProfile": "pico4ultra",
        "enableTexSubImage2D": "false",
        "useQuestColorWorkaround": "false",
    },
    "quest3": {
        **_VR_STREAM_BASE,
        "deviceProfile": "quest3",
        "enableTexSubImage2D": "true",
        "useQuestColorWorkaround": "true",
    },
    "quest3s": {
        **_VR_STREAM_BASE,
        "deviceProfile": "quest3s",
        "enableTexSubImage2D": "true",
        "useQuestColorWorkaround": "true",
    },
    "quest2": {
        **_VR_STREAM_BASE,
        "deviceProfile": "quest2",
        "enableTexSubImage2D": "true",
        "useQuestColorWorkaround": "false",
    },
}


class _StaleSnapshotCache:
    """Share one expensive filesystem snapshot across polling browsers."""

    def __init__(self, loader, *, ttl_seconds: float = 2.0):
        self._loader = loader
        self._ttl_seconds = ttl_seconds
        self._condition = threading.Condition()
        self._value = None
        self._loaded_at = 0.0
        self._refreshing = False
        self._generation = 0

    def get(self):
        while True:
            start_background = False
            with self._condition:
                if self._value is not None:
                    if (
                        time.monotonic() - self._loaded_at >= self._ttl_seconds
                        and not self._refreshing
                    ):
                        self._refreshing = True
                        start_background = True
                    value = self._value
                    break
                if not self._refreshing:
                    self._refreshing = True
                    return self._load_initial()
                self._condition.wait()
        if start_background:
            threading.Thread(target=self._refresh, daemon=True).start()
        return value

    def invalidate(self) -> None:
        with self._condition:
            self._generation += 1
            self._loaded_at = 0.0

    def _load_initial(self):
        try:
            value = self._loader()
        except BaseException:
            self._refreshing = False
            self._condition.notify_all()
            raise
        self._value = value
        self._loaded_at = time.monotonic()
        self._refreshing = False
        self._condition.notify_all()
        return value

    def _refresh(self) -> None:
        with self._condition:
            generation = self._generation
        try:
            value = self._loader()
        except BaseException:
            with self._condition:
                self._refreshing = False
                self._condition.notify_all()
            return
        with self._condition:
            if generation == self._generation:
                self._value = value
                self._loaded_at = time.monotonic()
            self._refreshing = False
            self._condition.notify_all()


def build_cloudxr_client_url(
    endpoint: str,
    signaling_port: int,
    device: VRDevice = "pico4ultra",
) -> str:
    """Build a validated device-specific CloudXR client URL."""
    host, port = endpoint.rsplit(":", 1)
    query = urlencode(
        {
            "serverIP": host,
            "port": port,
            "proxyUrl": f"https://{host}:{port}/_clothdojo/{signaling_port}",
            **_VR_DEVICE_PARAMS[device],
        }
    )
    return f"https://{host}:{port}/client/?{query}"


class AssetDecisionRequest(BaseModel):
    decision: Literal["accepted", "rejected"]


class AnnotationSegmentRequest(BaseModel):
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    label: Literal["bad", "good"]


class SaveAnnotationRequest(BaseModel):
    review_status: Literal["in_progress", "complete"]
    segments: list[AnnotationSegmentRequest]
    expected_revision: str | None = None


class SetCompletionRequest(BaseModel):
    sample_index: int | None = Field(default=None, ge=0)


class SetBreakpointRequest(BaseModel):
    sample_index: int = Field(ge=0)
    marked: bool


class BreakpointRequest(BaseModel):
    run_id: str = Field(min_length=1, max_length=160)
    episode_id: str = Field(min_length=1, max_length=160)
    sample_index: int = Field(ge=0)


class InterventionPlanRequest(BaseModel):
    breakpoints: list[BreakpointRequest] = Field(min_length=1, max_length=128)


class StartInterventionRequest(InterventionPlanRequest):
    profile_id: str = Field(min_length=1, max_length=160)
    slot_id: str | None = Field(default=None, max_length=160)


class StartSessionRequest(BaseModel):
    slot_id: str | None = Field(default=None, max_length=160)
    profile_id: str = Field(min_length=1, max_length=160)
    asset_id: str = Field(min_length=1, max_length=160)


def create_app(config_path: str | Path):
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import FileResponse, Response
    except ImportError as exc:
        raise RuntimeError(
            "FastAPI and Uvicorn are required to serve the collection console"
        ) from exc

    config = load_config(config_path)
    assets = AssetCatalog(config.assets_root)
    reviews = {
        profile_id: ReviewCatalog(profile) for profile_id, profile in config.profiles.items()
    }
    hil = {profile_id: HILCatalog(profile) for profile_id, profile in config.profiles.items()}
    supervisor = TeleopSupervisor(config)
    review_exports = ReviewExportManager(config)
    workspace = CollectionWorkspace(config, assets, reviews, hil, supervisor)
    snapshot_cache = _StaleSnapshotCache(workspace.snapshot)
    hil_cache = _StaleSnapshotCache(workspace.hil_snapshot)
    static_root = Path(__file__).with_name("static")

    @asynccontextmanager
    async def lifespan(_app):
        await asyncio.to_thread(supervisor.start_gateway)
        try:
            yield
        finally:
            await asyncio.to_thread(supervisor.stop_gateway)

    app = FastAPI(
        title="ClothDojo Collection Console",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    def require_idle_review_export(kind: str) -> None:
        if review_exports.status()["status"] in {"starting", "running"}:
            raise HTTPException(
                status_code=409,
                detail=f"cannot delete {kind} during review export",
            )

    def json_response(payload: dict, request: Request):
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        etag = f'"{hashlib.sha256(body).hexdigest()}"'
        if request.headers.get("if-none-match") == etag:
            return Response(
                status_code=304,
                headers={"ETag": etag, "Cache-Control": "no-cache"},
            )
        headers = {"Cache-Control": "no-cache", "ETag": etag, "Vary": "Accept-Encoding"}
        if "gzip" in request.headers.get("accept-encoding", "").lower():
            compressed = gzip.compress(body, compresslevel=5)
            if len(compressed) < len(body):
                body = compressed
                headers["Content-Encoding"] = "gzip"
        return Response(content=body, media_type="application/json", headers=headers)

    def review_catalog(profile_id: str) -> ReviewCatalog:
        try:
            return reviews[profile_id]
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="unknown collection profile") from exc

    def hil_catalog(profile_id: str) -> HILCatalog:
        try:
            return hil[profile_id]
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="unknown collection profile") from exc

    @app.get("/")
    async def index():
        return FileResponse(static_root / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/static/{name}")
    async def static(name: Literal["app.js", "styles.css"]):
        media = "text/javascript" if name.endswith(".js") else "text/css"
        return FileResponse(
            static_root / name,
            media_type=media,
            headers={"Cache-Control": "no-cache"},
        )

    # Browser response translation stays in CollectionWorkspace so canonical
    # data catalogs do not depend on presentation schemas.
    @app.get("/api/snapshot")
    async def workspace_snapshot(request: Request):
        payload = await asyncio.to_thread(snapshot_cache.get)
        return json_response(payload, request)

    @app.get("/api/assets")
    async def workspace_asset_page(
        status: AssetStatus = "pending",
        query: str = "",
        page: int = 1,
        page_size: int = 48,
    ):
        try:
            return await asyncio.to_thread(
                workspace.asset_page,
                status=status,
                query=query,
                page=page,
                page_size=page_size,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/assets/{asset_id}/preview/{view}")
    async def workspace_asset_preview(
        asset_id: str,
        view: Literal["front", "back"],
    ):
        try:
            path = assets.preview(asset_id, view)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(
            path,
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

    @app.get("/api/tasks/{profile_id}/assets/{asset_id}/preview/{view}")
    async def workspace_profile_asset_preview(
        profile_id: str,
        asset_id: str,
        view: Literal["front", "back"],
    ):
        review_catalog(profile_id)
        return await workspace_asset_preview(asset_id, view)

    @app.put("/api/assets/{asset_id}/decision")
    async def workspace_decide_asset(
        asset_id: str,
        request: AssetDecisionRequest,
    ):
        try:
            result = await asyncio.to_thread(assets.decide, asset_id, request.decision)
            snapshot_cache.invalidate()
            return result
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/tasks/{profile_id}/annotations")
    async def workspace_annotation_index(profile_id: str, request: Request):
        review_catalog(profile_id)
        try:
            payload = await asyncio.to_thread(workspace.annotation_index, profile_id)
            return json_response(payload, request)
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/tasks/{profile_id}/episodes/{episode_id}/annotation")
    async def workspace_annotation(profile_id: str, episode_id: str):
        try:
            return await asyncio.to_thread(review_catalog(profile_id).annotation, episode_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.put("/api/tasks/{profile_id}/episodes/{episode_id}/annotation")
    async def workspace_save_annotation(
        profile_id: str,
        episode_id: str,
        request: SaveAnnotationRequest,
    ):
        try:
            result = await asyncio.to_thread(
                review_catalog(profile_id).save_annotation,
                episode_id,
                review_status=request.review_status,
                segments=[segment.model_dump() for segment in request.segments],
                expected_revision=request.expected_revision,
            )
            snapshot_cache.invalidate()
            return result
        except AnnotationConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/tasks/{profile_id}/episodes/{episode_id}/videos/{camera}")
    async def workspace_review_video(
        profile_id: str,
        episode_id: str,
        camera: str,
    ):
        try:
            path = review_catalog(profile_id).video(episode_id, camera)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return FileResponse(path, media_type="video/mp4")

    @app.delete("/api/tasks/{profile_id}/episodes/{episode_id}")
    async def workspace_delete_trajectory(profile_id: str, episode_id: str):
        require_idle_review_export("a trajectory")
        try:
            result = await asyncio.to_thread(
                review_catalog(profile_id).delete_trajectory,
                episode_id,
            )
            snapshot_cache.invalidate()
            return result
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/hil/snapshot")
    async def workspace_hil_snapshot(request: Request):
        cached = await asyncio.to_thread(hil_cache.get)
        payload = {**cached, "review_export": review_exports.status()}
        return json_response(payload, request)

    @app.get("/api/hil/profiles/{profile_id}/runs/{run_id}/episodes/{episode_id}/videos/{view}")
    async def workspace_hil_video(
        profile_id: str,
        run_id: str,
        episode_id: str,
        view: str,
    ):
        try:
            camera = (
                workspace.primary_hil_camera(profile_id, run_id, episode_id)
                if view == "primary"
                else view
            )
            path = hil_catalog(profile_id).video(run_id, episode_id, camera)
        except (FileNotFoundError, KeyError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return FileResponse(path, media_type="video/mp4")

    @app.get(
        "/api/hil/profiles/{profile_id}/interventions/{trajectory_id}/videos/{view}"
    )
    async def workspace_hil_intervention_video(
        profile_id: str,
        trajectory_id: str,
        view: str,
    ):
        try:
            catalog = hil_catalog(profile_id)
            camera = (
                catalog.primary_intervention_camera(trajectory_id)
                if view == "primary"
                else view
            )
            path = catalog.intervention_video(trajectory_id, camera)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return FileResponse(path, media_type="video/mp4")

    @app.delete("/api/hil/profiles/{profile_id}/interventions/{trajectory_id}")
    async def workspace_delete_hil_intervention(
        profile_id: str,
        trajectory_id: str,
    ):
        require_idle_review_export("an HIL trajectory")
        try:
            result = await asyncio.to_thread(
                hil_catalog(profile_id).delete_intervention,
                trajectory_id,
            )
            hil_cache.invalidate()
            return result
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FileExistsError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/review-export")
    async def workspace_review_export():
        return review_exports.status()

    @app.post("/api/review-export", status_code=202)
    async def workspace_start_review_export():
        try:
            return await asyncio.to_thread(review_exports.start)
        except ReviewExportBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.put(
        "/api/hil/profiles/{profile_id}/runs/{run_id}/episodes/{episode_id}/completion"
    )
    async def workspace_set_hil_completion(
        profile_id: str,
        run_id: str,
        episode_id: str,
        request: SetCompletionRequest,
    ):
        try:
            result = await asyncio.to_thread(
                hil_catalog(profile_id).set_completion,
                run_id,
                episode_id,
                sample_index=request.sample_index,
            )
            hil_cache.invalidate()
            return result
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.put(
        "/api/hil/profiles/{profile_id}/runs/{run_id}/episodes/{episode_id}/breakpoint"
    )
    async def workspace_set_hil_breakpoint(
        profile_id: str,
        run_id: str,
        episode_id: str,
        request: SetBreakpointRequest,
    ):
        try:
            result = await asyncio.to_thread(
                hil_catalog(profile_id).set_breakpoint,
                run_id,
                episode_id,
                sample_index=request.sample_index,
                marked=request.marked,
            )
            hil_cache.invalidate()
            return result
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/hil/interventions")
    async def workspace_start_intervention(request: StartInterventionRequest):
        try:
            plan = await asyncio.to_thread(
                hil_catalog(request.profile_id).intervention_plan,
                [item.model_dump() for item in request.breakpoints],
            )
            asset_id = str(plan["asset_id"])
            collectable = await asyncio.to_thread(assets.library.collectable_ids)
            if asset_id not in set(collectable):
                raise RuntimeError("rollout garment is no longer converted and collectable")
            result = await asyncio.to_thread(
                supervisor.start_session,
                profile_id=request.profile_id,
                asset_id=asset_id,
                slot_id=request.slot_id,
                mode="intervention",
                intervention_plan=plan,
            )
            snapshot_cache.invalidate()
            hil_cache.invalidate()
            return result
        except NoAvailableSlotError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/sessions", status_code=201)
    async def workspace_start_session(request: StartSessionRequest):
        collectable = await asyncio.to_thread(assets.library.collectable_ids)
        if request.asset_id not in set(collectable):
            raise HTTPException(
                status_code=409,
                detail="asset is rejected, missing, or not converted",
            )
        try:
            result = await asyncio.to_thread(
                supervisor.start_session,
                profile_id=request.profile_id,
                asset_id=request.asset_id,
                slot_id=request.slot_id,
            )
            snapshot_cache.invalidate()
            hil_cache.invalidate()
            return result
        except NoAvailableSlotError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/sessions/{session_id}/stop")
    async def workspace_stop_session(session_id: str):
        try:
            result = await asyncio.to_thread(supervisor.stop_session, session_id)
            snapshot_cache.invalidate()
            hil_cache.invalidate()
            return result
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/sessions/{session_id}/log")
    async def workspace_session_log(session_id: str):
        try:
            return {"log": await asyncio.to_thread(supervisor.session_log, session_id)}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/sessions/{session_id}/qr")
    async def workspace_session_qr(session_id: str, device: VRDevice = "pico4ultra"):
        try:
            import qrcode
            from qrcode.image.svg import SvgPathImage
        except ImportError as exc:
            raise HTTPException(status_code=503, detail="QR support is not installed") from exc
        try:
            session = await asyncio.to_thread(supervisor.session, session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        slot = next(item for item in supervisor.slots() if item["id"] == session["slot_id"])
        client_url = build_cloudxr_client_url(
            str(session["endpoint"]),
            int(slot["signaling_port"]),
            device,
        )
        image = qrcode.make(client_url, image_factory=SvgPathImage)
        return Response(content=image.to_string(), media_type="image/svg+xml")

    app.state.collection_config = config
    app.state.asset_catalog = assets
    app.state.review_catalogs = reviews
    app.state.hil_catalogs = hil
    app.state.teleop_supervisor = supervisor
    app.state.review_exports = review_exports
    return app


def serve(config_path: str | Path) -> None:
    config: CollectionConfig = load_config(config_path)
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "FastAPI and Uvicorn are required to serve the collection console"
        ) from exc
    uvicorn.run(create_app(config_path), host=config.host, port=config.port, workers=1)
