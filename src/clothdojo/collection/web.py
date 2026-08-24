"""FastAPI control plane for the collection workspace."""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from urllib.parse import urlencode

from pydantic import BaseModel, Field

from clothdojo.collection.assets import AssetCatalog, AssetStatus
from clothdojo.collection.config import CollectionConfig, load_config
from clothdojo.collection.hil import HILCatalog
from clothdojo.collection.review import AnnotationConflictError, ReviewCatalog
from clothdojo.collection.supervisor import NoAvailableSlotError, TeleopSupervisor
from clothdojo.collection.workspace import CollectionWorkspace


def build_cloudxr_client_url(endpoint: str) -> str:
    """Build the validated Pico 4 Ultra client URL for a teleoperation slot."""
    host, port = endpoint.rsplit(":", 1)
    query = urlencode(
        {
            "serverIP": host,
            "port": port,
            "codec": "h264",
            "immersiveMode": "vr",
            "deviceFrameRate": "72",
            "maxStreamingBitrateMbps": "80",
            "panelHiddenAtStart": "true",
            "deviceProfile": "pico4ultra",
            "useQuestColorWorkaround": "false",
            "cloudxrAutoRefreshMode": "any",
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


class SetGoodRequest(BaseModel):
    good: bool


class BreakpointRequest(BaseModel):
    episode_id: str = Field(min_length=1, max_length=160)
    sample_index: int = Field(ge=0)


class InterventionPlanRequest(BaseModel):
    run_id: str = Field(min_length=1, max_length=160)
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
        from fastapi import FastAPI, HTTPException
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
    workspace = CollectionWorkspace(config, assets, reviews, hil, supervisor)
    static_root = Path(__file__).with_name("static")
    app = FastAPI(
        title="ClothDojo Collection Console",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

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

    @app.get("/api/v1/bootstrap")
    async def bootstrap():
        return {
            "profiles": [
                {
                    "id": profile.id,
                    "display_name": profile.display_name,
                    "scene": profile.scene,
                }
                for profile in config.profiles.values()
            ],
            "slots": supervisor.slots(),
        }

    # Stable routes used by the preserved newton-cloth-rl workspace.  The
    # response translation is isolated in CollectionWorkspace; canonical data
    # catalogs do not depend on browser-era schemas.
    @app.get("/api/snapshot")
    async def workspace_snapshot():
        return workspace.snapshot()

    @app.get("/api/assets")
    async def workspace_asset_page(
        status: AssetStatus = "pending",
        query: str = "",
        page: int = 1,
        page_size: int = 48,
    ):
        try:
            return workspace.asset_page(
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
            return assets.decide(asset_id, request.decision)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/tasks/{profile_id}/annotations")
    async def workspace_annotation_index(profile_id: str):
        review_catalog(profile_id)
        try:
            return workspace.annotation_index(profile_id)
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/tasks/{profile_id}/episodes/{episode_id}/annotation")
    async def workspace_annotation(profile_id: str, episode_id: str):
        try:
            return review_catalog(profile_id).annotation(episode_id)
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
            return review_catalog(profile_id).save_annotation(
                episode_id,
                review_status=request.review_status,
                segments=[segment.model_dump() for segment in request.segments],
                expected_revision=request.expected_revision,
            )
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

    @app.get("/api/hil/snapshot")
    async def workspace_hil_snapshot():
        return workspace.hil_snapshot()

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

    @app.put("/api/hil/profiles/{profile_id}/episodes/{episode_id}/good")
    async def workspace_set_hil_good(
        profile_id: str,
        episode_id: str,
        request: SetGoodRequest,
    ):
        try:
            run_id = workspace.rollout_run_for_episode(profile_id, episode_id)
            return hil_catalog(profile_id).set_good(
                run_id,
                episode_id,
                good=request.good,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/hil/interventions")
    async def workspace_start_intervention(request: StartInterventionRequest):
        try:
            plan = hil_catalog(request.profile_id).intervention_plan(
                request.run_id,
                [item.model_dump() for item in request.breakpoints],
            )
            asset_id = str(plan["asset_id"])
            if asset_id not in set(assets.library.collectable_ids()):
                raise RuntimeError("rollout garment is no longer converted and collectable")
            return supervisor.start_session(
                profile_id=request.profile_id,
                asset_id=asset_id,
                slot_id=request.slot_id,
                mode="intervention",
                intervention_plan=plan,
            )
        except NoAvailableSlotError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/sessions", status_code=201)
    async def workspace_start_session(request: StartSessionRequest):
        if request.asset_id not in set(assets.library.collectable_ids()):
            raise HTTPException(
                status_code=409,
                detail="asset is rejected, missing, or not converted",
            )
        try:
            return supervisor.start_session(
                profile_id=request.profile_id,
                asset_id=request.asset_id,
                slot_id=request.slot_id,
            )
        except NoAvailableSlotError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/sessions/{session_id}/stop")
    async def workspace_stop_session(session_id: str):
        try:
            return supervisor.stop_session(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/sessions/{session_id}/log")
    async def workspace_session_log(session_id: str):
        try:
            return {"log": supervisor.session_log(session_id)}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/sessions/{session_id}/qr")
    async def workspace_session_qr(session_id: str):
        try:
            import qrcode
            from qrcode.image.svg import SvgPathImage
        except ImportError as exc:
            raise HTTPException(status_code=503, detail="QR support is not installed") from exc
        try:
            session = supervisor.session(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        client_url = build_cloudxr_client_url(str(session["endpoint"]))
        image = qrcode.make(client_url, image_factory=SvgPathImage)
        return Response(content=image.to_string(), media_type="image/svg+xml")

    @app.get("/api/v1/assets")
    async def asset_page(
        status: AssetStatus = "pending",
        query: str = "",
        page: int = 1,
        page_size: int = 48,
    ):
        try:
            return assets.page(status=status, query=query, page=page, page_size=page_size)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/v1/assets/{asset_id}/preview/{view}")
    async def asset_preview(asset_id: str, view: Literal["front", "back"]):
        try:
            path = assets.preview(asset_id, view)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(
            path,
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

    @app.put("/api/v1/assets/{asset_id}/decision")
    async def decide_asset(asset_id: str, request: AssetDecisionRequest):
        try:
            return assets.decide(asset_id, request.decision)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/v1/profiles/{profile_id}/episodes")
    async def episode_page(profile_id: str, page: int = 1, page_size: int = 50):
        try:
            return review_catalog(profile_id).episodes(page=page, page_size=page_size)
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/v1/profiles/{profile_id}/episodes/{episode_id}/annotation")
    async def annotation(profile_id: str, episode_id: str):
        try:
            return review_catalog(profile_id).annotation(episode_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.put("/api/v1/profiles/{profile_id}/episodes/{episode_id}/annotation")
    async def save_annotation(
        profile_id: str,
        episode_id: str,
        request: SaveAnnotationRequest,
    ):
        try:
            return review_catalog(profile_id).save_annotation(
                episode_id,
                review_status=request.review_status,
                segments=[segment.model_dump() for segment in request.segments],
                expected_revision=request.expected_revision,
            )
        except AnnotationConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/v1/profiles/{profile_id}/episodes/{episode_id}/videos/{camera}")
    async def review_video(profile_id: str, episode_id: str, camera: str):
        try:
            path = review_catalog(profile_id).video(episode_id, camera)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return FileResponse(path, media_type="video/mp4")

    @app.get("/api/v1/profiles/{profile_id}/hil/runs")
    async def hil_runs(profile_id: str):
        return {"runs": hil_catalog(profile_id).runs()}

    @app.get(
        "/api/v1/profiles/{profile_id}/hil/runs/{run_id}/episodes/{episode_id}/videos/{camera}"
    )
    async def hil_video(
        profile_id: str,
        run_id: str,
        episode_id: str,
        camera: str,
    ):
        try:
            path = hil_catalog(profile_id).video(run_id, episode_id, camera)
        except (FileNotFoundError, KeyError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return FileResponse(path, media_type="video/mp4")

    @app.put("/api/v1/profiles/{profile_id}/hil/runs/{run_id}/episodes/{episode_id}/good")
    async def set_hil_good(
        profile_id: str,
        run_id: str,
        episode_id: str,
        request: SetGoodRequest,
    ):
        try:
            return hil_catalog(profile_id).set_good(run_id, episode_id, good=request.good)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/v1/profiles/{profile_id}/hil/plans")
    async def intervention_plan(profile_id: str, request: InterventionPlanRequest):
        try:
            return hil_catalog(profile_id).intervention_plan(
                request.run_id,
                [breakpoint.model_dump() for breakpoint in request.breakpoints],
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    app.state.collection_config = config
    app.state.asset_catalog = assets
    app.state.review_catalogs = reviews
    app.state.hil_catalogs = hil
    app.state.teleop_supervisor = supervisor
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
