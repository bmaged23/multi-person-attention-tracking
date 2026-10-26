"""Single-worker local API for bounded video jobs, with real shared inference models."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict
import json
import logging
import shutil
import time
from uuid import UUID, uuid4

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from attention_tracking.config import ROOT, Settings, ServiceSettings
from attention_tracking.detection.people_backend import PeopleBackend
from attention_tracking.video.pipeline import run
from attention_tracking.upload_limits import UploadLimits


def create_app(settings=None, limits=None, backend_factory=PeopleBackend):
    settings = settings or Settings()
    limits = limits or ServiceSettings()

    @asynccontextmanager
    async def lifespan(app):
        limits.data_dir.mkdir(parents=True, exist_ok=True)
        executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="video-inference"
        )
        app.state.executor = executor
        app.state.busy = False
        app.state.jobs = {}
        app.state.tasks = set()
        # Recover only this service's marked job directories and bound retained data.
        recovered = []
        for folder in limits.data_dir.iterdir():
            if not folder.is_dir() or not (folder / ".attention-job").is_file():
                continue
            try:
                if str(UUID(folder.name)) != folder.name:
                    continue
            except ValueError:
                continue
            completed = (folder / "summary.json").is_file() and (
                folder / "annotated.mp4"
            ).is_file()
            job = {
                "id": folder.name,
                "status": "completed" if completed else "failed",
                "created_at": folder.stat().st_mtime,
            }
            if not completed:
                job["error"] = "Previous process stopped before job completion"
            (folder / "input.video").unlink(missing_ok=True)
            recovered.append((job["created_at"], folder, job))
        for _, folder, job in sorted(recovered):
            app.state.jobs[job["id"]] = job
        while len(app.state.jobs) > limits.retained_jobs:
            oldest = next(iter(app.state.jobs))
            app.state.jobs.pop(oldest)
            shutil.rmtree(limits.data_dir / oldest)

        loop = asyncio.get_running_loop()

        def initialize():
            model = backend_factory(settings)
            try:
                model.detect(np.zeros((320, 320, 3), np.uint8), 0)
                return model
            except Exception:
                model.close()
                raise

        try:
            app.state.backend = await loop.run_in_executor(executor, initialize)
            yield
        finally:
            await asyncio.gather(*app.state.tasks, return_exceptions=True)
            if hasattr(app.state, "backend"):
                await loop.run_in_executor(executor, app.state.backend.close)
            executor.shutdown(wait=True)

    app = FastAPI(
        title="Multi-person visual attention API", version="1.0.0", lifespan=lifespan
    )

    app.add_middleware(
        UploadLimits,
        max_bytes=limits.max_upload_mb * 1024**2 + 65536,
        timeout_seconds=limits.upload_timeout_seconds,
    )

    def get_job(job_id):
        try:
            if str(UUID(job_id)) != job_id:
                raise ValueError()
        except ValueError:
            raise HTTPException(404, "Job not found") from None
        job = app.state.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        return job

    @app.get("/health")
    async def health():
        return {
            "ready": True,
            "busy": app.state.busy,
            "person_backend": settings.person_backend,
            "person_device": settings.person_device,
            "landmark_device": settings.device,
            "max_concurrent_jobs": 1,
        }

    @app.get("/config")
    async def configuration():
        return {
            "limits": {k: v for k, v in asdict(limits).items() if k != "data_dir"},
            "attention_method": "weighted visual heuristic; no behavioral attention accuracy claimed",
        }

    app.mount("/assets", StaticFiles(directory=ROOT / "web"), name="assets")

    @app.get("/")
    async def home():
        return FileResponse(ROOT / "web/index.html", media_type="text/html")

    @app.post("/jobs", status_code=202)
    async def submit(
        video: UploadFile = File(...), max_seconds: float | None = Query(None, gt=0)
    ):
        if max_seconds is not None and max_seconds > limits.max_clip_seconds:
            raise HTTPException(422, "max_seconds exceeds configured limit")
        if app.state.busy:
            raise HTTPException(409, "A video job is already running")
        app.state.busy = True
        job_id = str(uuid4())
        folder = limits.data_dir / job_id
        source = folder / "input.video"
        total = 0
        try:
            folder.mkdir()
            (folder / ".attention-job").touch()
            with source.open("wb") as stream:
                while chunk := await video.read(1024 * 1024):
                    total += len(chunk)
                    if total > limits.max_upload_mb * 1024**2:
                        raise HTTPException(413, "Upload exceeds configured limit")
                    stream.write(chunk)
            with source.open("rb") as stream:
                header = stream.read(16)
            if not (
                header[4:8] == b"ftyp"
                or header[:4] == b"RIFF"
                and header[8:12] == b"AVI "
                or header[:4] == b"\x1aE\xdf\xa3"
            ):
                raise HTTPException(
                    415, "Expected an MP4, MOV, AVI, WebM or MKV video container"
                )
            capture = cv2.VideoCapture(str(source))
            try:
                pixels = capture.get(cv2.CAP_PROP_FRAME_WIDTH) * capture.get(
                    cv2.CAP_PROP_FRAME_HEIGHT
                )
                fps = capture.get(cv2.CAP_PROP_FPS)
                if (
                    not capture.isOpened()
                    or not 0 < pixels <= limits.max_frame_pixels
                    or not 0 < fps <= limits.max_source_fps
                ):
                    raise HTTPException(
                        422, "Invalid video or resolution/FPS exceeds configured limits"
                    )
            finally:
                capture.release()
            # Retain a bounded number of completed jobs; there is only one active job.
            while len(app.state.jobs) >= limits.retained_jobs:
                oldest = next(iter(app.state.jobs))
                app.state.jobs.pop(oldest)
                shutil.rmtree(limits.data_dir / oldest)
            job = {"id": job_id, "status": "running", "created_at": time.time()}
            app.state.jobs[job_id] = job

            async def process():
                start = time.perf_counter()
                try:

                    def execute():
                        return run(
                            str(source),
                            settings,
                            folder / "summary.json",
                            folder / "annotated.mp4",
                            max_seconds=max_seconds or limits.max_clip_seconds,
                            backend=app.state.backend,
                            max_frame_pixels=limits.max_frame_pixels,
                        )

                    await asyncio.get_running_loop().run_in_executor(
                        app.state.executor, execute
                    )
                    job.update(status="completed", seconds=time.perf_counter() - start)
                except Exception:
                    logging.exception("Video job failed: %s", job_id)
                    job.update(
                        status="failed",
                        error="Video processing failed; inspect server logs",
                    )
                finally:
                    source.unlink(missing_ok=True)
                    app.state.busy = False

            task = asyncio.create_task(process())
            app.state.tasks.add(task)
            task.add_done_callback(app.state.tasks.discard)
            return dict(job)
        except BaseException:
            app.state.busy = False
            shutil.rmtree(folder, ignore_errors=True)
            raise
        finally:
            await video.close()

    @app.get("/jobs/{job_id}")
    async def status(job_id: str):
        return dict(get_job(job_id))

    @app.get("/jobs/{job_id}/{artifact}")
    async def artifact(job_id: str, artifact: str):
        job = get_job(job_id)
        if job["status"] != "completed":
            raise HTTPException(409, "Job has not completed")
        names = {
            "result": ("summary.json", "application/json"),
            "observations": ("summary_observations.jsonl", "application/x-ndjson"),
            "video": ("annotated.mp4", "video/mp4"),
        }
        if artifact not in names:
            raise HTTPException(404, "Artifact not found")
        name, kind = names[artifact]
        path = limits.data_dir / job_id / name
        if artifact == "result":
            report = json.loads(path.read_text())
            report.pop("source", None)
            return report
        return FileResponse(
            path,
            media_type=kind,
            filename=name,
            content_disposition_type="inline" if artifact == "video" else "attachment",
        )

    return app


app = create_app()
