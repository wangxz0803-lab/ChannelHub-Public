"""Local web service with isolated, bounded, persistent generation jobs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from importlib.resources import files
import importlib.metadata
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import uuid
import zipfile

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import Field

from . import __version__
from .config import RunConfig, Settings


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


class JobRequest(Settings):
    config: RunConfig
    custom_x_id: str | None = Field(None, pattern=r"^[a-f0-9]{32}$")


class Jobs:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.uploads = self.root / "uploads"
        self.uploads.mkdir(exist_ok=True)
        self.lock = threading.RLock()
        self.processes = {}
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="channelhub")
        for p in self.root.glob("*/status.json"):
            state = json.loads(p.read_text(encoding="utf-8"))
            if state["status"] in ("queued", "running"):
                state.update(status="interrupted", message="Service restarted; create a new run")
                write_json(p, state)

    def folder(self, job_id):
        if not re.fullmatch(r"[a-f0-9]{32}", job_id) or not (self.root / job_id / "status.json").exists():
            raise HTTPException(404, "Job not found")
        return self.root / job_id

    def state(self, job_id):
        with self.lock:
            return json.loads((self.folder(job_id) / "status.json").read_text(encoding="utf-8"))

    def update(self, job_id, **changes):
        with self.lock:
            value = self.state(job_id)
            if value["status"] == "cancelled":
                return
            value.update(changes)
            write_json(self.folder(job_id) / "status.json", value)

    def submit(self, request):
        with self.lock:
            active = sum(
                json.loads(p.read_text(encoding="utf-8"))["status"] in ("queued", "running")
                for p in self.root.glob("*/status.json")
            )
            if active >= 8:
                raise HTTPException(429, "Queue is full (8 active runs)")
            if request.config.tx.waveform.kind == "custom":
                if not request.custom_x_id or not (self.uploads / (request.custom_x_id + ".npy")).is_file():
                    raise HTTPException(422, "Upload a custom X .npy file first")
            elif request.custom_x_id:
                raise HTTPException(422, "A custom X upload requires the custom waveform")
            job_id = uuid.uuid4().hex
            folder = self.root / job_id
            folder.mkdir()
            (folder / "config.json").write_text(request.config.model_dump_json(indent=2), encoding="utf-8")
            write_json(
                folder / "status.json",
                {
                    "id": job_id,
                    "status": "queued",
                    "percent": 0,
                    "message": "Queued",
                    "created_at": time.time(),
                    "engine": request.config.engine,
                    "profile": request.config.profile,
                    "waveform": request.config.tx.waveform.kind,
                },
            )
            self.executor.submit(self.run, job_id, request.custom_x_id)
            return self.state(job_id)

    def run(self, job_id, upload_id):
        folder = self.folder(job_id)
        try:
            with self.lock:
                if self.state(job_id)["status"] == "cancelled":
                    return
                self.update(job_id, status="running", message="Starting engine")
                command = [
                    sys.executable,
                    "-m",
                    "channelhub.cli",
                    "generate",
                    "--config",
                    str(folder / "config.json"),
                    "--output",
                    str(folder / "result"),
                ]
                if upload_id:
                    command += ["--x", str(self.uploads / (upload_id + ".npy"))]
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                self.processes[job_id] = process
            last_error = "Generation failed; see worker.log"
            with (folder / "worker.log").open("w", encoding="utf-8") as log:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    try:
                        event = json.loads(line)
                        if "percent" in event:
                            self.update(job_id, percent=event["percent"], message=event["message"])
                        if "error" in event:
                            last_error = event["error"]
                    except (json.JSONDecodeError, TypeError):
                        pass
            code = process.wait()
            if code:
                self.update(job_id, status="failed", message=last_error)
            else:
                with zipfile.ZipFile(folder / "result.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
                    for name in ("observations.npz", "metadata.json"):
                        archive.write(folder / "result" / name, name)
                self.update(job_id, status="completed", percent=100, message="Observation package ready")
        except Exception as exc:
            self.update(job_id, status="failed", message=str(exc))
        finally:
            with self.lock:
                self.processes.pop(job_id, None)

    def cancel(self, job_id):
        with self.lock:
            state = self.state(job_id)
            if state["status"] in ("queued", "running"):
                self.update(job_id, status="cancelled", message="Cancelled")
                process = self.processes.get(job_id)
                if process and process.poll() is None:
                    process.terminate()
            return self.state(job_id)

    def close(self):
        with self.lock:
            for job_id in list(self.processes):
                self.cancel(job_id)
        self.executor.shutdown(wait=True, cancel_futures=True)


def create_app(runtime: Path | str = "runtime"):
    manager = Jobs(runtime)

    @asynccontextmanager
    async def lifespan(app):
        yield
        manager.close()

    app = FastAPI(title="ChannelHub Public", version=__version__, lifespan=lifespan)
    app.state.jobs = manager

    @app.middleware("http")
    async def local_requests(request, call_next):
        # Reject cross-origin browser mutations; no permissive CORS is enabled.
        if request.method in ("POST", "PUT", "DELETE"):
            origin = request.headers.get("origin")
            if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
                return JSONResponse({"detail": "Cross-origin writes are disabled"}, status_code=403)
        return await call_next(request)

    @app.get("/api/config")
    def config():
        return {"defaults": RunConfig().model_dump(mode="json"), "schema": RunConfig.model_json_schema()}

    @app.get("/api/capabilities")
    def capabilities():
        installed = {}
        for name in ("sionna", "sionna-rt"):
            try:
                installed[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                installed[name] = None
        return {
            "installed": installed,
            "note": "Package presence does not prove the native RT backend is operational",
        }

    @app.get("/api/jobs")
    def jobs():
        with manager.lock:
            states = [json.loads(p.read_text(encoding="utf-8")) for p in manager.root.glob("*/status.json")]
        return sorted(states, key=lambda s: s["created_at"], reverse=True)

    @app.post("/api/jobs", status_code=202)
    def create(request: JobRequest):
        return manager.submit(request)

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        return manager.state(job_id)

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        return manager.cancel(job_id)

    @app.post("/api/uploads/x", status_code=201)
    async def upload_x(request: Request):
        buffer = bytearray()
        async for chunk in request.stream():
            buffer.extend(chunk)
            if len(buffer) > 32 * 1024 * 1024:
                raise HTTPException(413, "X file exceeds 32 MiB")
        try:
            x = np.load(io.BytesIO(buffer), allow_pickle=False)
            if (
                not isinstance(x, np.ndarray)
                or x.ndim != 3
                or x.dtype.kind != "c"
                or not np.isfinite(x).all()
            ):
                raise ValueError("Expected finite complex X[time,tx_port,subcarrier]")
        except Exception as exc:
            raise HTTPException(422, "Invalid .npy file: " + str(exc)) from exc
        upload_id = uuid.uuid4().hex
        np.save(manager.uploads / (upload_id + ".npy"), x, allow_pickle=False)
        return {"id": upload_id, "shape": list(x.shape)}

    @app.get("/api/jobs/{job_id}/preview")
    def preview(job_id: str):
        folder = manager.folder(job_id)
        if manager.state(job_id)["status"] != "completed":
            raise HTTPException(409, "Result is not ready")
        metadata = json.loads((folder / "result/metadata.json").read_text(encoding="utf-8"))
        with np.load(folder / "result/observations.npz", allow_pickle=False) as data:
            channel_db = 20 * np.log10(np.maximum(np.abs(data["desired_h"][0, 0, 0]), 1e-150))
            powers = {
                name: (
                    10 * np.log10(np.maximum(np.mean(np.abs(data[name]) ** 2, axis=(1, 2)), 1e-150)) + 30
                ).tolist()
                for name in ("signal", "interference", "noise", "y")
            }
            return {
                "metadata": metadata,
                "frequency_mhz": (data["frequency_offset_hz"] / 1e6).tolist(),
                "channel_db": channel_db.tolist(),
                "time_ms": (data["time_s"] * 1000).tolist(),
                "powers_dbm": powers,
            }

    @app.get("/api/jobs/{job_id}/download/{filename}")
    def download(job_id: str, filename: str):
        folder = manager.folder(job_id)
        allowed = {
            "result.zip": folder / "result.zip",
            "observations.npz": folder / "result/observations.npz",
            "metadata.json": folder / "result/metadata.json",
            "config.json": folder / "config.json",
            "worker.log": folder / "worker.log",
        }
        if filename not in allowed or not allowed[filename].is_file():
            raise HTTPException(404, "File not found")
        if (
            filename in ("result.zip", "observations.npz", "metadata.json")
            and manager.state(job_id)["status"] != "completed"
        ):
            raise HTTPException(409, "Result is not ready")
        return FileResponse(allowed[filename], filename=filename)

    app.mount("/", StaticFiles(directory=str(files("channelhub").joinpath("web")), html=True), name="web")
    return app
