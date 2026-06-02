from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from sadar.serve.inference import ConformanceService

service = ConformanceService(os.environ.get("SADAR_SERVE_CONFIG", "configs/serve.yaml"))

app = FastAPI(title="SADAR Flight Conformance Monitor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SimulationRequest(BaseModel):
    id: int
    kind: str
    magnitude: float = 0.0
    onset: float = 0.5


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "windows": int(len(service.window_scores)),
        "threshold": service.threshold,
        "device": str(service.device),
    }


@app.get("/api/flights")
def flights(limit: int = 30, order: str = "anomalous") -> list[dict]:
    return service.list_flights(limit, order)


@app.get("/api/metrics")
def metrics() -> dict:
    return service.metrics()


@app.get("/api/scene")
def scene(count: int = 12) -> dict:
    return service.scene(count)


@app.get("/api/flights/{flight_id}")
def flight(flight_id: int) -> dict:
    if flight_id < 0 or flight_id >= len(service.window_scores):
        raise HTTPException(status_code=404, detail="unknown flight")
    return service.flight_detail(flight_id)


@app.post("/api/simulate")
def simulate(request: SimulationRequest) -> dict:
    if request.id < 0 or request.id >= len(service.window_scores):
        raise HTTPException(status_code=404, detail="unknown flight")
    try:
        return service.simulate(request.id, request.kind, request.magnitude, request.onset)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


_frontend_dir = Path(os.environ.get("SADAR_FRONTEND_DIR", "frontend/dist"))
if _frontend_dir.is_dir():
    app.mount(
        "/assets",
        StaticFiles(directory=_frontend_dir / "assets"),
        name="frontend-assets",
    )

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str) -> FileResponse:
        candidate = _frontend_dir / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_frontend_dir / "index.html")
