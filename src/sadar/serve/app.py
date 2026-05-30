from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
def flights(limit: int = 30) -> list[dict]:
    return service.list_flights(limit)


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
