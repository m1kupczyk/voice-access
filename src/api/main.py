"""FastAPI app and routes.

Audio is sent as the raw request body (any format ffmpeg or libsndfile can decode) and is read
into memory only; multipart uploads are not used because Starlette spools large parts to disk.

Run: python -m src.api.main
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response

from src.api.schemas import (
    ChallengeResponse,
    DeleteResponse,
    EnrollResponse,
    HealthResponse,
    StatsResponse,
    VerifyResponse,
)
from src.api.service import ApiError, VoiceAccessService, build_service
from src.config import load_config

STATIC_DIR = Path(__file__).resolve().parent / "static"


async def read_audio_body(request: Request, max_bytes: int) -> bytes:
    """Read the request body into memory, aborting once it exceeds `max_bytes`."""
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > max_bytes:
        raise ApiError(413, f"upload larger than {max_bytes} bytes")
    buf = bytearray()
    async for chunk in request.stream():
        buf.extend(chunk)
        if len(buf) > max_bytes:
            raise ApiError(413, f"upload larger than {max_bytes} bytes")
    return bytes(buf)


def create_app(
    cfg: dict[str, Any] | None = None, service: VoiceAccessService | None = None
) -> FastAPI:
    """Build the app; models are loaded once in the lifespan unless a service is injected."""
    cfg = cfg if cfg is not None else load_config()
    max_bytes = int(cfg["api"]["max_upload_bytes"])

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.service = service if service is not None else build_service(cfg)
        try:
            yield
        finally:
            app.state.service.db.close()

    app = FastAPI(title="voice-access", lifespan=lifespan)

    def svc(request: Request) -> VoiceAccessService:
        return request.app.state.service

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> dict[str, Any]:
        return svc(request).health()

    @app.post("/enroll", response_model=EnrollResponse)
    async def enroll(request: Request, name: str = Query(...)) -> dict[str, Any]:
        data = await read_audio_body(request, max_bytes)
        return await run_in_threadpool(svc(request).enroll, name, data)

    @app.get("/challenge", response_model=ChallengeResponse)
    async def challenge(request: Request, user_id: int = Query(...)) -> dict[str, Any]:
        return await run_in_threadpool(svc(request).challenge, user_id)

    @app.post("/verify", response_model=VerifyResponse)
    async def verify(
        request: Request, user_id: int = Query(...), nonce: str = Query(...),
        scenario: str | None = Query(None),
    ) -> dict[str, Any]:
        data = await read_audio_body(request, max_bytes)
        return await run_in_threadpool(svc(request).verify, user_id, nonce, data, scenario)

    @app.get("/stats", response_model=StatsResponse)
    async def stats(request: Request, user_id: int | None = Query(None)) -> dict[str, Any]:
        return await run_in_threadpool(svc(request).stats, user_id)

    @app.get("/stats.csv")
    async def stats_csv(request: Request, user_id: int | None = Query(None)) -> Response:
        body = await run_in_threadpool(svc(request).stats_csv, user_id)
        return Response(body, media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": 'attachment; filename="attempts.csv"'})

    @app.delete("/users/{user_id}", response_model=DeleteResponse)
    async def delete_user(request: Request, user_id: int) -> dict[str, Any]:
        return await run_in_threadpool(svc(request).delete_user, user_id)

    return app


def main() -> None:
    import uvicorn

    cfg = load_config()
    uvicorn.run(create_app(cfg), host=cfg["api"]["host"], port=int(cfg["api"]["port"]))


if __name__ == "__main__":
    main()
