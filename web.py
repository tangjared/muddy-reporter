"""FastAPI server: serves the static frontend + a streaming pipeline endpoint.

Single command to run:
    python web.py     (or)    uvicorn web:app --reload --port 8000

Endpoints:
- GET  /                       → index.html (single-page app)
- GET  /api/health             → provider info + heartbeat
- GET  /api/reports            → list of cached report tickers (from outputs/)
- GET  /api/reports/{ticker}   → cached report JSON for a ticker
- GET  /api/reports/{ticker}/html → cached printable HTML report
- POST /api/generate           → kick off pipeline, returns job_id
- GET  /api/stream/{job_id}    → Server-Sent Events stream of progress + final result
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# Bootstrap src/ layout so this works without `pip install -e .`
_SRC = Path(__file__).parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from muddy_reporter.llm import provider_info  # noqa: E402
from muddy_reporter.pipeline import generate_report  # noqa: E402


ROOT = Path(__file__).parent
FRONTEND_DIR = ROOT / "frontend"
OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


app = FastAPI(title="Muddy Reporter", docs_url="/api/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Job orchestration: each /api/generate call starts a background thread that
# pushes progress events into a queue. The /api/stream/{job_id} endpoint
# drains that queue as Server-Sent Events.
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    ticker: str
    max_filings: int = 5


class _Job:
    def __init__(self, ticker: str, max_filings: int):
        self.id = uuid.uuid4().hex[:12]
        self.ticker = ticker.strip().upper()
        self.max_filings = max_filings
        self.queue: asyncio.Queue = asyncio.Queue()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.done = False
        self.error: str | None = None
        self.report: dict | None = None
        self.created_at = time.time()


JOBS: dict[str, _Job] = {}


def _run_pipeline_blocking(job: _Job) -> None:
    """Runs in a worker thread. Pushes progress events into the asyncio queue."""

    def _push(event: str, data: dict) -> None:
        if job.loop is None:
            return
        payload = {"event": event, "data": data, "ts": time.time()}
        try:
            job.loop.call_soon_threadsafe(job.queue.put_nowait, payload)
        except RuntimeError:
            pass

    def _progress(message: str, fraction: float) -> None:
        _push("progress", {"message": message, "fraction": fraction})

    try:
        out_html = OUTPUTS_DIR / f"{job.ticker}.html"
        result = generate_report(
            ticker=job.ticker,
            max_filings=job.max_filings,
            out_html_path=str(out_html),
            progress=_progress,
        )
        report_dict = result.report.model_dump(mode="json")
        job.report = report_dict
        _push("complete", {"report": report_dict, "html_path": str(out_html.name)})
    except Exception as e:
        job.error = str(e)
        _push("error", {"message": str(e)})
    finally:
        job.done = True
        # Sentinel so the SSE generator knows to close cleanly.
        if job.loop is not None:
            job.loop.call_soon_threadsafe(job.queue.put_nowait, {"event": "_close", "data": {}})


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@app.get("/api/health")
async def health() -> dict:
    return {
        "ok": True,
        "provider": provider_info(),
    }


@app.get("/api/reports")
async def list_reports() -> dict:
    items = []
    for jpath in sorted(OUTPUTS_DIR.glob("*.json")):
        try:
            data = json.loads(jpath.read_text(encoding="utf-8"))
            items.append(
                {
                    "ticker": data.get("ticker") or jpath.stem,
                    "company_name": data.get("company_name"),
                    "generated_at_iso": data.get("generated_at_iso"),
                    "red_flags": len(data.get("red_flags", [])),
                    "anomalies": len(data.get("financial_anomalies", [])),
                    "provider": (data.get("provider_info") or {}).get("provider"),
                }
            )
        except Exception:
            continue
    items.sort(key=lambda x: x.get("generated_at_iso") or "", reverse=True)
    return {"reports": items}


@app.get("/api/reports/{ticker}")
async def get_report(ticker: str) -> JSONResponse:
    p = OUTPUTS_DIR / f"{ticker.upper()}.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="report not found")
    return JSONResponse(json.loads(p.read_text(encoding="utf-8")))


@app.get("/api/reports/{ticker}/html", response_class=HTMLResponse)
async def get_report_html(ticker: str) -> HTMLResponse:
    p = OUTPUTS_DIR / f"{ticker.upper()}.html"
    if not p.exists():
        raise HTTPException(status_code=404, detail="report html not found")
    return HTMLResponse(p.read_text(encoding="utf-8"))


@app.post("/api/generate")
async def kick_off(req: GenerateRequest) -> dict:
    if not req.ticker.strip():
        raise HTTPException(status_code=400, detail="ticker required")
    job = _Job(req.ticker, req.max_filings)
    job.loop = asyncio.get_event_loop()
    JOBS[job.id] = job
    threading.Thread(target=_run_pipeline_blocking, args=(job,), daemon=True).start()
    return {"job_id": job.id, "ticker": job.ticker}


@app.get("/api/stream/{job_id}")
async def stream(job_id: str) -> EventSourceResponse:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    async def gen():
        # Replay any already-queued events first, then keep streaming.
        while True:
            payload = await job.queue.get()
            if payload.get("event") == "_close":
                break
            yield {"event": payload["event"], "data": json.dumps(payload["data"])}
        # Send a terminal event so the client can close cleanly.
        yield {"event": "end", "data": "{}"}

    # ping= keeps the connection alive through Render/Cloudflare proxies that
    # otherwise close idle HTTP/1.1 streams after ~30s.
    return EventSourceResponse(gen(), ping=15)


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------


# Cleanest pattern: mount the entire frontend directory at /. We add a tiny
# wrapper so '/' still serves index.html.

@app.get("/", response_class=HTMLResponse)
async def index() -> FileResponse:
    # No-store so theme/style changes show up on the next refresh — the SPA is
    # tiny (single HTML file), so cache benefit is irrelevant here.
    return FileResponse(
        FRONTEND_DIR / "index.html",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        },
    )


if (FRONTEND_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR / "static"), name="static")


# ---------------------------------------------------------------------------
# Local dev runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import socket

    import uvicorn

    # Render injects PORT; locally we default to 8000.
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    # Force unbuffered stdout so Render's log stream shows progress in real time.
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

    if host in {"0.0.0.0", "::"} and not os.getenv("RENDER"):
        # Local-dev convenience: print the LAN URL for other devices on the Wi-Fi.
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            lan_ip = s.getsockname()[0]
            s.close()
        except Exception:
            lan_ip = None
        print()
        print("  Muddy Reporter — listening on:")
        print(f"    Local:    http://127.0.0.1:{port}")
        if lan_ip:
            print(f"    Network:  http://{lan_ip}:{port}   ← open this on another device on the same Wi-Fi")
        print()

    uvicorn.run("web:app", host=host, port=port, reload=False, log_level="info")
