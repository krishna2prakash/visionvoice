"""A minimal FastAPI app: upload an image, get a spoken-style answer.

Run with:  visionvoice serve   (or  uvicorn visionvoice.web.server:app --reload)
"""

from __future__ import annotations

from visionvoice.config import get_settings
from visionvoice.pipeline import Pipeline
from visionvoice.types import PerceptionSnapshot

try:
    from fastapi import FastAPI, File, Form, UploadFile
    from fastapi.responses import HTMLResponse, JSONResponse
except ImportError as exc:  # pragma: no cover
    raise RuntimeError('The web extra is required: pip install -e ".[web]"') from exc

app = FastAPI(title="VisionVoice", version="2.0.0")

# One shared pipeline for the process (providers/models are reused across requests).
_pipeline: Pipeline | None = None


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline(get_settings())
    return _pipeline


def _snapshot_from_bytes(pipe: Pipeline, data: bytes) -> tuple[PerceptionSnapshot, object | None]:
    frame = None
    detections = []
    try:
        import cv2
        import numpy as np

        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is not None:
            try:
                detections = pipe.detector.detect(frame)
            except Exception:
                pass
    except Exception:
        pass
    snapshot = PerceptionSnapshot(
        detections=detections,
        image_bytes=data if pipe.provider.supports_vision else None,
    )
    return snapshot, frame


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _PAGE


@app.post("/describe")
async def describe(image: UploadFile = File(...)) -> JSONResponse:
    pipe = get_pipeline()
    data = await image.read()
    snapshot, frame = _snapshot_from_bytes(pipe, data)
    caption = pipe.describe(snapshot=snapshot, frame=frame, speak=False)
    return JSONResponse({"caption": caption, "objects": snapshot.object_labels(),
                         "latency_ms": pipe.latency.averages_ms()})


@app.post("/ask")
async def ask(image: UploadFile = File(...), query: str = Form(...)) -> JSONResponse:
    pipe = get_pipeline()
    data = await image.read()
    snapshot, frame = _snapshot_from_bytes(pipe, data)
    answer = pipe.answer(query, snapshot=snapshot, frame=frame, speak=False)
    return JSONResponse({"answer": answer, "objects": snapshot.object_labels(),
                         "latency_ms": pipe.latency.averages_ms()})


@app.get("/health")
def health() -> JSONResponse:
    pipe = get_pipeline()
    ok, detail = pipe.provider.health()
    return JSONResponse({"ok": ok, "detail": detail, "provider": pipe.provider.name})


_PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>VisionVoice</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 640px; margin: 2rem auto; padding: 0 1rem; }
  h1 { margin-bottom: .2rem; } .sub { color:#666; margin-top:0; }
  input, button, textarea { font: inherit; padding:.5rem; margin:.25rem 0; width:100%; box-sizing:border-box; }
  button { background:#1f7a4d; color:#fff; border:0; border-radius:6px; cursor:pointer; }
  #out { white-space:pre-wrap; background:#f4f4f4; padding:1rem; border-radius:8px; min-height:2rem; }
  img#preview { max-width:100%; border-radius:8px; margin:.5rem 0; }
</style>
</head>
<body>
  <h1>VisionVoice 2.0</h1>
  <p class="sub">Upload a photo, then ask about it. Multimodal voice agent for accessibility.</p>
  <input type="file" id="file" accept="image/*">
  <img id="preview" hidden>
  <input type="text" id="query" placeholder="e.g. What's in front of me?  Is it safe to cross?" value="What's in front of me?">
  <button onclick="ask()">Ask</button>
  <button onclick="describe()" style="background:#37546b">Describe scene</button>
  <h3>Answer</h3>
  <div id="out">…</div>
<script>
const file = document.getElementById('file'), preview = document.getElementById('preview'), out = document.getElementById('out');
file.onchange = () => { if (file.files[0]) { preview.src = URL.createObjectURL(file.files[0]); preview.hidden = false; } };
function form() { const f = new FormData(); if (file.files[0]) f.append('image', file.files[0]); return f; }
async function post(url, f) { out.textContent = 'thinking…'; const r = await fetch(url, {method:'POST', body:f}); const j = await r.json(); out.textContent = (j.answer || j.caption || JSON.stringify(j)) + (j.objects && j.objects.length ? '\\n\\ndetected: ' + j.objects.join(', ') : ''); }
async function ask() { if (!file.files[0]) { out.textContent='Please choose an image first.'; return; } const f = form(); f.append('query', document.getElementById('query').value); post('/ask', f); }
async function describe() { if (!file.files[0]) { out.textContent='Please choose an image first.'; return; } post('/describe', form()); }
</script>
</body>
</html>
"""
