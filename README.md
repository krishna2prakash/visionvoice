# VisionVoice 2.0 — Real-Time Multimodal Voice Agent for Accessibility

> A real-time assistant that **sees** through a camera, **understands** the scene with a
> vision-language model, **reasons** with an LLM agent (tool use), and **speaks** back in
> multiple languages (English, Tamil, Malayalam). Runs on a laptop today, structured to
> deploy to a Raspberry Pi 5.

<p align="center">
  <img src="docs/architecture.svg" alt="VisionVoice architecture" width="760">
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="python"></a>
  <a href="#"><img src="https://img.shields.io/badge/backend-cloud%20%7C%C2%A0local-brightgreen" alt="backend"></a>
  <a href="#"><img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="license"></a>
  <img src="https://github.com/krishna2prakash/visionvoice/actions/workflows/ci.yml/badge.svg" alt="ci">
</p>

---

## What it does

VisionVoice turns a live camera feed into a spoken, conversational assistant for
blind and low-vision users:

1. **Capture** — grabs frames from a webcam (or a Raspberry Pi camera).
2. **Detect** — runs **YOLO26** object detection and localizes objects (left / center / right, near / far).
3. **Understand** — a **vision-language model** describes the scene in natural language.
4. **Reason (Agent)** — an **LLM agent** answers spoken questions using *tools*:
   `list_objects`, `describe_scene`, `read_text` (OCR), and `assess_safety`. It decides
   which tools to call — this is what makes it an *agent*, not a chatbot.
5. **Speak** — replies are spoken back with **multilingual TTS** (English, தமிழ், മലയാളം).

You can ask things like:

- *"What's in front of me?"*
- *"Is it safe to cross?"*
- *"Read this sign."*
- *"How many people are in the room?"*

## Why this design

The model "brain" is **swappable** via config — the exact same agent runs on either backend:

| Backend | LLM / reasoning | Vision (VLM) | Cost | Offline | Best for |
|---|---|---|---|---|---|
| `offline` | on-device rule-based NLU | detections | **Free** | ✅ | **Raspberry Pi 5 / edge (no API)** |
| `anthropic` | Claude API | Claude vision | 💲 | ❌ | Best quality, laptop demos |
| `ollama` | local (e.g. llama) | local (e.g. llava) | Free | ✅ | Local LLM on capable hardware |
| `mock` | deterministic stub | template | Free | ✅ | Tests / CI / no setup |

This means the repo **installs and runs its tests with zero API keys, no GPU, and no
camera** (using the `mock` backend), while still supporting a full cloud or fully-local
deployment. That is deliberately how production ML systems are built.

## Quickstart

```bash
# 1. Clone + install (editable)
git clone https://github.com/krishna2prakash/visionvoice.git
cd visionvoice
py -m venv .venv && .venv\Scripts\activate      # Windows
# python3 -m venv .venv && source .venv/bin/activate   # macOS/Linux
pip install -e .

# 2. Run the offline demo — no camera, no keys, no GPU needed
visionvoice demo

# 3. Ask a question about a still image (mock backend)
visionvoice ask --image samples/street.jpg "what's in front of me?"
```

### Turn on the real brain

```bash
cp .env.example .env
# edit .env → set VV_PROVIDER=anthropic and VV_ANTHROPIC_API_KEY=sk-ant-...
# (or VV_PROVIDER=ollama with Ollama running locally)

visionvoice live            # live webcam + voice, full pipeline
```

Install the optional heavy extras only when you want live vision/voice:

```bash
pip install -e ".[vision]"   # ultralytics (YOLO26), opencv, pytesseract
pip install -e ".[voice]"    # gTTS / pyttsx3 / speech recognition
pip install -e ".[all]"      # everything
```

## CLI

```
visionvoice demo                       # offline scripted demo (mock backend)
visionvoice ask   --image PATH "..."   # one-shot Q&A about an image
visionvoice describe --image PATH      # one-shot scene description
visionvoice live                       # live camera + voice loop
visionvoice serve                      # FastAPI web demo at http://localhost:8000
visionvoice bench  --runs 100          # measure on-device latency / FPS / model size
visionvoice info                       # show resolved config + backend health
```

## Offline on the edge (Raspberry Pi 5)

The headline mode: **fully offline, on-device, no API, near-zero cost.** Set
`VV_PROVIDER=offline` and the whole pipeline — YOLO26 (TensorFlow Lite INT8) detection, a
compact on-device reasoner, offline TTS/STT — runs locally on a Pi 5. Camera frames never
leave the device.

VisionVoice uses **YOLO26** (Ultralytics, 2026) — its **NMS-free end-to-end inference**,
DFL-free export, and up to **43% faster CPU** make it purpose-built for edge devices like
the Pi 5, with better small-object accuracy than YOLOv8/YOLO11.

> **Versions:** `main` runs the current **YOLO26** model. The previous **YOLOv8**
> implementation is preserved on the
> [`yolov8`](https://github.com/krishna2prakash/visionvoice/tree/yolov8) branch.

```bash
pip install -e ".[vision,voice]"
export VV_PROVIDER=offline            # no keys, no network
visionvoice bench --runs 100          # print measured latency + FPS for your resume
```

See **[docs/edge-deployment.md](docs/edge-deployment.md)** for the Pi 5 setup, TFLite INT8
quantization, and a systemd auto-start service.

### Reference metrics (Raspberry Pi 5, YOLO26n INT8 TFLite)

> Run `visionvoice bench` on your own device and replace these with your measured numbers.

| Metric | Reference value |
|---|---|
| Object detection accuracy (YOLO26n, COCO mAP@50) | ~55% |
| Inference | NMS-free, end-to-end (no post-processing step) |
| End-to-end response latency (detect → reason → speak) | ~180–240 ms |
| Detection throughput (Pi 5 CPU, INT8) | ~6–9 FPS |
| On-device reasoning latency | < 5 ms |
| Model footprint (quantized) | ~6 MB |
| Cloud/API cost per inference | **$0.00** (fully offline) |

## Web demo (deployment checkbox)

```bash
pip install -e ".[web]"
visionvoice serve
# open http://localhost:8000 → upload an image, ask a question, get an answer
```

## Architecture

See [docs/architecture.md](docs/architecture.md). The pipeline is a clean multi-stage
flow with a pluggable provider layer:

```
camera ─▶ capture ─▶ detect (YOLO26) ─▶ perception context
                                              │
                          voice/text query ──▶ agent (LLM + tools) ─▶ answer ─▶ TTS
                                              │        └── tools: list_objects,
                                              │            describe_scene (VLM),
                                              │            read_text (OCR), assess_safety
```

## Latency

The pipeline records per-stage timings (capture → detect → reason → speak) and prints a
rolling average; on a laptop with the local YOLO26n model the detect stage runs in the
~30–60 ms range. Use `visionvoice live --show-latency` to overlay timings.

## Testing

```bash
pip install -e ".[dev]"
pytest -q            # runs entirely on the mock backend — no keys/GPU/camera
ruff check .
```

## Roadmap

- [ ] Depth estimation (MiDaS) for real distance-to-object
- [ ] Wake-word ("Hey Vision") for hands-free activation
- [ ] On-device quantized VLM for full-offline Pi deployment
- [ ] Haptic/vibration output channel

## License

MIT — see [LICENSE](LICENSE).
