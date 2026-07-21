# Architecture

VisionVoice is a **multi-stage pipeline** with a **swappable model-provider layer**.
The design goal: the same agent logic runs unchanged on a cloud API, a fully-local
stack, or a deterministic offline mock — so the project installs and tests anywhere while
still supporting a real deployment.

```
                            ┌──────────────────────────────────────────┐
   camera / image ──▶ capture ──▶ detect (YOLOv8) ──▶ PerceptionSnapshot │
                            └───────────────────────────────┬──────────┘
                                                            │
  voice / text query ─▶ STT ─▶ ┌──────────── Agent (LLM loop) ───────────┐
                               │  observe → decide → act → answer         │
                               │  tools: list_objects, describe_scene,    │
                               │         read_text (OCR), assess_safety   │
                               └───────────────┬─────────────────────────┘
                                               │ answer text
                                     multilingual TTS (en / ta / ml) ─▶ 🔊
```

## Stages

| Stage | Module | Responsibility |
|---|---|---|
| Capture | `capture.py` | Threaded webcam grab, Pi camera, or still image; JPEG encode for the VLM |
| Detect | `detection.py` | YOLOv8 → `Detection` objects with direction + rough distance |
| Perceive | `types.py` | `PerceptionSnapshot` bundles detections + image bytes at one instant |
| Reason | `agent/` | The LLM agent loop and its perception tools |
| Understand | `vision/scene.py` | VLM caption, or a detection-derived summary as fallback |
| Speak | `speech/tts.py` | Multilingual TTS (translates non-primary languages via the provider) |
| Listen | `speech/stt.py` | Whisper / Google / keyboard voice input |
| Orchestrate | `pipeline.py` | Wires it together, measures per-stage latency |

## Provider layer (the swappable "brain")

`providers/base.py` defines `ModelProvider` with two methods — `chat()` (one agent turn,
optionally emitting tool calls) and `describe_image()` (VLM caption). Three
implementations:

- **`AnthropicProvider`** — Claude Messages API (tool use + base64 vision).
- **`OllamaProvider`** — local text model + local vision model (llava).
- **`MockProvider`** — deterministic, dependency-free; performs a genuine two-step
  tool-use cycle so the whole agent is testable offline.

`build_provider(settings)` selects one from `VV_PROVIDER`. Heavy SDKs are imported lazily,
so choosing `mock` never requires `anthropic` or `ollama` to be installed.

## Why the agent is an *agent*, not a chatbot

The model is not handed a blob of scene text. It is given **tools** and decides which to
call for a given question — read a sign (OCR), count objects (detections), judge safety
(heuristic over nearby hazards), or describe the whole scene (VLM). That observe → decide
→ act loop lives in `agent/assistant.py` and is completely provider-agnostic.

## Latency

`pipeline.LatencyMeter` keeps a rolling per-stage average (capture / detect / reason /
speak) so the "real-time" claim is measurable rather than asserted. `visionvoice live
--show-latency` prints it each turn.
