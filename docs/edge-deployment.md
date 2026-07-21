# Offline Edge Deployment (Raspberry Pi 5)

VisionVoice runs **fully offline on a Raspberry Pi 5** — no API keys, no network, no
per-request cost. The on-device stack is:

| Stage | On-device component | Notes |
|---|---|---|
| Detect | **YOLOv8-n**, exported to **TensorFlow Lite (INT8)** | quantized for ARM; ~6 MB weights |
| Reason | **`offline` provider** | compact rule-based NLU + tool routing — no LLM weights |
| Speak | **pyttsx3 / eSpeak-NG** | offline synthesis, multilingual |
| Listen | **Whisper (tiny/base) or Vosk** | offline speech-to-text |

Everything is local, so the only recurring cost is electricity.

## 1. Base setup on the Pi

```bash
sudo apt update && sudo apt install -y python3-venv espeak-ng tesseract-ocr libatlas-base-dev
git clone https://github.com/krishna2prakash/visionvoice.git
cd visionvoice
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[vision,voice]"
```

## 2. Configure for offline

```bash
cp .env.example .env
# .env already defaults to the offline edge stack:
#   VV_PROVIDER=offline
#   VV_TTS_ENGINE=pyttsx3
#   VV_STT_ENGINE=whisper
#   VV_YOLO_MODEL=yolov8n_int8.tflite
```

## 3. Export a quantized TFLite model (once, on any machine)

TensorFlow Lite INT8 quantization is what brings YOLOv8 into real-time range on the Pi's
CPU and shrinks the model for the edge:

```bash
pip install ultralytics
yolo export model=yolov8n.pt format=tflite int8=True
# produces yolov8n_full_integer_quant.tflite  → copy to the Pi and point VV_YOLO_MODEL at it
```

Ultralytics runs the exported `.tflite` directly via `YOLO("model.tflite")`, so no code
changes are needed — just set `VV_YOLO_MODEL`.

## 4. Measure it on the device

```bash
visionvoice bench --runs 100          # detection + reasoning latency, FPS, model size
visionvoice info                      # confirm backend health
visionvoice live --show-latency       # live loop with per-stage timings overlaid
```

Put the numbers `bench` prints straight into your resume — they are measured on *your*
hardware, not estimated.

## 5. Run as a service (auto-start on boot)

`/etc/systemd/system/visionvoice.service`:

```ini
[Unit]
Description=VisionVoice offline assistant
After=multi-user.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/visionvoice
ExecStart=/home/pi/visionvoice/.venv/bin/visionvoice live
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now visionvoice
```

## Why offline-on-edge is the differentiator

- **Privacy** — camera frames never leave the device.
- **Availability** — works with no internet (buses, rural areas, power-constrained sites).
- **Cost** — zero per-inference cost vs. a cloud VLM per frame.
- **Latency** — no network round-trip; the whole loop is on-device.

The same codebase still upgrades to a cloud VLM (`VV_PROVIDER=anthropic`) or a local LLM
(`VV_PROVIDER=ollama`) by changing one config value — so you can demo the richest version
on a laptop and ship the lean version to the Pi.
