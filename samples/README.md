# Sample images

Drop any JPEG/PNG here to try the one-shot commands:

```bash
visionvoice ask --image samples/your-photo.jpg "what's in front of me?"
visionvoice describe --image samples/your-photo.jpg
```

- With the **vision** extra installed (`pip install -e ".[vision]"`), YOLO26 runs local
  object detection on the image.
- With `VV_PROVIDER=anthropic` (or `ollama`), the vision-language model also describes the
  actual image.
- With the default `mock` backend, answers are generated from detections only — handy for
  wiring things up without any setup.

Images placed here are git-ignored except this file, so your test photos won't be
committed.
