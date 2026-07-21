# VisionVoice web demo — cloud/mock backend (no camera hardware in-container).
# For live camera + local models, run on the host with the [all] extra instead.
FROM python:3.12-slim

WORKDIR /app

# System deps for OpenCV / Tesseract (used when the vision extra is installed).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

# Install the web demo dependencies (swap for ".[all]" to add vision/voice).
RUN pip install --no-cache-dir -e ".[web,anthropic]"

ENV VV_PROVIDER=mock
EXPOSE 8000

CMD ["visionvoice", "serve", "--host", "0.0.0.0", "--port", "8000"]
