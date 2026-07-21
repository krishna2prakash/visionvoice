"""VisionVoice command-line interface."""

from __future__ import annotations

import logging

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from visionvoice import __version__
from visionvoice.config import get_settings
from visionvoice.pipeline import Pipeline
from visionvoice.types import Detection, PerceptionSnapshot

app = typer.Typer(add_completion=False, help="VisionVoice 2.0 — multimodal voice agent for accessibility.")
console = Console()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(message)s",
    )


# A scripted scene so `visionvoice demo` works with no camera, keys, or GPU.
_DEMO_SNAPSHOT = PerceptionSnapshot(
    detections=[
        Detection("person", 0.94, (0.05, 0.30, 0.28, 0.95), position="left", distance="close"),
        Detection("chair", 0.88, (0.42, 0.55, 0.63, 0.98), position="center", distance="mid"),
        Detection("car", 0.81, (0.70, 0.40, 0.99, 0.85), position="far-right", distance="very close"),
        Detection("laptop", 0.77, (0.45, 0.50, 0.58, 0.65), position="center", distance="far"),
    ],
    width=1280,
    height=720,
)
_DEMO_QUESTIONS = [
    "What's in front of me?",
    "Is it safe to move forward?",
    "How many people do you see?",
]


@app.command()
def demo(verbose: bool = typer.Option(True, help="Show pipeline logs.")) -> None:
    """Run a scripted, fully-offline demo (mock backend, no camera/keys/GPU)."""
    _setup_logging(verbose)
    settings = get_settings()
    console.print(
        Panel.fit(
            f"[bold]VisionVoice {__version__}[/] demo\n"
            f"backend: [cyan]{settings.provider}[/]   languages: [cyan]{settings.languages}[/]",
            border_style="green",
        )
    )
    pipe = Pipeline(settings)
    console.print("[dim]Scene:[/] a person to your left, a chair ahead, a car close on the right.\n")
    for q in _DEMO_QUESTIONS:
        console.print(f"[bold cyan]Q:[/] {q}")
        reply = pipe.answer(q, snapshot=_DEMO_SNAPSHOT, speak=True)
        console.print(f"[bold green]A:[/] {reply}\n")
    console.print(f"[dim]Latency: {pipe.latency.summary()}[/]")
    pipe.close()


@app.command()
def ask(
    query: str = typer.Argument(..., help="Question about the image/scene."),
    image: str = typer.Option(None, "--image", "-i", help="Path to an image file."),
    verbose: bool = typer.Option(False),
) -> None:
    """Ask a question about an image (or the live camera if --image is omitted)."""
    _setup_logging(verbose)
    pipe = Pipeline()
    if image:
        snapshot, frame = _snapshot_from_image(pipe, image)
        reply = pipe.answer(query, snapshot=snapshot, frame=frame)
    else:
        reply = pipe.answer(query)
    console.print(f"[bold green]A:[/] {reply}")
    pipe.close()


@app.command()
def describe(
    image: str = typer.Option(None, "--image", "-i", help="Path to an image file."),
    verbose: bool = typer.Option(False),
) -> None:
    """Describe an image (or the live camera if --image is omitted)."""
    _setup_logging(verbose)
    pipe = Pipeline()
    if image:
        snapshot, frame = _snapshot_from_image(pipe, image)
        caption = pipe.describe(snapshot=snapshot, frame=frame)
    else:
        caption = pipe.describe()
    console.print(f"[bold green]Scene:[/] {caption}")
    pipe.close()


@app.command()
def live(
    show_latency: bool = typer.Option(False, "--show-latency", help="Print stage timings each turn."),
    verbose: bool = typer.Option(True),
) -> None:
    """Live camera + voice loop. Ask questions; type 'quit' to exit."""
    _setup_logging(verbose)
    from visionvoice.speech.stt import build_stt

    settings = get_settings()
    pipe = Pipeline(settings)
    stt = build_stt(settings)
    console.print(Panel.fit("[bold]VisionVoice live[/] — ask about your surroundings. 'quit' to exit.", border_style="green"))
    try:
        while True:
            query = stt.listen("You: ")
            if not query:
                continue
            if query.lower() in {"quit", "exit", "stop"}:
                break
            snapshot, frame = pipe.perceive_live()
            reply = pipe.answer(query, snapshot=snapshot, frame=frame)
            console.print(f"[bold green]VisionVoice:[/] {reply}")
            if show_latency:
                console.print(f"[dim]{pipe.latency.summary()}[/]")
    except KeyboardInterrupt:
        pass
    finally:
        pipe.close()


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind host."),
    port: int = typer.Option(8000, help="Bind port."),
) -> None:
    """Launch the FastAPI web demo."""
    try:
        import uvicorn
    except ImportError:
        console.print('[red]Install the web extra:[/] pip install -e ".[web]"')
        raise typer.Exit(1) from None
    console.print(f"[green]Serving VisionVoice at[/] http://{host}:{port}")
    uvicorn.run("visionvoice.web.server:app", host=host, port=port, factory=False)


@app.command()
def info() -> None:
    """Show resolved configuration and backend health."""
    settings = get_settings()
    pipe = Pipeline(settings)
    ok, detail = pipe.provider.health()

    table = Table(title=f"VisionVoice {__version__}", show_header=False)
    table.add_row("provider", settings.provider)
    table.add_row("provider health", f"[green]ok[/] — {detail}" if ok else f"[red]not ready[/] — {detail}")
    table.add_row("vision (VLM)", "yes" if pipe.provider.supports_vision else "no (uses detections)")
    table.add_row("yolo model", settings.yolo_model)
    table.add_row("tts engine", settings.tts_engine)
    table.add_row("stt engine", settings.stt_engine)
    table.add_row("languages", settings.languages)
    console.print(table)
    pipe.close()


def _snapshot_from_image(pipe: Pipeline, path: str):
    """Build a snapshot from an image file: detect if possible, always attach bytes for VLM."""
    from pathlib import Path

    data = Path(path).read_bytes()
    frame = None
    detections = []
    try:
        import cv2  # noqa

        from visionvoice.capture import ImageFileSource

        frame = ImageFileSource(path).read()
        try:
            detections = pipe.detector.detect(frame)
        except Exception as exc:
            console.print(f"[yellow]Detection unavailable ({exc}); using VLM only.[/]")
    except Exception:
        console.print("[yellow]OpenCV not installed; skipping local detection.[/]")

    snapshot = PerceptionSnapshot(
        detections=detections,
        image_bytes=data if pipe.provider.supports_vision else None,
        image_media_type="image/jpeg",
    )
    return snapshot, frame


if __name__ == "__main__":
    app()
