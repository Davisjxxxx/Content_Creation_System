from __future__ import annotations

import json
import shutil
from pathlib import Path

import typer
import yaml

from .composer import build_job
from .models import AvatarManifest, ProviderConfig, RenderJob, ShotSpec
from .policy import evaluate_policy
from .providers import ComfyUIProvider
from .qa import ffprobe_video

app = typer.Typer(no_args_is_help=True, help="Avatar V2 reference-driven video orchestration CLI")


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_avatar(path: Path) -> AvatarManifest:
    return AvatarManifest.model_validate(_load_yaml(path))


def _load_shot(path: Path) -> ShotSpec:
    return ShotSpec.model_validate(_load_yaml(path))


def _load_provider(path: Path) -> ProviderConfig:
    return ProviderConfig.model_validate(_load_yaml(path))


@app.command()
def validate(
    avatar: Path = typer.Argument(..., exists=True, readable=True),
    shot: Path = typer.Argument(..., exists=True, readable=True),
) -> None:
    """Validate manifests and adult-content hard boundaries."""
    avatar_obj = _load_avatar(avatar)
    shot_obj = _load_shot(shot)
    decision = evaluate_policy(avatar_obj, shot_obj)
    typer.echo(json.dumps({"valid": decision.allowed, "reason": decision.reason}, indent=2))
    if not decision.allowed:
        raise typer.Exit(code=2)


@app.command()
def plan(
    avatar: Path = typer.Argument(..., exists=True, readable=True),
    shot: Path = typer.Argument(..., exists=True, readable=True),
    out: Path = typer.Option(Path("build/job.json"), "--out", "-o"),
) -> None:
    """Compose one renderer-neutral job from identity, scene and motion references."""
    job = build_job(_load_avatar(avatar), _load_shot(shot))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(job.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(f"planned {job.job_id} -> {out}")
    typer.echo(f"selected engine: {job.selected_engine}; seed: {job.shot.seed}")


@app.command("plan-batch")
def plan_batch(
    avatar: Path = typer.Argument(..., exists=True, readable=True),
    shot: Path = typer.Argument(..., exists=True, readable=True),
    count: int = typer.Option(4, "--count", "-n", min=1, max=32),
    out_dir: Path = typer.Option(Path("build/jobs"), "--out-dir"),
) -> None:
    """Create best-of-N jobs using consecutive deterministic seeds."""
    avatar_obj = _load_avatar(avatar)
    base = _load_shot(shot)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index in range(count):
        candidate_shot = base.model_copy(update={"seed": base.seed + index})
        job = build_job(avatar_obj, candidate_shot)
        path = out_dir / f"{base.shot_id}_seed_{candidate_shot.seed}.json"
        path.write_text(job.model_dump_json(indent=2), encoding="utf-8")
        manifest.append({"seed": candidate_shot.seed, "job": str(path), "engine": job.selected_engine})
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    typer.echo(json.dumps(manifest, indent=2))


@app.command()
def doctor(
    provider: Path = typer.Argument(..., exists=True, readable=True),
) -> None:
    """Check local rendering prerequisites and ComfyUI connectivity."""
    config = _load_provider(provider)
    report = {
        "ffmpeg": bool(shutil.which("ffmpeg")),
        "ffprobe": bool(shutil.which("ffprobe")),
        "provider": config.provider,
        "base_url": config.base_url,
        "comfyui": False,
    }
    try:
        ComfyUIProvider(config).health()
        report["comfyui"] = True
    except Exception as exc:  # diagnostic command should report, not hide, failures
        report["comfyui_error"] = str(exc)
    typer.echo(json.dumps(report, indent=2))
    if not report["comfyui"]:
        raise typer.Exit(code=2)


@app.command()
def render(
    job: Path = typer.Argument(..., exists=True, readable=True),
    provider: Path = typer.Argument(..., exists=True, readable=True),
    workflow: Path = typer.Option(..., "--workflow", "-w", exists=True, readable=True),
    dry_run: bool = typer.Option(False, "--dry-run", help="Resolve the workflow without queueing it."),
    resolved_out: Path = typer.Option(Path("build/resolved_workflow.json"), "--resolved-out"),
) -> None:
    """Resolve placeholders and render through a ComfyUI API-format workflow."""
    render_job = RenderJob.model_validate_json(job.read_text(encoding="utf-8"))
    config = _load_provider(provider)
    engine = ComfyUIProvider(config)

    if dry_run:
        resolved = engine.resolve_workflow(workflow, render_job, stage_assets=False)
        resolved_out.parent.mkdir(parents=True, exist_ok=True)
        resolved_out.write_text(json.dumps(resolved, indent=2), encoding="utf-8")
        typer.echo(f"resolved workflow -> {resolved_out}")
        return

    result = engine.render(workflow, render_job)
    typer.echo(json.dumps({"prompt_id": result["prompt_id"], "outputs": result["outputs"]}, indent=2))


@app.command("qa")
def qa_command(
    video: Path = typer.Argument(..., exists=True, readable=True),
) -> None:
    """Run structural video QA using ffprobe."""
    typer.echo(json.dumps(ffprobe_video(video), indent=2))


if __name__ == "__main__":
    app()
