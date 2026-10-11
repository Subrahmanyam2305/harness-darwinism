"""FastAPI backend for the evolution dashboard."""

from __future__ import annotations

import io
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from src.db import get_all_genomes, get_db, get_metrics, get_run, get_runs_for_genome

PROJECT_ROOT = Path(__file__).resolve().parent
REPLAY_DIR = PROJECT_ROOT / "data" / "replays"
RUNS_DIR = PROJECT_ROOT / "data" / "runs"
REPLAY_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Harness Darwinism", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Extract individual frame from GIF for synchronized replay
@app.get("/api/frame/{gif_name}/{frame_idx}")
def get_frame(gif_name: str, frame_idx: int):
    """Return a single PNG frame from a GIF replay."""
    gif_path = REPLAY_DIR / gif_name
    if not gif_path.exists():
        return Response(status_code=404)
    try:
        from PIL import Image

        with Image.open(str(gif_path)) as img:
            img.seek(min(frame_idx, img.n_frames - 1))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return Response(content=buf.getvalue(), media_type="image/png")
    except (OSError, EOFError):
        return Response(status_code=404)


# Serve GIF replays as static files
app.mount("/replays", StaticFiles(directory=str(REPLAY_DIR)), name="replays")


def _episode_artifact(run: dict, filename: str) -> Path | None:
    """Resolve one recorded artifact while keeping reads inside data/runs."""
    artifact_path = run.get("artifact_path")
    if not artifact_path:
        return None
    root = Path(artifact_path)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    root = root.resolve()
    allowed_root = RUNS_DIR.resolve()
    if not root.is_relative_to(allowed_root):
        return None
    target = (root / filename).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        return None
    return target


@app.get("/api/runs/{run_id}/replay.gif")
def episode_replay(run_id: str):
    """Serve the final GIF for a full Atlas episode record."""
    run = get_run(get_db(), run_id)
    artifact = _episode_artifact(run, "replay.gif") if run else None
    if artifact is None:
        return Response(status_code=404)
    return FileResponse(artifact, media_type="image/gif")


@app.get("/api/runs/{run_id}/frame/{frame_idx}")
def episode_frame(run_id: str, frame_idx: int):
    """Return one GIF frame for synchronized replay controls."""
    run = get_run(get_db(), run_id)
    artifact = _episode_artifact(run, "replay.gif") if run else None
    if artifact is None:
        return Response(status_code=404)
    try:
        from PIL import Image

        with Image.open(artifact) as image:
            image.seek(min(max(frame_idx, 0), image.n_frames - 1))
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return Response(content=buffer.getvalue(), media_type="image/png")
    except (OSError, EOFError):
        return Response(status_code=404)


@app.get("/api/health")
def health():
    return {"status": "ok"}


def _safe_timestamp(val):
    if hasattr(val, "isoformat"):
        return val.isoformat()
    return str(val) if val else None


def _serialize_run(run: dict) -> dict:
    """Expose legacy and full episode records through one dashboard contract."""
    document = dict(run)
    run_id = str(document.pop("_id", ""))
    trace = document.get("trace") if isinstance(document.get("trace"), dict) else {}
    history = trace.get("history", document.get("history", []))
    compact_history = [
        {
            "step": item.get("step"),
            "action": item.get("action"),
            "reward": item.get("reward", 0),
            "obs_snippet": item.get("obs_snippet", item.get("observation_after", ""))[
                :600
            ],
            "thought": item.get("thought", ""),
        }
        for item in history
        if isinstance(item, dict)
    ]
    document.update(
        {
            "id": run_id,
            "history": compact_history,
            "history_len": len(compact_history),
            "termination_reason": trace.get(
                "termination_reason", document.get("termination_reason")
            ),
        }
    )
    if document.get("artifact_path") and run_id:
        document["replay_url"] = f"/api/runs/{run_id}/replay.gif"
        document["frame_url"] = f"/api/runs/{run_id}/frame"
    timestamp = document.get("timestamp", document.get("created_at"))
    if timestamp is not None:
        document["timestamp"] = _safe_timestamp(timestamp)
    document.pop("trace", None)
    return document


@app.get("/api/metrics")
def metrics(arm: str = "full"):
    """Fitness curve: best/mean score per generation."""
    db = get_db()
    rows = get_metrics(db, arm=arm)
    for r in rows:
        r.pop("_id", None)
        if "timestamp" in r:
            r["timestamp"] = _safe_timestamp(r["timestamp"])
    return rows


@app.get("/api/genomes")
def genomes(arm: str | None = None):
    """All genomes (the lineage tree)."""
    db = get_db()
    gs = get_all_genomes(db, arm=arm)
    return [g.model_dump() for g in gs]


@app.get("/api/genomes/{genome_id}")
def genome_detail(genome_id: str):
    """Single genome with its runs."""
    db = get_db()
    from src.db import get_genome

    g = get_genome(db, genome_id)
    if not g:
        return {"error": "not found"}
    runs = get_runs_for_genome(db, genome_id)
    return {"genome": g.model_dump(), "runs": [_serialize_run(run) for run in runs]}


@app.get("/api/tree")
def lineage_tree(arm: str = "full"):
    """Lineage tree for D3 visualization."""
    db = get_db()
    gs = get_all_genomes(db, arm=arm)

    nodes = []
    edges = []
    for g in gs:
        nodes.append(
            {
                "id": g.id,
                "generation": g.generation,
                "status": g.status,
                "score": g.fitness.score,
                "rules_count": len(g.genome.rules),
                "selection_note": g.selection_note,
                "crossover_note": g.crossover_note,
                "mutation_note": g.mutation_note,
            }
        )
        for parent_id in g.parents:
            edges.append({"source": parent_id, "target": g.id})

    return {"nodes": nodes, "edges": edges}


@app.get("/api/achievements")
def achievement_grid(arm: str = "full"):
    """Achievement rates for the latest generation's champion vs baseline."""
    db = get_db()
    gs = get_all_genomes(db, arm=arm)
    if not gs:
        return {"baseline": {}, "champion": {}}

    gen0 = [g for g in gs if g.generation == 0]
    champions = [g for g in gs if g.status == "champion"]

    baseline_achs = gen0[0].fitness.achievements if gen0 else {}
    champion_achs = champions[-1].fitness.achievements if champions else {}

    return {"baseline": baseline_achs, "champion": champion_achs}
