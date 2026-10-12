"""FastAPI backend for the evolution dashboard."""

from __future__ import annotations

import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.db import get_all_genomes, get_db, get_metrics, get_runs_for_genome

app = FastAPI(title="Harness Darwinism", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


def _safe_timestamp(val):
    if hasattr(val, "isoformat"):
        return val.isoformat()
    return str(val) if val else None


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
    for r in runs:
        r.pop("_id", None)
        if "timestamp" in r:
            r["timestamp"] = _safe_timestamp(r["timestamp"])
    return {"genome": g.model_dump(), "runs": runs}


@app.get("/api/tree")
def lineage_tree(arm: str = "full"):
    """Lineage tree for D3 visualization."""
    db = get_db()
    gs = get_all_genomes(db, arm=arm)

    nodes = []
    edges = []
    for g in gs:
        nodes.append({
            "id": g.id,
            "generation": g.generation,
            "status": g.status,
            "score": g.fitness.score,
            "rules_count": len(g.genome.rules),
            "mutation_note": g.mutation_note,
        })
        for parent_id in g.parents:
            edges.append({"source": parent_id, "target": g.id})

    return {"nodes": nodes, "edges": edges}


def _run_replay(seed: int, steps: int, stride: int, genome_id: str | None) -> dict:
    from src.agent import play_episode
    from src.db import get_genome
    from src.genome import Genome, GenomeConfig

    if genome_id:
        genome = get_genome(get_db(), genome_id)
        if not genome:
            return {"error": f"genome {genome_id} not found"}
    else:
        genome = Genome(id="play_default", genome=GenomeConfig())

    trace = play_episode(genome, seed, max_steps=steps, record_stride=stride)
    unlocked = [k for k, v in trace.achievements.items() if v]
    return {
        "label": "scripted replay — separate from evolve scores",
        "genome_id": genome.id,
        "seed": seed,
        "steps": trace.steps,
        "score": trace.score,
        "unlocks": unlocked,
        "frames": trace.frames,
    }


@app.get("/api/replay")
async def replay(
    seed: int = 102,
    steps: int = 160,
    stride: int = 5,
    genome_id: str | None = None,
):
    """Play a short labeled replay off the event loop so /api/health stays up."""
    steps = max(50, min(int(steps), 400))
    stride = max(2, min(int(stride), 16))
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_run_replay, seed, steps, stride, genome_id),
            timeout=40,
        )
    except TimeoutError:
        return {"error": "replay timed out; try a smaller step budget"}


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
