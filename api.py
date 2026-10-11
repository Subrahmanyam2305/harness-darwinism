"""FastAPI backend for the evolution dashboard."""

from __future__ import annotations

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


@app.get("/api/metrics")
def metrics(arm: str = "full"):
    """Fitness curve: best/mean score per generation."""
    db = get_db()
    rows = get_metrics(db, arm=arm)
    for r in rows:
        r.pop("_id", None)
        if "timestamp" in r:
            r["timestamp"] = r["timestamp"].isoformat()
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
            r["timestamp"] = r["timestamp"].isoformat()
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
