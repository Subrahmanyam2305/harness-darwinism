"""Storage backend — uses MongoDB if available, falls back to local JSON files."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pymongo.errors import PyMongoError

from src.agent import EpisodeTrace
from src.config import settings
from src.genome import Genome

DATA_DIR = settings.project_root / "data" / "store"

_use_mongo: bool | None = None
_db = None


def _check_mongo():
    global _use_mongo, _db
    if _use_mongo is not None:
        return _use_mongo
    if settings.mongodb_uri == "mongodb://localhost:27017":
        _use_mongo = False
        return False
    try:
        from pymongo import MongoClient

        client = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=3000)
        client.server_info()
        _db = client[settings.db_name]
        _use_mongo = True
        return True
    except PyMongoError:
        _use_mongo = False
        return False


def get_db():
    if _check_mongo():
        return _db
    return None


# ── Local JSON helpers ──


def _collection_path(name: str) -> Path:
    p = DATA_DIR / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_docs(name: str) -> list[dict]:
    p = _collection_path(name)
    docs = []
    for f in sorted(p.glob("*.json")):
        docs.append(json.loads(f.read_text()))
    return docs


def _save_doc(name: str, doc_id: str, doc: dict):
    p = _collection_path(name)
    safe_id = doc_id.replace("/", "_")
    (p / f"{safe_id}.json").write_text(json.dumps(doc, default=str, indent=2))


def _find_doc(name: str, doc_id: str) -> dict | None:
    p = _collection_path(name) / f"{doc_id}.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


# ── Indexes ──


def ensure_indexes(db=None) -> None:
    if _check_mongo() and db is not None:
        db["harnesses"].create_index("generation")
        db["harnesses"].create_index("status")
        db["harnesses"].create_index("arm")
        db["runs"].create_index([("genome_id", 1), ("seed", 1)])
        db["runs"].create_index("generation")
        db["ledger"].create_index("edit_hash", unique=True)
        db["metrics"].create_index("generation")
    else:
        DATA_DIR.mkdir(parents=True, exist_ok=True)


# ── Harnesses ──


def upsert_genome(db, genome: Genome) -> None:
    if db is not None:
        db["harnesses"].replace_one({"_id": genome.id}, genome.to_doc(), upsert=True)
    else:
        _save_doc("harnesses", genome.id, genome.to_doc())


def get_genome(db, genome_id: str) -> Genome | None:
    if db is not None:
        doc = db["harnesses"].find_one({"_id": genome_id})
    else:
        doc = _find_doc("harnesses", genome_id)
    return Genome.from_doc(doc) if doc else None


def get_generation(db, gen: int, arm: str = "full") -> list[Genome]:
    if db is not None:
        docs = db["harnesses"].find({"generation": gen, "arm": arm, "status": "alive"})
        return [Genome.from_doc(d) for d in docs]
    else:
        all_docs = _load_docs("harnesses")
        return [
            Genome.from_doc(d)
            for d in all_docs
            if d.get("generation") == gen
            and d.get("arm") == arm
            and d.get("status") == "alive"
        ]


def get_archive(db, arm: str = "full") -> list[Genome]:
    if db is not None:
        docs = db["harnesses"].find({"arm": arm, "status": "archived"})
        return [Genome.from_doc(d) for d in docs]
    else:
        all_docs = _load_docs("harnesses")
        return [
            Genome.from_doc(d)
            for d in all_docs
            if d.get("arm") == arm and d.get("status") == "archived"
        ]


def get_all_genomes(db, arm: str | None = None) -> list[Genome]:
    if db is not None:
        query: dict = {}
        if arm:
            query["arm"] = arm
        docs = db["harnesses"].find(query).sort("generation", 1)
        return [Genome.from_doc(d) for d in docs]
    else:
        all_docs = _load_docs("harnesses")
        if arm:
            all_docs = [d for d in all_docs if d.get("arm") == arm]
        all_docs.sort(key=lambda d: d.get("generation", 0))
        return [Genome.from_doc(d) for d in all_docs]


# ── Runs ──


def insert_run(db, run_doc: dict) -> None:
    if db is not None:
        db["runs"].insert_one(run_doc)
    else:
        run_id = f"{run_doc['genome_id']}_s{run_doc['seed']}"
        _save_doc("runs", run_id, run_doc)


def insert_episode(db, trace: EpisodeTrace, *, artifact_path: str) -> str:
    """Persist a complete gameplay trace and return its stable run identifier."""
    run_id = uuid4().hex
    document = {
        "_id": run_id,
        "kind": "gameplay_episode",
        "genome_id": trace.genome_id,
        "seed": trace.seed,
        "score": trace.score,
        "achievements": trace.achievements,
        "steps": trace.steps,
        "termination_reason": trace.termination_reason,
        "provider": trace.provider,
        "model": trace.model,
        "is_stub": trace.is_stub,
        "artifact_path": artifact_path,
        "trace": trace.to_dict(),
        "created_at": datetime.now(UTC).isoformat(),
    }
    if db is not None:
        db["runs"].insert_one(document)
    else:
        _save_doc("runs", run_id, document)
    return run_id


def get_run(db, run_id: str) -> dict | None:
    """Load one gameplay or evaluation run by its stable identifier."""
    if db is not None:
        return db["runs"].find_one({"_id": run_id})
    return _find_doc("runs", run_id)


def get_runs_for_genome(db, genome_id: str) -> list[dict]:
    if db is not None:
        return list(db["runs"].find({"genome_id": genome_id}))
    else:
        all_docs = _load_docs("runs")
        return [d for d in all_docs if d.get("genome_id") == genome_id]


def get_failure_traces(db, genome_id: str, limit: int = 3) -> list[str]:
    if db is not None:
        runs = db["runs"].find({"genome_id": genome_id}).sort("score", 1).limit(limit)
        return [r.get("failure_summary", "") for r in runs if r.get("failure_summary")]
    else:
        runs = get_runs_for_genome(db, genome_id)
        runs.sort(key=lambda r: r.get("score", 0))
        return [
            r.get("failure_summary", "")
            for r in runs[:limit]
            if r.get("failure_summary")
        ]


# ── Ledger ──


def ledger_check(db, edit_hash: str) -> bool:
    if db is not None:
        return db["ledger"].find_one({"edit_hash": edit_hash}) is not None
    else:
        doc = _find_doc("ledger", edit_hash)
        return doc is not None


def ledger_add(db, edit_hash: str, note: str) -> None:
    doc = {
        "edit_hash": edit_hash,
        "note": note,
        "created_at": datetime.now(UTC).isoformat(),
    }
    if db is not None:
        db["ledger"].insert_one(doc)
    else:
        _save_doc("ledger", edit_hash, doc)


# ── Metrics ──


def record_metrics(
    db, gen: int, arm: str, best_score: float, mean_score: float, population_size: int
) -> None:
    doc = {
        "generation": gen,
        "arm": arm,
        "best_score": best_score,
        "mean_score": mean_score,
        "population_size": population_size,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    if db is not None:
        db["metrics"].insert_one(doc)
    else:
        _save_doc("metrics", f"gen_{gen}_{arm}", doc)


def get_metrics(db, arm: str = "full") -> list[dict]:
    if db is not None:
        return list(db["metrics"].find({"arm": arm}).sort("generation", 1))
    else:
        all_docs = _load_docs("metrics")
        docs = [d for d in all_docs if d.get("arm") == arm]
        docs.sort(key=lambda d: d.get("generation", 0))
        return docs
