"""MongoDB persistence: harnesses, runs, ledger, metrics."""

from __future__ import annotations

from datetime import datetime, timezone

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from src.config import settings
from src.genome import Genome


def get_db() -> Database:
    client = MongoClient(settings.mongodb_uri)
    return client[settings.db_name]


def ensure_indexes(db: Database) -> None:
    db["harnesses"].create_index("generation")
    db["harnesses"].create_index("status")
    db["harnesses"].create_index("arm")
    db["runs"].create_index([("genome_id", 1), ("seed", 1)])
    db["runs"].create_index("generation")
    db["ledger"].create_index("edit_hash", unique=True)
    db["metrics"].create_index("generation")


# ── Harnesses ──

def upsert_genome(db: Database, genome: Genome) -> None:
    db["harnesses"].replace_one({"_id": genome.id}, genome.to_doc(), upsert=True)


def get_genome(db: Database, genome_id: str) -> Genome | None:
    doc = db["harnesses"].find_one({"_id": genome_id})
    return Genome.from_doc(doc) if doc else None


def get_generation(db: Database, gen: int, arm: str = "full") -> list[Genome]:
    docs = db["harnesses"].find({"generation": gen, "arm": arm, "status": "alive"})
    return [Genome.from_doc(d) for d in docs]


def get_archive(db: Database, arm: str = "full") -> list[Genome]:
    docs = db["harnesses"].find({"arm": arm, "status": "archived"})
    return [Genome.from_doc(d) for d in docs]


def get_all_genomes(db: Database, arm: str | None = None) -> list[Genome]:
    query: dict = {}
    if arm:
        query["arm"] = arm
    docs = db["harnesses"].find(query).sort("generation", 1)
    return [Genome.from_doc(d) for d in docs]


# ── Runs ──

def insert_run(db: Database, run_doc: dict) -> None:
    db["runs"].insert_one(run_doc)


def get_runs_for_genome(db: Database, genome_id: str) -> list[dict]:
    return list(db["runs"].find({"genome_id": genome_id}))


def get_failure_traces(db: Database, genome_id: str, limit: int = 3) -> list[str]:
    runs = db["runs"].find({"genome_id": genome_id}).sort("score", 1).limit(limit)
    return [r.get("failure_summary", "") for r in runs if r.get("failure_summary")]


# ── Ledger ──

def ledger_check(db: Database, edit_hash: str) -> bool:
    return db["ledger"].find_one({"edit_hash": edit_hash}) is not None


def ledger_add(db: Database, edit_hash: str, note: str) -> None:
    db["ledger"].insert_one({
        "edit_hash": edit_hash,
        "note": note,
        "created_at": datetime.now(timezone.utc),
    })


# ── Metrics ──

def record_metrics(db: Database, gen: int, arm: str, best_score: float, mean_score: float, population_size: int) -> None:
    db["metrics"].insert_one({
        "generation": gen,
        "arm": arm,
        "best_score": best_score,
        "mean_score": mean_score,
        "population_size": population_size,
        "timestamp": datetime.now(timezone.utc),
    })


def get_metrics(db: Database, arm: str = "full") -> list[dict]:
    return list(db["metrics"].find({"arm": arm}).sort("generation", 1))
