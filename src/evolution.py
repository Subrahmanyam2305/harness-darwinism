"""Evolution loop: evaluate → select → crossover → mutate → repeat."""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timezone

import numpy as np
from rich.console import Console
from rich.table import Table

from src.agent import EpisodeTrace, play_episode
from src.config import settings
from src.db import (
    ensure_indexes,
    get_archive,
    get_db,
    get_failure_traces,
    get_generation,
    insert_run,
    ledger_add,
    ledger_check,
    record_metrics,
    upsert_genome,
)
from src.genome import Fitness, Genome, GenomeConfig
from src.meta_agent import llm_crossover, llm_mutate

console = Console()


def make_id(gen: int, idx: int) -> str:
    return f"h_g{gen}_{idx:02d}"


# ── Seed population ──

STARTER_RULE_POOL = [
    "Collect wood before anything else.",
    "Craft a wood pickaxe as soon as you have enough wood.",
    "Place a crafting table early.",
    "Sleep when energy is low and no enemies are nearby.",
    "Attack zombies and skeletons only if you have a sword.",
    "Collect stone after getting a wood pickaxe.",
    "Eat plants or cows when food is low.",
    "Drink water when thirst is low.",
    "Avoid moving at night without a weapon.",
    "Build a furnace after collecting stone and wood.",
]


def seed_population(n: int = 10) -> list[Genome]:
    """Create generation 0 with varied rule subsets."""
    population = []
    for i in range(n):
        k = random.randint(1, min(5, len(STARTER_RULE_POOL)))
        rules = random.sample(STARTER_RULE_POOL, k)
        genome = Genome(
            id=make_id(0, i),
            generation=0,
            genome=GenomeConfig(rules=rules),
        )
        population.append(genome)
    return population


# ── Evaluation ──

def evaluate_genome(genome: Genome, seeds: list[int]) -> tuple[Fitness, list[EpisodeTrace]]:
    """Play multiple seeds, return aggregate fitness and traces."""
    traces: list[EpisodeTrace] = []
    scores: list[float] = []
    all_achievements: dict[str, list[float]] = {}
    total_steps = 0

    for seed in seeds:
        console.print(f"  [dim]Playing seed {seed}...[/dim]", end=" ")
        trace = play_episode(genome, seed, max_steps=settings.max_steps_per_episode)
        traces.append(trace)
        scores.append(trace.score)
        total_steps += trace.steps
        console.print(f"score={trace.score:.1f}, steps={trace.steps}")

        for ach, val in trace.achievements.items():
            all_achievements.setdefault(ach, []).append(1.0 if val else 0.0)

    achievement_rates = {k: float(np.mean(v)) for k, v in all_achievements.items()}

    fitness = Fitness(
        score=float(np.mean(scores)),
        achievements=achievement_rates,
        mean_steps=total_steps / len(seeds),
    )
    return fitness, traces


# ── Selection ──

def select_and_archive(
    population: list[Genome],
    n_survivors: int,
) -> tuple[list[Genome], list[Genome]]:
    """Rank by fitness, keep top n_survivors, archive the rest."""
    ranked = sorted(population, key=lambda g: g.fitness.score, reverse=True)
    survivors = ranked[:n_survivors]
    culled = ranked[n_survivors:]

    for g in culled:
        g.status = "archived"

    return survivors, culled


# ── Pairing ──

def weighted_pairs(
    parents: list[Genome],
    archive: list[Genome],
    n_children: int,
    archive_ratio: float = 0.1,
) -> list[tuple[Genome, Genome]]:
    """Select parent pairs weighted by fitness."""
    pairs = []
    weights = np.array([max(g.fitness.score, 0.01) for g in parents])
    weights = weights / weights.sum()

    for _ in range(n_children):
        if archive and random.random() < archive_ratio:
            a = random.choices(parents, weights=weights, k=1)[0]
            b = random.choice(archive)
        else:
            chosen = random.choices(parents, weights=weights, k=2)
            a, b = chosen[0], chosen[1]
            if a.id == b.id and len(parents) > 1:
                others = [p for p in parents if p.id != a.id]
                b = random.choice(others)
        pairs.append((a, b))

    return pairs


# ── One generation ──

def run_generation(
    gen: int,
    population: list[Genome],
    seeds: list[int],
    db=None,
) -> list[Genome]:
    """Execute one full generation of the GA."""
    if db is None:
        db = get_db()

    console.rule(f"[bold blue]Generation {gen}[/bold blue]")

    # 1. Evaluate
    console.print("[bold]Phase 1: Evaluate[/bold]")
    for genome in population:
        console.print(f"\n[cyan]{genome.id}[/cyan] ({len(genome.genome.rules)} rules)")
        fitness, traces = evaluate_genome(genome, seeds)
        genome.fitness = fitness

        for trace in traces:
            insert_run(db, {
                "genome_id": genome.id,
                "generation": gen,
                "seed": trace.seed,
                "score": trace.score,
                "achievements": trace.achievements,
                "steps": trace.steps,
                "failure_summary": trace.failure_summary,
                "history": trace.history,
                "history_len": len(trace.history),
                "gif_path": trace.gif_path,
                "arm": genome.arm,
                "timestamp": datetime.now(timezone.utc),
            })

        upsert_genome(db, genome)

    # Print scoreboard
    _print_scoreboard(population, gen)

    # 2. Select
    console.print("\n[bold]Phase 2: Select[/bold]")
    n_survivors = settings.survivors
    survivors, culled = select_and_archive(population, n_survivors)

    for g in culled:
        upsert_genome(db, g)
        console.print(f"  [red]Archived[/red] {g.id} (score={g.fitness.score:.1f})")

    for g in survivors:
        console.print(f"  [green]Survived[/green] {g.id} (score={g.fitness.score:.1f})")

    # 3. Breed children
    console.print("\n[bold]Phase 3: Crossover + Mutate[/bold]")
    archive = get_archive(db, arm=population[0].arm if population else "full")
    n_children = settings.population_size - n_survivors
    pairs = weighted_pairs(survivors, archive, n_children)

    children: list[Genome] = []
    child_idx = 0

    for parent_a, parent_b in pairs:
        console.print(f"  Crossing {parent_a.id} × {parent_b.id}...", end=" ")

        child_config = llm_crossover(parent_a.genome, parent_b.genome)

        # Mutate based on failure traces
        traces_a = get_failure_traces(db, parent_a.id)
        traces_b = get_failure_traces(db, parent_b.id)
        failure_traces = traces_a + traces_b

        child_config, mutation_note = llm_mutate(child_config, failure_traces)

        # Check ledger
        edit_hash = child_config.content_hash()
        if ledger_check(db, edit_hash):
            console.print("[yellow]duplicate, re-mutating[/yellow]")
            child_config, mutation_note = llm_mutate(child_config, failure_traces)
            edit_hash = child_config.content_hash()

        ledger_add(db, edit_hash, mutation_note)

        child = Genome(
            id=make_id(gen + 1, child_idx),
            generation=gen + 1,
            parents=[parent_a.id, parent_b.id],
            genome=child_config,
            mutation_note=mutation_note,
            arm=parent_a.arm,
        )
        children.append(child)
        upsert_genome(db, child)
        console.print(f"→ {child.id}: {mutation_note}")
        child_idx += 1

    # Promote survivors to next generation
    next_population = []
    for i, g in enumerate(survivors):
        g.generation = gen + 1
        g.id = make_id(gen + 1, n_children + i)
        upsert_genome(db, g)
        next_population.append(g)

    next_population = children + next_population

    # 4. Record metrics
    scores = [g.fitness.score for g in population]
    record_metrics(
        db, gen,
        arm=population[0].arm if population else "full",
        best_score=max(scores),
        mean_score=float(np.mean(scores)),
        population_size=len(population),
    )

    return next_population


def _print_scoreboard(population: list[Genome], gen: int) -> None:
    table = Table(title=f"Generation {gen} Scoreboard")
    table.add_column("ID", style="cyan")
    table.add_column("Score", justify="right", style="bold")
    table.add_column("Steps", justify="right")
    table.add_column("Rules", justify="right")
    table.add_column("Top Achievement")

    ranked = sorted(population, key=lambda g: g.fitness.score, reverse=True)
    for g in ranked:
        top_ach = max(g.fitness.achievements.items(), key=lambda x: x[1], default=("none", 0))
        table.add_row(
            g.id,
            f"{g.fitness.score:.1f}",
            f"{g.fitness.mean_steps:.0f}",
            str(len(g.genome.rules)),
            f"{top_ach[0]} ({top_ach[1]:.0%})",
        )
    console.print(table)


# ── Full evolution run ──

def run_evolution(
    max_generations: int | None = None,
    train_seeds: list[int] | None = None,
) -> Genome:
    """Run the full GA loop. Returns the champion."""
    db = get_db()
    ensure_indexes(db)

    max_gen = max_generations or settings.max_generations
    all_train_seeds = train_seeds or list(range(100, 130))

    # Seed generation 0
    population = seed_population(settings.population_size)
    for g in population:
        upsert_genome(db, g)

    console.print(f"[bold green]Starting evolution: {max_gen} generations, pop={settings.population_size}[/bold green]\n")

    best_score = 0.0
    stale_count = 0

    for gen in range(max_gen):
        seeds = random.sample(all_train_seeds, min(settings.seeds_per_gen, len(all_train_seeds)))
        population = run_generation(gen, population, seeds, db)

        gen_best = max(g.fitness.score for g in population)
        if gen_best > best_score + 0.5:
            best_score = gen_best
            stale_count = 0
        else:
            stale_count += 1

        if stale_count >= 3:
            console.print("[yellow]No improvement for 3 generations — stopping early.[/yellow]")
            break

    # Crown champion
    champion = max(population, key=lambda g: g.fitness.score)
    champion.status = "champion"
    upsert_genome(db, champion)

    console.print(f"\n[bold green]🏆 Champion: {champion.id} — score={champion.fitness.score:.1f}[/bold green]")
    console.print(f"Rules: {champion.genome.rules}")

    return champion
