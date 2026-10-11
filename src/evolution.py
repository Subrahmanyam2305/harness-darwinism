"""Evolution loop: evaluate → select → crossover → mutate → repeat."""

from __future__ import annotations

import random
import re
from datetime import UTC, datetime

import numpy as np
from rich.console import Console
from rich.table import Table

from src.agent import EpisodeTrace, play_episode
from src.config import settings
from src.db import (
    ensure_indexes,
    get_all_genomes,
    get_archive,
    get_db,
    get_failure_traces,
    get_runs_for_genome,
    insert_run,
    ledger_add,
    ledger_check,
    record_metrics,
    upsert_genome,
)
from src.evaluator import summarize_traces
from src.genome import Fitness, Genome, GenomeConfig
from src.meta_agent import llm_crossover, llm_mutate

console = Console()


def make_id(gen: int, idx: int, experiment_id: str = "default") -> str:
    """Return an experiment-scoped harness identifier safe for persistence."""
    safe_experiment = re.sub(r"[^a-zA-Z0-9_-]", "-", experiment_id).strip("-")
    if not safe_experiment:
        raise ValueError("experiment_id must contain a letter or number")
    return f"{safe_experiment}_g{gen}_{idx:02d}"


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


def seed_population(n: int = 10, *, experiment_id: str = "default") -> list[Genome]:
    """Create generation 0 with varied rule subsets."""
    population = []
    for i in range(n):
        k = random.randint(1, min(5, len(STARTER_RULE_POOL)))
        rules = random.sample(STARTER_RULE_POOL, k)
        genome = Genome(
            id=make_id(0, i, experiment_id),
            generation=0,
            genome=GenomeConfig(rules=rules),
            arm=experiment_id,
        )
        population.append(genome)
    return population


# ── Evaluation ──


def evaluate_genome(
    genome: Genome, seeds: list[int], *, max_steps: int | None = None
) -> tuple[Fitness, list[EpisodeTrace]]:
    """Play multiple seeds, return aggregate fitness and traces."""
    traces: list[EpisodeTrace] = []
    for seed in seeds:
        console.print(f"  [dim]Playing seed {seed}...[/dim]", end=" ")
        trace = play_episode(
            genome,
            seed,
            max_steps=(
                max_steps if max_steps is not None else settings.max_steps_per_episode
            ),
        )
        traces.append(trace)
        console.print(f"score={trace.score:.1f}, steps={trace.steps}")

    evaluation = summarize_traces(traces)
    score = evaluation.score if evaluation.score is not None else 0.0
    if not evaluation.complete:
        console.print(
            f"  [yellow]{genome.id} ended with agent_error; assigning score=0 "
            "and retaining its trace[/yellow]"
        )

    fitness = Fitness(
        score=score,
        achievements=evaluation.achievement_rates,
        mean_steps=evaluation.mean_steps,
        cost_usd=evaluation.cost_usd or 0.0,
    )
    return fitness, traces


def _run_document(genome: Genome, trace: EpisodeTrace, generation: int) -> dict:
    """Build the complete Atlas evidence document for one evaluated episode."""
    return {
        "genome_id": genome.id,
        "generation": generation,
        "seed": trace.seed,
        "score": trace.score,
        "achievements": trace.achievements,
        "steps": trace.steps,
        "failure_summary": trace.failure_summary,
        "history": trace.history,
        "history_len": len(trace.history),
        "attempts": trace.attempts,
        "termination_reason": trace.termination_reason,
        "error": trace.error,
        "provider": trace.provider,
        "model": trace.model,
        "input_tokens": trace.input_tokens,
        "output_tokens": trace.output_tokens,
        "cost_usd": trace.cost_usd,
        "latency_seconds": trace.latency_seconds,
        "harness": trace.harness,
        "arm": genome.arm,
        "timestamp": datetime.now(UTC),
    }


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
    *,
    population_size: int | None = None,
    n_survivors: int | None = None,
    max_steps: int | None = None,
    max_mutation_attempts: int = 3,
    reuse_completed: bool = False,
) -> list[Genome]:
    """Execute one full generation of the GA."""
    if db is None:
        db = get_db()

    target_size = (
        population_size if population_size is not None else settings.population_size
    )
    survivor_count = n_survivors if n_survivors is not None else settings.survivors
    if len(population) != target_size:
        raise ValueError(
            f"Expected population size {target_size}, received {len(population)}"
        )
    if not 1 <= survivor_count < target_size:
        raise ValueError("survivors must be between 1 and population_size - 1")
    if not seeds:
        raise ValueError("At least one shared training seed is required")

    console.rule(f"[bold blue]Generation {gen}[/bold blue]")

    # 1. Evaluate
    console.print("[bold]Phase 1: Evaluate[/bold]")
    for genome in population:
        console.print(f"\n[cyan]{genome.id}[/cyan] ({len(genome.genome.rules)} rules)")
        if reuse_completed:
            completed_seeds = {
                run.get("seed")
                for run in get_runs_for_genome(db, genome.id)
                if run.get("generation") == gen
            }
            requested_seeds = set(seeds)
            if completed_seeds == requested_seeds:
                console.print(
                    f"  [green]Reusing stored evaluation for seeds "
                    f"{sorted(requested_seeds)}[/green]"
                )
                continue
            if completed_seeds:
                raise RuntimeError(
                    f"Cannot resume partially evaluated seed set for {genome.id}: "
                    f"stored={sorted(completed_seeds)}, requested={sorted(requested_seeds)}"
                )
        fitness, traces = evaluate_genome(genome, seeds, max_steps=max_steps)
        genome.fitness = fitness

        for trace in traces:
            insert_run(db, _run_document(genome, trace, gen))

        upsert_genome(db, genome)

    # Print scoreboard
    _print_scoreboard(population, gen)

    # 2. Select
    console.print("\n[bold]Phase 2: Select[/bold]")
    survivors, culled = select_and_archive(population, survivor_count)

    for g in culled:
        upsert_genome(db, g)
        console.print(f"  [red]Archived[/red] {g.id} (score={g.fitness.score:.1f})")

    for g in survivors:
        console.print(f"  [green]Survived[/green] {g.id} (score={g.fitness.score:.1f})")

    # 3. Breed children
    console.print("\n[bold]Phase 3: Crossover + Mutate[/bold]")
    archive = get_archive(db, arm=population[0].arm if population else "full")
    n_children = target_size - survivor_count
    pairs = weighted_pairs(survivors, archive, n_children)

    children: list[Genome] = []
    for child_idx, (parent_a, parent_b) in enumerate(pairs):
        console.print(f"  Crossing {parent_a.id} × {parent_b.id}...", end=" ")

        child_config = llm_crossover(parent_a.genome, parent_b.genome)

        # Mutate based on failure traces
        traces_a = get_failure_traces(db, parent_a.id)
        traces_b = get_failure_traces(db, parent_b.id)
        failure_traces = traces_a + traces_b

        tried_edits: list[str] = []
        for attempt in range(1, max_mutation_attempts + 1):
            child_config, mutation_note = llm_mutate(
                child_config, failure_traces, tried_edits=tried_edits
            )
            edit_hash = child_config.content_hash()
            if not ledger_check(db, edit_hash):
                ledger_add(db, edit_hash, mutation_note)
                break
            tried_edits.append(mutation_note)
            console.print(
                f"[yellow]duplicate mutation {attempt}/{max_mutation_attempts}[/yellow]"
            )
        else:
            raise RuntimeError(
                f"Could not produce a unique child after {max_mutation_attempts} attempts"
            )

        child = Genome(
            id=make_id(gen + 1, child_idx, parent_a.arm),
            generation=gen + 1,
            parents=[parent_a.id, parent_b.id],
            genome=child_config,
            mutation_note=mutation_note,
            arm=parent_a.arm,
        )
        children.append(child)
        upsert_genome(db, child)
        console.print(f"→ {child.id}: {mutation_note}")

    # Preserve evaluated survivors and carry cloned descendants into the next generation.
    carried: list[Genome] = []
    for i, survivor in enumerate(survivors):
        survivor.status = "archived"
        upsert_genome(db, survivor)
        promoted = Genome(
            id=make_id(gen + 1, n_children + i, survivor.arm),
            generation=gen + 1,
            parents=[survivor.id],
            genome=survivor.genome.model_copy(deep=True),
            fitness=survivor.fitness.model_copy(deep=True),
            mutation_note=f"Carried forward from {survivor.id}",
            arm=survivor.arm,
        )
        upsert_genome(db, promoted)
        carried.append(promoted)

    next_population = children + carried
    if len(next_population) != target_size:
        raise RuntimeError(
            f"Generation {gen + 1} has {len(next_population)} genomes, expected {target_size}"
        )

    # 4. Record metrics
    scores = [g.fitness.score for g in population]
    record_metrics(
        db,
        gen,
        arm=population[0].arm if population else "full",
        best_score=max(scores),
        mean_score=float(np.mean(scores)),
        population_size=len(population),
        seeds=seeds,
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
        top_ach = max(
            g.fitness.achievements.items(), key=lambda x: x[1], default=("none", 0)
        )
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
    *,
    experiment_id: str = "default",
    population_size: int | None = None,
    survivors: int | None = None,
    seeds_per_generation: int | None = None,
    max_steps: int | None = None,
    resume: bool = False,
) -> Genome:
    """Run evolutionary transitions, evaluate the final generation, and return its champion."""
    db = get_db()
    if db is None:
        raise RuntimeError("Atlas is required for evolution runs")
    ensure_indexes(db)

    max_gen = (
        max_generations if max_generations is not None else settings.max_generations
    )
    target_size = (
        population_size if population_size is not None else settings.population_size
    )
    survivor_count = survivors if survivors is not None else settings.survivors
    seed_count = (
        seeds_per_generation
        if seeds_per_generation is not None
        else settings.seeds_per_gen
    )
    step_limit = max_steps if max_steps is not None else settings.max_steps_per_episode
    all_train_seeds = train_seeds or list(range(100, 130))
    if max_gen <= 0:
        raise ValueError("max_generations must be positive")
    if target_size < 2:
        raise ValueError("population_size must be at least 2")
    if not 1 <= survivor_count < target_size:
        raise ValueError("survivors must be between 1 and population_size - 1")
    if step_limit <= 0:
        raise ValueError("max_steps must be positive")
    if seed_count <= 0 or seed_count > len(all_train_seeds):
        raise ValueError("seeds_per_generation must fit within train_seeds")
    existing_genomes = get_all_genomes(db, arm=experiment_id)
    if existing_genomes and not resume:
        raise ValueError(
            f"Experiment {experiment_id!r} already exists; choose a new experiment ID "
            "or pass --resume"
        )
    if resume:
        if not existing_genomes:
            raise ValueError(f"Experiment {experiment_id!r} does not exist")
        generations = {genome.generation for genome in existing_genomes}
        if generations != {0}:
            raise ValueError(
                "Resume currently requires an interrupted generation 0 experiment"
            )
        population = [genome for genome in existing_genomes if genome.status == "alive"]
        if len(population) != target_size:
            raise ValueError(
                f"Stored generation 0 has {len(population)} live genomes, "
                f"expected {target_size}"
            )
        stored_seeds = {
            run.get("seed")
            for genome in population
            for run in get_runs_for_genome(db, genome.id)
            if run.get("generation") == 0 and run.get("seed") is not None
        }
        if len(stored_seeds) != seed_count:
            raise ValueError(
                f"Stored run evidence has seeds {sorted(stored_seeds)}, "
                f"expected {seed_count} shared seed(s)"
            )
        shared_seeds = sorted(stored_seeds)
    else:
        shared_seeds = random.sample(all_train_seeds, seed_count)
        population = seed_population(target_size, experiment_id=experiment_id)
        for g in population:
            upsert_genome(db, g)

    console.print(
        f"[bold green]Starting evolution: {max_gen} transitions, pop={target_size}, "
        f"seeds={shared_seeds}, steps={step_limit}[/bold green]\n"
    )

    for gen in range(max_gen):
        population = run_generation(
            gen,
            population,
            shared_seeds,
            db,
            population_size=target_size,
            n_survivors=survivor_count,
            max_steps=step_limit,
            reuse_completed=resume and gen == 0,
        )

    # Evaluate offspring before crowning a champion.
    final_generation = max_gen
    for genome in population:
        fitness, traces = evaluate_genome(genome, shared_seeds, max_steps=step_limit)
        genome.fitness = fitness
        for trace in traces:
            insert_run(db, _run_document(genome, trace, final_generation))
        upsert_genome(db, genome)

    final_scores = [genome.fitness.score for genome in population]
    record_metrics(
        db,
        final_generation,
        arm=experiment_id,
        best_score=max(final_scores),
        mean_score=float(np.mean(final_scores)),
        population_size=len(population),
        seeds=shared_seeds,
    )

    champion = max(population, key=lambda g: g.fitness.score)
    champion.status = "champion"
    upsert_genome(db, champion)

    console.print(
        f"\n[bold green]🏆 Champion: {champion.id} — score={champion.fitness.score:.1f}[/bold green]"
    )
    console.print(f"Rules: {champion.genome.rules}")

    return champion
