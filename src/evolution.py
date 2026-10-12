"""Evolution loop: evaluate → select → crossover → mutate → repeat."""

from __future__ import annotations

import random
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
from src.genome import Fitness, Genome, GenomeConfig, PlannerPolicy
from src.meta_agent import crossover, mutate

console = Console()


def make_id(gen: int, idx: int, tag: str = "h") -> str:
    return f"{tag}_g{gen}_{idx:02d}"


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

STARTER_POLICIES = [
    PlannerPolicy(),
    PlannerPolicy(relocate_workstation=True),
    PlannerPolicy(prefer_sword=True, health_flee=3),
    PlannerPolicy(sleep_at_night=True, energy_sleep=3),
    PlannerPolicy(drink_threshold=4, food_threshold=3),
    PlannerPolicy(relocate_workstation=True, prefer_sword=True),
    PlannerPolicy(attack_unarmed=True, health_flee=1),
    PlannerPolicy(wood_for_table=3, stone_for_furnace=4),
]


TECH_TIERS = [
    ("collect_wood",),
    ("place_table",),
    ("make_wood_pickaxe",),
    ("collect_stone",),
    ("make_stone_pickaxe", "place_furnace"),
    ("collect_coal", "collect_iron"),
    ("make_iron_pickaxe",),
    ("collect_diamond",),
]


def tech_depth(achievements: dict[str, bool | float]) -> int:
    depth = 0
    for i, names in enumerate(TECH_TIERS, start=1):
        if any(bool(achievements.get(name)) for name in names):
            depth = i
        else:
            break
    return depth


def seed_population(n: int = 10, tag: str = "h") -> list[Genome]:
    """Create generation 0 with varied planner skills."""
    population = []
    for i in range(n):
        policy = STARTER_POLICIES[i % len(STARTER_POLICIES)].model_copy()
        k = random.randint(0, min(3, len(STARTER_RULE_POOL)))
        rules = random.sample(STARTER_RULE_POOL, k) if k else []
        genome = Genome(
            id=make_id(0, i, tag),
            generation=0,
            genome=GenomeConfig(rules=rules, planner=policy),
        )
        population.append(genome)
    return population


# ── Evaluation ──

def evaluate_genome(
    genome: Genome,
    seeds: list[int],
    max_steps: int | None = None,
) -> tuple[Fitness, list[EpisodeTrace]]:
    """Play multiple seeds, return aggregate fitness and traces."""
    max_steps = max_steps if max_steps is not None else settings.max_steps_per_episode
    traces: list[EpisodeTrace] = []
    scores: list[float] = []
    all_achievements: dict[str, list[float]] = {}
    total_steps = 0

    for seed in seeds:
        console.print(f"  [dim]Playing seed {seed}...[/dim]", end=" ")
        trace = play_episode(genome, seed, max_steps=max_steps)
        traces.append(trace)
        scores.append(trace.score)
        total_steps += trace.steps
        console.print(f"score={trace.score:.1f}, steps={trace.steps}")

        for ach, val in trace.achievements.items():
            all_achievements.setdefault(ach, []).append(1.0 if val else 0.0)

    achievement_rates = {k: float(np.mean(v)) for k, v in all_achievements.items()}
    unlocks = [sum(1 for v in t.achievements.values() if v) for t in traces]
    depths = [tech_depth(t.achievements) for t in traces]

    iron_rate = float(
        np.mean([1.0 if t.achievements.get("make_iron_pickaxe") else 0.0 for t in traces])
    )
    fitness = Fitness(
        score=float(np.mean(scores)),
        hafner=hafner_score(achievement_rates),
        unlocks=float(np.mean(unlocks)),
        iron_rate=iron_rate,
        tech_depth=float(np.mean(depths)),
        achievements=achievement_rates,
        mean_steps=total_steps / len(seeds),
    )
    return fitness, traces


def hafner_score(achievement_rates: dict[str, float]) -> float:
    """Official Crafter score: exp(mean ln(1 + s_i)) - 1 with s_i in [0, 100]."""
    from src.crafter_env import ALL_ACHIEVEMENTS

    rates = [float(achievement_rates.get(name, 0.0)) * 100.0 for name in ALL_ACHIEVEMENTS]
    if not rates:
        return 0.0
    return float(np.exp(np.mean(np.log1p(rates))) - 1.0)


# ── Selection ──

def select_and_archive(
    population: list[Genome],
    n_survivors: int,
) -> tuple[list[Genome], list[Genome]]:
    """Rank by iron / unlocks / steps / depth / score; keep top n_survivors."""
    ranked = sorted(population, key=lambda g: g.fitness.rank_key(), reverse=True)
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
    """Select parent pairs. High unlocks and few children are favored (DGM-style)."""
    pairs = []

    def weights_for(pool: list[Genome]) -> np.ndarray:
        raw = np.array([
            max(g.fitness.unlocks, 0.1) / (1.0 + g.n_children)
            for g in pool
        ])
        return raw / raw.sum()

    parent_w = weights_for(parents)
    archive_w = weights_for(archive) if archive else None

    for _ in range(n_children):
        if archive and archive_w is not None and random.random() < archive_ratio:
            a = random.choices(parents, weights=parent_w, k=1)[0]
            b = random.choices(archive, weights=archive_w, k=1)[0]
        else:
            chosen = random.choices(parents, weights=parent_w, k=2)
            a, b = chosen[0], chosen[1]
            if a.id == b.id and len(parents) > 1:
                others = [p for p in parents if p.id != a.id]
                b = random.choice(others)
        a.n_children += 1
        b.n_children += 1
        pairs.append((a, b))

    return pairs


# ── One generation ──

def run_generation(
    gen: int,
    population: list[Genome],
    seeds: list[int],
    db=None,
    tag: str = "h",
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
                "unlocks": sum(1 for v in trace.achievements.values() if v),
                "tech_depth": tech_depth(trace.achievements),
                "achievements": trace.achievements,
                "steps": trace.steps,
                "failure_summary": trace.failure_summary,
                "history_len": len(trace.history),
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
        console.print(
            f"  [red]Archived[/red] {g.id} (unlocks={g.fitness.unlocks:.1f} score={g.fitness.score:.1f})"
        )

    for g in survivors:
        console.print(
            f"  [green]Survived[/green] {g.id} (unlocks={g.fitness.unlocks:.1f} score={g.fitness.score:.1f})"
        )

    # 3. Breed children
    console.print("\n[bold]Phase 3: Crossover + Mutate[/bold]")
    archive = get_archive(db, arm=population[0].arm if population else "full")
    n_children = settings.population_size - n_survivors
    arm = population[0].arm if population else "full"
    archive_ratio = 0.0 if arm == "no_archive" else settings.archive_sample_ratio
    if arm == "no_archive":
        archive = []
    pairs = weighted_pairs(survivors, archive, n_children, archive_ratio=archive_ratio)

    children: list[Genome] = []
    child_idx = 0

    for parent_a, parent_b in pairs:
        console.print(f"  Crossing {parent_a.id} × {parent_b.id}...", end=" ")

        child_config = crossover(parent_a.genome, parent_b.genome)

        # Mutate based on failure traces
        traces_a = get_failure_traces(db, parent_a.id)
        traces_b = get_failure_traces(db, parent_b.id)
        failure_traces = traces_a + traces_b

        child_config, mutation_note = mutate(
            child_config, failure_traces, arm=parent_a.arm
        )

        # Check ledger
        edit_hash = child_config.content_hash()
        if ledger_check(db, edit_hash):
            console.print("[yellow]duplicate, re-mutating[/yellow]")
            child_config, mutation_note = mutate(
                child_config, failure_traces, arm=parent_a.arm
            )
            edit_hash = child_config.content_hash()
        if parent_a.arm == "no_relocate":
            child_config.planner.relocate_workstation = False
            child_config.planner.pair_stations = False

        ledger_add(db, edit_hash, mutation_note)

        child = Genome(
            id=make_id(gen + 1, child_idx, tag),
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

    # Survivors keep their IDs so lineage stays intact (DGM archive style).
    next_population = []
    for g in survivors:
        g.status = "alive"
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
    table.add_column("Unlocks", justify="right", style="bold")
    table.add_column("Iron", justify="right")
    table.add_column("Depth", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Steps", justify="right")
    table.add_column("Skill")

    ranked = sorted(population, key=lambda g: g.fitness.rank_key(), reverse=True)
    for g in ranked:
        flags = []
        if g.genome.planner.relocate_workstation:
            flags.append("reloc")
        if g.genome.planner.pair_stations:
            flags.append("pair")
        if g.genome.planner.kite_combat:
            flags.append("kite")
        skill = "+".join(flags) or "default"
        table.add_row(
            g.id,
            f"{g.fitness.unlocks:.1f}",
            f"{g.fitness.iron_rate:.2f}",
            f"{g.fitness.tech_depth:.1f}",
            f"{g.fitness.score:.1f}",
            f"{g.fitness.mean_steps:.0f}",
            skill,
        )
    console.print(table)


# ── Full evolution run ──

def apply_arm(genome: Genome, arm: str) -> Genome:
    genome.arm = arm
    if arm == "no_relocate":
        genome.genome.planner.relocate_workstation = False
        genome.genome.planner.pair_stations = False
    return genome


def run_evolution(
    max_generations: int | None = None,
    train_seeds: list[int] | None = None,
    arm: str = "full",
    tag: str = "h",
) -> Genome:
    """Run the full GA loop. Returns the champion."""
    db = get_db()
    ensure_indexes(db)

    max_gen = max_generations or settings.max_generations
    all_train_seeds = train_seeds or list(range(100, 130))

    # Seed generation 0
    population = seed_population(settings.population_size, tag=tag)
    for g in population:
        apply_arm(g, arm)
        upsert_genome(db, g)

    console.print(
        f"[bold green]Starting evolution: {max_gen} generations, "
        f"pop={settings.population_size}, arm={arm}[/bold green]\n"
    )

    best_score = 0.0
    stale_count = 0

    for gen in range(max_gen):
        seeds = random.sample(all_train_seeds, min(settings.seeds_per_gen, len(all_train_seeds)))
        population = run_generation(gen, population, seeds, db, tag=tag)

        gen_best = max(g.fitness.unlocks for g in population)
        if gen_best > best_score + 0.25:
            best_score = gen_best
            stale_count = 0
        else:
            stale_count += 1

        if stale_count >= 3:
            console.print("[yellow]No improvement for 3 generations — stopping early.[/yellow]")
            break

    # Crown champion
    champion = max(population, key=lambda g: g.fitness.rank_key())
    champion.status = "champion"
    upsert_genome(db, champion)

    console.print(
        f"\n[bold green]Champion: {champion.id} — "
        f"unlocks={champion.fitness.unlocks:.1f} score={champion.fitness.score:.1f}[/bold green]"
    )
    console.print(f"Planner: {champion.genome.planner.model_dump()}")
    console.print(f"Rules: {champion.genome.rules}")

    return champion
