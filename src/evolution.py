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
from src.crafter_env import crafter_score
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
    # Core resource loop (wood → tools → stone → iron)
    "PRIORITY 1: Immediately move to the nearest tree and use 'do' to collect wood. Get at least 4 wood before doing anything else.",
    "PRIORITY 2: Once you have 2+ wood, use place_table. Then use make_wood_pickaxe and make_wood_sword.",
    "After getting wood pickaxe, find stone (grey tiles) and use 'do' adjacent to mine it. Get 4+ stone.",
    "After getting 4 stone, use place_furnace. Then make_stone_pickaxe if you have wood+stone.",
    # Survival
    "If food ≤ 3, find a cow or plant and use 'do' adjacent to eat it.",
    "If drink ≤ 3, find water (blue tiles) and use 'do' adjacent to drink.",
    "If energy ≤ 2 and no enemies nearby, use sleep action.",
    "At NIGHT, either sleep or stay near your table/furnace. Don't explore without a sword.",
    # Interaction discipline
    "Use 'do' whenever adjacent to ANY resource — tree, stone, coal, water, cow, plant. Never walk past resources.",
    "Never take more than 4 consecutive move actions without using 'do'. If stuck, try 'do' on the tile you're facing.",
    # Advanced
    "After stone pickaxe, go underground (move_down) to find coal and iron. Use 'do' to mine them.",
    "Collect saplings from trees (use 'do' near trees) and place them with place_plant for the achievement.",
    "If you see a zombie or skeleton and have a sword, use 'do' adjacent to fight it.",
]


# Pre-built strong genome templates for Gen 0 diversity
STRONG_TEMPLATE_RULES = [
    # Template A: Resource-focused
    [
        "ALWAYS: Move to nearest tree first, use 'do' repeatedly until 4+ wood collected.",
        "With 2 wood: place_table. With 1 wood + table nearby: make_wood_pickaxe, then make_wood_sword.",
        "With wood pickaxe: find and mine stone with 'do'. Get 4+ stone for place_furnace.",
        "Use 'do' on EVERY adjacent resource — trees, stone, water, cows, plants. Never walk past anything useful.",
        "If health < 5 or food < 4: find food (cow/plant) and use 'do'. If drink < 4: find water and 'do'.",
        "At night without sword: use sleep. With sword: fight any adjacent enemies with 'do'.",
        "After furnace: go underground (move_down), mine coal and iron, make stone/iron tools.",
    ],
    # Template B: Achievement-chasing  
    [
        "Follow this achievement order: collect_wood → place_table → make_wood_pickaxe → make_wood_sword → collect_drink → collect_sapling → eat_cow → collect_stone → place_furnace.",
        "For each achievement, identify what you need and go get it. Check 'Can craft NOW' in observations.",
        "Use 'do' action constantly — at least every 2-3 steps. It's how you interact with everything.",
        "Keep food ≥ 3, drink ≥ 3, energy ≥ 2. Prioritize survival over achievements when vitals are low.",
        "Place crafting table and keep track of where it is. Return to it when you need to craft.",
        "Explore systematically: don't backtrack. Move in one direction until you find what you need.",
    ],
    # Template C: Survival-focused
    [
        "Step 1-10: Sprint to nearest tree, collect 4 wood with 'do'. Immediately place_table and make_wood_pickaxe + make_wood_sword.",
        "Step 11-30: Find stone, mine 4+ with 'do'. Find water, drink with 'do'. Find cow/plant, eat with 'do'.",
        "Step 31+: place_furnace, go underground for coal/iron. Make better tools.",
        "SURVIVAL: Always keep health > 3, food > 3, drink > 3. Drop everything to eat/drink when low.",
        "COMBAT: Only fight with sword equipped. Use 'do' when adjacent to enemy.",
        "NIGHT: Sleep if no sword. Fight or hide if you have one.",
        "Use 'do' on every single adjacent resource you see. The reward comes from 'do' interactions.",
    ],
]


def seed_population(n: int = 10) -> list[Genome]:
    """Create generation 0 with strong starter templates + random variations."""
    population = []

    # First few genomes use the strong templates
    for i, template_rules in enumerate(STRONG_TEMPLATE_RULES):
        if i >= n:
            break
        genome = Genome(
            id=make_id(0, i),
            generation=0,
            genome=GenomeConfig(
                rules=template_rules,
                system_prompt="You are a Crafter survival agent. Your goal is to unlock as many achievements as possible. Follow the strategy rules carefully. Use 'do' to interact with everything adjacent.",
                context_budget={"history_steps": 15},
            ),
        )
        population.append(genome)

    # Remaining genomes: random subsets of the starter pool
    for i in range(len(STRONG_TEMPLATE_RULES), n):
        k = random.randint(4, min(8, len(STARTER_RULE_POOL)))
        rules = random.sample(STARTER_RULE_POOL, k)
        genome = Genome(
            id=make_id(0, i),
            generation=0,
            genome=GenomeConfig(
                rules=rules,
                context_budget={"history_steps": random.choice([10, 15, 20])},
            ),
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

    # Use upstream Crafter scoring: exp(mean(log(1 + 100*rate))) - 1
    aggregate_score = crafter_score(achievement_rates)

    fitness = Fitness(
        score=aggregate_score,
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

    # Promote survivors to next generation — preserve original identity
    # Survivors carry forward as-is; their original id/generation stays intact
    # so lineage is traceable. Create lightweight references for the next round.
    next_population = list(children)
    for g in survivors:
        # Clone survivor with new generation ID but keep parents pointing to original
        promoted = Genome(
            id=make_id(gen + 1, n_children + survivors.index(g)),
            generation=gen + 1,
            parents=[g.id],
            genome=g.genome.model_copy(),
            fitness=g.fitness.model_copy(),
            mutation_note=f"Carried forward from {g.id}",
            arm=g.arm,
        )
        upsert_genome(db, promoted)
        next_population.append(promoted)

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
