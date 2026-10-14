#!/usr/bin/env python3
"""Main entry point for running Harness Darwinism evolution."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console

from src.settings import settings

console = Console()


def cmd_evolve(args):
    """Run the full evolution loop."""
    from src.evolution import calculate_elite_count, run_evolution

    episode_count = (
        settings.population_size * settings.seeds_per_gen * (args.generations + 1)
    )
    decision_upper_bound = episode_count * args.max_steps
    retry_upper_bound = decision_upper_bound * 3
    elite_count = calculate_elite_count(settings.population_size, settings.elite_ratio)
    children_per_generation = settings.population_size - elite_count
    crossover_calls = children_per_generation * args.generations
    expected_mutation_calls = crossover_calls * settings.mutation_rate
    console.print(
        "[bold]Experiment call estimate[/bold]\n"
        f"  controller: {args.controller}\n"
        f"  workers: {args.workers}\n"
        f"  episodes: {episode_count:,}\n"
        f"  genetic policy: {elite_count} elites + "
        f"{children_per_generation} crossover children per transition\n"
        f"  expected meta-agent calls: {crossover_calls:,} crossover + "
        f"~{expected_mutation_calls:.1f} mutation (before retries)\n"
        f"  decision upper bound: {decision_upper_bound:,} gameplay decisions\n"
        f"  worst-case retry upper bound: {retry_upper_bound:,} gameplay calls"
    )
    if args.controller == "skills":
        console.print(
            "  optimistic full-chunk estimate: "
            f"{episode_count * math.ceil(args.max_steps / settings.max_skill_steps):,} "
            "gameplay calls (actual calls depend on interruptions and retries)"
        )
    champion = run_evolution(
        max_generations=args.generations,
        train_seeds=args.train_seeds,
        experiment_id=args.experiment_id,
        population_size=settings.population_size,
        elite_ratio=settings.elite_ratio,
        mutation_rate=settings.mutation_rate,
        seeds_per_generation=settings.seeds_per_gen,
        max_steps=args.max_steps,
        resume=args.resume,
        seed_profile=args.seed_profile,
        controller=args.controller,
        workers=args.workers,
        max_skill_steps=settings.max_skill_steps,
    )
    console.print("\n[bold]Champion rules:[/bold]")
    for i, rule in enumerate(champion.genome.rules, 1):
        console.print(f"  {i}. {rule}")


def cmd_eval_held_out(args):
    """Evaluate the champion on held-out seeds."""
    from src.db import get_db
    from src.evaluator import evaluate_harness

    db = get_db()
    from src.db import get_genome

    champion = get_genome(db, args.genome_id)
    if not champion:
        console.print(f"[red]Genome {args.genome_id} not found[/red]")
        sys.exit(1)

    held_out_seeds = args.seeds or list(range(200, 230))
    console.print(
        f"[bold]Evaluating {champion.id} on {len(held_out_seeds)} held-out seeds[/bold]"
    )
    result = evaluate_harness(champion, held_out_seeds, max_steps=args.max_steps)
    output = args.output or _default_output(f"heldout-{champion.id}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result.to_dict(), indent=2))
    if not result.complete:
        console.print(
            "[red]Held-out evaluation incomplete because an episode ended with agent_error[/red]"
        )
        sys.exit(2)
    console.print(f"\n[bold green]Held-out score: {result.score:.1f}[/bold green]")
    console.print(f"Evidence: {output}")


def _default_output(prefix: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("data/runs") / f"{prefix}-{timestamp}.json"


def cmd_play(args):
    """Run the fixed baseline with live frames and replay evidence."""
    from src.gameplay import run_recorded_episode, scripted_provider

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path("data/runs") / f"baseline-{timestamp}"
    provider = scripted_provider() if args.scripted else None
    if args.scripted:
        console.print(
            "[yellow]OFFLINE SCRIPTED CHECK — this is not LLM gameplay[/yellow]"
        )
    if args.serve:
        console.print(f"Viewer: http://127.0.0.1:{args.port}/")
    trace, run_id, server = run_recorded_episode(
        seed=args.seed,
        max_steps=args.max_steps,
        output=output,
        provider=provider,
        serve_port=args.port if args.serve else None,
        pace_seconds=args.pace,
        persist=not args.scripted,
    )
    console.print(
        f"[green]Episode complete[/green]: {trace.termination_reason}, "
        f"{trace.steps} steps, score={trace.score:.2f}"
    )
    console.print(f"Replay: {output / 'index.html'}")
    if run_id:
        console.print(f"Atlas run: {run_id}")
    if trace.termination_reason == "agent_error":
        sys.exit(2)
    if server:
        console.print("Press Ctrl-C to stop the viewer.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            server.shutdown()
            server.server_close()


def cmd_compare(args):
    """Evaluate a frozen baseline and champion on identical held-out seeds."""
    from src.db import get_db, get_genome
    from src.evaluator import compare_paired, evaluate_harness
    from src.gameplay import baseline_genome

    database = get_db()
    champion = get_genome(database, args.champion_id)
    if champion is None:
        console.print(f"[red]Genome {args.champion_id} not found[/red]")
        sys.exit(1)
    baseline = (
        get_genome(database, args.baseline_id)
        if args.baseline_id
        else baseline_genome()
    )
    if baseline is None:
        console.print(f"[red]Genome {args.baseline_id} not found[/red]")
        sys.exit(1)
    seeds = args.seeds or list(range(200, 230))
    baseline_result = evaluate_harness(baseline, seeds, max_steps=args.max_steps)
    champion_result = evaluate_harness(champion, seeds, max_steps=args.max_steps)
    comparison = compare_paired(baseline_result, champion_result)
    output = args.output or _default_output(f"paired-{baseline.id}-vs-{champion.id}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "comparison": comparison.to_dict(),
                "baseline": baseline_result.to_dict(),
                "champion": champion_result.to_dict(),
            },
            indent=2,
        )
    )
    console.print(
        f"[green]Paired score delta: {comparison.score_delta:+.2f}[/green] "
        f"over {comparison.sample_size} held-out seeds"
    )
    console.print(f"Evidence: {output}")


def cmd_compare_controllers(args):
    """Compare primitive and skill controllers on one frozen harness."""
    from src.db import get_db, get_genome, insert_run
    from src.evaluator import compare_controllers, evaluate_harness

    database = get_db()
    if database is None:
        raise RuntimeError("Atlas is required for paired controller evidence")
    genome = get_genome(database, args.genome_id)
    if genome is None:
        console.print(f"[red]Genome {args.genome_id} not found[/red]")
        sys.exit(1)
    seeds = args.seeds or list(range(200, 230))
    comparison_id = (
        f"controllers-{genome.id}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    artifact_root = Path("data/runs/controller-comparisons") / comparison_id
    common = {"max_steps": args.max_steps}
    primitive = evaluate_harness(
        genome,
        seeds,
        controller="primitive",
        max_skill_steps=1,
        artifact_root=artifact_root,
        **common,
    )
    skills = evaluate_harness(
        genome,
        seeds,
        controller="skills",
        max_skill_steps=args.max_skill_steps,
        artifact_root=artifact_root,
        **common,
    )
    comparison = compare_controllers(primitive, skills)
    for arm_result in (primitive, skills):
        for trace in arm_result.traces:
            insert_run(
                database,
                {
                    "_id": f"{comparison_id}-{arm_result.controller}-s{trace.seed}",
                    "kind": "controller_comparison_episode",
                    "comparison_id": comparison_id,
                    "controller_arm": arm_result.controller,
                    "paired_seed": trace.seed,
                    "genome_id": genome.id,
                    "seed": trace.seed,
                    "score": trace.score,
                    "steps": trace.steps,
                    "artifact_path": trace.artifact_path,
                    "trace": trace.to_dict(),
                    "created_at": datetime.now(UTC).isoformat(),
                },
            )
    output = args.output or artifact_root / "comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(comparison.to_dict(), indent=2))
    console.print(
        "[green]Paired controller comparison complete[/green]: "
        f"calls {comparison.model_call_delta:+d}, "
        f"latency {comparison.latency_delta_seconds:+.2f}s"
    )
    console.print(f"Evidence: {output}")


def cmd_serve(args):
    """Start the API server."""
    import uvicorn

    console.print(
        f"[bold green]Starting dashboard API on port {args.port}[/bold green]"
    )
    uvicorn.run("api:app", host="0.0.0.0", port=args.port, reload=True)


def cmd_seed_only(args):
    """Just seed the initial population (for debugging)."""
    from src.db import ensure_indexes, get_db, upsert_genome
    from src.evolution import seed_population

    db = get_db()
    ensure_indexes(db)
    population = seed_population(args.n)
    for g in population:
        upsert_genome(db, g)
        console.print(f"[cyan]{g.id}[/cyan]: {g.genome.rules}")
    console.print(f"\n[green]Seeded {len(population)} genomes[/green]")


def main():
    parser = argparse.ArgumentParser(
        description="Harness Darwinism — evolve LLM agent harnesses"
    )
    sub = parser.add_subparsers(dest="command")

    p_evolve = sub.add_parser("evolve", help="Run the full evolution loop")
    p_evolve.add_argument(
        "-g",
        "--generations",
        type=int,
        default=5,
        help="Evolutionary transitions to run (additional transitions with --resume)",
    )
    p_evolve.add_argument("--experiment-id", required=True)
    p_evolve.add_argument(
        "--train-seeds",
        type=int,
        nargs="+",
        help="Explicit training-seed pool; sampled identically for every candidate",
    )
    p_evolve.add_argument(
        "--seed-profile",
        choices=["legacy", "strong-v2"],
        default="legacy",
        help="Generation-zero harness initialization profile",
    )
    p_evolve.add_argument("--max-steps", type=int, default=100)
    p_evolve.add_argument(
        "--controller",
        choices=["primitive", "skills"],
        default="primitive",
        help="Gameplay controller used for every candidate episode",
    )
    p_evolve.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Maximum candidate episodes evaluated concurrently",
    )
    p_evolve.add_argument(
        "--resume",
        action="store_true",
        help="Continue the latest stored generation using Atlas evidence",
    )
    p_evolve.set_defaults(func=cmd_evolve)

    p_eval = sub.add_parser("eval", help="Evaluate a genome on held-out seeds")
    p_eval.add_argument("genome_id", help="ID of the genome to evaluate")
    p_eval.add_argument("--seeds", type=int, nargs="+")
    p_eval.add_argument("--max-steps", type=int, default=200)
    p_eval.add_argument("--output", type=Path)
    p_eval.set_defaults(func=cmd_eval_held_out)

    p_play = sub.add_parser("play", help="Run and record the fixed baseline harness")
    p_play.add_argument("--seed", type=int, default=42)
    p_play.add_argument("--max-steps", type=int, default=200)
    p_play.add_argument("--output", type=Path)
    p_play.add_argument(
        "--scripted", action="store_true", help="Use labeled offline actions"
    )
    p_play.add_argument("--serve", action="store_true", help="Serve the live viewer")
    p_play.add_argument("--port", type=int, default=8765)
    p_play.add_argument("--pace", type=float, default=0.0)
    p_play.set_defaults(func=cmd_play)

    p_compare = sub.add_parser(
        "compare", help="Compare baseline and champion on paired held-out seeds"
    )
    p_compare.add_argument("champion_id")
    p_compare.add_argument("--baseline-id")
    p_compare.add_argument("--seeds", type=int, nargs="+")
    p_compare.add_argument("--max-steps", type=int, default=200)
    p_compare.add_argument("--output", type=Path)
    p_compare.set_defaults(func=cmd_compare)

    p_controllers = sub.add_parser(
        "compare-controllers",
        help="Compare primitive and skill control on the same genome and seeds",
    )
    p_controllers.add_argument("genome_id")
    p_controllers.add_argument("--seeds", type=int, nargs="+")
    p_controllers.add_argument("--max-steps", type=int, default=200)
    p_controllers.add_argument("--max-skill-steps", type=int, default=5)
    p_controllers.add_argument("--output", type=Path)
    p_controllers.set_defaults(func=cmd_compare_controllers)

    p_serve = sub.add_parser("serve", help="Start the dashboard API")
    p_serve.add_argument("-p", "--port", type=int, default=8001)
    p_serve.set_defaults(func=cmd_serve)

    p_seed = sub.add_parser("seed", help="Seed initial population only")
    p_seed.add_argument("-n", type=int, default=10)
    p_seed.set_defaults(func=cmd_seed_only)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
