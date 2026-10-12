#!/usr/bin/env python3
"""Main entry point for running Harness Darwinism evolution."""

from __future__ import annotations

import argparse
import sys

from rich.console import Console

console = Console()


def cmd_evolve(args):
    """Run the full evolution loop."""
    from src.evolution import run_evolution
    champion = run_evolution(
        max_generations=args.generations,
        arm=args.arm,
        tag=args.tag,
    )
    console.print(f"\n[bold]Champion rules:[/bold]")
    for i, rule in enumerate(champion.genome.rules, 1):
        console.print(f"  {i}. {rule}")


def cmd_eval_held_out(args):
    """Evaluate a genome on held-out seeds. Traces are not fed back into evolution."""
    from src.db import get_db, get_genome
    from src.evolution import evaluate_genome
    from src.genome import Genome, GenomeConfig

    if args.local:
        champion = Genome(id="play_default", genome=GenomeConfig())
    else:
        db = get_db()
        champion = get_genome(db, args.genome_id)
        if not champion:
            console.print(f"[red]Genome {args.genome_id} not found[/red]")
            sys.exit(1)

    if args.seeds:
        held_out_seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    else:
        held_out_seeds = list(range(200, 200 + args.n))

    console.print(
        f"[bold]Held-out eval {champion.id}: {len(held_out_seeds)} seeds, "
        f"{args.steps} steps[/bold]"
    )
    fitness, traces = evaluate_genome(champion, held_out_seeds, max_steps=args.steps)
    console.print(
        f"unlocks={fitness.unlocks:.1f}  iron_rate={fitness.iron_rate:.2f}  "
        f"depth={fitness.tech_depth:.1f}  steps={fitness.mean_steps:.0f}  "
        f"score={fitness.score:.1f}  hafner={fitness.hafner:.1f}"
    )
    unlocked = [k for k, v in fitness.achievements.items() if v]
    console.print(f"Achievement rates > 0: {unlocked}")
    for t in traces:
        n = sum(1 for v in t.achievements.values() if v)
        console.print(f"  seed {t.seed}: unlocks={n} score={t.score:.1f} steps={t.steps}")


def cmd_serve(args):
    """Start the API server."""
    import uvicorn
    console.print(f"[bold green]Starting dashboard API on port {args.port}[/bold green]")
    uvicorn.run("api:app", host="0.0.0.0", port=args.port, reload=True)


def cmd_play(args):
    """Play Crafter with the scripted body (no evolution, no Atlas writes)."""
    from rich.table import Table

    from src.agent import play_episode
    from src.genome import Genome, GenomeConfig, PlannerPolicy

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    planner = PlannerPolicy(relocate_workstation=args.relocate)
    genome = Genome(id="play_default", genome=GenomeConfig(rules=[], planner=planner))
    table = Table(title="Scripted body play")
    table.add_column("Seed")
    table.add_column("Score", justify="right")
    table.add_column("Steps", justify="right")
    table.add_column("Unlocks", justify="right")
    table.add_column("Achievements")

    traces = []
    for seed in seeds:
        console.print(f"[cyan]Seed {seed}[/cyan]")
        trace = play_episode(genome, seed, max_steps=args.steps)
        traces.append(trace)
        unlocked = [k for k, v in trace.achievements.items() if v]
        table.add_row(
            str(seed),
            f"{trace.score:.1f}",
            str(trace.steps),
            str(len(unlocked)),
            ", ".join(unlocked) if unlocked else "none",
        )
        console.print(trace.failure_summary)
        console.print()

    console.print(table)
    unlock_counts = [sum(1 for v in t.achievements.values() if v) for t in traces]
    median = sorted(unlock_counts)[len(unlock_counts) // 2]
    console.print(f"[bold]Median unlocks: {median}[/bold] across {len(seeds)} seeds")
    if not any(t.achievements.get("collect_wood") for t in traces):
        console.print("[red]No seed collected wood — body needs more work.[/red]")
        sys.exit(1)


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
    parser = argparse.ArgumentParser(description="Harness Darwinism — evolve LLM agent harnesses")
    sub = parser.add_subparsers(dest="command")

    p_evolve = sub.add_parser("evolve", help="Run the full evolution loop")
    p_evolve.add_argument("-g", "--generations", type=int, default=5)
    p_evolve.add_argument(
        "--arm",
        default="full",
        choices=["full", "no_archive", "random_mutate", "no_relocate"],
        help="Ablation arm (tagged on genomes and metrics)",
    )
    p_evolve.add_argument(
        "--tag",
        default="h",
        help="ID prefix so parallel/later runs do not overwrite genomes (e.g. s3)",
    )
    p_evolve.set_defaults(func=cmd_evolve)

    p_eval = sub.add_parser("eval", help="Evaluate a genome on held-out seeds")
    p_eval.add_argument("genome_id", nargs="?", default="play_default")
    p_eval.add_argument("--local", action="store_true", help="Use the default body, skip Atlas")
    p_eval.add_argument("--seeds", default="", help="Comma-separated seeds (default 200..)")
    p_eval.add_argument("-n", type=int, default=5, help="Held-out seed count if --seeds omitted")
    p_eval.add_argument("--steps", type=int, default=2000, help="Env steps (use 10000 for official horizon)")
    p_eval.set_defaults(func=cmd_eval_held_out)

    p_serve = sub.add_parser("serve", help="Start the dashboard API")
    p_serve.add_argument("-p", "--port", type=int, default=8001)
    p_serve.set_defaults(func=cmd_serve)

    p_play = sub.add_parser("play", help="Smoke-test the scripted Crafter body")
    p_play.add_argument("--seeds", default="100,101,102", help="Comma-separated Crafter seeds")
    p_play.add_argument("--steps", type=int, default=2000, help="Env steps per episode")
    p_play.add_argument(
        "--relocate",
        action="store_true",
        help="Enable relocate_workstation planner skill",
    )
    p_play.set_defaults(func=cmd_play)

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
