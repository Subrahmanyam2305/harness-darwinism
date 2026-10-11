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
    )
    console.print(f"\n[bold]Champion rules:[/bold]")
    for i, rule in enumerate(champion.genome.rules, 1):
        console.print(f"  {i}. {rule}")


def cmd_eval_held_out(args):
    """Evaluate the champion on held-out seeds."""
    from src.db import get_db
    from src.evolution import evaluate_genome
    from src.genome import Genome

    db = get_db()
    from src.db import get_genome
    champion = get_genome(db, args.genome_id)
    if not champion:
        console.print(f"[red]Genome {args.genome_id} not found[/red]")
        sys.exit(1)

    held_out_seeds = list(range(200, 230))
    console.print(f"[bold]Evaluating {champion.id} on {len(held_out_seeds)} held-out seeds[/bold]")
    fitness, traces = evaluate_genome(champion, held_out_seeds)
    console.print(f"\n[bold green]Held-out score: {fitness.score:.1f}[/bold green]")
    console.print(f"Achievements: {fitness.achievements}")


def cmd_serve(args):
    """Start the API server."""
    import uvicorn
    console.print(f"[bold green]Starting dashboard API on port {args.port}[/bold green]")
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
    parser = argparse.ArgumentParser(description="Harness Darwinism — evolve LLM agent harnesses")
    sub = parser.add_subparsers(dest="command")

    p_evolve = sub.add_parser("evolve", help="Run the full evolution loop")
    p_evolve.add_argument("-g", "--generations", type=int, default=5)
    p_evolve.set_defaults(func=cmd_evolve)

    p_eval = sub.add_parser("eval", help="Evaluate a genome on held-out seeds")
    p_eval.add_argument("genome_id", help="ID of the genome to evaluate")
    p_eval.set_defaults(func=cmd_eval_held_out)

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
