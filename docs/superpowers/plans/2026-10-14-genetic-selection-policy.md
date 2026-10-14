# Genetic Selection Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace top-k-only reproduction with a reproducible 30% elite, 70% crossover genetic algorithm using rank-weighted parent selection and probabilistic semantic mutation.

**Architecture:** Keep evaluation, Atlas persistence, and lineage unchanged. Separate deterministic elite preservation from parent selection: every evaluated candidate enters a rank-weighted roulette pool, while only the top elite fraction is cloned. Apply mutation to crossover children using an experiment-and-generation-seeded RNG, forcing mutation only when an unmutated crossover duplicates an existing genome.

**Tech Stack:** Python 3.12, Pydantic, pytest, argparse, MongoDB Atlas persistence.

**Spec:** `docs/harness-darwinism-crafter.md` plus the user-approved population policy in this session.

## Global Constraints

- Keep the population constant after every transition.
- Preserve every evaluated genome and its lineage in Atlas.
- Keep existing `--survivors` commands valid as an exact elite-count override.
- Do not launch an evolution run or make paid provider calls.
- Keep random selection reproducible for the same experiment and generation.

## Review Focus

- Equal-fitness and zero-fitness populations must keep every candidate selectable.
- Parent pairs must be distinct when the population has at least two candidates.
- Population sizes that do not divide cleanly at 30% must still produce at least one elite and one child.
- Invalid elite and mutation ratios must fail before evaluation or Atlas writes.
- Duplicate crossover configurations must enter bounded mutation recovery even when mutation was not initially sampled.

---

### Task 1: Selection policy

**Files:**
- Modify: `src/evolution.py`
- Test: `tests/test_evolution.py`

**Interfaces:**
- Produces: `calculate_elite_count(population_size, elite_ratio) -> int` and rank-weighted `weighted_pairs(...)` over the complete population.
- Consumes: evaluated `Genome.fitness.score` values.

- [x] Write failing tests for 3/7 population composition, all-candidate parent eligibility, distinct pairing, and deterministic pairing.
- [x] Run the focused tests and confirm the new expectations fail.
- [x] Implement elite calculation and rank-weighted pairing with an injected seeded RNG.
- [x] Run the focused tests and confirm they pass.

### Task 2: Mutation policy and uniqueness

**Files:**
- Modify: `src/evolution.py`
- Modify: `src/meta_agent.py`
- Test: `tests/test_evolution.py`

**Interfaces:**
- Consumes: crossover configuration, mutation rate, ledger, parent failure traces, and seeded RNG.
- Produces: seven crossover children for population ten, with mutation sampled independently and final configurations checked for uniqueness.

- [x] Write failing tests for zero mutation, full mutation, and forced recovery of a duplicate crossover.
- [x] Run the focused tests and confirm they fail for the expected behavior differences.
- [x] Implement probabilistic mutation and final-child novelty validation; correct failed-mutation evidence wording.
- [x] Run the focused tests and confirm they pass.

### Task 3: Configuration and CLI

**Files:**
- Modify: `src/config.py`
- Modify: `run.py`
- Test: `tests/test_run.py`

**Interfaces:**
- Produces: `--elite-ratio` default `0.3`, `--mutation-rate` default `0.4`, and optional `--survivors` override.
- Consumes: `run_evolution(..., elite_ratio, mutation_rate, survivors)`.

- [x] Write failing CLI-forwarding and default tests.
- [x] Run the focused tests and confirm the missing arguments fail.
- [x] Wire settings, parser arguments, call estimate, and evolution validation.
- [x] Run focused and full verification suites.
- [x] Commit the implementation without generated runs or secrets.
