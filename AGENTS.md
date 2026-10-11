# Harness Darwinism

## Purpose and priorities

Read `docs/harness-darwinism-crafter.md` before changing architecture, evolution,
evaluation, or the demo. It is the project brief; distinguish its proposed
features from behavior verified in the current code.

This project evolves agent harness configurations on Crafter for the Recursive
Harnessing hackathon track. The gameplay model stays fixed across variants;
prompts, rules, allowed actions, memory policies, and context budgets can evolve.
MongoDB Atlas is the primary persistence and memory integration. Use the event's
Atlas Hackathon Sandbox for the submission.

Priorities are a working evolution loop, paired held-out evidence, and a visual
demo of an LLM-controlled agent playing real Crafter. Comparisons remain part of
the deliverable, with their depth determined by the agreed scope. Text input to
the agent is compatible with visual gameplay for the audience. Verify the
environment and rendered gameplay before extending harness evolution.

## Repository boundaries

| Area | Current responsibility |
| --- | --- |
| `src/crafter_env.py`, `src/agent.py` | Environment adapter and gameplay episodes |
| `src/evolution.py`, `src/meta_agent.py` | Selection, breeding, mutation, and orchestration |
| `src/genome.py`, `src/config.py` | Shared domain models and runtime settings |
| `src/db.py`, `api.py` | MongoDB persistence and dashboard API |
| `dashboard/` | React/TypeScript visualizations and demo UI |
| `run.py`, `data/seeds.json` | CLI entry point and seed data; verify which seed source the runner uses |

Three people share this project. Establish file ownership before parallel
implementation and preserve others' changes. Coordinate changes to shared
models, configuration, dependencies, API responses, and database schemas before
editing them. Ownership assignments will be agreed separately.

For independent work, agree on typed input/output contracts first. Each component
should be developable and verifiable against labeled fixtures or a stub of that
contract. Keep fixtures visibly separate from live data. If a contract change is
needed, explain its impact and coordinate with the other owners; avoid silently
changing both sides of an interface.

## Development conventions

- Use simple, cohesive modules with explicit dependencies. Apply SOLID, DRY,
  KISS, and YAGNI without building abstractions for hypothetical requirements.
- Keep selection, validation, score computation, and experiment bookkeeping
  deterministic where possible. Use LLMs for gameplay decisions and genome edits.
- Keep provider calls, environment mechanics, storage, and UI concerns behind
  narrow interfaces. Prefer existing code and dependencies before adding tools.
- Use typed Python contracts and Pydantic models at external boundaries. Use
  TypeScript interfaces for API data; validate model output before accepting it.
- Handle expected provider, database, and action failures explicitly. Record
  invalid output, retries, and fallback actions so they remain visible in traces.
- Start new source files with a concise responsibility description. Use
  Google-style docstrings for public or non-trivial Python functions and classes;
  comments explain non-obvious reasons, not implementation history.
- Keep edits within the assigned scope. Avoid broad refactors or incidental
  formatting of other owners' files.

## Evolution and evaluation integrity

- Keep the active population constant: survivors plus children refill the
  configured population. Preserve culled variants and parent lineage in Atlas.
- Validate genome bounds and allowed actions before a run. Keep duplicate-edit
  prevention in the mutation ledger; bound retries when breeding cannot refill.
- Use identical training seeds within a generation. Keep held-out seeds and their
  traces inaccessible to mutation, selection, and memory retrieval during
  evolution. Freeze the champion before held-out evaluation.
- Fix gameplay model, sampling settings, observation mode, action interface, and
  episode limits across compared variants unless explicitly running an ablation.
- Verify the score definition and aggregation against upstream Crafter before
  calling a metric the standard benchmark score. Track achievements, survival,
  model calls, tokens, and measured cost separately.
- Compare baseline and champion on paired held-out worlds and report sample size
  and uncertainty. Parent/child scores from different seed batches are not a
  controlled per-edit gain; causal rule claims require a controlled ablation.
- Crafter seeds reproduce initial worlds, but exact trajectories can diverge
  because creature handling uses unordered sets. Extra night renders consume the
  gameplay RNG: preserve its state around recording renders or capture existing
  observations. Verify that recording does not change the evaluated run.
- Label scripted checks, fixtures, recorded replays, and live LLM runs accurately.
  Report only observed scores, costs, achievements, and deployment status.

## Environment and verification

Use Python 3.12 in the root `.venv`; dependencies and dev tools are declared in
`pyproject.toml`. Initial setup from the repository root:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r pyproject.toml --extra dev
npm --prefix dashboard ci
```

Inspect `run.py --help` and `dashboard/package.json` for current commands. VS Code
uses the root interpreter and offers current-file and dashboard-API debugging.
Run the API on port 8001 and the dashboard with `npm --prefix dashboard run dev`.

Add focused tests for domain invariants, state transitions, scoring, and changed
contracts. Use real seeded Crafter runs for environment checks. For UI or replay
changes, inspect actual rendered gameplay and verify playback; a successful
build alone does not establish that the demo works.

Run checks appropriate to the change:

```sh
.venv/bin/ruff check <changed-python-paths>
.venv/bin/ruff format --check <changed-python-paths>
.venv/bin/python -m pytest <relevant-test-paths>
npm --prefix dashboard run lint
npm --prefix dashboard run build
```

Use concrete paths for placeholders. Report the commands and results, including
checks that could not run. Finish with changed behavior and remaining gaps;
distinguish static, test, rendered, provider, and live-database evidence.

## Credentials, artifacts, and external actions

Keep credentials in the ignored `.env` and placeholders in `.env.example`. Use
relative workspace paths in shared editor settings. Keep generated recordings,
traces, and verification output under ignored `data/runs/`; commit reproducible
code and small labeled fixtures rather than generated media or secrets.

Verify the selected Atlas target before writes. Run offline checks before paid
inference; establish provider access, spend limits, and expected run size before
launching an experiment. Never start evolution automatically during editor or
test discovery. Commit, push, publish, and deploy only within the user's stated
authorization.
