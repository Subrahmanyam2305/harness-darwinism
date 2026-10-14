# Efficient Crafter Skill Controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a compact, guarded multi-step Crafter controller, bounded candidate concurrency, dashboard evidence, and paired controller comparison.

**Architecture:** Keep primitive control intact and introduce a run-level skills controller backed by typed execution state, structured Crafter snapshots, a fixed six-skill catalog, and per-step guards. Persist a decision timeline alongside action history, then evaluate candidate episodes concurrently while serializing population updates and Atlas writes.

**Tech Stack:** Python 3.12, Pydantic 2, Crafter, boto3 Bedrock Converse, PyMongo, FastAPI, React 19, TypeScript.

**Spec:** `docs/superpowers/specs/2026-10-14-efficient-skill-controller-design.md`

## Global Constraints

- Preserve the primitive controller and existing run documents.
- Fixed skill code and frozen gameplay model across genomes.
- Every local action uses the real environment and emits a replay frame.
- No paid evolution run during implementation or verification.
- Atlas writes occur in the coordinating thread.

## Review Focus

- Skill actions disallowed by the genome must stop before stepping.
- Death and the global episode limit must override remaining queued actions.
- World-specific state must reset between seeds.
- Concurrent evaluation must not reuse recorder paths or reorder persistence.
- Legacy runs with no decision timeline must still render.

---

### Task 1: Typed state, snapshots, and skill catalog

**Files:**
- Create: `src/execution_state.py`
- Create: `src/skills.py`
- Modify: `src/crafter_env.py`
- Test: `tests/test_skills.py`

**Interfaces:**
- Produces: `ExecutionState`, `StatePatch`, `SkillRequest`, `SkillResult`, `CrafterSnapshot`, `feasible_skills()`, and `execute_skill()`.

- [ ] Write fixture tests for snapshot extraction, affordance filtering, skill guards, tool allowlists, progress, and reset-safe state.
- [ ] Implement typed snapshots and the bounded six-skill catalog.
- [ ] Run `pytest tests/test_crafter_env.py tests/test_skills.py -q` and commit.

### Task 2: Skill-controlled episodes and compact prompts

**Files:**
- Modify: `src/agent.py`
- Modify: `src/gameplay_provider.py`
- Modify: `src/llm.py`
- Modify: `src/gameplay.py`
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: Task 1 state and skills.
- Produces: `play_episode(..., controller="primitive"|"skills", max_skill_steps=5)` and decision/action evidence.

- [ ] Add failing tests proving one model decision executes multiple real steps and compact requests omit transcript reasoning.
- [ ] Implement controller dispatch, state patches, decision IDs, guarded local execution, usage/cache fields, and compatibility fallbacks.
- [ ] Run focused agent, recording, and DB tests and commit.

### Task 3: Bounded concurrent evolution and CLI controls

**Files:**
- Modify: `src/evolution.py`
- Modify: `run.py`
- Test: `tests/test_evolution.py`
- Test: `tests/test_run.py`

**Interfaces:**
- Consumes: Task 2 episode controller.
- Produces: `evaluate_population()` with stable result ordering and CLI `--controller`, `--workers`, and `--max-skill-steps`.

- [ ] Add tests for forwarding controls, one-worker compatibility, stable concurrent ordering, and coordinator-only persistence.
- [ ] Implement bounded episode concurrency and print a labeled call upper bound before execution.
- [ ] Run focused evolution and CLI tests and commit.

### Task 4: API and dashboard evidence

**Files:**
- Modify: `api.py`
- Modify: `dashboard/src/api.ts`
- Modify: `dashboard/src/components/GameReplay.tsx`
- Modify: `dashboard/src/App.css`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: Task 2 trace documents.
- Produces: backward-compatible run summaries with decisions and efficiency metrics.

- [ ] Add API serialization tests for skill and legacy records.
- [ ] Expose and render controller, grouped decisions, calls, steps per call, tokens per step, latency, outcomes, and interruption reasons.
- [ ] Run API tests plus dashboard lint/build and commit.

### Task 5: Paired controller evidence command

**Files:**
- Modify: `src/evaluator.py`
- Modify: `run.py`
- Test: `tests/test_evaluator.py`
- Test: `tests/test_run.py`

**Interfaces:**
- Produces: explicitly invoked paired primitive-versus-skills report with identical seeds and settings.

- [ ] Add aggregation tests covering calls, tokens, latency, score, errors, and unavailable cost.
- [ ] Add a comparison command that never runs during discovery or dashboard startup.
- [ ] Run focused tests and commit.

### Task 6: Full verification and ticket handoff

**Files:**
- Modify: `docs/superpowers/plans/2026-10-14-efficient-skill-controller.md`

- [ ] Run Python tests, Ruff checks, dashboard lint, and dashboard build.
- [ ] Run an offline seeded Crafter skill-controller smoke check and inspect its replay without paid inference.
- [ ] Verify the final diff, commit remaining changes, and move issues #10-#15 to In review.

