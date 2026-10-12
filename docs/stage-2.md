# Stage 2: Evolution on the Crafter body

Stage 1 gave the agent a scripted **body** (macros + tech-tree planner) so it can walk, face a tile, and craft without one LLM call per environment step. Stage 2 makes the genetic algorithm operate on that body.

The frozen gameplay model stays Haiku-free in the inner loop. What evolves is the **planner skill config**, not English slogans.

## Why this stage exists

After Stage 1, a default genome already unlocks about 8–12 achievements (wood through furnace / coal / iron collect). The old geometric score (~1.8) was a 200-step, no-facing artifact.

The remaining failures are structural, for example:

- `make_iron_pickaxe` with **no path to table+furnace** (stations placed far apart)
- combat death while the goal is still `attack`
- thirst/hunger deaths because the planner kept mining

Mutating “Collect wood first” does not fix those. Mutating `relocate_workstation`, `prefer_sword`, or drink thresholds can.

## What evolves

`GenomeConfig.planner` (`PlannerPolicy` in [`src/genome.py`](../src/genome.py)) is the primary gene:

| Field | Default | Effect |
|---|---|---|
| `energy_sleep` | 2 | Sleep when energy is at or below this |
| `drink_threshold` | 2 | Drink when thirst is at or below this |
| `food_threshold` | 2 | Eat when food is at or below this |
| `health_flee` | 2 | Flee or fight when health is at or below this |
| `prefer_sword` | false | Craft a sword before pushing the cave |
| `attack_unarmed` | false | Allow `attack` with no sword |
| `relocate_workstation` | false | If table and furnace are not adjacent, collect stone and `relocate_furnace` |
| `sleep_at_night` | false | Sleep during night even if energy is higher |
| `wood_for_table` | 2 | Wood required before `place_table` |
| `stone_for_furnace` | 4 | Stone required before `place_furnace` |
| `disabled_goals` | [] | Goals the planner will not pick |

Rules are still stored and can be keyword-hints (`sword` still biases the planner). They are secondary.

Generation 0 seeds mixed `PlannerPolicy` variants (relocate on, prefer sword, sleep at night, higher drink/food thresholds), not random tip subsets only.

## New skill: relocate furnace

[`src/skills.py`](../src/skills.py) implements `relocate_furnace`: stand next to the table, face an allowed tile, `place_furnace`.

The planner only schedules it when `relocate_workstation` is on, iron tools are still missing, and the known table and furnace are more than Chebyshev distance 2 apart. If stone is below 4, it collects stone first.

## Fitness and selection

Selection no longer uses geometric score alone. [`Fitness.rank_key`](../src/genome.py) is:

1. **unlocks** — mean count of distinct achievements across the generation’s seeds
2. **tech_depth** — how far the ordered tech tree went (wood → table → wood pickaxe → stone → stone pickaxe/furnace → coal/iron → iron pickaxe → diamond)
3. **score** — existing per-episode geometric mean (dashboard curve still uses this)

A genome with 9 unlocks and score 6.6 beats one with 3 unlocks and score 20.

Early-stop in `run_evolution` watches **unlocks**, not a 0.5 score delta.

## Archive and lineage (DGM-style)

- Culled genomes stay `archived` and can be sampled as parents.
- Pairing weight is `unlocks / (1 + n_children)` so high performers with few children and archive stepping stones get another look.
- **Survivors keep their IDs.** They are not rewritten as `h_g{n+1}_*`. Children get new IDs; parents stay `h_g0_02` etc. so `$graphLookup` / the dashboard tree remains honest.

## Meta-agent

[`src/meta_agent.py`](../src/meta_agent.py) crossover and mutation edit **planner fields** from failure traces (`Last goal: make_iron_pickaxe (no path to table+furnace)`). Inventing a new flag combination is allowed.

| `META_MODE` | Behavior |
|---|---|
| `llm` (default) | Opus crossover/mutation; heuristic fallback on parse/provider failure |
| `scripted` | No Bedrock. Heuristic mutate (e.g. iron-path fail → `relocate_workstation=true`) |

Gameplay stays `PLANNER_MODE=scripted` unless you opt into LLM goals when stuck.

## How to run

Smoke-test the body (no Atlas, no Bedrock):

```bash
python run.py play --seeds 100,101,102
python run.py play --seeds 100 --relocate
```

Evolve (writes Atlas; Bedrock only for the 5 children per generation):

```bash
python run.py evolve -g 5
```

Dashboard: API on port 8001, UI on [http://localhost:5174](http://localhost:5174).

## What Stage 2 is not

- Not a model upgrade. The inner loop is still the scripted body.
- Not the official 10k-step Crafter protocol or Hafner’s published score formula.
- Not a claim of SPRING 27 or human 50. Stage 2 is “evolution can move skills that change play,” with a realistic next band of mid-teens unlocks if relocate/combat genes stick.

## Files

| File | Role |
|---|---|
| [`src/genome.py`](../src/genome.py) | `PlannerPolicy`, `Fitness.unlocks` / `tech_depth` / `rank_key`, `n_children` |
| [`src/planner.py`](../src/planner.py) | Reads policy; `relocate_furnace` goal |
| [`src/skills.py`](../src/skills.py) | `relocate_furnace` macro |
| [`src/meta_agent.py`](../src/meta_agent.py) | Skill crossover/mutation + scripted fallback |
| [`src/evolution.py`](../src/evolution.py) | Rank by unlocks, stable survivor IDs, archive weights |
| [`src/config.py`](../src/config.py) | `meta_mode`, 2000-step episodes |
| [`run.py`](../run.py) | `play --relocate` |

## Stage 3

See [stage-3.md](stage-3.md): paired stations, kite combat, held-out / Hafner eval, ablation arms.
