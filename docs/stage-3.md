# Stage 3: Comparable eval and stronger body skills

Stage 1 made the agent able to play. Stage 2 made evolution edit planner skills. Stage 3 closes the remaining **demo and evidence** gaps: first-class paired workstations, kite combat, a held-out protocol that can use the official 10k-step horizon, Hafner’s published score formula, and ablation arms.

The gameplay model is still frozen. The inner loop is still the scripted body (no per-step Haiku). Stage 3 does **not** claim human 50.5 or SPRING 27 until you run held-out 10k-step evals and report them as such.

## What changed

### Paired workstations

Iron tools need a table **and** a furnace in the same 3×3. Stage 2 only relocated after a failed `make_iron_pickaxe`.

Now `place_furnace` prefers a tile **adjacent to the existing table** whenever `PlannerPolicy.pair_stations` is true (default). `relocate_workstation` remains the recover-after-the-fact gene.

Ablation `no_relocate` forces both `pair_stations=false` and `relocate_workstation=false`.

### Kite combat

Most Stage 1/2 deaths were `Last goal: attack`. `kite` hits an adjacent zombie/skeleton once, then steps away if health is ≤ 7. The planner picks `kite` instead of `attack` when `kite_combat` is true (default) and the agent has a sword.

### Held-out evaluation

`python run.py eval` never writes traces back into mutation. Training seeds stay in 100–129; held-out defaults start at 200.

```bash
# Default body, 5 unseen seeds, 2000 steps
python run.py eval --local -n 5 --steps 2000

# Official-ish horizon (slow, still scripted / no LLM per tick)
python run.py eval --local -n 5 --steps 10000

# Frozen champion from Atlas
python run.py eval h_g0_02 --steps 2000
```

Prints mean **unlocks**, **tech_depth**, project **score**, and **Hafner** score.

### Hafner score

[`hafner_score`](../src/evolution.py) is the published Crafter aggregate:

`S = exp(mean_i ln(1 + s_i)) − 1` with success rates `s_i ∈ [0, 100]` over the 22 achievements.

The dashboard curve still uses this repo’s per-episode geometric mean (`Fitness.score`). Use Hafner when comparing to SPRING / Dreamer / human tables.

### Ablation arms

`python run.py evolve -g 5 --arm full`

| Arm | Difference |
|---|---|
| `full` | Archive sampling + trace-driven mutate + paired stations |
| `no_archive` | Parents only from the live top 5 |
| `random_mutate` | Flip a random planner flag; ignore traces |
| `no_relocate` | Pairing and relocate genes forced off |

Arm is stored on genomes and metrics so all four can live in one Atlas DB.

## Commands

```bash
python run.py play --seeds 100,101,102
python run.py eval --local -n 5 --steps 2000
python run.py evolve -g 5 --arm full
python run.py evolve -g 5 --arm no_archive
```

Do not start evolve from tests or editor discovery. Held-out traces must not be passed to `mutate`.

## Realistic targets

| Setup | What to expect |
|---|---|
| Stage 1 body, 2000 steps | ~8–12 unlocks / seed |
| Stage 3 body + kite/pair, 2000 steps | Fewer hug-deaths; better chance of iron tools if stations pair |
| Held-out 10k steps | More time for diamond *if* iron pickaxe exists; not automatic 50+ |
| Official human / SPRING numbers | Require Hafner score + 10k-step protocol + many seeds |

60 on the official scale is above human experts (50.5). Stage 3 gives you the **measurement** and the last body skills; it does not guarantee 60.

## Files

| File | Role |
|---|---|
| [`src/skills.py`](../src/skills.py) | `kite`; `place_furnace` next to table |
| [`src/planner.py`](../src/planner.py) | `kite` vs `attack` |
| [`src/genome.py`](../src/genome.py) | `pair_stations`, `kite_combat`, `Fitness.hafner` |
| [`src/evolution.py`](../src/evolution.py) | Hafner score, `--arm` constraints |
| [`src/meta_agent.py`](../src/meta_agent.py) | `random_mutate` ablation |
| [`run.py`](../run.py) | `eval --local --steps`, `evolve --arm` |

## Stage 4

See [stage-4.md](stage-4.md): camp, night, `craft_iron`, iron/survival fitness, stuck-only LLM, cheap unlocks.

## Still out of scope

Upgrading the frozen gameplay model to Opus, visual observations, Atlas Agent Engine memory genes, and a full 30-seed × 10k-step paper table. Run those only with an explicit budget.
