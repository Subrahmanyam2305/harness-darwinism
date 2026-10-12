# Stage 4: Survive, craft iron, measure honestly

Stages 1–3 got the agent to ~10 unlocks. Games still died around step 180 (kite / thirst / hunger), so the 2000-step budget was unused and iron tools stayed rare. Evolution only raised `health_flee`. Stage 4 changes the **body and fitness**, not another generation of tips.

Do not start `evolve` from this document. Verify with `play` and `eval --local`.

## The six items

### 1. Stop dying — camp and night

`WorldMap` stores **camp** as the first table seen. New goal `retreat_to_camp` BFS-walks home.

If drink or food is below `safe_vitals` (default 5), or it is night and `retreat_at_night` is true, the planner will not kite, mine coal/iron, or explore. It drinks, eats, goes home, or sleeps.

`kite` only runs with a sword **and** at least one walkable escape neighbor; otherwise `flee`.

### 2. Iron as one primitive

`craft_iron` is a single macro: pair table and furnace (rebuild furnace on a table neighbor if needed), then `make_iron_pickaxe`. The planner calls it when a stone pickaxe, coal, iron, and wood are in inventory. The old relocate-then-spin loop is not the primary path.

### 3. Fitness that can see iron and survival

`Fitness.rank_key` is:

`(iron_rate, unlocks, mean_steps, tech_depth, score)`

Nine wood-tier unlocks and death at 180 lose to fewer unlocks plus an iron pickaxe and a long life.

Heuristic mutation may enable allow-listed macros only: `retreat_to_camp`, `craft_iron`, `farm_plants`. No free Python self-edits.

### 4. Official-ish measurement

Held-out eval never feeds traces into mutation.

```bash
python run.py play --seeds 100,101,102
python run.py eval --local -n 3 --steps 2000
python run.py eval --local -n 5 --steps 10000   # official-ish horizon; slow
```

Compare Hafner scores to SPRING / human only after 10k-step, multi-seed held-out.

### 5. LLM only when stuck

Default `PLANNER_MODE=scripted`. If the same goal fails three times **and** `PLANNER_MODE=llm`, the planner may ask Haiku for one allow-listed goal, capped by `max_llm_calls_per_episode`. The default path spends no Bedrock on gameplay.

### 6. Cheap unlocks

After wood-tier, or while waiting out night at camp: `farm_plants` (`collect_sapling` → `place_plant` → eat ripe plant), `sleep_safe` for `wake_up`, and sword fights at camp edge only (`defeat_zombie`).

## Success check

On seeds 100–102, median **steps** should rise above ~180, and at least one seed should attempt `craft_iron` without a 12-repeat spin. `wake_up` / sapling / plant may appear if the agent lives past night.

## Files

| File | Role |
|---|---|
| [`src/skills.py`](../src/skills.py) | Camp, retreat, `craft_iron`, safer kite, `farm_plants` |
| [`src/planner.py`](../src/planner.py) | Safe vitals, night, `craft_iron`, farm |
| [`src/genome.py`](../src/genome.py) | `safe_vitals`, `retreat_at_night`, `extra_goals`, `iron_rate` |
| [`src/evolution.py`](../src/evolution.py) | New rank key |
| [`src/meta_agent.py`](../src/meta_agent.py) | Allow-list mutate |
| [`src/agent.py`](../src/agent.py) | Stuck → optional LLM; retreat instead of spin |

## Still out of scope

Per-step Opus, DGM rewriting Python files, 30×10k paper tables, claiming 50–60 official Crafter.
