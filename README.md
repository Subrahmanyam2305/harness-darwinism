# 🧬 Harness Darwinism

A genetic algorithm that **evolves LLM agent harnesses** on the [Crafter](https://github.com/danijar/crafter) survival game. The LLM is frozen — only the harness (system prompt, rules, memory policy, context budget) changes. Inspired by the [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954).

## How It Works

```
Population of 10 harness genomes
        │
        ▼
   ┌─ Evaluate ──── each genome plays Crafter on sampled seeds
   │                 score = achievement-based Crafter metric
   │
   ├─ Select ─────── top 5 survive, bottom 5 → archive
   │
   ├─ Crossover ──── LLM merges two parent genomes
   │
   ├─ Mutate ─────── LLM reads failure traces, makes one targeted edit
   │                 (ledger prevents retrying failed ideas)
   │
   └─ Repeat ─────── fresh seeds each generation
```

**What evolves:** rules, system prompt, memory policy, context budget  
**What's fixed:** the LLM model, Crafter environment, action space  
**What evaluates:** Crafter's built-in achievement score (objective, no LLM judge)

## Quick Start

### 1. Clone & install

```bash
git clone git@github.com:Subrahmanyam2305/harness-darwinism.git
cd harness-darwinism
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Set up credentials

```bash
cp .env.example .env
# Edit .env with your AWS Bedrock creds and MongoDB URI
```

The `.env` file needs:
- **AWS Bedrock** — `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, `AWS_DEFAULT_REGION`
- **MongoDB Atlas** — `MONGODB_URI` (free tier works fine)

> **Note:** AWS session tokens expire. If you get auth errors, refresh your credentials and update `.env`.

### 3. Run evolution

```bash
# Quick test (2 genomes, 1 seed, 15 steps — ~2 min)
python run.py evolve -g 1

# Full run (default: 10 genomes, 5 seeds, 200 steps, 10 generations)
python run.py evolve -g 5
```

### 4. Start the dashboard

In a separate terminal:

```bash
source .venv/bin/activate
python run.py serve
```

In another terminal:

```bash
cd dashboard
npm install
npm run dev -- --port 5174
```

Open [http://localhost:5174](http://localhost:5174) — the dashboard auto-polls the API every 5 seconds.

### 5. Evaluate champion on held-out seeds

```bash
python run.py eval h_g5_00  # replace with your champion's ID
```

## Project Structure

```
harness-darwinism/
├── run.py                  # CLI: evolve | eval | serve | seed
├── api.py                  # FastAPI dashboard backend (port 8001)
├── .env                    # Credentials (not committed)
│
├── src/
│   ├── config.py           # Settings from .env
│   ├── genome.py           # Genome dataclass (the evolvable harness)
│   ├── crafter_env.py      # Crafter → text observation wrapper
│   ├── llm.py              # Bedrock Converse API client
│   ├── agent.py            # LLM agent that plays Crafter
│   ├── meta_agent.py       # Crossover + trace-driven mutation
│   ├── evolution.py        # GA loop: evaluate → select → breed
│   └── db.py               # MongoDB Atlas (falls back to local JSON)
│
├── dashboard/              # React + TypeScript + Vite
│   └── src/
│       ├── components/
│       │   ├── FitnessCurve.tsx      # Best/mean score per generation
│       │   ├── LineageTree.tsx       # Interactive lineage graph
│       │   ├── AchievementGrid.tsx   # 22-achievement baseline vs champion
│       │   └── GenomeDetail.tsx      # Inspect any genome's rules
│       └── hooks/
│           └── useEvolutionData.ts   # Auto-polling data hook
│
└── data/
    └── seeds.json          # 30 train + 30 held-out seeds
```

## Models

| Role | Default Model | Why |
|------|--------------|-----|
| **Agent** (plays Crafter) | Claude Haiku 4.5 | Fast + cheap — 50 LLM calls per episode |
| **Meta-agent** (crossover/mutation) | Claude Opus 4.6 | Smart — only called once per child genome |

To change models, edit `AGENT_MODEL` and `META_MODEL` in `.env`. Use Bedrock inference profile IDs (e.g., `us.anthropic.claude-sonnet-4-6`).

## MongoDB Collections

| Collection | Purpose |
|---|---|
| `harnesses` | Genomes with parents, fitness, status (alive/archived/champion) |
| `runs` | Per genome × seed: score, achievements, failure summary |
| `ledger` | Hash of every tried mutation (prevents duplicates) |
| `metrics` | Best/mean score per generation (powers the fitness curve) |

If MongoDB is unavailable, the system automatically falls back to local JSON files in `data/store/`.

## Customizing Evolution

Edit parameters in `src/config.py` or override at runtime:

```python
from src.config import settings
settings.population_size = 10   # genomes per generation
settings.survivors = 5          # keep top N
settings.seeds_per_gen = 5      # Crafter seeds per evaluation
settings.max_steps_per_episode = 200  # steps per Crafter game
settings.max_generations = 10   # when to stop
```

## Architecture

```
┌─────────────────────────────────────────────┐
│  Orchestrator (evolution.py)                 │
│  Generation loop: eval → select → breed      │
└──────┬──────────────────┬───────────────────┘
       │                  │
┌──────▼──────┐    ┌──────▼──────────┐
│ Agent        │    │ Meta-agent       │
│ (Haiku 4.5)  │    │ (Opus 4.6)       │
│ Plays Crafter│    │ Crossover/Mutate │
│ via Bedrock  │    │ via Bedrock      │
└──────┬──────┘    └──────┬──────────┘
       │                  │
┌──────▼──────────────────▼───────────────────┐
│  MongoDB Atlas                               │
│  harnesses │ runs │ ledger │ metrics         │
└──────┬──────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────┐
│  Dashboard (React + Vite)                    │
│  Fitness curve │ Lineage tree │ Achievements │
└─────────────────────────────────────────────┘
```

## References

- Zhang et al. (2025). [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954). arXiv:2505.22954
- Hafner (2021). [Benchmarking the Spectrum of Agent Capabilities](https://github.com/danijar/crafter) (Crafter)
