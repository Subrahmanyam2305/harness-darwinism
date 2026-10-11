# Harness Darwinism system design

The gameplay model stays fixed while harness configurations evolve. This diagram
shows the intended system, grouped into the three workstreams.

```mermaid
flowchart TB
    Train[Training seeds]
    LLM[LLM inference provider]

    subgraph W2[Workstream 2 — Evolution and MongoDB backend]
        Orchestrator[Generation orchestrator]
        Select[Rank fitness: keep 5 and archive 5]
        Meta[Meta-agent: crossover and trace-driven mutation]
        Validate[Validate children and check mutation ledger]
        Refill[5 survivors + 5 children = 10 genomes]
        Atlas[(MongoDB Atlas: genomes, runs, lineage, ledger, metrics)]
        API[Dashboard API]
    end

    subgraph W1[Workstream 1 — Crafter agent and evaluation]
        Evaluate[Evaluate genome on shared seeds]
        Agent[Harness runtime: prompt, rules, memory, context]
        Game[Real Crafter environment]
        Results[Scores, achievements, traces, usage]
        Recordings[Rendered gameplay recordings]
        HeldOut[Paired evaluation: frozen baseline and champion]
    end

    subgraph W3[Workstream 3 — Visual demo and dashboard]
        Dashboard[React dashboard]
        Playback[Baseline and champion gameplay]
        Charts[Lineage, fitness, achievements, evolved rules]
        Evidence[Held-out results and comparisons]
    end

    Train --> Orchestrator
    Orchestrator -->|genomes and seed batch| Evaluate
    Evaluate --> Agent
    Agent <-->|observations and validated actions| Game
    Agent <-->|fixed gameplay model| LLM
    Game --> Results
    Game --> Recordings
    Results --> Orchestrator
    Orchestrator -->|persist generation results| Atlas
    Orchestrator --> Select
    Select -->|parent genomes| Meta
    Atlas -->|training failure traces| Meta
    Meta <-->|genome edits| LLM
    Meta --> Validate
    Validate <-->|duplicate checks| Atlas
    Validate --> Refill
    Refill -->|next generation| Orchestrator

    Orchestrator -->|stop evolution and freeze champion| HeldOut
    Unseen[Unseen test seeds] --> HeldOut
    HeldOut -->|same evaluator for both harnesses| Evaluate
    Results -->|persist held-out results separately| Atlas

    Atlas --> API
    API -->|poll results initially| Dashboard
    Recordings -->|served media references| Playback
    Dashboard --> Playback
    Dashboard --> Charts
    Dashboard --> Evidence
```

Each generation evaluates all 10 genomes on the same training seed batch. The
orchestrator records results, retains the top 5, archives the rest, and produces
5 validated children using crossover and failure-informed mutation. The process
repeats until the configured stopping condition is reached.

After evolution, the frozen baseline and champion play the same unseen test
worlds. Test results and traces never feed back into evolution. The dashboard
shows real gameplay alongside performance, achievements, and lineage. Additional
experimental arms use the same evaluation path.

Agree on evaluator results, database schemas, and API response contracts before
parallel implementation. The backend can use a stub evaluator and the dashboard
can use labeled API fixtures while real gameplay is developed.

The current implementation includes evolution, MongoDB persistence, API routes,
and dashboard polling. LLM gameplay verification, recording delivery, complete
trace and usage persistence, and paired held-out reporting remain integration
work. Recordings must preserve gameplay randomness around extra renders.
