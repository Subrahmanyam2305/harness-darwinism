const ACTIONS = [
  { name: "noop", desc: "Do nothing" },
  { name: "move_left", desc: "Move west" },
  { name: "move_right", desc: "Move east" },
  { name: "move_up", desc: "Move north" },
  { name: "move_down", desc: "Move south" },
  { name: "do", desc: "Interact with the tile you face" },
  { name: "sleep", desc: "Rest to regain energy" },
  { name: "place_stone", desc: "Place a stone block" },
  { name: "place_table", desc: "Place a crafting table (2 wood)" },
  { name: "place_furnace", desc: "Place a furnace (4 stone)" },
  { name: "place_plant", desc: "Plant a sapling" },
  { name: "make_wood_pickaxe", desc: "Craft wooden pickaxe (1 wood, near table)" },
  { name: "make_stone_pickaxe", desc: "Craft stone pickaxe (1 wood + 1 stone)" },
  { name: "make_iron_pickaxe", desc: "Craft iron pickaxe (1 wood + 1 iron)" },
  { name: "make_wood_sword", desc: "Craft wooden sword (1 wood, near table)" },
  { name: "make_stone_sword", desc: "Craft stone sword (1 wood + 1 stone)" },
  { name: "make_iron_sword", desc: "Craft iron sword (1 wood + 1 iron)" },
];

const TECH_TREE = [
  { tier: "Tier 0", color: "var(--green)", items: ["collect_wood", "collect_drink", "collect_sapling"] },
  { tier: "Tier 1", color: "var(--accent)", items: ["place_table", "eat_cow", "place_plant", "eat_plant"] },
  { tier: "Tier 2", color: "var(--yellow)", items: ["make_wood_pickaxe", "make_wood_sword", "place_stone"] },
  { tier: "Tier 3", color: "var(--orange)", items: ["collect_stone", "collect_coal", "defeat_zombie"] },
  { tier: "Tier 4", color: "var(--red)", items: ["place_furnace", "make_stone_pickaxe", "make_stone_sword", "defeat_skeleton"] },
  { tier: "Tier 5", color: "var(--purple)", items: ["collect_iron", "make_iron_pickaxe", "make_iron_sword", "collect_diamond"] },
];

const EVOLUTION_STEPS = [
  {
    step: "1",
    title: "Seed Population",
    desc: "Generate N initial harness configurations. Each genome contains: a system prompt, strategy rules, allowed action set, memory policy, and context budget.",
    icon: "🌱",
  },
  {
    step: "2",
    title: "Evaluate",
    desc: "Each genome plays Crafter across multiple random seeds. The scripted planner picks macros; score tracks unlocks, iron rate, steps, and the geometric achievement metric.",
    icon: "🎮",
  },
  {
    step: "3",
    title: "Select & Archive",
    desc: "Rank by iron rate, unlocks, mean steps, tech depth, then score. Top survivors carry forward; the rest are archived and can still be sampled as parents.",
    icon: "🏆",
  },
  {
    step: "4",
    title: "Crossover",
    desc: "Pair survivors and archived genomes. An Opus-class meta-LLM merges complementary planner knobs and rules, informed by failure traces.",
    icon: "🧬",
  },
  {
    step: "5",
    title: "Mutate",
    desc: "The meta-LLM or a heuristic tweaks one skill. Extra goals may only come from an allow-list (retreat_to_camp, craft_iron, farm_plants). A ledger blocks duplicate edits.",
    icon: "⚡",
  },
  {
    step: "6",
    title: "Repeat",
    desc: "The new population (children + survivors) enters the next generation. Early stopping triggers after 3 stale generations. The champion is crowned at the end.",
    icon: "🔄",
  },
];

export function Architecture() {
  return (
    <div className="architecture-tab">
      <section className="panel arch-section">
        <h2>The Crafter Environment</h2>
        <div className="arch-text">
          <p>
            <strong>Crafter</strong> is a 2D open-world survival game designed as an RL benchmark.
            The agent starts in a procedurally generated 64×64 world with resources (trees, stone, iron, diamond),
            water, lava, and hostile mobs (zombies, skeletons).
          </p>
          <div className="arch-grid-3" style={{ marginTop: 14 }}>
            <div className="arch-card">
              <div className="arch-card-title">Observation</div>
              <p>Structured state: vitals, inventory, facing, a local semantic view, nearby tiles, and achievements. Macros turn that into walk / face / do.</p>
            </div>
            <div className="arch-card">
              <div className="arch-card-title">Episode</div>
              <p>Default 2000 steps (official Crafter is 10k). The agent dies if health reaches 0. Score is based on 22 binary achievements.</p>
            </div>
            <div className="arch-card">
              <div className="arch-card-title">Goal</div>
              <p>Unlock as many achievements as possible. Progress through a tech tree from basic collection → crafting → combat → rare resources.</p>
            </div>
          </div>
        </div>
      </section>

      <section className="panel arch-section">
        <h2>Achievement Tech Tree</h2>
        <div className="tech-tree">
          {TECH_TREE.map((tier) => (
            <div className="tech-tier" key={tier.tier}>
              <div className="tech-tier-label" style={{ color: tier.color }}>
                {tier.tier}
              </div>
              <div className="tech-tier-items">
                {tier.items.map((item) => (
                  <span
                    className="tech-chip"
                    key={item}
                    style={{ borderColor: tier.color, color: tier.color }}
                  >
                    {item.replace(/_/g, " ")}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="panel arch-section">
        <h2>Available Actions ({ACTIONS.length})</h2>
        <div className="actions-grid">
          {ACTIONS.map((a) => (
            <div className="action-item" key={a.name}>
              <code>{a.name}</code>
              <span>{a.desc}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="panel arch-section">
        <h2>Scoring</h2>
        <div className="arch-text">
          <p>The <strong>upstream Crafter score</strong> aggregates binary achievement rates across multiple episodes:</p>
          <div className="formula-box">
            score = exp( mean( log(1 + 100 × rate<sub>i</sub>) ) ) − 1
          </div>
          <p style={{ marginTop: 10, fontSize: 13, color: "var(--text-secondary)" }}>
            Where rate<sub>i</sub> is the fraction of episodes where achievement i was unlocked (0.0–1.0).
            With 22 achievements, a perfect score is <strong>100.0</strong>. This dashboard also shows the
            project geometric score and ranks evolution by iron rate, unlocks, and survival.
          </p>
        </div>
      </section>

      <section className="panel arch-section">
        <h2>How Harness Evolution Works</h2>
        <p style={{ fontSize: 13, color: "var(--text-secondary)", marginBottom: 16 }}>
          We use a <strong>genetic algorithm</strong> to evolve the agent&apos;s <em>harness</em> — planner
          knobs, strategy rules, action allowlist, and memory policy — while keeping the gameplay model frozen.
        </p>
        <div className="evolution-pipeline">
          {EVOLUTION_STEPS.map((s, i) => (
            <div className="evo-step" key={s.step}>
              <div className="evo-connector">
                <div className="evo-dot">{s.icon}</div>
                {i < EVOLUTION_STEPS.length - 1 && <div className="evo-line" />}
              </div>
              <div className="evo-content">
                <div className="evo-step-title">
                  <span className="evo-num">{s.step}</span>
                  {s.title}
                </div>
                <p>{s.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="panel arch-section">
        <h2>What Evolves (The Genome)</h2>
        <div className="arch-grid-2">
          <div className="arch-card">
            <div className="arch-card-title">System Prompt</div>
            <p>Base instructions for the agent. Includes fixed game mechanics and the evolvable strategy rules.</p>
          </div>
          <div className="arch-card">
            <div className="arch-card-title">Planner Policy</div>
            <p>Thresholds and flags the scripted planner reads: drink/food/sleep, kite, camp retreat, extra allow-listed macros.</p>
          </div>
          <div className="arch-card">
            <div className="arch-card-title">Strategy Rules</div>
            <p>Short natural-language hints. Keyword-matched or used by the meta-agent; not a free Python rewrite.</p>
          </div>
          <div className="arch-card">
            <div className="arch-card-title">Memory Policy</div>
            <p>How traces are stored: <code>notes</code>, <code>summary</code>, or <code>none</code>, plus max items retained.</p>
          </div>
        </div>
      </section>

      <section className="panel arch-section">
        <h2>Tech Stack</h2>
        <div className="arch-grid-3">
          <div className="arch-card">
            <div className="arch-card-title">Gameplay</div>
            <p>Scripted planner + macros by default. Haiku is optional and only used when <code>PLANNER_MODE=llm</code> and a goal is stuck.</p>
          </div>
          <div className="arch-card">
            <div className="arch-card-title">Meta Model</div>
            <p>Claude Opus 4.6 via Bedrock. Performs crossover and mutation of harness configs.</p>
          </div>
          <div className="arch-card">
            <div className="arch-card-title">Storage</div>
            <p>MongoDB Atlas for genomes, runs, metrics. JSON file fallback for local development.</p>
          </div>
        </div>
      </section>
    </div>
  );
}
