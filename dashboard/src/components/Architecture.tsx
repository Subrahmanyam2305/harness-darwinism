import { motion } from "framer-motion";

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

const ACHIEVEMENTS = [
  "collect_wood", "place_table", "make_wood_pickaxe", "collect_stone",
  "place_furnace", "make_stone_pickaxe", "collect_iron", "make_iron_pickaxe",
  "collect_diamond", "collect_coal", "collect_drink", "collect_sapling",
  "place_plant", "eat_plant", "eat_cow", "defeat_zombie",
  "defeat_skeleton", "make_wood_sword", "make_stone_sword", "make_iron_sword",
  "place_stone", "wake_up",
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
    desc: "Each genome plays Crafter across multiple random seeds. The LLM agent receives text observations and must return JSON actions. Score = upstream Crafter metric: exp(mean(log(1 + 100 × rate))) − 1.",
    icon: "🎮",
  },
  {
    step: "3",
    title: "Select & Archive",
    desc: "Rank by fitness score. Top survivors carry forward; the rest are archived. Archived genomes can still be sampled as crossover partners.",
    icon: "🏆",
  },
  {
    step: "4",
    title: "Crossover",
    desc: "Pair survivors and archived genomes. An Opus-class meta-LLM merges the best ideas from two parent harnesses into a child configuration, informed by failure traces.",
    icon: "🧬",
  },
  {
    step: "5",
    title: "Mutate",
    desc: "The meta-LLM tweaks the child: adding, removing, or rewording strategy rules. A dedup ledger prevents repeating mutations that already failed.",
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
      {/* Environment */}
      <motion.section
        className="panel arch-section"
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.05 }}
      >
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
              <p>Text summary: vitals (health, food, drink, energy), inventory, 9×7 semantic grid, nearby items with direction &amp; distance, achievements unlocked so far.</p>
            </div>
            <div className="arch-card">
              <div className="arch-card-title">Episode</div>
              <p>200 steps max. The agent dies if health reaches 0 (from mobs, lava, or starvation). Score is based on 22 binary achievements.</p>
            </div>
            <div className="arch-card">
              <div className="arch-card-title">Goal</div>
              <p>Unlock as many achievements as possible. Progress through a tech tree from basic collection → crafting → combat → rare resources.</p>
            </div>
          </div>
        </div>
      </motion.section>

      {/* Tech Tree */}
      <motion.section
        className="panel arch-section"
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
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
      </motion.section>

      {/* Actions */}
      <motion.section
        className="panel arch-section"
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.15 }}
      >
        <h2>Available Actions ({ACTIONS.length})</h2>
        <div className="actions-grid">
          {ACTIONS.map((a) => (
            <div className="action-item" key={a.name}>
              <code>{a.name}</code>
              <span>{a.desc}</span>
            </div>
          ))}
        </div>
      </motion.section>

      {/* Scoring */}
      <motion.section
        className="panel arch-section"
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <h2>Scoring</h2>
        <div className="arch-text">
          <p>The <strong>upstream Crafter score</strong> aggregates binary achievement rates across multiple episodes:</p>
          <div className="formula-box">
            score = exp( mean( log(1 + 100 × rate<sub>i</sub>) ) ) − 1
          </div>
          <p style={{ marginTop: 10, fontSize: 13, color: "var(--text-secondary)" }}>
            Where rate<sub>i</sub> is the fraction of episodes where achievement i was unlocked (0.0–1.0).
            With 22 achievements, a perfect score is <strong>100.0</strong>. Random policy scores ~1.6.
            State-of-the-art RL agents reach 15–20+.
          </p>
        </div>
      </motion.section>

      {/* Evolution Loop */}
      <motion.section
        className="panel arch-section"
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.25 }}
      >
        <h2>How Harness Evolution Works</h2>
        <p style={{ fontSize: 13, color: "var(--text-secondary)", marginBottom: 16 }}>
          We use a <strong>genetic algorithm</strong> to evolve the LLM agent's <em>harness</em> — the system prompt,
          strategy rules, action allowlist, and memory policy — while keeping the underlying model (Claude Haiku 4.5) frozen.
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
      </motion.section>

      {/* What Evolves */}
      <motion.section
        className="panel arch-section"
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
      >
        <h2>What Evolves (The Genome)</h2>
        <div className="arch-grid-2">
          <div className="arch-card">
            <div className="arch-card-title">System Prompt</div>
            <p>Base instructions telling the LLM it's a Crafter agent. Includes fixed game mechanics, response format, and the evolvable strategy rules.</p>
          </div>
          <div className="arch-card">
            <div className="arch-card-title">Strategy Rules</div>
            <p>Ordered list of natural-language rules (e.g., "Collect at least 3 wood before placing a table"). These are the primary targets of crossover and mutation.</p>
          </div>
          <div className="arch-card">
            <div className="arch-card-title">Action Allowlist</div>
            <p>Subset of the 17 Crafter actions the agent is permitted to use. The meta-agent can restrict actions to force focus on early tech tree progression.</p>
          </div>
          <div className="arch-card">
            <div className="arch-card-title">Memory Policy</div>
            <p>How the agent remembers past events: <code>notes</code> (bullet points), <code>summary</code> (compressed), or <code>none</code>. Plus max items retained.</p>
          </div>
        </div>
      </motion.section>

      {/* Stack */}
      <motion.section
        className="panel arch-section"
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.35 }}
      >
        <h2>Tech Stack</h2>
        <div className="arch-grid-3">
          <div className="arch-card">
            <div className="arch-card-title">Agent Model</div>
            <p>Claude Haiku 4.5 via AWS Bedrock. ~200 calls per episode at temperature 0.3.</p>
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
      </motion.section>
    </div>
  );
}
