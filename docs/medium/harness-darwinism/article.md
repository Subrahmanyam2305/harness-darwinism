# Harness Darwinism: Evolving Agent Harnesses with Genetic Algorithms

## 1. Introduction

As frontier models become more capable of completing tasks from single-shot prompts, the natural extension is to ask whether they can also improve performance across repeated attempts without changing their weights.

Prior work has treated external memory as one way to support this form of adaptation [2–5], but few have explored agentic memory through an evolutionary lens [7]. In this work, we propose **Harness Darwinism**, a framework for studying harness evolution as a form of recursive self-improvement.

Although our experiments did not produce a high absolute score per episode, after eight generations in Crafter, the best score per episode increased from **1.3** to **5.7**, indicating that the harness evolved over time.

## 2. Methodology

### 2.1 Primer on genetic algorithms

A genetic algorithm mimics Darwin's principle of evolution. It begins with a population of candidate solutions, each with a genome that defines its behavior [6].

This process has five main stages:

1. **Initialization:** The process starts with an initial set of candidates that take part in evolution. These candidates can be considered the *seeds*.
2. **Fitness evaluation:** Every candidate is evaluated using a fitness function and assigned a score.
3. **Selection:** Candidates are selected through a process similar to a roulette wheel, where the probability of selection is proportional to their fitness scores.
4. **Crossover:** This is the crux of evolution, where two strong candidates cross to produce an offspring that can inherit dominant traits from either parent.
5. **Mutation:** After crossover, a small part of the offspring's genome is randomly updated to preserve diversity among the new candidates.

The next population contains survivors and children. Repeating this cycle searches the genome space without testing every possible genome.

### 2.2 Our method

We evaluated Harness Darwinism in Crafter, a procedurally generated 2D survival environment, and represented each agent harness as a genome whose rules, memory policy, action interface, and context budget could be changed and validated by the evolution loop.

```json
{
  "system_prompt": "You are a Crafter survival agent...",
  "rules": [
    "Collect wood before anything else.",
    "Place a crafting table early."
  ],
  "memory_policy": {
    "type": "notes",
    "max_items": 20
  },
  "context_budget": {
    "history_steps": 10
  }
}
```

The gameplay model receives this genome together with the current observation and returns one validated Crafter action, while the model and its sampling settings remain fixed across every candidate.

For each child, the meta-agent receives two selected genomes, produces a typed configuration with a note explaining which traits it retained, and then reads both parents' failure summaries before proposing one targeted mutation. These summaries contain the termination reason, step count, and unlocked achievements from each run.

The system records every genome, run, score, parent link, crossover note, mutation note, and tried configuration hash so that the search remains resumable and its lineage remains visible.

![Figure 1: System architecture for selection, crossover, mutation, Crafter evaluation, and experiment evidence.](assets/architecture.png)

### 2.3 Initialization

Generation 0 began with ten hand-written rules covering resource collection, tool crafting, health, water, sleep, combat, and furnace construction, and each initial genome received a random subset of one to five of those rules. This produced ten different harnesses without asking the meta-agent to design the first population.

Because the starting pool was small and simple, later genomes could only combine its rules or add strategies that the mutation model inferred from failures, which limited the space explored by the run.

### 2.4 Fitness function

Crafter tracks 22 achievements, and we computed its standard score from their success rates:

```text
score = exp(mean(log(1 + 100 × achievement_rate))) - 1
```

Every candidate in a generation played the same world, which reduced within-generation noise and made selection consistent; however, because the run used only one world and one episode per generation, each achievement rate was either zero or one, producing a valid training signal but a weak estimate of general performance.

### 2.5 Selection and elitism

We ranked all ten candidates by fitness, retained the top five as survivors, and moved the bottom five to the archive. Each survivor was then copied into the next generation, an elitism step that protected strong genomes from poor children and explains why some lineage nodes have one parent and no crossover.

### 2.6 Crossover

Parent pairs were sampled with fitness-based weights, although the system could select an archived genome with a small probability to restore diversity. The meta-agent then merged complementary rules, memory settings, and context budgets into an output that had to match the genome schema, retrying invalid output up to three times.

### 2.7 Mutation

After crossover, the meta-agent read failure summaries from both parents and either added, removed, or changed one rule, although it could also adjust the memory policy or context budget.

A global ledger stored the hash of every tried configuration, duplicate mutations were retried three times, and a deterministic context-budget change served as the final novelty fallback. Five elite copies and five children then formed the next population, keeping the active population fixed at ten.

## 3. Experiments

### 3.1 Crafter environment

Crafter is a 2D survival game set in a procedurally generated world, where the agent collects resources, manages health and energy, fights enemies, builds stations, and crafts better tools [1]. Its action space contains 17 discrete actions, while evaluation covers 22 achievements ranging from collecting wood to crafting advanced tools and finding diamond.

We optimized the geometric-mean Crafter score because it rewards progress across the full achievement set while giving additional value to achievements that few episodes unlock.

### 3.2 Experiment setup

**Table 1: Experiment setup.**

| Setting | Value |
|---|---|
| Experiment | `demo-evolution-3` |
| Generations | 8, numbered 0–7 |
| Candidates per generation | 10 |
| Survivors per generation | 5 |
| Children per generation | 5 |
| Training worlds per generation | 1 |
| Training world | Seed 105 |
| Timesteps per episode | 200 steps |
| Gameplay model | Claude Haiku 4.5 on Amazon Bedrock |
| Meta-agent | Claude Opus 4.6 on Amazon Bedrock |
| Persistence | MongoDB Atlas |

At each step, the gameplay model received a text observation of the world and selected one allowlisted action, while the runner captured rendered frames for the replay GIF. The meta-agent never played Crafter and handled only crossover and mutation, keeping task execution and evolutionary search in separate model calls.

Atlas used four collections: `harnesses` stored genomes and lineage, `runs` stored episode evidence, `metrics` stored generation scores, and `ledger` stored configuration hashes.

### 3.3 Measurement protocol

Every candidate was evaluated with the same 200-step episode budget. Because we did not run a paired held-out test, we report only training-world fitness and observed behavior rather than a generalization result.

## 4. Results

### 4.1 Fitness over generations

![Figure 2: Best and mean fitness across eight generations, with every episode evaluated for 200 steps.](assets/score-by-generation.png)

Generation 0 began with a best score of 1.314 and a population mean of 0.901. The best score rose to 1.854 by Generation 2, remained there through Generation 4, and then reached 5.606 in Generation 5, where it stayed through Generation 7.

The population mean fluctuated during the earlier generations before rising from 1.299 in Generation 4 to 1.606 in Generation 5 and 2.740 by Generation 7, showing that later generations improved the overall population even after the best score stopped increasing.

### 4.2 Lineage

![Figure 3: Interactive lineage of 80 candidates. Node color shows score. The red outline marks the final champion's ancestry.](assets/lineage-overview.png)

The lineage contains 80 evaluated harnesses, with pale edges showing every parent link and the red path tracing the ancestors of the final champion. Single-parent edges represent elite copies, while two-parent edges represent crossover children, preventing an inherited survivor from being mistaken for a new crossover.

Candidate `g2_00` provides one clear crossover example: its parents scored 1.314 and 0.876, with one carrying concise survival rules and the other a detailed technology plan. The crossover retained both traits, after which mutation added immediate stone mining, inventory checks, goal reassessment, and a larger history budget, producing a child that scored 1.854.

Because this child changed several fields at once, the result does not establish that any single rule caused the gain; that claim would require a rule-level ablation.

### 4.3 Observed behavior

Recorded replay evidence is available for Generations 3–5, and the following examples show three measured candidates on world seed 105.

![Figure 4: Generation 3 candidate g3_00. Score 0.521 with a 200-step budget.](assets/generation-3-score-0.521.gif)

Candidate `g3_00` collected wood and placed a table but did not reach stone tools, spending much of the episode near the beginning of the technology tree.

![Figure 5: Generation 4 candidate g4_05. Score 1.854 with a 200-step budget.](assets/generation-4-score-1.854.gif)

Candidate `g4_05` collected wood, stone, and coal, placed a table, and made a wooden pickaxe, producing an action sequence with a clearer progression from resources to tools.

![Figure 6: Generation 5 candidate g5_02. Score 5.606 with a 200-step budget.](assets/generation-5-score-5.606.gif)

Candidate `g5_02` unlocked nine achievements by collecting wood, stone, coal, water, and a sapling, making wooden and stone pickaxes, and placing both a table and a furnace.

### 4.4 Rule evolution

The rules became more specific as the meta-agent encountered failures and carried their lessons into later generations.

**Table 2: Example policy changes across generations.**

| Stage | Example policy change | Score |
|---|---|---|
| Generation 0 | “Collect wood before anything else.” | Best 1.314 |
| Generation 2 | Mine stone after the wooden pickaxe; check inventory; reassess the next goal. | 1.854 |
| Generation 4 | Add step limits, milestone notes, and direct cave or downward search. | 1.854 |
| Generation 5 | Add anti-stall thresholds, survival thresholds, and a 25-step history window. | 5.606 |

The champion first appeared as `g5_02`, after which elitism copied it to `g6_05` and then `g7_05`, with both copies retaining the same score and genome.

The final policy remained imperfect because one rule claimed that stone was always two to four blocks below the surface, a statement that may fit seed 105 but is unsafe for unseen worlds. This suggests possible overfitting, since evolution can preserve a useful shortcut before there is evidence that it generalizes.

### 4.5 Comparison with published Crafter results

CrafterDojo reports 31.8 for its strongest prior baseline and 61.4 for its large behavioral-cloning model [8], while the official Crafter scoreboard lists humans at 50.5 [9]. Our score of 5.606 is substantially lower, but it is also not directly comparable because the published results use different observations, training budgets, policies, and many evaluation episodes.

Our run used one training world and 200 steps per episode because its purpose was to test harness evolution, lineage, and replay rather than establish a Crafter benchmark result.

## 5. Conclusion

Harness Darwinism turned an agent harness into a searchable genome by using selection to retain stronger policies, crossover to combine parents, and mutation to act on failure evidence, while preserving the complete history of the search.

Across eight generations with a fixed 200-step budget, best fitness rose from 1.314 to 5.606, while mean fitness rose from 0.901 to 2.740. Although the experiment did not reach published Crafter scores, it showed that a fixed model can produce better training behavior when the harness itself becomes the object of evolution.

The experiment had two main limits: it used one training world and did not run a paired evaluation on unseen worlds. API usage created another constraint because one 200-step episode could send more than two million input tokens as each action repeated recent history, meaning that larger context genes increased both capability and cost.

The starter rules also restricted the search because they mainly covered survival and the crafting sequence, whereas a stronger initial population could include navigation, spatial memory, recovery, and tool-use policies. A follow-up study should therefore use several shared training seeds, freeze the champion, compare it with the initial harness on paired held-out seeds, and remove one evolved rule at a time.

The experiment also exposed a basic evaluation requirement: generating another prompt is easy, but retaining enough evidence to determine whether that prompt is better is harder.

## References

1. Danijar Hafner. [Benchmarking the Spectrum of Agent Capabilities](https://arxiv.org/abs/2109.06780), 2021.
2. Joon Sung Park et al. [Generative Agents: Interactive Simulacra of Human Behavior](https://doi.org/10.1145/3586183.3606763), 2023.
3. Noah Shinn et al. [Reflexion: Language Agents with Verbal Reinforcement Learning](https://papers.neurips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html), 2023.
4. Andrew Zhao et al. [ExpeL: LLM Agents Are Experiential Learners](https://ojs.aaai.org/index.php/AAAI/article/view/29936), 2024.
5. Guanzhi Wang et al. [Voyager: An Open-Ended Embodied Agent with Large Language Models](https://arxiv.org/abs/2305.16291), 2023.
6. A. E. Eiben and J. E. Smith. [Introduction to Evolutionary Computing](https://doi.org/10.1007/978-3-662-44874-8), 2015.
7. Jenny Zhang et al. [Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents](https://arxiv.org/abs/2505.22954), 2025.
8. Junyeong Park, Hyeonseo Cho, and Sungjin Ahn. [CrafterDojo: A Suite of Foundation Models for Building Open-Ended Embodied Agents in Crafter](https://arxiv.org/abs/2508.13530), 2025.
9. Danijar Hafner. [Crafter repository README and official scoreboard](https://github.com/danijar/crafter/blob/main/README.md), accessed October 2026.
