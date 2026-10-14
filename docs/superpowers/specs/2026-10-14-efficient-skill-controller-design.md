# Efficient Crafter Skill Controller Design

## Intent

Reduce Bedrock gameplay calls, prompt tokens, and evolution wall time while preserving real Crafter execution, fixed-model comparisons, Atlas evidence, and replayable frames.

## Design

The existing primitive controller remains available. A run-level `skills` controller asks the frozen gameplay model for a compact execution-state patch and one parameterized skill request. A fixed local skill catalog filters requests against the current Crafter snapshot, executes bounded primitive actions, observes the real result after every action, and stops on success, failure, danger, death, or the episode limit.

The six skills are `explore`, `acquire`, `evade`, `engage`, `build_or_craft`, and `recover`. Their Python implementations remain fixed across genomes. This is inspired by SayCan affordance filtering, ReAct feedback-driven replanning, SKILL.state compact execution state, and Voyager temporally extended skills without claiming exact reproduction of those methods.

Candidate episodes may run concurrently with bounded workers. Each worker owns its environment, controller state, and recording directory. The coordinator applies fitness and Atlas writes in population order. The dashboard groups local steps under model decisions and reports calls, steps per call, tokens, latency, skill outcomes, and replay evidence.

## Constraints

- Keep the gameplay model, temperature, observation mode, scoring, and episode limits fixed across controller comparisons.
- Reset execution state between seeds and keep held-out traces outside evolution.
- Preserve every real action and frame; recording must not change gameplay RNG.
- Keep primitive-controller records readable.
- Do not launch paid evolution automatically.
- Prompt caching is optional and must not pad prompts to reach provider minimums.

