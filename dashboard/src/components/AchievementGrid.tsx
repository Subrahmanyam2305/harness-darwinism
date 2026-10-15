import { useState } from "react";
import type { GenerationAchievements } from "../api";

interface Props { generations: GenerationAchievements[]; }

const ALL_ACHIEVEMENTS = [
  "collect_coal", "collect_diamond", "collect_drink", "collect_iron",
  "collect_sapling", "collect_stone", "collect_wood", "defeat_skeleton",
  "defeat_zombie", "eat_cow", "eat_plant", "make_iron_pickaxe",
  "make_iron_sword", "make_stone_pickaxe", "make_stone_sword",
  "make_wood_pickaxe", "make_wood_sword", "place_furnace", "place_plant",
  "place_stone", "place_table", "wake_up",
];

function AchCell({ label, rate }: { label: string; rate: number }) {
  const bg =
    rate >= 0.8 ? "#00d4aa" :
    rate >= 0.4 ? "#6366f1" :
    rate > 0   ? "#444"    :
                 "#1a1a2e";
  return (
    <div
      className="ach-cell"
      style={{ background: bg }}
      title={`${label}: ${(rate * 100).toFixed(0)}%`}
    >
      <span className="ach-label">{label.replace(/_/g, " ")}</span>
      <span className="ach-pct">{(rate * 100).toFixed(0)}%</span>
    </div>
  );
}

export function AchievementGrid({ generations }: Props) {
  const [selectedGeneration, setSelectedGeneration] = useState<number | null>(null);
  const selected =
    generations.find((item) => item.generation === selectedGeneration) ??
    generations.at(-1);

  if (!selected) {
    return <div className="panel empty">Waiting for a complete generation…</div>;
  }

  return (
    <div className="panel achievement-panel">
      <div className="achievement-heading">
        <div>
          <span>Population achievement rate</span>
          <h2>🏅 Generation {selected.generation} achievements</h2>
        </div>
        <p>
          {selected.evaluated_genomes} genomes · {selected.episode_count} episodes · best {selected.best_score.toFixed(1)}
        </p>
      </div>
      <div className="generation-tabs" role="tablist" aria-label="Completed generations">
        {generations.map((item) => (
          <button
            type="button"
            role="tab"
            aria-selected={item.generation === selected.generation}
            className={item.generation === selected.generation ? "active" : ""}
            key={item.generation}
            onClick={() => setSelectedGeneration(item.generation)}
          >
            Gen {item.generation}
          </button>
        ))}
      </div>
      <div className="ach-grid generation-achievements">
        {ALL_ACHIEVEMENTS.map((achievement) => (
          <AchCell
            key={achievement}
            label={achievement}
            rate={selected.achievement_rates[achievement] ?? 0}
          />
        ))}
      </div>
    </div>
  );
}
