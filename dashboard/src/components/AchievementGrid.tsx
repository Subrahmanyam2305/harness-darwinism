interface Props {
  baseline: Record<string, number>;
  champion: Record<string, number>;
}

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

export function AchievementGrid({ baseline, champion }: Props) {
  return (
    <div className="panel">
      <h2>🏅 Achievement Grid</h2>
      <div className="ach-comparison">
        <div>
          <h3>Generation 0 (Baseline)</h3>
          <div className="ach-grid">
            {ALL_ACHIEVEMENTS.map((a) => (
              <AchCell key={a} label={a} rate={baseline[a] ?? 0} />
            ))}
          </div>
        </div>
        <div>
          <h3>🏆 Champion</h3>
          <div className="ach-grid">
            {ALL_ACHIEVEMENTS.map((a) => (
              <AchCell key={a} label={a} rate={champion[a] ?? 0} />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
