import { useState } from "react";
import { useEvolutionData } from "./hooks/useEvolutionData";
import { FitnessCurve } from "./components/FitnessCurve";
import { LineageTree } from "./components/LineageTree";
import { AchievementGrid } from "./components/AchievementGrid";
import { GenomeDetail } from "./components/GenomeDetail";
import type { GenomeSummary } from "./api";
import "./App.css";

function App() {
  const { metrics, tree, achievements, genomes, loading, refresh } =
    useEvolutionData(5000);
  const [selected, setSelected] = useState<GenomeSummary | null>(null);

  const handleSelectGenome = (id: string) => {
    const g = genomes.find((g) => g.id === id) ?? null;
    setSelected(g);
  };

  const champion = genomes.find((g) => g.status === "champion");
  const currentGen = metrics.length > 0 ? metrics[metrics.length - 1].generation : 0;
  const bestScore = metrics.length > 0 ? metrics[metrics.length - 1].best_score : 0;

  return (
    <div className="app">
      <header>
        <div className="logo">
          <span className="logo-icon">🧬</span>
          <h1>Harness Darwinism</h1>
        </div>
        <div className="header-stats">
          <div className="stat">
            <span className="stat-label">Generation</span>
            <span className="stat-value">{currentGen}</span>
          </div>
          <div className="stat">
            <span className="stat-label">Best Score</span>
            <span className="stat-value">{bestScore.toFixed(1)}</span>
          </div>
          <div className="stat">
            <span className="stat-label">Population</span>
            <span className="stat-value">{genomes.filter(g => g.status === "alive").length}</span>
          </div>
          <button className="refresh-btn" onClick={refresh}>
            ↻ Refresh
          </button>
        </div>
      </header>

      {loading ? (
        <div className="loading">
          <div className="spinner" />
          <p>Connecting to evolution server...</p>
        </div>
      ) : (
        <main>
          <div className="grid-2">
            <FitnessCurve data={metrics} />
            <LineageTree data={tree} onSelectGenome={handleSelectGenome} />
          </div>

          <AchievementGrid
            baseline={achievements.baseline}
            champion={achievements.champion}
          />

          {selected && (
            <GenomeDetail
              genome={selected}
              onClose={() => setSelected(null)}
            />
          )}

          {champion && !selected && (
            <div className="panel champion-banner">
              <h2>🏆 Champion: {champion.id}</h2>
              <p>Score: {champion.fitness.score.toFixed(1)} | Rules: {champion.genome.rules.length}</p>
              <ol className="rules-list">
                {champion.genome.rules.map((r, i) => (
                  <li key={i}>{r}</li>
                ))}
              </ol>
              {champion.mutation_note && (
                <p className="mutation-note">
                  <strong>Last mutation:</strong> {champion.mutation_note}
                </p>
              )}
            </div>
          )}
        </main>
      )}
    </div>
  );
}

export default App;
