import { useCallback, useState } from "react";
import { useEvolutionData } from "./hooks/useEvolutionData";
import { FitnessCurve } from "./components/FitnessCurve";
import { LineageTree } from "./components/LineageTree";
import { AchievementGrid } from "./components/AchievementGrid";
import { GenomeDetail } from "./components/GenomeDetail";
import { ReplayCard } from "./components/ReplayCard";
import { Architecture } from "./components/Architecture";
import type { GenomeSummary } from "./api";
import "./App.css";

type Tab = "evolution" | "architecture";

function App() {
  const { metrics, tree, achievements, genomes, loading, error, refresh } =
    useEvolutionData(5000);
  const [selected, setSelected] = useState<GenomeSummary | null>(null);
  const [activeTab, setActiveTab] = useState<Tab>("evolution");

  const handleSelectGenome = useCallback(
    (id: string) => {
      const match = genomes.find((g) => g.id === id) ?? null;
      setSelected(match);
      window.requestAnimationFrame(() => {
        document.getElementById("genome-detail")?.scrollIntoView({
          behavior: "smooth",
          block: "nearest",
        });
      });
    },
    [genomes]
  );

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
          <div className="tab-switcher">
            <button
              className={`tab-btn ${activeTab === "evolution" ? "active" : ""}`}
              onClick={() => setActiveTab("evolution")}
            >
              Evolution
            </button>
            <button
              className={`tab-btn ${activeTab === "architecture" ? "active" : ""}`}
              onClick={() => setActiveTab("architecture")}
            >
              Architecture
            </button>
          </div>
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
      ) : error && genomes.length === 0 && metrics.length === 0 ? (
        <div className="loading">
          <p className="replay-error">{error}</p>
          <button className="refresh-btn" onClick={refresh}>
            ↻ Retry
          </button>
        </div>
      ) : (
        <main>
          {activeTab === "architecture" ? (
            <Architecture />
          ) : (
          <>
          <div className="grid-2">
            <FitnessCurve data={metrics} />
            <LineageTree
              data={tree}
              selectedId={selected?.id}
              onSelectGenome={handleSelectGenome}
            />
          </div>

          {selected && (
            <div id="genome-detail">
              <GenomeDetail
                genome={selected}
                onClose={() => setSelected(null)}
              />
            </div>
          )}

          <ReplayCard genomeId={selected?.id ?? champion?.id} />

          <AchievementGrid
            baseline={achievements.baseline}
            champion={achievements.champion}
          />

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
          </>
          )}
        </main>
      )}
    </div>
  );
}

export default App;
