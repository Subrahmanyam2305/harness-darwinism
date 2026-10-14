import { useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useEvolutionData } from "./hooks/useEvolutionData";
import { FitnessCurve } from "./components/FitnessCurve";
import { LineageTree } from "./components/LineageTree";
import { AchievementGrid } from "./components/AchievementGrid";
import { GenomeDetail } from "./components/GenomeDetail";
import { GameReplay } from "./components/GameReplay";
import { fetchGenomeDetail } from "./api";
import type { GenomeSummary } from "./api";
import "./App.css";

function App() {
  const arm =
    new URLSearchParams(window.location.search).get("arm") ?? "full";
  const { metrics, tree, achievements, genomes, loading, refresh } =
    useEvolutionData(arm, 4000);
  const [selected, setSelected] = useState<GenomeSummary | null>(null);
  const [replayRuns, setReplayRuns] = useState<any[]>([]);
  const [replayGenomeId, setReplayGenomeId] = useState<string>("");
  const detailRef = useRef<HTMLDivElement>(null);

  const handleSelectGenome = async (id: string) => {
    const g = genomes.find((g) => g.id === id) ?? null;
    setSelected(g);
    // Fetch runs for replay
    try {
      const detail = await fetchGenomeDetail(id);
      setReplayRuns(detail.runs || []);
      setReplayGenomeId(id);
      // Auto-scroll to detail after a brief render delay
      setTimeout(() => {
        detailRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 150);
    } catch {
      setReplayRuns([]);
    }
  };

  const champion = genomes.find((g) => g.status === "champion");
  const aliveGenomes = genomes.filter((g) => g.status === "alive");
  const currentGen =
    metrics.length > 0 ? metrics[metrics.length - 1].generation : 0;
  const bestScore =
    metrics.length > 0 ? metrics[metrics.length - 1].best_score : 0;
  const currentMetric = metrics.length > 0 ? metrics[metrics.length - 1] : null;
  const evaluatedCount = currentMetric?.evaluated_count ?? 0;
  const populationSize = currentMetric?.population_size ?? aliveGenomes.length;

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
            <motion.span
              className="stat-value"
              key={currentGen}
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ type: "spring", bounce: 0, duration: 0.35 }}
            >
              {currentGen}
            </motion.span>
          </div>
          <div className="stat">
            <span className="stat-label">Best Score</span>
            <motion.span
              className="stat-value"
              key={bestScore.toFixed(1)}
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ type: "spring", bounce: 0, duration: 0.35 }}
            >
              {bestScore.toFixed(1)}
            </motion.span>
          </div>
          <div className="stat">
            <span className="stat-label">Evaluated</span>
            <span className="stat-value">
              {evaluatedCount}/{populationSize}
            </span>
          </div>
          <button className="refresh-btn" onClick={refresh}>
            ↻ Refresh
          </button>
        </div>
      </header>

      {loading ? (
        <div className="loading">
          <div className="spinner" />
          <p>Connecting to evolution server…</p>
        </div>
      ) : (
        <main>
          <div className="grid-2">
            <FitnessCurve data={metrics} />
            <LineageTree data={tree} onSelectGenome={handleSelectGenome} />
          </div>

          {/* Game Replay — show when a genome is selected */}
          <div ref={detailRef} className="detail-anchor">
          <AnimatePresence>
            {replayRuns.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ type: "spring", bounce: 0, duration: 0.4 }}
              >
                <GameReplay
                  runs={replayRuns}
                  genomeId={replayGenomeId}
                />
              </motion.div>
            )}
          </AnimatePresence>

          <AchievementGrid
            baseline={achievements.baseline}
            champion={achievements.champion}
          />

          <AnimatePresence>
            {selected && (
              <motion.div
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ type: "spring", bounce: 0, duration: 0.4 }}
              >
                <GenomeDetail
                  genome={selected}
                  genomes={genomes}
                  onClose={() => {
                    setSelected(null);
                    setReplayRuns([]);
                  }}
                />
              </motion.div>
            )}
          </AnimatePresence>
          </div>

          {champion && !selected && (
            <motion.div
              className="panel champion-banner"
              initial={{ opacity: 0, scale: 0.97 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ type: "spring", bounce: 0.1, duration: 0.5 }}
            >
              <h2>🏆 Champion: {champion.id}</h2>
              <p
                style={{
                  fontSize: 13,
                  color: "var(--text-secondary)",
                  marginBottom: 12,
                }}
              >
                Score: {champion.fitness.score.toFixed(1)} · Rules:{" "}
                {champion.genome.rules.length}
              </p>
              <ol className="rules-list">
                {champion.genome.rules.map((r, i) => (
                  <li key={i}>{r}</li>
                ))}
              </ol>
              {champion.mutation_note && (
                <div className="mutation-note">
                  <strong>Last mutation:</strong> {champion.mutation_note}
                </div>
              )}
            </motion.div>
          )}
        </main>
      )}
    </div>
  );
}

export default App;
