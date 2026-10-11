import type { GenomeSummary } from "../api";

interface Props {
  genome: GenomeSummary | null;
  onClose: () => void;
}

export function GenomeDetail({ genome, onClose }: Props) {
  if (!genome) return null;

  return (
    <div className="panel genome-detail">
      <div className="detail-header">
        <h2>
          {genome.status === "champion" ? "🏆 " : ""}
          {genome.id}
        </h2>
        <button onClick={onClose}>✕</button>
      </div>

      <div className="detail-meta">
        <span className={`status-badge ${genome.status}`}>{genome.status}</span>
        <span>Gen {genome.generation}</span>
        <span>Score: {genome.fitness.score.toFixed(1)}</span>
        <span>Steps: {genome.fitness.mean_steps.toFixed(0)}</span>
        {genome.parents.length > 0 && (
          <span>Parents: {genome.parents.join(" × ")}</span>
        )}
      </div>

      {genome.selection_note && (
        <div className="mutation-note">
          <strong>Selection:</strong> {genome.selection_note}
        </div>
      )}

      {genome.crossover_note && (
        <div className="mutation-note">
          <strong>Crossover:</strong> {genome.crossover_note}
        </div>
      )}

      {genome.crossover_config && (
        <>
          <h3>Crossover Output</h3>
          <ol className="rules-list">
            {genome.crossover_config.rules.map((rule, index) => (
              <li key={index}>{rule}</li>
            ))}
          </ol>
        </>
      )}

      {genome.mutation_note && (
        <div className="mutation-note">
          <strong>Mutation:</strong> {genome.mutation_note}
        </div>
      )}

      <h3>Evolved Rules</h3>
      <ol className="rules-list">
        {genome.genome.rules.map((r, i) => (
          <li key={i}>{r}</li>
        ))}
      </ol>

      <h3>Config</h3>
      <div className="config-grid">
        <div>
          <strong>Memory:</strong> {genome.genome.memory_policy.type} (max{" "}
          {genome.genome.memory_policy.max_items})
        </div>
        <div>
          <strong>Context:</strong> {genome.genome.context_budget.history_steps}{" "}
          steps
        </div>
      </div>
    </div>
  );
}
