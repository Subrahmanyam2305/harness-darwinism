import { useState, useEffect, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import type { ReplayStep, RunSummary } from "../api";

interface Props {
  runs: RunSummary[];
  genomeId: string;
}

const API_BASE = "http://localhost:8001";

function pickDefaultRun(runs: RunSummary[]): RunSummary | null {
  if (!runs.length) return null;
  // Prefer a run that has a GIF, then highest score
  const withGif = runs.filter((r) => r.replay_url || r.gif_path);
  if (withGif.length) return [...withGif].sort((a, b) => b.score - a.score)[0];
  return [...runs].sort((a, b) => b.score - a.score)[0];
}

export function GameReplay({ runs, genomeId }: Props) {
  const [selectedRun, setSelectedRun] = useState<RunSummary | null>(
    pickDefaultRun(runs)
  );
  const [steps, setSteps] = useState<ReplayStep[]>([]);
  const [currentStep, setCurrentStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Re-pick default run when genome changes
  useEffect(() => {
    setSelectedRun(pickDefaultRun(runs));
    setSteps([]);
    setCurrentStep(0);
    setPlaying(false);
  }, [genomeId, runs]);

  useEffect(() => {
    if (!selectedRun) return;
    if (selectedRun.history && selectedRun.history.length > 0) {
      setSteps(selectedRun.history);
      setCurrentStep(0);
      return;
    }
    fetch(`${API_BASE}/api/genomes/${selectedRun.genome_id}`)
      .then((r) => r.json())
      .then((data) => {
        const run = data.runs?.find((candidate: RunSummary) =>
          selectedRun.id ? candidate.id === selectedRun.id : candidate.seed === selectedRun.seed
        );
        if (run?.history) {
          setSteps(run.history);
          setCurrentStep(0);
        }
      })
      .catch(() => {});
  }, [selectedRun]);

  const play = useCallback(() => setPlaying(true), []);
  const pause = useCallback(() => setPlaying(false), []);

  useEffect(() => {
    if (playing && steps.length > 0) {
      timerRef.current = setInterval(() => {
        setCurrentStep((prev) => {
          if (prev >= steps.length - 1) {
            setPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, 500);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [playing, steps.length]);

  const step = steps[currentStep];
  const currentDecision = selectedRun?.decisions?.find(
    (decision) => decision.decision_id === step?.decision_id
  );

  if (!runs.length) return null;

  const gifUrl = selectedRun?.replay_url
    ? `${API_BASE}${selectedRun.replay_url}`
    : selectedRun?.gif_path
      ? `${API_BASE}/replays/${selectedRun.gif_path}`
      : null;
  const frameUrl = selectedRun?.frame_url
    ? `${API_BASE}${selectedRun.frame_url}`
    : selectedRun?.gif_path
      ? `${API_BASE}/api/frame/${selectedRun.gif_path}`
      : null;

  return (
    <div className="panel replay-panel">
      <div className="replay-header">
        <h2>▶ Game Replay — {genomeId}</h2>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          {runs.map((run) => (
            <button
              key={run.id ?? run.seed}
              className={`replay-btn ${
                selectedRun?.id && run.id
                  ? selectedRun.id === run.id
                    ? "active"
                    : ""
                  : selectedRun?.seed === run.seed
                    ? "active"
                    : ""
              }`}
              onClick={() => {
                setSelectedRun(run);
                setPlaying(false);
              }}
              title={`Seed ${run.seed} — Score ${run.score?.toFixed(1)}`}
            >
              {run.controller_arm === "primitive"
                ? "P"
                : run.controller_arm === "skills"
                  ? "S"
                  : "🌱"}
            </button>
          ))}
        </div>
      </div>

      {selectedRun && (
        <>
        <div className="run-evidence-label">
          <strong>
            {selectedRun.is_stub === true
              ? "OFFLINE STUB"
              : selectedRun.is_stub === false
                ? "LIVE LLM"
                : "EVIDENCE TYPE UNAVAILABLE"}
          </strong>
          {selectedRun.comparison_id && <span>Paired experiment: {selectedRun.comparison_id}</span>}
          <span>Seed {selectedRun.paired_seed ?? selectedRun.seed}</span>
        </div>
        <div className="efficiency-strip">
          <div><span>Controller</span><strong>{selectedRun.controller_arm ?? selectedRun.controller ?? "primitive"}</strong></div>
          <div><span>Model calls</span><strong>{selectedRun.model_calls ?? "Unavailable"}</strong></div>
          <div><span>Environment steps</span><strong>{selectedRun.environment_steps ?? selectedRun.steps}</strong></div>
          <div><span>Steps / call</span><strong>{selectedRun.steps_per_model_call?.toFixed(2) ?? "Unavailable"}</strong></div>
          <div><span>Tokens / step</span><strong>{selectedRun.tokens_per_environment_step?.toFixed(1) ?? "Unavailable"}</strong></div>
          <div>
            <span>Cache read / write</span>
            <strong>
              {selectedRun.cache_read_tokens ?? "Unavailable"} / {selectedRun.cache_write_tokens ?? "Unavailable"}
            </strong>
          </div>
          <div><span>Latency</span><strong>{selectedRun.latency_seconds != null ? `${selectedRun.latency_seconds.toFixed(1)}s` : "Unavailable"}</strong></div>
        </div>
        </>
      )}

      {/* Video + Text side by side */}
      <div style={{ display: "grid", gridTemplateColumns: gifUrl ? "256px 1fr" : "1fr", gap: 20 }}>
        {/* Frame-synced replay */}
        {gifUrl && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ type: "spring", bounce: 0, duration: 0.4 }}
            style={{
              borderRadius: "var(--radius-md)",
              overflow: "hidden",
              border: "1px solid var(--border)",
              background: "#000",
              alignSelf: "start",
            }}
          >
            <img
              src={
                steps.length > 0 && frameUrl
                  ? `${frameUrl}/${step?.step ?? currentStep}`
                  : gifUrl
              }
              alt={`Crafter replay: ${genomeId} seed ${selectedRun?.seed}`}
              style={{ width: 256, height: 256, display: "block", imageRendering: "pixelated" }}
            />
            <div
              style={{
                padding: "8px 12px",
                background: "rgba(0,0,0,0.6)",
                fontSize: 11,
                color: "var(--text-tertiary)",
                display: "flex",
                justifyContent: "space-between",
              }}
            >
              <span>Seed {selectedRun?.seed} — Step {currentStep + 1}</span>
              <span style={{ color: "var(--green)", fontWeight: 600 }}>
                Score {selectedRun?.score?.toFixed(1)}
              </span>
            </div>
          </motion.div>
        )}

        {/* Step-by-step text replay */}
        <div>
          {/* Timeline */}
          <div className="replay-timeline">
            {steps.map((s, i) => (
              <div
                key={i}
                className={`replay-tick ${i < currentStep ? "past" : ""} ${
                  i === currentStep ? "current" : ""
                } ${s.reward > 0 ? "reward" : ""}`}
                onClick={() => {
                  setCurrentStep(i);
                  setPlaying(false);
                }}
              />
            ))}
          </div>

          {/* Controls */}
          <div className="replay-controls" style={{ marginBottom: 14 }}>
            <button className="replay-btn" onClick={() => setCurrentStep((p) => Math.max(0, p - 1))}>
              ◀
            </button>
            <button className="replay-btn active" onClick={playing ? pause : play}>
              {playing ? "⏸" : "▶"}
            </button>
            <button className="replay-btn" onClick={() => setCurrentStep((p) => Math.min(steps.length - 1, p + 1))}>
              ▶
            </button>
            <span className="replay-step-counter">
              Step {currentStep + 1} / {steps.length}
            </span>
          </div>

          {/* Current frame */}
          <AnimatePresence mode="wait">
            {step && (
              <motion.div
                key={currentStep}
                className="replay-frame"
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -4 }}
                transition={{ type: "spring", bounce: 0, duration: 0.25 }}
                style={{ gridTemplateColumns: "1fr 1fr" }}
              >
                <div className="replay-observation">{step.obs_snippet}</div>
                <div className="replay-action">
                  {step.decision_id && (
                    <div className="replay-decision">
                      <span>
                        {step.skill ?? "skill"}
                        {currentDecision && (currentDecision.skill.target ?? currentDecision.skill.goal ?? currentDecision.skill.need)
                          ? `: ${currentDecision.skill.target ?? currentDecision.skill.goal ?? currentDecision.skill.need}`
                          : ""}
                      </span>
                      <code>{step.decision_id}</code>
                      {step.skill_status && <em>{step.skill_status}</em>}
                      {currentDecision?.reason && <p>{currentDecision.reason}</p>}
                    </div>
                  )}
                  <div className="replay-action-badge">→ {step.action}</div>
                  <div className={`replay-reward ${step.reward > 0 ? "positive" : ""}`}>
                    {step.reward > 0 ? `🎉 Reward: +${step.reward}` : `Reward: ${step.reward}`}
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
