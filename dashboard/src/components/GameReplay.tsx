import { useState, useEffect, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface Step {
  step: number;
  action: string;
  reward: number;
  obs_snippet: string;
}

interface Run {
  genome_id: string;
  seed: number;
  score: number;
  steps: number;
  failure_summary: string;
  history_len: number;
}

interface Props {
  runs: Run[];
  genomeId: string;
}

export function GameReplay({ runs, genomeId }: Props) {
  const [selectedRun, setSelectedRun] = useState<Run | null>(
    runs.length > 0 ? runs[0] : null
  );
  const [steps, setSteps] = useState<Step[]>([]);
  const [currentStep, setCurrentStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (!selectedRun) return;
    // Fetch full run detail with history
    fetch(
      `http://localhost:8001/api/genomes/${selectedRun.genome_id}`
    )
      .then((r) => r.json())
      .then((data) => {
        const run = data.runs?.find(
          (r: any) => r.seed === selectedRun.seed
        );
        if (run?.history) {
          setSteps(run.history);
          setCurrentStep(0);
        }
      })
      .catch(() => {});
  }, [selectedRun]);

  const play = useCallback(() => {
    setPlaying(true);
  }, []);

  const pause = useCallback(() => {
    setPlaying(false);
  }, []);

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
      }, 600);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [playing, steps.length]);

  const step = steps[currentStep];

  if (!runs.length) {
    return (
      <div className="panel empty">
        <p>No runs to replay.</p>
      </div>
    );
  }

  return (
    <div className="panel replay-panel">
      <div className="replay-header">
        <h2>▶ Game Replay — {genomeId}</h2>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          {runs.map((run) => (
            <button
              key={run.seed}
              className={`replay-btn ${selectedRun?.seed === run.seed ? "active" : ""}`}
              onClick={() => {
                setSelectedRun(run);
                setPlaying(false);
              }}
              title={`Seed ${run.seed} — Score ${run.score.toFixed(1)}`}
            >
              🌱
            </button>
          ))}
        </div>
      </div>

      {/* Timeline scrubber */}
      <div className="replay-timeline">
        {steps.map((s, i) => (
          <div
            key={i}
            className={`replay-tick ${
              i < currentStep ? "past" : ""
            } ${i === currentStep ? "current" : ""} ${
              s.reward > 0 ? "reward" : ""
            }`}
            onClick={() => {
              setCurrentStep(i);
              setPlaying(false);
            }}
          />
        ))}
      </div>

      {/* Controls */}
      <div className="replay-controls" style={{ marginBottom: 16 }}>
        <button
          className="replay-btn"
          onClick={() =>
            setCurrentStep((p) => Math.max(0, p - 1))
          }
        >
          ◀
        </button>
        <button
          className="replay-btn active"
          onClick={playing ? pause : play}
        >
          {playing ? "⏸" : "▶"}
        </button>
        <button
          className="replay-btn"
          onClick={() =>
            setCurrentStep((p) => Math.min(steps.length - 1, p + 1))
          }
        >
          ▶
        </button>
        <span className="replay-step-counter">
          Step {currentStep + 1} / {steps.length}
        </span>
        {selectedRun && (
          <span
            style={{
              fontSize: 12,
              color: "var(--text-tertiary)",
              marginLeft: "auto",
            }}
          >
            Seed {selectedRun.seed} · Score{" "}
            <strong style={{ color: "var(--green)" }}>
              {selectedRun.score.toFixed(1)}
            </strong>
          </span>
        )}
      </div>

      {/* Frame */}
      <AnimatePresence mode="wait">
        {step && (
          <motion.div
            key={currentStep}
            className="replay-frame"
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ type: "spring", bounce: 0, duration: 0.25 }}
          >
            <div className="replay-observation">{step.obs_snippet}</div>
            <div className="replay-action">
              <div className="replay-action-badge">
                → {step.action}
              </div>
              <div
                className={`replay-reward ${
                  step.reward > 0 ? "positive" : ""
                }`}
              >
                {step.reward > 0
                  ? `🎉 Reward: +${step.reward}`
                  : `Reward: ${step.reward}`}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
