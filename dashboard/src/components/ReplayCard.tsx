import { useEffect, useMemo, useState } from "react";
import { fetchReplay, type ReplayData } from "../api";

interface Props {
  genomeId?: string;
}

const VITALS = ["health", "food", "drink", "energy"] as const;

export function ReplayCard({ genomeId }: Props) {
  const [seed, setSeed] = useState("102");
  const [data, setData] = useState<ReplayData | null>(null);
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const frame = data?.frames[index];

  const load = async () => {
    const parsed = Number(seed);
    if (!Number.isFinite(parsed)) {
      setError("Seed must be a number.");
      return;
    }
    setLoading(true);
    setError(null);
    setPlaying(false);
    try {
      const replay = await fetchReplay({
        seed: parsed,
        genomeId,
        steps: 160,
        stride: 4,
      });
      if (replay.error) {
        setError(replay.error);
        setData(null);
        return;
      }
      setData(replay);
      setIndex(0);
      setPlaying(replay.frames.length > 1);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Replay failed");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!playing || !data?.frames.length) return;
    const timer = window.setInterval(() => {
      setIndex((i) => {
        if (i + 1 >= data.frames.length) {
          setPlaying(false);
          return i;
        }
        return i + 1;
      });
    }, 120);
    return () => window.clearInterval(timer);
  }, [playing, data]);

  const src = useMemo(() => {
    if (!frame?.image) return "";
    return frame.image.startsWith("data:")
      ? frame.image
      : `data:image/png;base64,${frame.image}`;
  }, [frame]);

  return (
    <div className="panel replay-card">
      <div className="replay-header">
        <h2>🎮 Game Replay</h2>
        <span className="replay-label">
          {data?.label ?? "scripted Crafter playback under the lineage tree"}
        </span>
      </div>

      <div className="replay-controls">
        <label>
          Seed
          <input
            value={seed}
            onChange={(e) => setSeed(e.target.value)}
            inputMode="numeric"
          />
        </label>
        <button className="refresh-btn" onClick={load} disabled={loading}>
          {loading ? "Recording…" : "Load replay"}
        </button>
        <button
          className="replay-btn"
          onClick={() => setPlaying((p) => !p)}
          disabled={!data?.frames.length}
        >
          {playing ? "Pause" : "Play"}
        </button>
        {genomeId && <span className="replay-genome">genome {genomeId}</span>}
      </div>

      {error && <p className="replay-error">{error}</p>}

      <div className="replay-body">
        <div className="replay-screen">
          {src ? (
            <img src={src} alt={`Crafter step ${frame?.step ?? 0}`} />
          ) : (
            <p>Load a replay to watch the scripted body on a real Crafter world.</p>
          )}
        </div>
        <div className="replay-hud">
          <p>
            Step {frame?.step ?? 0}
            {data ? ` / ${data.steps}` : ""}
          </p>
          <p className="replay-goal">Goal: {frame?.goal ?? "—"}</p>
          {VITALS.map((key) => {
            const value = frame?.vitals?.[key] ?? 0;
            return (
              <div key={key} className="vital-row">
                <span>{key}</span>
                <div className="vital-bar">
                  <div style={{ width: `${(value / 9) * 100}%` }} />
                </div>
                <span>{value}</span>
              </div>
            );
          })}
          {data && (
            <p className="replay-meta">
              score {data.score.toFixed(1)} · {data.unlocks.length} unlocks
              {data.unlocks.length ? ` (${data.unlocks.slice(0, 6).join(", ")})` : ""}
            </p>
          )}
        </div>
      </div>

      {data && data.frames.length > 0 && (
        <input
          className="replay-slider"
          type="range"
          min={0}
          max={data.frames.length - 1}
          value={index}
          onChange={(e) => {
            setPlaying(false);
            setIndex(Number(e.target.value));
          }}
        />
      )}
    </div>
  );
}
