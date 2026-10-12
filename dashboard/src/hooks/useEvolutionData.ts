import { useEffect, useState, useCallback } from "react";
import {
  fetchMetrics,
  fetchTree,
  fetchAchievements,
  fetchGenomes,
  type MetricPoint,
  type TreeData,
  type GenomeSummary,
} from "../api";

export function useEvolutionData(pollInterval = 5000) {
  const [metrics, setMetrics] = useState<MetricPoint[]>([]);
  const [tree, setTree] = useState<TreeData>({ nodes: [], edges: [] });
  const [achievements, setAchievements] = useState<{
    baseline: Record<string, number>;
    champion: Record<string, number>;
  }>({ baseline: {}, champion: {} });
  const [genomes, setGenomes] = useState<GenomeSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    const signal = AbortSignal.timeout(8000);
    try {
      const [m, t, a, g] = await Promise.all([
        fetchMetrics("full", signal),
        fetchTree("full", signal),
        fetchAchievements("full", signal),
        fetchGenomes(undefined, signal),
      ]);
      setMetrics(Array.isArray(m) ? m : []);
      setTree(t?.nodes ? t : { nodes: [], edges: [] });
      setAchievements(a ?? { baseline: {}, champion: {} });
      setGenomes(Array.isArray(g) ? g : []);
      setError(null);
    } catch (e) {
      console.error("Failed to fetch data:", e);
      setError(
        "Cannot reach the dashboard API on port 8001. Start it with python run.py serve."
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, pollInterval);
    return () => clearInterval(interval);
  }, [refresh, pollInterval]);

  return { metrics, tree, achievements, genomes, loading, error, refresh };
}
