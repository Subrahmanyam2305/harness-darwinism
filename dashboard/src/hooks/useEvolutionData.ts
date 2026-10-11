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

  const refresh = useCallback(async () => {
    try {
      const [m, t, a, g] = await Promise.all([
        fetchMetrics(),
        fetchTree(),
        fetchAchievements(),
        fetchGenomes(),
      ]);
      setMetrics(m);
      setTree(t);
      setAchievements(a);
      setGenomes(g);
    } catch (e) {
      console.error("Failed to fetch data:", e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, pollInterval);
    return () => clearInterval(interval);
  }, [refresh, pollInterval]);

  return { metrics, tree, achievements, genomes, loading, refresh };
}
