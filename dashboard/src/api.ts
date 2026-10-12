const API_BASE = "http://localhost:8001";

async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { signal });
  if (!res.ok) {
    throw new Error(`${path} failed (${res.status})`);
  }
  return res.json();
}

export interface MetricPoint {
  generation: number;
  arm: string;
  best_score: number;
  mean_score: number;
  population_size: number;
  timestamp: string;
}

export interface GenomeSummary {
  id: string;
  generation: number;
  status: string;
  parents: string[];
  fitness: {
    score: number;
    achievements: Record<string, number>;
    mean_steps: number;
    cost_usd: number;
  };
  genome: {
    system_prompt: string;
    rules: string[];
    tools: string[];
    memory_policy: { type: string; max_items: number };
    context_budget: { history_steps: number };
  };
  mutation_note: string;
  arm: string;
}

export interface TreeData {
  nodes: {
    id: string;
    generation: number;
    status: string;
    score: number;
    rules_count: number;
    mutation_note: string;
  }[];
  edges: { source: string; target: string }[];
}

export async function fetchMetrics(
  arm = "full",
  signal?: AbortSignal
): Promise<MetricPoint[]> {
  return getJson(`/api/metrics?arm=${arm}`, signal);
}

export async function fetchGenomes(
  arm?: string,
  signal?: AbortSignal
): Promise<GenomeSummary[]> {
  const qs = arm ? `?arm=${arm}` : "";
  return getJson(`/api/genomes${qs}`, signal);
}

export async function fetchTree(arm = "full", signal?: AbortSignal): Promise<TreeData> {
  return getJson(`/api/tree?arm=${arm}`, signal);
}

export async function fetchAchievements(
  arm = "full",
  signal?: AbortSignal
): Promise<{ baseline: Record<string, number>; champion: Record<string, number> }> {
  return getJson(`/api/achievements?arm=${arm}`, signal);
}

export async function fetchGenomeDetail(
  id: string,
  signal?: AbortSignal
): Promise<{ genome: GenomeSummary; runs: any[] }> {
  return getJson(`/api/genomes/${id}`, signal);
}

export interface ReplayFrame {
  step: number;
  goal: string;
  vitals: Record<string, number>;
  image: string;
}

export interface ReplayData {
  label: string;
  genome_id: string;
  seed: number;
  steps: number;
  score: number;
  unlocks: string[];
  frames: ReplayFrame[];
  error?: string;
}

export async function fetchReplay(opts: {
  seed: number;
  steps?: number;
  stride?: number;
  genomeId?: string;
}): Promise<ReplayData> {
  const params = new URLSearchParams({
    seed: String(opts.seed),
    steps: String(opts.steps ?? 400),
    stride: String(opts.stride ?? 4),
  });
  if (opts.genomeId) params.set("genome_id", opts.genomeId);
  const res = await fetch(`${API_BASE}/api/replay?${params}`, {
    signal: AbortSignal.timeout(45000),
  });
  if (!res.ok) {
    throw new Error(`Replay failed (${res.status})`);
  }
  return res.json();
}
