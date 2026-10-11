const API_BASE = "http://localhost:8001";

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
  selection_note: string;
  crossover_note: string;
  crossover_config: GenomeSummary["genome"] | null;
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
    selection_note: string;
    crossover_note: string;
    mutation_note: string;
  }[];
  edges: { source: string; target: string }[];
}

export async function fetchMetrics(arm = "full"): Promise<MetricPoint[]> {
  const res = await fetch(`${API_BASE}/api/metrics?arm=${arm}`);
  return res.json();
}

export async function fetchGenomes(arm?: string): Promise<GenomeSummary[]> {
  const url = arm
    ? `${API_BASE}/api/genomes?arm=${arm}`
    : `${API_BASE}/api/genomes`;
  const res = await fetch(url);
  return res.json();
}

export async function fetchTree(arm = "full"): Promise<TreeData> {
  const res = await fetch(`${API_BASE}/api/tree?arm=${arm}`);
  return res.json();
}

export async function fetchAchievements(
  arm = "full"
): Promise<{ baseline: Record<string, number>; champion: Record<string, number> }> {
  const res = await fetch(`${API_BASE}/api/achievements?arm=${arm}`);
  return res.json();
}

export async function fetchGenomeDetail(
  id: string
): Promise<{ genome: GenomeSummary; runs: any[] }> {
  const res = await fetch(`${API_BASE}/api/genomes/${id}`);
  return res.json();
}
