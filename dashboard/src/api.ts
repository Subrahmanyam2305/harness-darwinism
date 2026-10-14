const API_BASE = "http://localhost:8001";

export interface MetricPoint {
  generation: number;
  arm: string;
  best_score: number;
  mean_score: number;
  population_size: number;
  evaluated_count?: number;
  state?: "evaluating" | "complete";
  timestamp?: string;
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

export interface ReplayStep {
  step: number;
  decision_id?: string | null;
  skill?: string | null;
  skill_status?: string | null;
  action: string;
  reward: number;
  obs_snippet: string;
}

export interface SkillDecision {
  decision_id: string;
  step_started?: number;
  step_ended?: number;
  skill: {
    name: string;
    target?: string | null;
    goal?: string | null;
    need?: string | null;
  };
  actions: string[];
  status: string;
  reason?: string;
  state_before?: ExecutionState;
  state_patch?: Partial<ExecutionState>;
  state_after?: ExecutionState;
  affordances?: SkillAffordance[];
  guard_checks?: GuardCheck[];
  observation_before?: string;
  observation_after?: string;
  attempts?: number;
}

export interface ExecutionState {
  phase: string;
  goal?: string | null;
  subgoal?: string | null;
  known_locations: Record<string, unknown>;
  failed_skills: Record<string, unknown>;
  no_progress_steps: number;
}

export interface SkillAffordance {
  option: string;
  feasible: boolean;
  reason: string;
}

export interface GuardCheck {
  step: number;
  check: string;
  passed: boolean;
  detail: string;
}

export interface RunSummary {
  id?: string;
  genome_id: string;
  seed: number;
  score: number;
  steps: number;
  environment_steps?: number | null;
  controller?: "primitive" | "skills";
  model_calls?: number | null;
  steps_per_model_call?: number | null;
  tokens_per_environment_step?: number | null;
  latency_seconds?: number | null;
  cache_read_tokens?: number | null;
  cache_write_tokens?: number | null;
  comparison_id?: string;
  controller_arm?: string;
  paired_seed?: number;
  is_stub?: boolean;
  provider?: string;
  decisions?: SkillDecision[];
  failure_summary: string;
  history_len: number;
  gif_path?: string;
  replay_url?: string;
  frame_url?: string;
  history?: ReplayStep[];
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
): Promise<{ genome: GenomeSummary; runs: RunSummary[] }> {
  const res = await fetch(`${API_BASE}/api/genomes/${id}`);
  return res.json();
}
