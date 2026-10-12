import { useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import type { MetricPoint } from "../api";

interface Props {
  data: MetricPoint[];
}

function byGeneration(data: MetricPoint[]) {
  const groups = new Map<number, MetricPoint[]>();
  for (const point of data) {
    const gen = Number(point.generation);
    if (!Number.isFinite(gen)) continue;
    const list = groups.get(gen) ?? [];
    list.push(point);
    groups.set(gen, list);
  }
  return [...groups.entries()]
    .sort(([a], [b]) => a - b)
    .map(([generation, points]) => ({
      generation,
      best_score: Math.max(...points.map((p) => p.best_score)),
      mean_score:
        points.reduce((sum, p) => sum + p.mean_score, 0) / points.length,
    }));
}

export function FitnessCurve({ data }: Props) {
  const series = useMemo(() => byGeneration(data), [data]);

  if (!series.length) {
    return (
      <div className="panel empty">
        <p>No evolution data yet. Run <code>python run.py evolve</code> to start.</p>
      </div>
    );
  }

  const ticks = series.map((p) => p.generation);

  return (
    <div className="panel">
      <h2>📈 Fitness Curve</h2>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={series}>
          <CartesianGrid strokeDasharray="3 3" stroke="#333" />
          <XAxis
            dataKey="generation"
            type="number"
            ticks={ticks}
            domain={[ticks[0], ticks[ticks.length - 1]]}
            allowDecimals={false}
            tickFormatter={(value) => String(Math.round(Number(value)))}
            label={{ value: "Generation", position: "insideBottom", offset: -5 }}
            stroke="#888"
          />
          <YAxis
            label={{ value: "Score", angle: -90, position: "insideLeft" }}
            stroke="#888"
          />
          <Tooltip
            contentStyle={{ backgroundColor: "#1a1a2e", border: "1px solid #444" }}
          />
          <Legend />
          <Line
            type="monotone"
            dataKey="best_score"
            stroke="#00d4aa"
            strokeWidth={2}
            name="Best"
            dot={{ r: 4 }}
          />
          <Line
            type="monotone"
            dataKey="mean_score"
            stroke="#6366f1"
            strokeWidth={2}
            name="Mean"
            dot={{ r: 4 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
