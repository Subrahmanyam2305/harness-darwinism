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

export function FitnessCurve({ data }: Props) {
  if (!data.length) {
    return (
      <div className="panel empty">
        <p>No evolution data yet. Run <code>python run.py evolve</code> to start.</p>
      </div>
    );
  }

  return (
    <div className="panel">
      <h2>📈 Fitness Curve</h2>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#333" />
          <XAxis
            dataKey="generation"
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
