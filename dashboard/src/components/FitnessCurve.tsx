import { motion } from "framer-motion";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  AreaChart,
} from "recharts";
import type { MetricPoint } from "../api";

interface Props {
  data: MetricPoint[];
}

export function FitnessCurve({ data }: Props) {
  if (!data.length) {
    return (
      <div className="panel empty">
        <p>
          Waiting for first generation…
        </p>
      </div>
    );
  }

  return (
    <motion.div
      className="panel"
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ type: "spring", bounce: 0, duration: 0.5 }}
    >
      <h2>Fitness Curve</h2>
      <ResponsiveContainer width="100%" height={280}>
        <AreaChart data={data}>
          <defs>
            <linearGradient id="bestGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#30d158" stopOpacity={0.2} />
              <stop offset="95%" stopColor="#30d158" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="meanGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#0a84ff" stopOpacity={0.15} />
              <stop offset="95%" stopColor="#0a84ff" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="rgba(255,255,255,0.04)"
          />
          <XAxis
            dataKey="generation"
            stroke="rgba(235,235,245,0.3)"
            tick={{ fontSize: 11, fill: "rgba(235,235,245,0.4)" }}
            axisLine={{ stroke: "rgba(255,255,255,0.06)" }}
          />
          <YAxis
            stroke="rgba(235,235,245,0.3)"
            tick={{ fontSize: 11, fill: "rgba(235,235,245,0.4)" }}
            axisLine={{ stroke: "rgba(255,255,255,0.06)" }}
          />
          <Tooltip
            contentStyle={{
              background: "rgba(28,28,30,0.9)",
              backdropFilter: "blur(20px)",
              border: "1px solid rgba(255,255,255,0.1)",
              borderRadius: 10,
              fontSize: 12,
              color: "#f5f5f7",
            }}
          />
          <Area
            type="monotone"
            dataKey="best_score"
            stroke="#30d158"
            strokeWidth={2}
            fill="url(#bestGrad)"
            name="Best"
            dot={{ r: 4, fill: "#30d158", strokeWidth: 0 }}
            activeDot={{ r: 6, fill: "#30d158" }}
          />
          <Area
            type="monotone"
            dataKey="mean_score"
            stroke="#0a84ff"
            strokeWidth={1.5}
            fill="url(#meanGrad)"
            name="Mean"
            dot={{ r: 3, fill: "#0a84ff", strokeWidth: 0 }}
          />
        </AreaChart>
      </ResponsiveContainer>
    </motion.div>
  );
}
