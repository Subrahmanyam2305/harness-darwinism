import { useCallback, useMemo } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  type Node,
  type Edge,
  Position,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import type { TreeData } from "../api";

interface Props {
  data: TreeData;
  onSelectGenome?: (id: string) => void;
}

const STATUS_COLORS: Record<string, string> = {
  alive: "#6366f1",
  archived: "#666",
  champion: "#00d4aa",
};

export function LineageTree({ data, onSelectGenome }: Props) {
  const { nodes, edges } = useMemo(() => {
    const genGroups: Record<number, typeof data.nodes> = {};
    for (const n of data.nodes) {
      (genGroups[n.generation] ??= []).push(n);
    }

    const rfNodes: Node[] = data.nodes.map((n) => {
      const genNodes = genGroups[n.generation] || [];
      const idx = genNodes.indexOf(n);
      const total = genNodes.length;

      return {
        id: n.id,
        position: {
          x: (idx - total / 2) * 200 + 600,
          y: n.generation * 200 + 50,
        },
        data: {
          label: (
            <div style={{ textAlign: "center", fontSize: 11 }}>
              <div style={{ fontWeight: 700 }}>{n.id}</div>
              <div>score: {n.score.toFixed(1)}</div>
              <div>{n.rules_count} rules</div>
              {n.status === "champion" && <div>🏆</div>}
            </div>
          ),
        },
        sourcePosition: Position.Bottom,
        targetPosition: Position.Top,
        style: {
          background: STATUS_COLORS[n.status] || "#444",
          color: "#fff",
          border: n.status === "champion" ? "2px solid #00d4aa" : "1px solid #555",
          borderRadius: 8,
          padding: "6px 10px",
          width: 120,
        },
      };
    });

    const rfEdges: Edge[] = data.edges.map((e, i) => ({
      id: `e-${i}`,
      source: e.source,
      target: e.target,
      style: { stroke: "#555" },
      animated: false,
    }));

    return { nodes: rfNodes, edges: rfEdges };
  }, [data]);

  const onNodeClick = useCallback(
    (_: any, node: Node) => onSelectGenome?.(node.id),
    [onSelectGenome]
  );

  if (!data.nodes.length) {
    return (
      <div className="panel empty">
        <p>No lineage data yet.</p>
      </div>
    );
  }

  return (
    <div className="panel" style={{ height: 700 }}>
      <h2>🌳 Lineage Tree</h2>
      <div style={{ height: "calc(100% - 40px)" }}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodeClick={onNodeClick}
          fitView
          proOptions={{ hideAttribution: true }}
        >
          <Background color="#333" gap={20} />
          <Controls />
        </ReactFlow>
      </div>
    </div>
  );
}
