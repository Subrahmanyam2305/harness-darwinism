import { useCallback, useEffect, useMemo, useState } from "react";
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
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    if (!expanded) return;
    const previousOverflow = document.body.style.overflow;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setExpanded(false);
    };
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", closeOnEscape);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", closeOnEscape);
    };
  }, [expanded]);

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
          x: (idx - total / 2) * 160 + 400,
          y: n.generation * 180 + 50,
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
    <div className={`panel lineage-panel ${expanded ? "expanded" : ""}`}>
      <div className="lineage-heading">
        <div>
          <span>Evolution map</span>
          <h2>🌳 Lineage Tree</h2>
        </div>
        <div className="lineage-heading-actions">
          <p>{data.nodes.length} candidates · click a candidate to inspect its run</p>
          <button
            className="lineage-expand-btn"
            type="button"
            aria-label={expanded ? "Collapse lineage tree" : "Expand lineage tree"}
            aria-pressed={expanded}
            onClick={() => setExpanded((value) => !value)}
          >
            <span aria-hidden="true">{expanded ? "↙" : "↗"}</span>
            {expanded ? "Collapse" : "Expand"}
          </button>
        </div>
      </div>
      <div className="lineage-canvas">
        <ReactFlow
          key={expanded ? "expanded" : "inline"}
          nodes={nodes}
          edges={edges}
          onNodeClick={onNodeClick}
          fitView
          fitViewOptions={{ padding: 0.08 }}
          minZoom={0.08}
          maxZoom={2}
        >
          <Background color="#333" gap={20} />
          <Controls />
        </ReactFlow>
      </div>
    </div>
  );
}
