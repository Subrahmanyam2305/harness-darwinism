import { useCallback, useMemo } from "react";
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  Controls,
  type Node,
  type Edge,
  type NodeProps,
  Position,
  Handle,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import type { TreeData } from "../api";

interface Props {
  data: TreeData;
  selectedId?: string | null;
  onSelectGenome?: (id: string) => void;
}

const STATUS_COLORS: Record<string, string> = {
  alive: "#6366f1",
  archived: "#666",
  champion: "#00d4aa",
};

type GenomeNodeData = {
  label: string;
  score: number;
  rulesCount: number;
  status: string;
  selected: boolean;
  onSelect: () => void;
};

function GenomeNode({ data }: NodeProps) {
  const node = data as GenomeNodeData;
  return (
    <button
      type="button"
      className={`genome-node${node.selected ? " selected" : ""}`}
      onClick={(event) => {
        event.stopPropagation();
        node.onSelect();
      }}
      style={{
        background: STATUS_COLORS[node.status] || "#444",
        borderColor:
          node.selected ? "#fff" : node.status === "champion" ? "#00d4aa" : "#555",
      }}
    >
      <Handle type="target" position={Position.Top} />
      <strong>{node.label}</strong>
      <span>score: {node.score.toFixed(1)}</span>
      <span>{node.rulesCount} rules</span>
      {node.status === "champion" && <span>🏆</span>}
      <Handle type="source" position={Position.Bottom} />
    </button>
  );
}

const nodeTypes = { genome: GenomeNode };

function LineageFlow({ data, selectedId, onSelectGenome }: Props) {
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
        type: "genome",
        position: {
          x: (idx - total / 2) * 160 + 400,
          y: n.generation * 180 + 50,
        },
        data: {
          label: n.id,
          score: n.score,
          rulesCount: n.rules_count,
          status: n.status,
          selected: n.id === selectedId,
          onSelect: () => onSelectGenome?.(n.id),
        },
        sourcePosition: Position.Bottom,
        targetPosition: Position.Top,
        draggable: false,
        connectable: false,
      };
    });

    const rfEdges: Edge[] = data.edges.map((e, i) => ({
      id: `e-${i}`,
      source: e.source,
      target: e.target,
      style: { stroke: "#555" },
    }));

    return { nodes: rfNodes, edges: rfEdges };
  }, [data, selectedId, onSelectGenome]);

  const onNodeClick = useCallback(
    (_event: unknown, node: Node) => {
      onSelectGenome?.(node.id);
    },
    [onSelectGenome]
  );

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      onNodeClick={onNodeClick}
      nodesDraggable={false}
      nodesConnectable={false}
      elementsSelectable
      selectNodesOnDrag={false}
      fitView
      minZoom={0.3}
      proOptions={{ hideAttribution: true }}
    >
      <Background color="#333" gap={20} />
      <Controls />
    </ReactFlow>
  );
}

export function LineageTree({ data, selectedId, onSelectGenome }: Props) {
  if (!data.nodes.length) {
    return (
      <div className="panel empty">
        <p>No lineage data yet.</p>
      </div>
    );
  }

  return (
    <div className="panel lineage-panel">
      <h2>🌳 Lineage Tree</h2>
      <p className="lineage-hint">Click a genome to inspect its rules and score.</p>
      <div className="lineage-canvas">
        <ReactFlowProvider>
          <LineageFlow
            data={data}
            selectedId={selectedId}
            onSelectGenome={onSelectGenome}
          />
        </ReactFlowProvider>
      </div>
    </div>
  );
}
