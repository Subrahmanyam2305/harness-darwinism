"use strict";

const bar = document.getElementById('progress');
    addEventListener('scroll', () => {
      const range = document.documentElement.scrollHeight - innerHeight;
      bar.style.width = `${range > 0 ? scrollY / range * 100 : 0}%`;
    }, { passive:true });

(async () => {
  const graph = document.getElementById("lineage-graph");
  const detail = document.getElementById("lineage-detail");
  if (!graph || !detail) return;

  let records;
  try {
    const response = await fetch("./lineage-data.json");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    records = await response.json();
  } catch (error) {
    detail.replaceChildren();
    detail.appendChild(htmlElement("p", "detail-empty", "Lineage data could not be loaded."));
    console.error("Failed to load lineage data", error);
    return;
  }
  const byId = new Map(records.map((record) => [record.id, record]));
  const positions = new Map();
  const svgNamespace = "http://www.w3.org/2000/svg";
  const selectedEdges = new Set();

  function svgElement(tag, attributes = {}) {
    const element = document.createElementNS(svgNamespace, tag);
    for (const [name, value] of Object.entries(attributes)) {
      element.setAttribute(name, String(value));
    }
    return element;
  }

  function htmlElement(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
  }

  function shortId(identifier) {
    return identifier.split("_").at(-1);
  }

  function scoreColor(score) {
    if (score >= 5) return "#8b1e1e";
    if (score >= 3) return "#d4952f";
    if (score >= 1.8) return "#1f5f68";
    if (score > 0) return "#9b9186";
    return "#d7d0c8";
  }

  function addText(parent, x, y, text, attributes = {}) {
    const label = svgElement("text", {
      x,
      y,
      "text-anchor": "middle",
      "font-family": "Source Code Pro, monospace",
      ...attributes,
    });
    label.textContent = text;
    parent.appendChild(label);
  }

  function appendSection(title, text) {
    detail.appendChild(htmlElement("h4", "", title));
    detail.appendChild(htmlElement("p", "", text || "No note was recorded."));
  }

  function appendJson(title, value) {
    detail.appendChild(htmlElement("h4", "", title));
    const pre = htmlElement("pre");
    const code = htmlElement("code", "", JSON.stringify(value, null, 2));
    pre.appendChild(code);
    detail.appendChild(pre);
  }

  function appendParents(child, clickedParentId) {
    detail.appendChild(htmlElement("h4", "", "Parents"));
    for (const parentId of child.parents) {
      const parent = byId.get(parentId);
      const card = htmlElement("div", "parent-card");
      const marker = parentId === clickedParentId ? "Selected connection · " : "";
      card.appendChild(htmlElement("strong", "", `${marker}${parentId}`));
      card.appendChild(
        htmlElement(
          "span",
          "",
          parent ? `Generation ${parent.generation} · score ${parent.score.toFixed(3)}` : "Record unavailable",
        ),
      );
      if (parent) {
        const parentConfig = htmlElement("details");
        parentConfig.appendChild(htmlElement("summary", "", "Parent genome"));
        const pre = htmlElement("pre");
        pre.appendChild(htmlElement("code", "", JSON.stringify(parent.genome, null, 2)));
        parentConfig.appendChild(pre);
        card.appendChild(parentConfig);
      }
      detail.appendChild(card);
    }
  }

  function showConnection(parentId, childId) {
    const child = byId.get(childId);
    if (!child) return;

    for (const selectedEdge of selectedEdges) {
      const oldLine = selectedEdge.querySelector(".edge-visible");
      oldLine.setAttribute("stroke", "#d8d0c7");
      oldLine.setAttribute("stroke-width", "1.5");
    }
    selectedEdges.clear();
    for (const candidate of edgeLayer.children) {
      if (candidate.dataset.child !== childId) continue;
      selectedEdges.add(candidate);
      const activeLine = candidate.querySelector(".edge-visible");
      activeLine.setAttribute("stroke", "#8b1e1e");
      activeLine.setAttribute("stroke-width", "4");
    }

    detail.replaceChildren();
    detail.appendChild(htmlElement("h3", "", child.id));
    const operation = child.parents.length > 1 ? "Crossover + mutation" : "Elite carry-forward";
    detail.appendChild(htmlElement("span", "detail-badge", operation));
    detail.appendChild(
      htmlElement(
        "p",
        "",
        `${child.parents.join(" + ")} → ${child.id} · resulting score ${child.score.toFixed(3)}`,
      ),
    );
    appendSection("Selection", child.selection_note);
    appendParents(child, parentId);

    if (child.parents.length > 1) {
      appendSection("Crossover decision", child.crossover_note);
      appendJson("Exact crossover output", child.crossover_config);
    } else {
      appendSection(
        "Crossover decision",
        "No crossover occurred. The selected genome was copied into the next generation by elitism.",
      );
    }

    appendSection("Mutation", child.mutation_note);
    appendJson("Final child genome", child.genome);
  }

  function showInitial(record) {
    detail.replaceChildren();
    detail.appendChild(htmlElement("h3", "", record.id));
    detail.appendChild(htmlElement("span", "detail-badge", "Initial genome"));
    detail.appendChild(
      htmlElement("p", "", `Generation ${record.generation} · score ${record.score.toFixed(3)}`),
    );
    appendJson("Genome", record.genome);
  }

  function curvedPath(start, end) {
    const midpoint = (start.x + end.x) / 2;
    return `M ${start.x + 21} ${start.y} C ${midpoint} ${start.y}, ${midpoint} ${end.y}, ${end.x - 21} ${end.y}`;
  }

  for (let generation = 0; generation < 8; generation += 1) {
    const generationRecords = records
      .filter((record) => record.generation === generation)
      .sort((left, right) => left.id.localeCompare(right.id));
    const x = 85 + generation * 176;
    addText(graph, x, 42, `GEN ${generation}`, {
      fill: "#8b1e1e",
      "font-size": 16,
      "font-weight": 700,
    });
    generationRecords.forEach((record, index) => {
      positions.set(record.id, { x, y: 92 + index * 70 });
    });
  }

  const edgeLayer = svgElement("g");
  const nodeLayer = svgElement("g");
  graph.append(edgeLayer, nodeLayer);

  for (const child of records) {
    const childPosition = positions.get(child.id);
    for (const parentId of child.parents) {
      const parentPosition = positions.get(parentId);
      if (!parentPosition || !childPosition) continue;
      const group = svgElement("g", {
        class: "lineage-edge",
        role: "button",
        tabindex: 0,
        "data-parent": parentId,
        "data-child": child.id,
        "aria-label": `Inspect connection from ${parentId} to ${child.id}`,
      });
      const pathDefinition = curvedPath(parentPosition, childPosition);
      const visible = svgElement("path", {
        class: "edge-visible",
        d: pathDefinition,
        fill: "none",
        stroke: "#d8d0c7",
        "stroke-width": 1.5,
      });
      const target = svgElement("path", {
        d: pathDefinition,
        fill: "none",
        stroke: "transparent",
        "stroke-width": 16,
      });
      group.append(visible, target);
      group.addEventListener("mouseenter", () => {
        if (!selectedEdges.has(group)) visible.setAttribute("stroke", "#1f5f68");
      });
      group.addEventListener("mouseleave", () => {
        if (!selectedEdges.has(group)) visible.setAttribute("stroke", "#d8d0c7");
      });
      group.addEventListener("click", () => showConnection(parentId, child.id));
      group.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          showConnection(parentId, child.id);
        }
      });
      edgeLayer.appendChild(group);
    }
  }

  for (const record of records) {
    const position = positions.get(record.id);
    const group = svgElement("g", {
      class: "lineage-node",
      role: "button",
      tabindex: 0,
      "aria-label": `Inspect candidate ${record.id}`,
    });
    const circle = svgElement("circle", {
      cx: position.x,
      cy: position.y,
      r: record.status === "champion" ? 24 : 19,
      fill: scoreColor(record.score),
      stroke: record.status === "champion" ? "#8b1e1e" : "#fffdf9",
      "stroke-width": record.status === "champion" ? 5 : 2,
    });
    group.appendChild(circle);
    addText(group, position.x, position.y + 4, shortId(record.id), {
      fill: "#ffffff",
      "font-size": 10,
      "font-weight": 700,
      "pointer-events": "none",
    });
    group.addEventListener("click", () => {
      if (record.parents.length) {
        showConnection(record.parents[0], record.id);
      } else {
        showInitial(record);
      }
    });
    group.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        group.dispatchEvent(new MouseEvent("click"));
      }
    });
    nodeLayer.appendChild(group);
  }

  const parameters = new URLSearchParams(window.location.search);
  const initialChild = parameters.get("child") || "demo-evolution-3_g2_00";
  const initialParent = parameters.get("parent") || "demo-evolution-3_g1_06";
  const initialEdge = [...edgeLayer.children].find(
    (candidate) =>
      candidate.dataset.parent === initialParent &&
      candidate.dataset.child === initialChild,
  );
  if (initialEdge) showConnection(initialParent, initialChild);
  if (parameters.has("child") || parameters.has("parent")) {
    window.setTimeout(() => {
      document.getElementById("lineage-title").scrollIntoView({ block: "start" });
    }, 100);
  }
})();
