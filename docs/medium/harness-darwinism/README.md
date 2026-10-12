# Medium article package

This folder contains the publication draft and its supporting evidence.

## Files

- `article.md`: Medium-ready research article.
- `index.html`: Local research-paper preview in the visual style of the author's RLVR article.
- `champion-rules.md`: Exact champion rules exported from Atlas.
- `evidence.json`: Metrics and experiment constraints used by the article.
- `lineage.json`: Detailed Atlas evidence for the two lineage case studies.
- `lineage-all.json`: Parent links, scores, crossover output, mutations, and final genomes for all 80 evaluated harnesses.
- `lineage.js`: Interactive HTML lineage graph and per-connection evidence panel.
- `build_assets.py`: Rebuilds diagrams and compressed generation replays.
- `assets/architecture.drawio`: Editable source for the system architecture.
- `assets/architecture.png`: Medium-ready system architecture.
- `assets/lineage-overview.png`: Static eight-generation lineage figure for Medium and browsers without JavaScript.
- `assets/generation-*.gif`: Recorded behavior at Generations 3, 4, and 5.
- `assets/champion-replay.gif`: Compressed champion replay for Medium.
- `assets/champion-frame.png`: Still image from the champion replay.
- `assets/score-by-generation.png`: Score chart with the step-budget change marked.

## Medium upload order

1. Copy the text from `article.md`.
2. Upload `architecture.png` in the methodology section.
3. Upload `score-by-generation.png` and `lineage-overview.png` in the results section.
4. Upload the three `generation-*.gif` files in their listed order.
5. Check code blocks, tables, captions, and links in Medium's editor.

The article separates the 85-step and 200-step phases. Do not remove that distinction when editing the results.
