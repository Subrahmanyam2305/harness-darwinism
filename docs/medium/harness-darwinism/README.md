# Medium article package

This folder contains the publication draft and its supporting evidence.

## Files

- `article.md`: Medium-ready article copy.
- `index.html`: Local article preview in the visual style of the author's RLVR article.
- `champion-rules.md`: Exact champion rules exported from Atlas.
- `evidence.json`: Metrics and experiment constraints used by the article.
- `assets/champion-replay.gif`: Compressed champion replay for Medium.
- `assets/champion-frame.png`: Still image from the champion replay.
- `assets/score-by-generation.png`: Score chart with the step-budget change marked.

## Medium upload order

1. Copy the text from `article.md`.
2. Upload `champion-replay.gif` after the introduction.
3. Upload `score-by-generation.png` in the results section.
4. Upload `champion-frame.png` after the champion rules section.
5. Check code blocks, tables, captions, and links in Medium's editor.

The article separates the 85-step and 200-step phases. Do not remove that distinction when editing the results.
