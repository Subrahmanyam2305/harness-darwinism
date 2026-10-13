# Harness Darwinism article endpoint

This folder is a self-contained static site for the Harness Darwinism article.
Publish the folder at the desired website path without bundling or server-side rendering.

Required files:

- `index.html`: article markup
- `styles.css`: page and lineage layout
- `app.js`: scroll progress and interactive lineage behavior
- `lineage-data.json`: lineage records loaded by the interactive graph
- `assets/`: figures and Crafter replay GIFs used by the article

For local preview, serve the repository root over HTTP because the lineage graph fetches its JSON file:

```sh
python -m http.server 4173
```

Then open `http://localhost:4173/docs/harness-darwinism/`.
