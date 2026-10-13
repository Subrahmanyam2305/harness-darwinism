"""Build the static Medium article preview from the Markdown source."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def inline(text: str) -> str:
    """Render the small inline Markdown subset used by the article."""
    escaped = html.escape(text, quote=False)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", escaped)
    escaped = re.sub(
        r"\[([^]]+)]\(([^)]+)\)",
        r'<a href="\2">\1</a>',
        escaped,
    )
    return escaped


def render_markdown(source: str) -> tuple[str, str, str]:
    """Return title, author, and rendered article body."""
    lines = source.splitlines()
    title = lines[0].removeprefix("# ").strip()
    author = lines[2].strip("*")
    lines = lines[3:]
    out: list[str] = []
    paragraph: list[str] = []
    list_items: list[str] = []
    list_tag: str | None = None
    table_rows: list[list[str]] = []
    in_code = False
    code_lines: list[str] = []
    code_language = ""

    def flush_paragraph() -> None:
        if paragraph:
            out.append(f"<p>{inline(' '.join(paragraph))}</p>")
            paragraph.clear()

    def flush_list() -> None:
        nonlocal list_tag
        if list_tag:
            items = "".join(f"<li>{inline(item)}</li>" for item in list_items)
            out.append(f"<{list_tag}>{items}</{list_tag}>")
            list_items.clear()
            list_tag = None

    def flush_table() -> None:
        if not table_rows:
            return
        rows = [
            row for row in table_rows if not all(re.fullmatch(r"[-:]+", c) for c in row)
        ]
        head = "".join(f"<th>{inline(cell)}</th>" for cell in rows[0])
        body = "".join(
            "<tr>" + "".join(f"<td>{inline(cell)}</td>" for cell in row) + "</tr>"
            for row in rows[1:]
        )
        out.append(
            f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'
        )
        table_rows.clear()

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()

        if in_code:
            if stripped == "```":
                out.append(
                    f'<pre data-language="{html.escape(code_language)}"><code>'
                    f"{html.escape(chr(10).join(code_lines))}</code></pre>"
                )
                in_code = False
                code_lines.clear()
                code_language = ""
            else:
                code_lines.append(line)
            continue

        if stripped.startswith("```"):
            flush_paragraph()
            flush_list()
            flush_table()
            in_code = True
            code_language = stripped[3:]
            continue

        image = re.fullmatch(r"!\[([^]]*)]\(([^)]+)\)", stripped)
        if image:
            flush_paragraph()
            flush_list()
            flush_table()
            alt, src = image.groups()
            if src == "assets/lineage-overview.png":
                out.append(
                    '<section class="lineage-explorer" aria-labelledby="lineage-title">'
                    '<div class="lineage-heading">'
                    '<h3 id="lineage-title">Interactive lineage</h3>'
                    "<p>Click any connection to highlight both crossover parents and inspect the exact crossover output, mutation, and final child genome.</p>"
                    "</div>"
                    '<div class="lineage-layout">'
                    '<div class="lineage-canvas"><svg id="lineage-graph" viewBox="0 0 1400 820" role="img" '
                    'aria-label="Interactive graph of 80 harness candidates across eight generations"></svg></div>'
                    '<aside class="lineage-detail" id="lineage-detail" aria-live="polite">'
                    '<p class="detail-empty">Select a parent-to-child connection.</p>'
                    "</aside></div></section>"
                    f'<noscript><figure><img src="{html.escape(src)}" alt="{html.escape(alt)}">'
                    f"<figcaption>{inline(alt)}</figcaption></figure></noscript>"
                )
                continue
            loading = "eager" if "champion-replay" in src else "lazy"
            out.append(
                f'<figure><img src="{html.escape(src)}" alt="{html.escape(alt)}" '
                f'loading="{loading}"><figcaption>{inline(alt)}</figcaption></figure>'
            )
            continue

        if stripped.startswith("|"):
            flush_paragraph()
            flush_list()
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            table_rows.append(cells)
            continue
        flush_table()

        ordered = re.match(r"^\d+\.\s+(.+)", stripped)
        unordered = re.match(r"^-\s+(.+)", stripped)
        if ordered or unordered:
            flush_paragraph()
            wanted = "ol" if ordered else "ul"
            if list_tag and list_tag != wanted:
                flush_list()
            list_tag = wanted
            list_items.append((ordered or unordered).group(1))
            continue
        flush_list()

        if stripped.startswith("### "):
            flush_paragraph()
            out.append(f"<h3>{inline(stripped[4:])}</h3>")
        elif stripped.startswith("## "):
            flush_paragraph()
            heading = stripped[3:]
            section_id = re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")
            out.append(f'<h2 id="{section_id}">{inline(heading)}</h2>')
        elif stripped.startswith("> "):
            flush_paragraph()
            out.append(f"<blockquote>{inline(stripped[2:])}</blockquote>")
        elif stripped:
            paragraph.append(stripped)
        else:
            flush_paragraph()

    flush_paragraph()
    flush_list()
    flush_table()
    return title, author, "\n".join(out)


def build() -> None:
    """Build index.html beside the source article."""
    title, author, body = render_markdown((ROOT / "article.md").read_text())
    lineage_json = json.dumps(
        json.loads((ROOT / "lineage-all.json").read_text())
    ).replace("</", "<\\/")
    lineage_script = (ROOT / "lineage.js").read_text()
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="A research report on evolving an LLM agent harness with genetic algorithms in Crafter.">
  <title>{html.escape(title)}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=Source+Code+Pro:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {{ --paper:#fffdf9; --ink:#211d19; --muted:#6b6258; --accent:#8b1e1e; --teal:#1f5f68; --rule:#ddd4ca; --soft:#f3eee7; }}
    * {{ box-sizing:border-box; }}
    html {{ scroll-behavior:smooth; }}
    body {{ margin:0; background:#eee9e2; color:var(--ink); font-family:"Libre Baskerville", Georgia, serif; font-size:16px; line-height:1.75; }}
    .progress {{ position:fixed; z-index:10; top:0; left:0; width:0; height:3px; background:var(--accent); }}
    .paper {{ width:min(900px, calc(100% - 32px)); margin:24px auto 64px; background:var(--paper); box-shadow:0 18px 55px rgba(67,53,39,.12); }}
    .masthead {{ padding:64px 62px 34px; text-align:center; }}
    h1 {{ max-width:760px; margin:0 auto 20px; font-size:40px; line-height:1.18; letter-spacing:-.025em; }}
    .byline {{ color:var(--muted); font-size:14px; }}
    article {{ padding:40px 62px 72px; }}
    p {{ margin:0 0 18px; }}
    article > p:first-child {{ font-size:19px; line-height:1.7; }}
    h2 {{ margin:48px 0 17px; padding-bottom:8px; border-bottom:1px solid var(--rule); color:var(--accent); font-size:22px; line-height:1.35; }}
    h3 {{ margin:30px 0 10px; font-size:17px; }}
    ul, ol {{ margin:0 0 20px 24px; padding:0; }}
    li {{ margin:7px 0; padding-left:4px; }}
    blockquote {{ margin:26px 0; padding:18px 22px; background:var(--soft); border-left:4px solid var(--accent); font-size:18px; font-style:italic; }}
    code {{ padding:2px 5px; border-radius:3px; background:#efeae3; font:500 13px/1.5 "Source Code Pro", monospace; }}
    pre {{ overflow:auto; margin:24px 0; padding:22px; border:1px solid #2c3334; border-radius:4px; background:#172123; color:#e7efeb; font-size:13px; line-height:1.65; }}
    pre code {{ padding:0; background:none; color:inherit; }}
    a {{ color:var(--teal); text-decoration-thickness:1px; text-underline-offset:3px; }}
    .table-wrap {{ overflow-x:auto; margin:22px 0 28px; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; line-height:1.55; }}
    th, td {{ padding:10px 12px; border:1px solid var(--rule); text-align:left; vertical-align:top; }}
    th {{ background:var(--soft); font:500 11px/1.4 "Source Code Pro", monospace; letter-spacing:.04em; text-transform:uppercase; }}
    figure {{ margin:34px -20px; text-align:center; }}
    figure img {{ display:block; max-width:100%; max-height:720px; margin:auto; border:1px solid var(--rule); background:#fff; }}
    figure img[src$=".gif"] {{ width:min(100%, 520px); image-rendering:pixelated; }}
    figcaption {{ max-width:760px; margin:10px auto 0; color:var(--muted); font-size:12px; line-height:1.55; text-align:left; }}
    .lineage-explorer {{ position:relative; left:50%; width:min(1320px, calc(100vw - 32px)); margin:38px 0; transform:translateX(-50%); overflow:hidden; border:1px solid var(--rule); background:#fff; }}
    .lineage-heading {{ padding:20px 24px 16px; border-bottom:1px solid var(--rule); }}
    .lineage-heading h3 {{ margin:0 0 4px; color:var(--accent); }}
    .lineage-heading p {{ margin:0; color:var(--muted); font-size:13px; }}
    .lineage-layout {{ display:grid; grid-template-columns:minmax(0, 1fr) 400px; min-height:680px; }}
    .lineage-canvas {{ overflow:auto; border-right:1px solid var(--rule); background:var(--paper); }}
    #lineage-graph {{ display:block; min-width:900px; width:100%; height:680px; }}
    .lineage-edge {{ cursor:pointer; }}
    .lineage-edge:focus {{ outline:none; }}
    .lineage-edge:focus .edge-visible {{ stroke:var(--accent); stroke-width:4; }}
    .lineage-node {{ cursor:pointer; }}
    .lineage-detail {{ max-height:680px; overflow:auto; overflow-wrap:anywhere; padding:20px; background:#faf7f2; font-family:"Source Code Pro", monospace; font-size:11px; line-height:1.55; }}
    .lineage-detail h3 {{ margin:0 0 8px; overflow-wrap:anywhere; font-family:"Libre Baskerville", Georgia, serif; font-size:17px; }}
    .lineage-detail h4 {{ margin:18px 0 7px; padding-top:12px; border-top:1px solid var(--rule); color:var(--accent); font-size:11px; letter-spacing:.05em; text-transform:uppercase; }}
    .lineage-detail p {{ margin:0 0 10px; }}
    .lineage-detail pre {{ max-height:260px; margin:8px 0 14px; padding:12px; white-space:pre-wrap; overflow:auto; font-size:10px; line-height:1.5; }}
    .detail-empty {{ color:var(--muted); font-family:"Libre Baskerville", Georgia, serif; font-style:italic; }}
    .detail-badge {{ display:inline-block; margin:0 0 12px; padding:3px 7px; border-radius:999px; background:var(--accent); color:#fff; font-size:10px; text-transform:uppercase; }}
    .parent-card {{ margin:8px 0; padding:10px; border:1px solid var(--rule); background:#fff; }}
    .parent-card strong {{ display:block; overflow-wrap:anywhere; color:var(--teal); }}
    footer {{ padding:24px 62px 36px; border-top:1px solid var(--rule); color:var(--muted); font-size:12px; }}
    @media (max-width:1180px) {{
      .lineage-layout {{ grid-template-columns:1fr; }}
      .lineage-canvas {{ border-right:0; border-bottom:1px solid var(--rule); }}
      .lineage-detail {{ max-height:620px; }}
    }}
    @media (max-width:700px) {{
      .paper {{ width:100%; margin:0; box-shadow:none; }}
      .masthead, article {{ padding-left:22px; padding-right:22px; }}
      h1 {{ font-size:29px; }}
      figure {{ margin-left:0; margin-right:0; }}
      .lineage-explorer {{ width:calc(100vw - 20px); }}
      .lineage-detail {{ max-height:none; }}
      footer {{ padding-left:22px; padding-right:22px; }}
    }}
    @media print {{ body {{ background:#fff; }} .paper {{ width:100%; margin:0; box-shadow:none; }} .progress {{ display:none; }} }}
  </style>
</head>
<body>
  <div class="progress" id="progress"></div>
  <main class="paper">
    <header class="masthead">
      <h1>{html.escape(title)}</h1>
      <div class="byline">{html.escape(author)}</div>
    </header>
    <article>{body}</article>
    <footer>Research report prepared from Atlas experiment records and recorded Crafter episodes. Training world: seed 105.</footer>
  </main>
  <script id="lineage-data" type="application/json">{lineage_json}</script>
  <script>
    const bar = document.getElementById('progress');
    addEventListener('scroll', () => {{
      const range = document.documentElement.scrollHeight - innerHeight;
      bar.style.width = `${{range > 0 ? scrollY / range * 100 : 0}}%`;
    }}, {{ passive:true }});
  </script>
  <script>{lineage_script}</script>
</body>
</html>
"""
    (ROOT / "index.html").write_text(document)


if __name__ == "__main__":
    build()
