"""Build publication figures and compressed gameplay replays for the article."""

from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
PROJECT_ROOT = ROOT.parents[2]
SCALE = 2

INK = "#201c18"
MUTED = "#6f665d"
PAPER = "#fffdf9"
SOFT = "#f3eee7"
RED = "#8b1e1e"
TEAL = "#1f5f68"
GOLD = "#d4952f"
RULE = "#d8cec2"

FONT_REGULAR = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FONT_SERIF_BOLD = "/System/Library/Fonts/Supplemental/Georgia Bold.ttf"


def font(
    size: int, *, bold: bool = False, serif: bool = False
) -> ImageFont.FreeTypeFont:
    """Return a publication font at the scaled drawing size."""
    path = FONT_SERIF_BOLD if serif else FONT_BOLD if bold else FONT_REGULAR
    return ImageFont.truetype(path, size * SCALE)


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    color: str = TEAL,
    width: int = 3,
) -> None:
    """Draw a line with a triangular arrow head."""
    start = tuple(value * SCALE for value in start)
    end = tuple(value * SCALE for value in end)
    draw.line((start, end), fill=color, width=width * SCALE)
    ex, ey = end
    sx, sy = start
    dx, dy = ex - sx, ey - sy
    length = max((dx * dx + dy * dy) ** 0.5, 1)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    size = 9 * SCALE
    points = [
        (ex, ey),
        (ex - ux * size + px * size * 0.55, ey - uy * size + py * size * 0.55),
        (ex - ux * size - px * size * 0.55, ey - uy * size - py * size * 0.55),
    ]
    draw.polygon(points, fill=color)


def poly_arrow(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[int, int]],
    *,
    color: str,
    width: int = 3,
) -> None:
    """Draw an orthogonal connector with an arrow on its last segment."""
    scaled = [tuple(value * SCALE for value in point) for point in points]
    draw.line(scaled, fill=color, width=width * SCALE, joint="curve")
    arrow(draw, points[-2], points[-1], color=color, width=width)


def box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    title: str,
    lines: list[str],
    *,
    accent: str = TEAL,
    fill: str = PAPER,
) -> None:
    """Draw one rounded architecture block."""
    scaled = tuple(value * SCALE for value in xy)
    draw.rounded_rectangle(
        scaled, radius=14 * SCALE, fill=fill, outline=accent, width=3 * SCALE
    )
    x1, y1, x2, _ = xy
    draw.rounded_rectangle(
        (x1 * SCALE, y1 * SCALE, x2 * SCALE, (y1 + 38) * SCALE),
        radius=14 * SCALE,
        fill=accent,
    )
    draw.rectangle(
        (x1 * SCALE, (y1 + 24) * SCALE, x2 * SCALE, (y1 + 38) * SCALE),
        fill=accent,
    )
    draw.text(
        ((x1 + 16) * SCALE, (y1 + 9) * SCALE),
        title,
        fill="white",
        font=font(15, bold=True),
    )
    y = y1 + 54
    for line in lines:
        draw.text(((x1 + 16) * SCALE, y * SCALE), line, fill=INK, font=font(13))
        y += 22


def build_architecture() -> None:
    """Create the architecture PNG and its editable draw.io source."""
    canvas = Image.new("RGB", (1600 * SCALE, 960 * SCALE), PAPER)
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (70 * SCALE, 48 * SCALE),
        "Harness Darwinism",
        fill=INK,
        font=font(34, serif=True),
    )
    draw.text(
        (70 * SCALE, 98 * SCALE),
        "A fixed gameplay model inside an evolving harness loop",
        fill=MUTED,
        font=font(16),
    )

    box(
        draw,
        (70, 175, 305, 315),
        "1  POPULATION",
        ["N harness genomes", "Rules + memory + context"],
        accent=RED,
    )
    box(
        draw,
        (365, 175, 600, 315),
        "2  EVALUATE",
        ["Shared Crafter world", "Score + trace + replay"],
        accent=TEAL,
    )
    box(
        draw,
        (660, 175, 895, 315),
        "3  SELECT",
        ["Keep top K", "Archive N − K"],
        accent=GOLD,
    )
    box(
        draw,
        (955, 175, 1190, 315),
        "4  CROSSOVER",
        ["Merge two parents", "Meta-agent operator"],
        accent=RED,
    )
    box(
        draw,
        (1250, 175, 1485, 315),
        "5  MUTATE",
        ["Read failure summaries", "Make one targeted edit"],
        accent=TEAL,
    )

    for start, end in [
        ((305, 245), (365, 245)),
        ((600, 245), (660, 245)),
        ((895, 245), (955, 245)),
        ((1190, 245), (1250, 245)),
    ]:
        arrow(draw, start, end)

    box(
        draw,
        (365, 405, 600, 545),
        "FIXED PLAYER",
        ["Gameplay model", "One validated action/step"],
        accent=TEAL,
        fill="#edf6f4",
    )
    box(
        draw,
        (660, 405, 895, 545),
        "CRAFTER",
        ["22 achievements", "17 discrete actions"],
        accent=GOLD,
        fill="#fbf3df",
    )
    arrow(draw, (600, 475), (660, 475))
    arrow(draw, (660, 505), (600, 505), color=GOLD)
    draw.text((604 * SCALE, 443 * SCALE), "action", fill=MUTED, font=font(11))
    draw.text((604 * SCALE, 512 * SCALE), "observation", fill=MUTED, font=font(11))
    arrow(draw, (482, 405), (482, 315), color=TEAL)

    box(
        draw,
        (70, 655, 520, 830),
        "EXPERIMENT EVIDENCE",
        [
            "Genomes and parent links",
            "Scores, achievements, and traces",
            "Best and mean fitness",
            "Previously tried configurations",
        ],
        accent=RED,
        fill="#fbf0ee",
    )
    box(
        draw,
        (630, 655, 990, 830),
        "VALIDATE + REFILL",
        [
            "Reject duplicate genomes",
            "K elites + (N − K) children",
            "Population returns to N",
        ],
        accent=GOLD,
        fill="#fbf3df",
    )
    box(
        draw,
        (1100, 655, 1485, 830),
        "NEXT GENERATION",
        ["Evaluate the new population", "Repeat for the run budget"],
        accent=TEAL,
        fill="#edf6f4",
    )

    arrow(draw, (1368, 315), (1368, 655), color=TEAL)
    arrow(draw, (520, 742), (630, 742), color=RED)
    arrow(draw, (990, 742), (1100, 742), color=GOLD)
    arrow(draw, (1292, 655), (1292, 585), color=TEAL)
    draw.line(
        (
            (1292 * SCALE, 585 * SCALE),
            (188 * SCALE, 585 * SCALE),
            (188 * SCALE, 315 * SCALE),
        ),
        fill=TEAL,
        width=3 * SCALE,
    )
    arrow(draw, (188, 585), (188, 315), color=TEAL)
    poly_arrow(draw, [(482, 315), (330, 360), (330, 620), (295, 655)], color=RED)
    poly_arrow(
        draw, [(1072, 315), (1220, 360), (1220, 610), (410, 610), (410, 655)], color=RED
    )

    draw.text(
        (70 * SCALE, 886 * SCALE),
        "The model weights stay fixed. Only the harness genome changes.",
        fill=RED,
        font=font(16, bold=True),
    )
    canvas.resize((1600, 960), Image.Resampling.LANCZOS).save(
        ASSETS / "architecture.png", optimize=True
    )
    (ASSETS / "architecture.drawio").write_text(drawio_xml())


def drawio_xml() -> str:
    """Return an editable draw.io version of the architecture figure."""
    nodes = [
        (
            "pop",
            "1  POPULATION<br>N harness genomes<br>Rules + memory + context",
            70,
            175,
            235,
            140,
            RED,
        ),
        (
            "eval",
            "2  EVALUATE<br>Shared Crafter world<br>Score + trace + replay",
            365,
            175,
            235,
            140,
            TEAL,
        ),
        (
            "select",
            "3  SELECT<br>Keep top K<br>Archive N − K",
            660,
            175,
            235,
            140,
            GOLD,
        ),
        (
            "cross",
            "4  CROSSOVER<br>Merge two parents<br>Meta-agent operator",
            955,
            175,
            235,
            140,
            RED,
        ),
        (
            "mutate",
            "5  MUTATE<br>Read failure summaries<br>Make one targeted edit",
            1250,
            175,
            235,
            140,
            TEAL,
        ),
        (
            "player",
            "FIXED PLAYER<br>Gameplay model<br>One validated action/step",
            365,
            405,
            235,
            140,
            TEAL,
        ),
        (
            "crafter",
            "CRAFTER<br>22 achievements<br>17 discrete actions",
            660,
            405,
            235,
            140,
            GOLD,
        ),
        (
            "atlas",
            "EXPERIMENT EVIDENCE<br>Genomes · runs · metrics · tried configurations",
            70,
            655,
            450,
            175,
            RED,
        ),
        (
            "validate",
            "VALIDATE + REFILL<br>Reject duplicates<br>K elites + (N − K) children<br>Population returns to N",
            630,
            655,
            360,
            175,
            GOLD,
        ),
        (
            "next",
            "NEXT GENERATION<br>Evaluate the new population<br>Repeat",
            1100,
            655,
            385,
            175,
            TEAL,
        ),
    ]
    cells = []
    for node_id, label, x, y, width, height, color in nodes:
        cells.append(
            f'<mxCell id="{node_id}" value="{escape(label)}" '
            f'style="rounded=1;whiteSpace=wrap;html=1;fillColor={color};fontColor=#ffffff;strokeColor={color};fontSize=15;" '
            f'vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{width}" height="{height}" as="geometry"/></mxCell>'
        )
    edges = [
        ("e1", "pop", "eval"),
        ("e2", "eval", "select"),
        ("e3", "select", "cross"),
        ("e4", "cross", "mutate"),
        ("e5", "player", "crafter"),
        ("e6", "crafter", "player"),
        ("e7", "eval", "atlas"),
        ("e8", "cross", "atlas"),
        ("e9", "mutate", "validate"),
        ("e10", "atlas", "validate"),
        ("e11", "validate", "next"),
        ("e12", "next", "pop"),
    ]
    for edge_id, source, target in edges:
        cells.append(
            f'<mxCell id="{edge_id}" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor={TEAL};strokeWidth=2;" '
            f'edge="1" parent="1" source="{source}" target="{target}"><mxGeometry relative="1" as="geometry"/></mxCell>'
        )
    return (
        '<mxGraphModel adaptiveColors="auto" page="1" pageScale="1" pageWidth="1600" pageHeight="960">'
        '<root><mxCell id="0"/><mxCell id="1" parent="0"/>'
        + "".join(cells)
        + "</root></mxGraphModel>\n"
    )


def score_color(score: float) -> str:
    """Map a fitness score to a lineage node color."""
    if score >= 5:
        return RED
    if score >= 3:
        return GOLD
    if score >= 1.8:
        return TEAL
    if score > 0:
        return "#9b9186"
    return "#d7d0c8"


def build_lineage() -> None:
    """Draw the full Atlas lineage and highlight the champion ancestry."""
    rows = json.loads((ROOT / "lineage-all.json").read_text())
    by_id = {row["id"]: row for row in rows}
    ancestors = {"demo-evolution-3_g7_05"}
    queue = list(ancestors)
    while queue:
        current = queue.pop()
        for parent in by_id.get(current, {}).get("parents", []):
            if parent not in ancestors:
                ancestors.add(parent)
                queue.append(parent)

    canvas = Image.new("RGB", (1600 * SCALE, 1030 * SCALE), PAPER)
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (70 * SCALE, 42 * SCALE),
        "Evolution lineage",
        fill=INK,
        font=font(34, serif=True),
    )
    draw.text(
        (70 * SCALE, 92 * SCALE),
        "80 evaluated harnesses across eight generations",
        fill=MUTED,
        font=font(16),
    )

    positions: dict[str, tuple[int, int]] = {}
    for generation in range(8):
        generation_rows = sorted(
            (row for row in rows if row["generation"] == generation),
            key=lambda row: row["id"],
        )
        x = 115 + generation * 195
        draw.text(
            (x * SCALE, 140 * SCALE),
            f"GEN {generation}",
            fill=RED,
            font=font(14, bold=True),
        )
        for index, row in enumerate(generation_rows):
            y = 205 + index * 70
            positions[row["id"]] = (x, y)

    for row in rows:
        target = positions[row["id"]]
        for parent in row["parents"]:
            if parent in positions:
                highlighted = row["id"] in ancestors and parent in ancestors
                arrow(
                    draw,
                    (positions[parent][0] + 54, positions[parent][1]),
                    (target[0] - 54, target[1]),
                    color=RED if highlighted else "#d8d0c7",
                    width=3 if highlighted else 1,
                )

    for row in rows:
        x, y = positions[row["id"]]
        highlighted = row["id"] in ancestors
        radius = 25 if highlighted else 19
        draw.ellipse(
            (
                (x - radius) * SCALE,
                (y - radius) * SCALE,
                (x + radius) * SCALE,
                (y + radius) * SCALE,
            ),
            fill=score_color(row["score"]),
            outline=RED if highlighted else PAPER,
            width=(4 if highlighted else 2) * SCALE,
        )
        short_id = row["id"].rsplit("_", 2)[-1]
        draw.text(
            (x * SCALE, y * SCALE),
            short_id,
            fill="white",
            font=font(10, bold=True),
            anchor="mm",
        )

    draw.rounded_rectangle(
        (70 * SCALE, 930 * SCALE, 1530 * SCALE, 985 * SCALE),
        radius=10 * SCALE,
        fill=SOFT,
    )
    legend = [
        ("0", "0"),
        ("#9b9186", "0–1.8"),
        (TEAL, "1.8–3"),
        (GOLD, "3–5"),
        (RED, "5+"),
    ]
    x = 105
    for color, label in legend:
        actual = "#d7d0c8" if color == "0" else color
        draw.ellipse(
            ((x - 9) * SCALE, 950 * SCALE, (x + 9) * SCALE, 968 * SCALE), fill=actual
        )
        draw.text(((x + 16) * SCALE, 948 * SCALE), label, fill=INK, font=font(12))
        x += 135
    draw.text(
        (850 * SCALE, 948 * SCALE),
        "Red outline: ancestry of the final champion",
        fill=RED,
        font=font(13, bold=True),
    )
    canvas.resize((1600, 1030), Image.Resampling.LANCZOS).save(
        ASSETS / "lineage-overview.png", optimize=True
    )


def build_score_chart() -> None:
    """Draw best and mean fitness across the eight 200-step generations."""
    metrics = json.loads((ROOT / "evidence.json").read_text())["metrics"]
    canvas = Image.new("RGB", (1200 * SCALE, 650 * SCALE), PAPER)
    draw = ImageDraw.Draw(canvas)

    left, top, right, bottom = 100, 65, 1135, 555
    for value in range(7):
        y = bottom - (value / 6) * (bottom - top)
        draw.line(
            ((left * SCALE, y * SCALE), (right * SCALE, y * SCALE)),
            fill=RULE,
            width=SCALE,
        )
        draw.text(
            ((left - 18) * SCALE, y * SCALE),
            str(value),
            fill=MUTED,
            font=font(12),
            anchor="rm",
        )

    def point(generation: int, score: float) -> tuple[int, int]:
        x = left + generation * (right - left) / (len(metrics) - 1)
        y = bottom - score / 6 * (bottom - top)
        return round(x * SCALE), round(y * SCALE)

    best_points = [point(row["generation"], row["best"]) for row in metrics]
    mean_points = [point(row["generation"], row["mean"]) for row in metrics]
    draw.line(best_points, fill=RED, width=4 * SCALE, joint="curve")
    draw.line(mean_points, fill=TEAL, width=4 * SCALE, joint="curve")

    for row, best_point, mean_point in zip(
        metrics, best_points, mean_points, strict=True
    ):
        for chart_point, color in ((best_point, RED), (mean_point, TEAL)):
            x, y = chart_point
            radius = 6 * SCALE
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=color,
                outline=PAPER,
                width=2 * SCALE,
            )
        draw.text(
            (best_point[0], (bottom + 24) * SCALE),
            str(row["generation"]),
            fill=MUTED,
            font=font(12),
            anchor="ma",
        )

    draw.text(
        (((left + right) / 2) * SCALE, 615 * SCALE),
        "Generation",
        fill=INK,
        font=font(13, bold=True),
        anchor="mm",
    )
    draw.text(
        (24 * SCALE, ((top + bottom) / 2) * SCALE),
        "Fitness",
        fill=INK,
        font=font(13, bold=True),
        anchor="mm",
    )

    legend_y = 35
    for start_x, color, label in ((825, RED, "Best"), (970, TEAL, "Mean")):
        draw.line(
            (
                (start_x * SCALE, legend_y * SCALE),
                ((start_x + 34) * SCALE, legend_y * SCALE),
            ),
            fill=color,
            width=4 * SCALE,
        )
        draw.text(
            ((start_x + 44) * SCALE, legend_y * SCALE),
            label,
            fill=INK,
            font=font(13, bold=True),
            anchor="lm",
        )

    canvas.resize((1200, 650), Image.Resampling.LANCZOS).save(
        ASSETS / "score-by-generation.png", optimize=True
    )


def compress_replay(source: Path, destination: Path) -> None:
    """Resize and subsample one replay while preserving its playback duration."""
    with Image.open(source) as replay:
        source_duration = replay.info.get("duration", 200)
        frames = []
        for index in range(0, replay.n_frames, 2):
            replay.seek(index)
            frame = replay.convert("RGB").resize((384, 384), Image.Resampling.NEAREST)
            frames.append(frame.quantize(colors=128))
    frames[0].save(
        destination,
        save_all=True,
        append_images=frames[1:],
        duration=source_duration * 2,
        loop=0,
        optimize=True,
        disposal=2,
    )


def build_replays() -> None:
    """Create three compact replays that show measured behavioral progression."""
    sources = {
        "generation-3-score-0.521.gif": "demo-evolution-3_g3_00",
        "generation-4-score-1.854.gif": "demo-evolution-3_g4_05",
        "generation-5-score-5.606.gif": "demo-evolution-3_g5_02",
    }
    for filename, genome_id in sources.items():
        source = (
            PROJECT_ROOT
            / "data"
            / "runs"
            / "evolution"
            / genome_id
            / "seed-105"
            / "replay.gif"
        )
        compress_replay(source, ASSETS / filename)


def main() -> None:
    """Build all research-article assets."""
    ASSETS.mkdir(exist_ok=True)
    build_architecture()
    build_score_chart()
    build_lineage()
    build_replays()


if __name__ == "__main__":
    main()
