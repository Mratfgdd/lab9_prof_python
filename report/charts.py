"""Графіки звіту: час виконання й speedup (matplotlib, SVG)."""

import io
from typing import Any

import matplotlib
from matplotlib.figure import Figure

BLUE = "#2f6db3"
ORANGE = "#d9741f"
GREEN = "#3f8f5a"
GRAY = "#8a8a84"
INK = "#1a1a18"
COLORS = {
    "Python": GRAY,
    "Threads": BLUE,
    "Processes": ORANGE,
    "multiprocessing": "#b5651d",
    "NumPy": GREEN,
}


def use_fonts() -> None:
    """Шрифт Times New Roman і текст у SVG замість контурів."""
    matplotlib.rcParams.update(
        {"font.family": "Times New Roman", "svg.fonttype": "none"}
    )


def to_svg(figure: Any) -> str:
    """Серіалізує фігуру matplotlib у SVG без заголовка XML."""
    buffer = io.StringIO()
    figure.savefig(buffer, format="svg")
    svg = buffer.getvalue()
    start = svg.index("<svg")
    return svg[start:]


def style(axis: Any) -> None:
    """Спільне оформлення осей."""
    axis.grid(axis="both", color="#e4e2dc", linewidth=0.6)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def time_chart(rows: list[dict[str, Any]], title: str) -> str:
    """Горизонтальні стовпчики середнього часу для одного датасету."""
    use_fonts()
    figure = Figure(figsize=(6.2, 2.0))
    axis = figure.add_subplot()
    names = [row["method"] for row in rows][::-1]
    values = [row["mean"] for row in rows][::-1]
    colors = [COLORS.get(name, GRAY) for name in names]
    bars = axis.barh(names, values, color=colors, height=0.62)
    for patch, value in zip(bars, values, strict=True):
        axis.text(
            value,
            patch.get_y() + patch.get_height() / 2,
            f" {value:.3f} s".replace(".", ","),
            va="center",
            fontsize=9,
            color=INK,
        )
    axis.set_xlim(0, max(values) * 1.18)
    axis.set_xlabel(title, fontsize=9)
    axis.tick_params(labelsize=9)
    style(axis)
    figure.tight_layout()
    return to_svg(figure)


def speedup_chart(rows: list[dict[str, Any]], numpy_speedup: float) -> str:
    """Speedup проти кількості workers для потоків і процесів."""
    use_fonts()
    figure = Figure(figsize=(6.2, 2.5))
    axis = figure.add_subplot()
    for prefix, color, label in (
        ("Processes", ORANGE, "ProcessPoolExecutor"),
        ("Threads", BLUE, "ThreadPoolExecutor"),
    ):
        points = [
            (row["workers"], row["speedup"])
            for row in rows
            if row["method"].startswith(prefix)
        ]
        xs, ys = zip(*points, strict=True)
        axis.plot(
            xs,
            ys,
            color=color,
            linewidth=2,
            marker="o",
            markersize=5,
            label=label,
        )
        best = max(points, key=lambda point: point[1])
        axis.annotate(
            f"{best[1]:.2f}×".replace(".", ","),
            best,
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=9,
            color=INK,
        )
    axis.axhline(numpy_speedup, color=GREEN, linewidth=1.3, linestyle="--")
    axis.text(
        16,
        numpy_speedup,
        f"NumPy {numpy_speedup:.1f}×".replace(".", ","),
        va="bottom",
        ha="right",
        fontsize=9,
        color=INK,
    )
    axis.axhline(1, color="#9a9890", linewidth=0.8)
    axis.set_xscale("log", base=2)
    axis.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
    axis.set_xlabel("кількість workers", fontsize=9)
    axis.set_ylabel("speedup, ×", fontsize=9)
    top = max(numpy_speedup, max(row["speedup"] for row in rows))
    axis.set_ylim(0, top * 1.22)
    axis.tick_params(labelsize=9)
    axis.legend(frameon=False, loc="upper left", fontsize=9)
    style(axis)
    figure.tight_layout()
    return to_svg(figure)
