from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def _draw_node(ax, x: float, y: float, w: float, h: float, text: str, face: str, edge: str) -> None:
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=10, color="#111111")


def _draw_arrow(ax, x1: float, y1: float, x2: float, y2: float) -> None:
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", lw=1.2, color="#3f3f3f"),
    )


def _setup_canvas(title: str):
    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=14)
    return fig, ax


def plot_traditional_rag(output_file: Path) -> None:
    title = (
        "Traditional RAG pipeline for table question answering, "
        "from retrieval through validation to final output."
    )
    fig, ax = _setup_canvas(title)

    w, h = 0.22, 0.08
    x = 0.5
    ys = [0.92, 0.75, 0.58, 0.41, 0.24, 0.08]
    labels = [
        "Question",
        "Retriever",
        "Context Assembly",
        "LLM Answer Generation",
        "Validation",
        "Final Output",
    ]
    fills = ["#eef3ff", "#f8f9fb", "#f8f9fb", "#f8f9fb", "#fff4e6", "#e9f7ef"]
    edges = ["#3b5b92", "#4a4a4a", "#4a4a4a", "#4a4a4a", "#a56a00", "#1e7f4f"]

    for y, label, face, edge in zip(ys, labels, fills, edges):
        _draw_node(ax, x, y, w, h, label, face, edge)

    for y1, y2 in zip(ys[:-1], ys[1:]):
        _draw_arrow(ax, x, y1 - h / 2, x, y2 + h / 2)

    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=220)
    plt.close(fig)


def plot_agentic_multi_tool(output_file: Path) -> None:
    title = (
        "Agentic multi-tool pipeline with branching tool execution, "
        "evidence fusion, validation, and final output."
    )
    fig, ax = _setup_canvas(title)

    # Shared style palette aligned with existing plots.
    start_face, start_edge = "#eef3ff", "#3b5b92"
    proc_face, proc_edge = "#f8f9fb", "#4a4a4a"
    tool_face, tool_edge = "#fff7eb", "#c56a00"
    val_face, val_edge = "#fff4e6", "#a56a00"
    out_face, out_edge = "#e9f7ef", "#1e7f4f"

    w, h = 0.20, 0.09

    nodes = {
        "Q": (0.50, 0.90, "Question", start_face, start_edge),
        "R": (0.50, 0.76, "Retriever", proc_face, proc_edge),
        "DL": (0.24, 0.58, "Direct Table Lookup", tool_face, tool_edge),
        "SR": (0.50, 0.58, "Structured Retriever", tool_face, tool_edge),
        "GPT": (0.76, 0.58, "GPT Tool", tool_face, tool_edge),
        "F": (0.50, 0.40, "Evidence Fusion", proc_face, proc_edge),
        "V": (0.50, 0.24, "Validation", val_face, val_edge),
        "O": (0.50, 0.10, "Final Output", out_face, out_edge),
    }

    for _, (x, y, label, face, edge) in nodes.items():
        _draw_node(ax, x, y, w, h, label, face, edge)

    _draw_arrow(ax, 0.50, 0.90 - h / 2, 0.50, 0.76 + h / 2)

    _draw_arrow(ax, 0.47, 0.76 - h / 2, 0.24, 0.58 + h / 2)
    _draw_arrow(ax, 0.50, 0.76 - h / 2, 0.50, 0.58 + h / 2)
    _draw_arrow(ax, 0.53, 0.76 - h / 2, 0.76, 0.58 + h / 2)

    _draw_arrow(ax, 0.24, 0.58 - h / 2, 0.47, 0.40 + h / 2)
    _draw_arrow(ax, 0.50, 0.58 - h / 2, 0.50, 0.40 + h / 2)
    _draw_arrow(ax, 0.76, 0.58 - h / 2, 0.53, 0.40 + h / 2)

    _draw_arrow(ax, 0.50, 0.40 - h / 2, 0.50, 0.24 + h / 2)
    _draw_arrow(ax, 0.50, 0.24 - h / 2, 0.50, 0.10 + h / 2)

    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pipeline diagram figures")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/full_nonlive_now/plots_final",
        help="Directory for output PNG files",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    rag_file = output_dir / "figure6_traditional_rag_pipeline.png"
    agentic_file = output_dir / "figure7_agentic_multi_tool_pipeline.png"

    plot_traditional_rag(rag_file)
    plot_agentic_multi_tool(agentic_file)

    print(f"Saved: {rag_file}")
    print(f"Saved: {agentic_file}")


if __name__ == "__main__":
    main()
