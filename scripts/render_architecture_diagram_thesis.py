"""
Render the corrected benchmark architecture diagram with thesis-ready styling.
White background, professional colors, icons, and typography.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.patheffects as pe

# ── Color Palette (Professional + Accessible) ─────────────────────────────────
BG       = "#ffffff"           # White background
BOX      = "#f5f7fa"           # Light blue-gray
BDR_MAIN = "#2563eb"           # Bright blue (core flow)
BDR_OPT  = "#059669"           # Emerald green (optional)
BOX_OPT  = "#f0fdf4"           # Very light green
TXT_MAIN = "#1f2937"           # Dark gray
TXT_SUB  = "#6b7280"           # Medium gray
ARR_MAIN = "#2563eb"           # Blue
ARR_OPT  = "#059669"           # Green (dashed)

fig, ax = plt.subplots(figsize=(12, 17))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 10)
ax.set_ylim(0, 17.5)
ax.axis("off")

# ── Helper: styled rounded box with icon ──────────────────────────────────────
def box(ax, cx, cy, w, h, label, icon="", color=BOX, border=BDR_MAIN, 
        fontsize=9, sublabel=None, shadow=True):
    """Draw a box with optional icon and shadow."""
    # Shadow
    if shadow:
        shadow_rect = FancyBboxPatch(
            (cx - w / 2 + 0.05, cy - h / 2 - 0.05), w, h,
            boxstyle="round,pad=0.08",
            linewidth=0,
            facecolor="#00000008",
            zorder=1,
        )
        ax.add_patch(shadow_rect)
    
    # Main box
    rect = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.08",
        linewidth=2,
        edgecolor=border,
        facecolor=color,
        zorder=3,
    )
    ax.add_patch(rect)
    
    # Icon (if provided)
    if icon:
        ax.text(cx - w / 2 + 0.35, cy + 0.12, icon, 
                fontsize=12, va="center", zorder=4,
                color=border)
    
    # Label
    y_text = cy + (0.20 if sublabel else 0)
    label_text = ax.text(cx + (0.15 if icon else 0), y_text, label, 
                         ha="left" if icon else "center", 
                         va="center",
                         color=TXT_MAIN, fontsize=fontsize, 
                         fontweight="600", zorder=4,
                         linespacing=1.5)
    
    # Sublabel
    if sublabel:
        ax.text(cx + (0.15 if icon else 0), cy - 0.25, sublabel, 
                ha="left" if icon else "center", 
                va="center",
                color=TXT_SUB, fontsize=7.5, 
                style="italic", zorder=4)


# ── Helper: solid arrow ───────────────────────────────────────────────────────
def arrow(ax, x1, y1, x2, y2, dashed=False, color=ARR_MAIN, lw=2):
    """Draw an arrow with optional dashing."""
    style = "dashed" if dashed else "solid"
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            linestyle=style,
            connectionstyle="arc3,rad=0.0",
        ),
        zorder=2,
    )


# ── Helper: vertical connector ────────────────────────────────────────────────
def connect(ax, cx, from_y, to_y, offset=0, dashed=False, color=ARR_MAIN):
    """Vertical connector between boxes."""
    col = color
    arrow(ax, cx + offset, from_y, cx + offset, to_y, dashed=dashed, color=col)


# ── Title ─────────────────────────────────────────────────────────────────────
title_text = ax.text(5, 17.05, "GRI-QA Benchmark Architecture", 
                     ha="center", va="top", color=TXT_MAIN,
                     fontsize=14, fontweight="bold", zorder=4)
subtitle_text = ax.text(5, 16.65, "Traditional RAG vs. Agentic Multi-Tool Pipeline", 
                        ha="center", va="top", color=TXT_SUB,
                        fontsize=10, style="italic", zorder=4)

# ── BOXES ──────────────────────────────────────────────────────────────────────

# Layer 0: Data Input
box(ax, 5.0, 15.5, 4.2, 0.60, "Source GRI-QA Dataset", 
    icon="📊", border=BDR_MAIN)

# Layer 1: Preparation
box(ax, 5.0, 14.5, 4.2, 0.60, "Data Preparation", 
    icon="⚙️", border=BDR_MAIN)

# Layer 2: Benchmark Data
box(ax, 5.0, 13.3, 5.0, 0.75, "Benchmark Data Splits",
    icon="📁", sublabel="data/benchmark + manifest",
    border=BDR_MAIN)

# Layer 3: Config & Retriever
box(ax, 5.0, 11.9, 5.4, 0.95, "Config-Driven Runner",
    icon="⚡", sublabel="strict_mode retriever built internally",
    border=BDR_MAIN, fontsize=9.5)

# Layer 4: Pipelines (diverge)
box(ax, 2.3, 10.35, 3.6, 0.60, "Traditional RAG Pipeline",
    icon="🔄", border=BDR_MAIN)
box(ax, 7.7, 10.35, 3.6, 0.65, "Agentic Multi-Tool\nPipeline",
    icon="🤖", border=BDR_MAIN, fontsize=8.5)

# Layer 5: Predictions (converge)
box(ax, 5.0, 8.85, 5.2, 0.80, "Per-Question Predictions + Metadata",
    icon="📝", sublabel="Citations, Trace Steps, Tool Calls",
    border=BDR_MAIN, fontsize=9)

# Layer 6: Analysis (split)
box(ax, 1.8, 7.45, 3.0, 0.60, "Metrics Aggregation",
    icon="📊", border=BDR_MAIN)
box(ax, 5.0, 7.45, 3.0, 0.60, "Error Taxonomy Labeling",
    icon="🏷️", border=BDR_MAIN)

# Layer 7: Summary (converge)
box(ax, 5.0, 6.0, 4.4, 0.60, "Summary JSON",
    icon="✅", border=BDR_MAIN, fontsize=10)

# Layer 8: Reports
box(ax, 5.0, 4.8, 4.4, 0.60, "Generated Reports & Tables",
    icon="📈", border=BDR_MAIN)

# Layer 9 (Optional): McNemar Branch
box(ax, 8.3, 5.5, 3.0, 0.70, "Paired Outcome Analysis",
    icon="📊", sublabel="+ McNemar Tests (optional)",
    color=BOX_OPT, border=BDR_OPT)

# Layer 10 (Optional): Live Summary
box(ax, 8.3, 4.2, 3.0, 0.60, "live_summary.json",
    icon="📄", sublabel="separate output",
    color=BOX_OPT, border=BDR_OPT, fontsize=8.5)

# ── ARROWS ──────────────────────────────────────────────────────────────────────

# Main flow (vertical)
connect(ax, 5, 15.5 - 0.30, 14.5 + 0.30)          # Source → Prep
connect(ax, 5, 14.5 - 0.30, 13.3 + 0.38)          # Prep → Benchmark
connect(ax, 5, 13.3 - 0.38, 11.9 + 0.48)          # Benchmark → Runner

# Runner → Pipelines (diverge)
arrow(ax, 3.6, 11.9 - 0.48, 2.3, 10.35 + 0.30, color=ARR_MAIN, lw=2)
arrow(ax, 6.4, 11.9 - 0.48, 7.7, 10.35 + 0.33, color=ARR_MAIN, lw=2)

# Pipelines → Predictions (converge)
arrow(ax, 2.3, 10.35 - 0.30, 3.5, 8.85 + 0.40, color=ARR_MAIN, lw=2)
arrow(ax, 7.7, 10.35 - 0.33, 6.5, 8.85 + 0.40, color=ARR_MAIN, lw=2)

# Predictions → Analysis (split left & centre)
arrow(ax, 3.8, 8.85 - 0.40, 1.8, 7.45 + 0.30, color=ARR_MAIN, lw=2)
arrow(ax, 5.0, 8.85 - 0.40, 5.0, 7.45 + 0.30, color=ARR_MAIN, lw=2)

# Analysis → Summary (converge)
arrow(ax, 1.8, 7.45 - 0.30, 3.4, 6.0 + 0.30, color=ARR_MAIN, lw=2)
arrow(ax, 5.0, 7.45 - 0.30, 5.0, 6.0 + 0.30, color=ARR_MAIN, lw=2)

# Summary → Reports
connect(ax, 5, 6.0 - 0.30, 4.8 + 0.30, color=ARR_MAIN)

# Optional McNemar branch (dashed)
arrow(ax, 6.6, 8.85 - 0.40, 8.3, 5.5 + 0.35, 
      dashed=True, color=ARR_OPT, lw=1.8)
connect(ax, 8.3, 5.5 - 0.35, 4.2 + 0.30, dashed=True, color=ARR_OPT)

# ── LEGEND ────────────────────────────────────────────────────────────────────
legend_x, legend_y = 0.5, 2.2
legend_w, legend_h = 4.0, 1.0

# Legend box
legend_box = FancyBboxPatch(
    (legend_x - 0.05, legend_y - legend_h), legend_w, legend_h,
    boxstyle="round,pad=0.1",
    linewidth=1.5,
    edgecolor="#d1d5db",
    facecolor="#fafafa",
    zorder=3,
)
ax.add_patch(legend_box)

# Legend items
y_offset = 0.32
ax.plot([legend_x + 0.2, legend_x + 0.7], 
        [legend_y - y_offset, legend_y - y_offset],
        color=ARR_MAIN, lw=2.5, zorder=4)
ax.text(legend_x + 0.95, legend_y - y_offset, 
        "Core Pipeline Flow",
        color=TXT_MAIN, fontsize=8, va="center", fontweight="600", zorder=4)

ax.plot([legend_x + 0.2, legend_x + 0.7], 
        [legend_y - y_offset - 0.30, legend_y - y_offset - 0.30],
        color=ARR_OPT, lw=2.5, linestyle="--", zorder=4)
ax.text(legend_x + 0.95, legend_y - y_offset - 0.30, 
        "Optional Post-Processing",
        color=TXT_MAIN, fontsize=8, va="center", fontweight="600", zorder=4)

# Green box swatch
green_box = FancyBboxPatch(
    (legend_x + 0.2, legend_y - y_offset - 0.60), 0.45, 0.18,
    boxstyle="round,pad=0.04",
    linewidth=1.5,
    edgecolor=BDR_OPT,
    facecolor=BOX_OPT,
    zorder=4,
)
ax.add_patch(green_box)
ax.text(legend_x + 0.95, legend_y - y_offset - 0.51, 
        "Optional Node",
        color=TXT_MAIN, fontsize=8, va="center", fontweight="600", zorder=4)

# ── FOOTER ──────────────────────────────────────────────────────────────────────
ax.text(5, 0.25, "Energy Sector GRI-QA Benchmark | Agentic vs. Traditional RAG Comparison", 
        ha="center", va="center", color=TXT_SUB,
        fontsize=7.5, style="italic", zorder=4)

# ── SAVE ────────────────────────────────────────────────────────────────────────
out_png = "/Users/J.Panda/Downloads/Cert_Masters/TRADITIONAL RAG VS MULTI TOOL AGENT PIPELINES_gri_QA_dataset/Energy_Sector_Agentic_vs_RAG/docs/architecture_thesis.png"
out_pdf = "/Users/J.Panda/Downloads/Cert_Masters/TRADITIONAL RAG VS MULTI TOOL AGENT PIPELINES_gri_QA_dataset/Energy_Sector_Agentic_vs_RAG/docs/architecture_thesis.pdf"

plt.tight_layout()
plt.savefig(out_png, dpi=300, bbox_inches="tight",
            facecolor=BG, edgecolor="none")
plt.savefig(out_pdf, bbox_inches="tight",
            facecolor=BG, edgecolor="none")

print(f"✓ PNG saved: {out_png} (300 DPI)")
print(f"✓ PDF saved: {out_pdf}")
print("\nThesis-ready features:")
print("  • White background (professional printing)")
print("  • High-resolution PNG (300 DPI) and vector PDF")
print("  • Accessible color palette (blue/green)")
print("  • Icons for visual interest")
print("  • Clear distinction: core flow vs. optional branches")
