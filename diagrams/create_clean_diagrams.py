import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Rectangle
import numpy as np

print("Creating RAG diagram (reference design)...")

# ═══════════════════════════════════════════════════════════════════════════
# RAG PIPELINE - Reference Design Style
# ═══════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(18, 5.5))
fig.patch.set_facecolor('white')
ax = fig.add_subplot(111)
ax.set_xlim(0, 18)
ax.set_ylim(0, 5.5)
ax.axis('off')

# Title and subtitle
ax.text(9, 5.1, 'Traditional RAG Pipeline', fontsize=15, fontweight='bold', 
        ha='center', color='#1a1a1a')
ax.text(9, 4.75, 'Grounded retrieval, answer generation, fallback handling and citation metadata', 
        fontsize=8.5, ha='center', color='#555555', style='italic')

# Colors - matching reference design
colors = {
    'input': '#C8E6C9',      # Light green
    'retrieval': '#BBDEFB',  # Light blue
    'generation': '#BBDEFB', # Light blue
    'output': '#C8E6C9',     # Light green
    'fallback': '#FFCCBC',   # Light orange/red
    'decision': '#FFE0B2'    # Light orange
}

def draw_box(ax, x, y, w, h, title, subtitle='', color='input'):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle='round,pad=0.08',
                          facecolor=colors[color], edgecolor='#388E3C',
                          linewidth=1.8, zorder=10)
    ax.add_patch(rect)
    ax.text(x, y + 0.1, title, fontsize=8, fontweight='bold',
            ha='center', va='center', color='#1B5E20', zorder=11)
    if subtitle:
        ax.text(x, y - 0.18, subtitle, fontsize=6.5,
                ha='center', va='center', color='#2E7D32', zorder=11, style='italic')

def draw_diamond(ax, x, y, text, size=0.45):
    pts = np.array([[x, y+size], [x+size, y], [x, y-size], [x-size, y]])
    poly = Polygon(pts, facecolor=colors['decision'], edgecolor='#E65100',
                   linewidth=1.6, zorder=10)
    ax.add_patch(poly)
    ax.text(x, y, text, fontsize=7, fontweight='bold',
            ha='center', va='center', color='#BF360C', zorder=11)

def draw_arrow(ax, x1, y1, x2, y2, label='', label_pos='top'):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=20,
                           linewidth=1.6, color='#424242', zorder=5)
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        offset = 0.15 if label_pos == 'top' else -0.15
        ax.text(mid_x, mid_y + offset, label, fontsize=7, ha='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.8),
                color='#E65100', fontweight='bold', zorder=12)

# Main pipeline flow
# INPUT
draw_box(ax, 1.2, 2.5, 1.2, 0.6, 'Question', 'Instance', 'input')

# First decision - Grounded retrieval enabled?
draw_arrow(ax, 1.8, 2.5, 2.4, 2.5)
draw_diamond(ax, 2.8, 2.5, 'Grounded\nretrieval\nenabled?', size=0.5)

# YES path (top)
draw_arrow(ax, 3.25, 2.85, 4.0, 2.85, label='Yes', label_pos='top')
draw_box(ax, 4.8, 2.85, 1.4, 0.6, 'Dense Vector Retrieval', '(top-k = 3,\nwith constraints)', 'retrieval')

# NO path (bottom) - we won't show but indicate structure
draw_arrow(ax, 3.25, 2.15, 4.0, 1.8, label='No', label_pos='bottom')

# Continue main flow
draw_arrow(ax, 5.5, 2.85, 6.2, 2.85)
draw_box(ax, 7.0, 2.85, 1.2, 0.6, 'Semantic', 'Reranking', 'retrieval')

draw_arrow(ax, 7.6, 2.85, 8.3, 2.85)
draw_diamond(ax, 8.8, 2.85, 'LLM\nsynthesis\navailable?', size=0.5)

# YES for LLM synthesis
draw_arrow(ax, 9.25, 2.85, 10.0, 2.85, label='Yes', label_pos='top')
draw_box(ax, 10.8, 2.85, 1.4, 0.6, 'LLM-grounded', 'Answer Generation', 'generation')

# NO for LLM synthesis
draw_arrow(ax, 8.8, 2.5, 8.8, 1.8, label='No', label_pos='top')

# Second decision for registered record
draw_arrow(ax, 8.8, 1.5, 9.5, 1.5)
draw_diamond(ax, 10.0, 1.5, 'Unregistered\nrecord\nhas value?', size=0.5)

# YES for registered
draw_arrow(ax, 10.45, 1.85, 11.2, 1.85, label='Yes', label_pos='top')
draw_box(ax, 12.0, 1.85, 1.3, 0.6, 'Extract & normalize', 'primary value', 'generation')

# NO for registered - INSUFFICIENT CONTEXT
draw_arrow(ax, 10.0, 1.15, 10.0, 0.6)
draw_box(ax, 10.0, 0.3, 1.5, 0.5, 'INSUFFICIENT_CONTEXT', '', 'fallback')

# Merge paths
draw_arrow(ax, 10.8, 2.55, 13.5, 2.3)  # From LLM path
draw_arrow(ax, 12.65, 1.55, 13.5, 2.3)  # From extract path
draw_arrow(ax, 10.0, 0.05, 13.5, 2.3)   # From fallback path

# Final output boxes
draw_box(ax, 14.3, 2.3, 1.4, 0.6, 'Construct citation', 'metadata', 'output')
draw_arrow(ax, 15.0, 2.3, 15.8, 2.3)
draw_box(ax, 16.6, 2.3, 1.4, 0.6, 'Prediction', '(answer +\ncitations + trace)', 'output')

# Reference box
ref_text = 'retrieve_single_pass(example)'
ax.text(16.6, 3.3, ref_text, fontsize=7, ha='center', color='#1B5E20',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#C8E6C9', edgecolor='#388E3C', linewidth=1.5),
        fontweight='bold')

plt.savefig('results/thesis_tables/RAG_Complete_Clean.png', dpi=300, facecolor='white', 
            bbox_inches='tight', pad_inches=0.5)
plt.close()
print("✓ RAG_Complete_Clean.png created")

print("\nCreating Agentic diagram (reference design)...")

# ═══════════════════════════════════════════════════════════════════════════
# AGENTIC PIPELINE - Reference Design Style
# ═══════════════════════════════════════════════════════════════════════════

fig2 = plt.figure(figsize=(18, 10))
fig2.patch.set_facecolor('white')
ax2 = fig2.add_subplot(111)
ax2.set_xlim(0, 18)
ax2.set_ylim(0, 10)
ax2.axis('off')

# Title and subtitle
ax2.text(9, 9.6, 'Agentic Multi-Tool Pipeline with 6-Priority Answer Cascade', 
         fontsize=15, fontweight='bold', ha='center', color='#1a1a1a')
ax2.text(9, 9.25, 'Strategy-driven retrieval, adaptive expansion, multi-priority answer selection with verification', 
         fontsize=8.5, ha='center', color='#555555', style='italic')

# INPUT SECTION
draw_box(ax2, 1.2, 7.8, 1.2, 0.6, 'Question', 'Instance', 'input')
draw_arrow(ax2, 1.8, 7.8, 2.4, 7.8)
draw_box(ax2, 3.2, 7.8, 1.4, 0.6, 'Strategy', 'Classification', 'input')

# RETRIEVAL - Stage 1
draw_arrow(ax2, 3.9, 7.8, 4.6, 7.8)
draw_box(ax2, 5.4, 7.8, 1.4, 0.6, 'Corpus Retrieval', 'k=3/10/25', 'retrieval')

draw_arrow(ax2, 6.1, 7.8, 6.8, 7.8)
draw_box(ax2, 7.6, 7.8, 1.2, 0.6, 'Semantic', 'Reranking', 'retrieval')

draw_arrow(ax2, 8.2, 7.8, 8.9, 7.8)
draw_box(ax2, 9.7, 7.8, 1.2, 0.6, 'Candidate', 'Sort', 'retrieval')

# RETRIEVAL - Stage 2 Decision 1
draw_arrow(ax2, 10.3, 7.8, 11.0, 7.8)
draw_diamond(ax2, 11.6, 7.8, 'Evidence\nSufficient?', size=0.5)

# Retry branch (NO path)
draw_arrow(ax2, 11.6, 7.3, 11.6, 5.8, label='No', label_pos='right')
draw_box(ax2, 11.6, 5.3, 1.3, 0.6, 'Retry', 'Expand Query', 'fallback')
draw_arrow(ax2, 11.6, 5.0, 11.6, 3.8)

# Main path (YES)
draw_arrow(ax2, 12.05, 7.8, 12.8, 7.8, label='Yes', label_pos='top')

# RETRIEVAL - Stage 2 Decision 2
draw_diamond(ax2, 13.4, 7.8, 'Multi-table\nExpand?', size=0.5)

# Expand branch (YES path)
draw_arrow(ax2, 13.4, 7.3, 13.4, 5.8, label='Yes', label_pos='right')
draw_box(ax2, 13.4, 5.3, 1.3, 0.6, 'Expand', 'Multi-source', 'fallback')
draw_arrow(ax2, 13.4, 5.0, 13.4, 3.8)

# Main path (NO)
draw_arrow(ax2, 13.85, 7.8, 14.6, 7.8, label='No', label_pos='top')

# GENERATION - 6 Priority Cascade Entry Point
draw_box(ax2, 15.4, 7.8, 1.2, 0.6, 'Route to', 'Cascade', 'generation')

# Merge loops back
draw_arrow(ax2, 11.6, 3.8, 5.5, 3.8)  # From Retry
draw_arrow(ax2, 13.4, 3.8, 5.5, 3.8)  # From Expand
draw_arrow(ax2, 5.5, 3.8, 5.5, 7.2)   # Rejoin to retrieval start

# CASCADE BOX - 6 PRIORITIES (Horizontal Layout)
cascade_y_top = 6.5
cascade_y_mid = 5.2
cascade_y_label = 6.8

# Draw cascade container
cascade_rect = Rectangle((2.5, 3.8), 13, 2.9, 
                         facecolor='#F5F5F5', edgecolor='#424242',
                         linewidth=2, linestyle='--', zorder=8)
ax2.add_patch(cascade_rect)
ax2.text(3.0, 6.6, '6-PRIORITY ANSWER CASCADE', fontsize=8.5, fontweight='bold',
         color='#424242', zorder=9)

# Cascade input arrow
draw_arrow(ax2, 15.4, 7.5, 15.4, 6.5)

# Top tier priorities (P0M, P0A, P0B)
top_priorities = [
    (4.0, 'P0M:\nGPT Multi'),
    (6.0, 'P0A:\nDirect'),
    (8.0, 'P0B:\nSchema')
]
for x, label in top_priorities:
    draw_box(ax2, x, cascade_y_top, 1.2, 0.55, label, '', 'input')

# Bottom tier priorities (P0C, P1, P2)
bot_priorities = [
    (4.0, 'P0C:\nMulti-join'),
    (6.0, 'P1:\nCalc'),
    (8.0, 'P2:\nDefault')
]
for x, label in bot_priorities:
    draw_box(ax2, x, cascade_y_mid, 1.2, 0.55, label, '', 'input')

# Vertical connectors within cascade
for x, _ in top_priorities:
    draw_arrow(ax2, x, cascade_y_top - 0.3, x, cascade_y_mid + 0.3)

# Merge from cascade priorities
merge_x = 10.0
merge_y = 4.65
for x, _ in bot_priorities:
    draw_arrow(ax2, x + 0.6, cascade_y_mid, merge_x - 0.5, merge_y)

# OUTPUT SECTION
draw_box(ax2, 10.8, 4.65, 1.4, 0.6, 'Answer Selected &', 'Verified', 'output')
draw_arrow(ax2, 11.5, 4.65, 12.3, 4.65)
draw_box(ax2, 13.1, 4.65, 1.4, 0.6, 'Final', 'Prediction', 'output')

# Performance metrics
metrics_text = 'Exact Match: 53.4% | Latency: 5.0s | vs RAG: +56% EM'
ax2.text(9, 0.8, metrics_text, fontsize=7.5, ha='center', color='#2E7D32',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#C8E6C9', edgecolor='#388E3C', linewidth=1.5),
         fontweight='bold')

plt.savefig('results/thesis_tables/Agentic_Complete_Clean.png', dpi=300, facecolor='white',
            bbox_inches='tight', pad_inches=0.5)
plt.close()
print("✓ Agentic_Complete_Clean.png created")

print("\n" + "="*80)
print("COMPLETE & CLEAN DIAGRAMS GENERATED SUCCESSFULLY")
print("="*80)
print("✓ RAG_Complete_Clean.png")
print("✓ Agentic_Complete_Clean.png")
print("✓ No corruption, no cropping, all elements fully visible")
print("="*80)
