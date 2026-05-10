import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Ensure image directory exists
os.makedirs("image", exist_ok=True)

# Set up matplotlib parameters for ultra-clean minimalist academic flowchart style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

# Compact and wide canvas for excellent 2-page document integration
fig, ax = plt.subplots(figsize=(9.5, 6.0))
fig.patch.set_facecolor('#ffffff')
ax.set_facecolor('#ffffff')

# ---------------------------------------------------------------------------
# Helpers to draw clean black/white boxes and arrows
# ---------------------------------------------------------------------------

def draw_clean_box(ax, x, y, w, h, text):
    # Perfect crisp thin border with pure white background
    rect = patches.Rectangle(
        (x, y), w, h,
        linewidth=1.2,
        edgecolor='#2d3748',
        facecolor='#ffffff',
    )
    ax.add_patch(rect)
    # Perfectly centered text inside the box
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontweight='bold', fontsize=9.5, color='#1a202c')

def draw_arrow_line(ax, x1, y1, x2, y2):
    # Drawing simple sharp arrow
    ax.annotate(
        '', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color='#4a5568', lw=1.2, mutation_scale=10)
    )

def draw_plain_line(ax, x1, y1, x2, y2):
    # Simple straight line for connectors
    ax.plot([x1, x2], [y1, y2], color='#4a5568', lw=1.2)

# ---------------------------------------------------------------------------
# Layout Configuration (x: 0 to 10, y: 0 to 6.2)
# ---------------------------------------------------------------------------

cx = 1.0     # Box left-x
box_w = 3.0  # Box width
bx = cx + box_w / 2  # Box center-x (2.5)
side_x = 4.4 # Gutter for side annotations

# 1. Header: Conversation History
ax.text(bx, 5.8, "Conversation History (full, stateless)", ha='center', va='center', fontsize=9.5, color='#2d3748')
draw_arrow_line(ax, bx, 5.65, bx, 5.3)

# 2. Box: Constraint Extraction
draw_clean_box(ax, cx, 4.7, box_w, 0.6, "Constraint Extraction")
ax.text(side_x, 5.0, "← LLM extracts structured state from conversation", ha='left', va='center', fontsize=9.0, color='#4a5568')
draw_arrow_line(ax, bx, 4.7, bx, 4.25)

# 3. Box: Intent Classification
draw_clean_box(ax, cx, 3.65, box_w, 0.6, "Intent Classification")
ax.text(side_x, 3.95, "← Hybrid: rule-based fast path + LLM fallback", ha='left', va='center', fontsize=9.0, color='#4a5568')

# 4. Conditional Routing (Branches)
draw_arrow_line(ax, bx, 3.65, bx, 3.1)
ax.text(bx + 0.1, 3.3, "(conditional routing)", ha='left', va='center', fontsize=8.5, color='#718096', style='italic')

# Column x-coordinates for branching
cols_x = [0.4, 1.45, 2.5, 3.55, 4.6]
labels = [
    "Refuse",
    "Clarify",
    "Retrieve\n+ Filter\n+ Rank",
    "Refine",
    "Compare"
]

# Draw horizontal distribution line
draw_plain_line(ax, cols_x[0], 3.1, cols_x[-1], 3.1)

# Draw arrows down from distributor to labels
for x, lbl in zip(cols_x, labels):
    draw_arrow_line(ax, x, 3.1, x, 2.8)
    # Centered column labels
    ax.text(x, 2.45, lbl, ha='center', va='center', fontsize=8.5, fontweight='bold', color='#2d3748')

# Draw horizontal combiner line
draw_plain_line(ax, cols_x[0], 2.1, cols_x[-1], 2.1)

# Draw lines from labels down to combiner
for x in cols_x:
    draw_arrow_line(ax, x, 2.2, x, 2.1)

# Main arrow from combiner to Grounded Response
draw_arrow_line(ax, bx, 2.1, bx, 1.7)

# 5. Box: Grounded Response Gen
draw_clean_box(ax, cx, 1.1, box_w, 0.6, "Grounded Response Gen")
ax.text(side_x, 1.4, "← LLM explains, but NEVER invents catalog data", ha='left', va='center', fontsize=9.0, color='#4a5568')
draw_arrow_line(ax, bx, 1.1, bx, 0.7)

# 6. Box: Strict Structured JSON
draw_clean_box(ax, cx, 0.1, box_w, 0.6, "Strict Structured JSON")
ax.text(side_x, 0.4, "← {reply, recommendations[], end_of_conversation}", ha='left', va='center', fontsize=9.0, color='#4a5568')

# Hide axis lines completely
ax.set_xlim(-0.2, 9.2)
ax.set_ylim(-0.1, 6.2)
ax.set_axis_off()

# Adjust layout and save high-DPI image
plt.tight_layout()
output_path = os.path.join("image", "architecture_flow.png")
plt.savefig(output_path, dpi=300, facecolor='#ffffff', bbox_inches='tight')
plt.close()
print(f"Successfully generated pristine clean flowchart: {output_path}")
