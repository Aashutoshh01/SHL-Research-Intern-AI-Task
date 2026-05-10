import os
import matplotlib.pyplot as plt
import numpy as np

# Ensure target directory exists
os.makedirs("image", exist_ok=True)

# Set global matplotlib parameters for clean academic style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

# Curated, professional color palette (Academic/Muted)
COLORS = {
    'recall': '#2b5c8f',       # Slate Blue
    'precision': '#4682b4',    # Steel Blue
    'mrr': '#3b9a9c',          # Teal/Sage Muted
}

# ---------------------------------------------------------------------------
# PLOT 1: Embedding Model Comparison (Grouped Bar Chart)
# ---------------------------------------------------------------------------
def generate_embedding_comparison_plot():
    models = ['all-mpnet-base-v2', 'all-MiniLM-L6-v2\n(Selected)', 'bge-small-en-v1.5']
    recall_10 = [0.900, 0.833, 0.783]
    precision_5 = [0.500, 0.470, 0.473]
    mrr = [0.555, 0.438, 0.471]

    x = np.arange(len(models))
    width = 0.24  # Width of each bar

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Plot grouped bars
    rects1 = ax.bar(x - width, recall_10, width, label='Recall@10', color=COLORS['recall'], edgecolor='none', alpha=0.9)
    rects2 = ax.bar(x, precision_5, width, label='Precision@5', color=COLORS['precision'], edgecolor='none', alpha=0.9)
    rects3 = ax.bar(x + width, mrr, width, label='MRR', color=COLORS['mrr'], edgecolor='none', alpha=0.9)

    # Clean styling
    ax.set_title('Retrieval Quality by Embedding Model', pad=15, fontweight='bold', color='#333333')
    ax.set_xticks(x)
    ax.set_xticklabels(models, color='#333333', fontweight='medium')
    ax.set_ylabel('Metric Value (0.0 - 1.0)', labelpad=10, color='#333333')
    ax.set_ylim(0, 1.05)

    # Subtle horizontal gridlines only
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='#cccccc')
    ax.xaxis.grid(False)

    # Remove outer spines
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('#888888')

    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8.5, color='#444444')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    # Clean legend
    ax.legend(frameon=True, facecolor='white', edgecolor='#e0e0e0', loc='upper right')

    plt.tight_layout()
    output_path = os.path.join("image", "embedding_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

# ---------------------------------------------------------------------------
# PLOT 2: Confidence Threshold Sweep (Multi-Line Plot)
# ---------------------------------------------------------------------------
def generate_threshold_sweep_plot():
    thresholds = [0.35, 0.45, 0.55, 0.65, 0.75]
    recall_10 = [0.900, 0.867, 0.867, 0.867, 0.867]
    precision_5 = [0.500, 0.493, 0.493, 0.493, 0.473]
    mrr = [0.549, 0.538, 0.538, 0.538, 0.481]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Plot lines with distinct markers and high-end styling
    ax.plot(thresholds, recall_10, marker='o', markersize=6, linewidth=2, 
            label='Recall@10', color=COLORS['recall'])
    ax.plot(thresholds, precision_5, marker='s', markersize=5.5, linewidth=2, 
            label='Precision@5', color=COLORS['precision'])
    ax.plot(thresholds, mrr, marker='D', markersize=5, linewidth=2, 
            label='MRR', color=COLORS['mrr'])

    # Highlight optimal threshold 0.55
    ax.axvline(x=0.55, color='#d9534f', linestyle=':', alpha=0.8, linewidth=1.5, 
               label='Optimal Threshold (0.55)')

    # Clean styling
    ax.set_title('Ranking Optimization Sweep', pad=15, fontweight='bold', color='#333333')
    ax.set_xlabel('Confidence Cut-off Threshold', labelpad=10, color='#333333')
    ax.set_ylabel('Metric Value (0.0 - 1.0)', labelpad=10, color='#333333')
    ax.set_xticks(thresholds)
    ax.set_xticklabels([f'{t:.2f}' for t in thresholds], color='#333333')
    ax.set_ylim(0.4, 0.95)

    # Subtle gridlines
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='#cccccc')
    ax.xaxis.grid(True, linestyle='--', alpha=0.3, color='#dddddd')

    # Remove outer spines
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('#888888')

    # Add numeric annotations at 0.55
    ax.annotate('Optimal Cut-off\n(0.55)', 
                xy=(0.55, 0.867), 
                xytext=(0.58, 0.75),
                arrowprops=dict(facecolor='#444444', arrowstyle='->', lw=1),
                fontsize=9.5, fontweight='medium', color='#333333')

    # Clean legend
    ax.legend(frameon=True, facecolor='white', edgecolor='#e0e0e0', loc='upper right')

    plt.tight_layout()
    output_path = os.path.join("image", "threshold_sweep.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    generate_embedding_comparison_plot()
    generate_threshold_sweep_plot()
