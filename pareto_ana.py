import numpy as np
import matplotlib.pyplot as plt

# ─── Your data ────────────────────────────────────────────────────────────────
# DTD dataset (Base)
lambdas_dtd = np.array([1.0, 3.0, 5.0, 8.0, 10.0])
acc_dtd     = np.array([79.44, 79.17, 80.75, 80.13, 80.90])
ece_dtd     = np.array([ 3.47,  3.31,  2.93,  3.04,  3.10])

methods_dtd = {
    'Maple':           {'acc': 79.63, 'ece':  4.18},
    'Temp. Scaling':   {'acc': 78.60, 'ece':  5.98},
    'ECCV Penalty':    {'acc': 64.60, 'ece': 19.67},
    'ECCV Zero-shot':  {'acc': 80.43, 'ece':  7.02},
    'MDCA':            {'acc': 80.53, 'ece':  3.71},
    'MBLS':            {'acc': 80.03, 'ece':  4.79},
}

# Food dataset (Base)
lambdas_food = np.array([1.0, 3.0, 5.0, 8.0, 10.0])
acc_food     = np.array([90.72, 90.61, 90.43, 90.56, 90.57])
ece_food     = np.array([ 0.97,  0.80,  0.67,  0.73,  0.74])

methods_food = {
    'Maple':           {'acc': 90.80, 'ece': 0.78},
    'Temp. Scaling':   {'acc': 90.63, 'ece': 0.71},
    'ECCV Penalty':    {'acc': 90.73, 'ece': 3.87},   # renamed from your eccv_Penalty_dtd
    'ECCV Zero-shot':  {'acc': 90.57, 'ece': 1.13},  # renamed from eccv_Zeroshot_dtd
    'MDCA':            {'acc': 90.70, 'ece': 0.84},
    'MBLS':            {'acc': 90.80, 'ece': 6.55},
}
# ────────────────────────────────────────────────────────────────────────────────

def plot_panel(ax, title, lambdas, acc, ece, fixed_methods):
    markers = ['x', 'D', '^', 's', 'P', 'X']
    colors  = ['red', 'blue', 'green', 'magenta', 'orange', 'purple']

    # 1) plot fixed-λ methods
    for (label, mtd), m, c in zip(fixed_methods.items(), markers, colors):
        ax.scatter(mtd['acc'], mtd['ece'],
                   marker=m, color=c, edgecolor='k', s=100, label=label)

    # 2) plot your Text-Momentum series
    sc = ax.scatter(acc, ece,
                    c=lambdas, cmap='viridis',
                    edgecolor='k', s=80, label='Text-Momentum')

    # 3) connect in ascending-λ order
    idx = np.argsort(lambdas)
    ax.plot(acc[idx], ece[idx], '--', color='gray', alpha=0.6)

    # 4) arrow for λ ↑
    start = (acc[idx[0]], ece[idx[0]])
    end   = (acc[idx[-1]], ece[idx[-1]])
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
    ax.text(end[0], end[1], 'λ ↑', color='blue',
            fontsize=12, ha='right', va='bottom')

    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Accuracy (%)', fontsize=12)
    ax.set_ylabel('ECE', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.4)
    return sc

# ─── Draw the 1×2 figure ───────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

sc1 = plot_panel(ax1, 'DTD (Base)',  lambdas_dtd,  acc_dtd,  ece_dtd,  methods_dtd)
sc2 = plot_panel(ax2, 'Food (Base)', lambdas_food, acc_food, ece_food, methods_food)

# shared legend
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False, fontsize=12)

# single colorbar for λ
cbar = fig.colorbar(sc1, ax=[ax1, ax2], location='right', pad=0.02)
cbar.set_label('λ', fontsize=12)

fig.suptitle('Pareto Front Analysis (Base) — All Methods + Text-Momentum', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.93])

# ─── Save high-resolution PNG ─────────────────────────────────────────────────
fig.savefig('pareto_front_base.png', dpi=300, bbox_inches='tight')
print("Saved figure as pareto_front_base.png")

plt.show()
