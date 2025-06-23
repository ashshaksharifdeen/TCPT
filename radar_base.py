import numpy as np
import matplotlib.pyplot as plt

# 1. --- YOUR DATA HERE ---------------------------------------------------
labels = [
     "DTD", "OxfordFlowers", "Food", "SUN397",
    "Aircraft", "OxfordPets", "Caltech", "UCF101", "EuroSAT", "Cars"
]

ours =               [ 2.42,   4.8, 0.3,   1.22, 
                      4.96,  1.94,   1.01,  1.11,   4.9,   7.1]
temperature_scaling = [ 3.86, 4.6, 0.71, 0.5,
                       2.01, 3.43, 2.54, 1.2,  1.57,  4.76, 6.63]
coop =              [ 4.18, 4.28, 0.78,  1.27,
                      3.86, 2.68,   1.54,2.68, 3.42,  7.25]
# -------------------------------------------------------------------------

# 2. Compute angles and close the loops
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

def close_loop(data):
    return data + data[:1]

ours_loop   = close_loop(ours)
temp_loop   = close_loop(temperature_scaling)
maple_loop  = close_loop(coop)

# 3. Create the radar chart
fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
plt.tight_layout()

# 4. Plot each series
ax.plot(angles, ours_loop,   color="C3", linewidth=3, marker="o", markersize=8, label="Ours")
ax.fill(angles, ours_loop,   color="C3", alpha=0.25)

ax.plot(angles, temp_loop,   color="C1", linewidth=3, marker="s", markersize=8, label="Temperature Scaling")
ax.fill(angles, temp_loop,   color="C1", alpha=0.25)

ax.plot(angles, maple_loop,  color="C2", linewidth=3, marker="^", markersize=8, label="MAPLE")
ax.fill(angles, maple_loop,  color="C2", alpha=0.25)

# 5. Styling: start at top and go clockwise
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# 6. Dataset labels (axes)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=18, fontweight='bold')

# 7. Radial gridlines only (no labels)
max_val = max(max(ours), max(temperature_scaling), max(coop))
step    = max_val / 5
r_ticks = np.arange(0, max_val + step, step)

ax.set_yticks(r_ticks)
ax.set_yticklabels([])            # <-- remove radial tick labels
ax.yaxis.grid(True, color='gray', linestyle='--', linewidth=0.5)
ax.set_ylim(0, max_val)

# 8. Annotate each point with its value in black, staggering offsets
base_offset = max_val * 0.02
offsets = [base_offset, base_offset * 2, base_offset * 3]

for angle, value in zip(angles, ours_loop):
    ax.text(angle, value + offsets[0], f"{value:.1f}",
            color="black", fontsize=20, ha='center', va='bottom')

for angle, value in zip(angles, temp_loop):
    ax.text(angle, value + offsets[1], f"{value:.1f}",
            color="black", fontsize=20, ha='center', va='bottom')

for angle, value in zip(angles, maple_loop):
    ax.text(angle, value + offsets[2], f"{value:.1f}",
            color="black", fontsize=20, ha='center', va='bottom')

# 9. Title and legend
ax.set_title("Base Classes", y=1.12,
             fontsize=22, fontweight='bold')
leg = ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
for text in leg.get_texts():
    text.set_fontsize(16)

# 10. Save high-res for Overleaf
plt.savefig("radar_plot.pdf", dpi=300, bbox_inches="tight")
plt.savefig("radar_plot.png", dpi=300, bbox_inches="tight")

plt.show()
