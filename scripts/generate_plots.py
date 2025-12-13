import matplotlib.pyplot as plt
import numpy as np
import os

# Style
plt.style.use('dark_background')
plt.rcParams.update({'font.size': 12, 'axes.facecolor': '#1e1e1e', 'savefig.facecolor': '#111111'})

# Data (Manual entry from observed results)
metrics = {
    "Model Size (MB)": {"FP16": 5.3, "Atomic-1Bit": 2.0},
    "Inference Speed (TPS)": {"FP16": 555, "Atomic Py": 125, "Atomic C++": 60} 
}

def save_plot(name):
    plt.tight_layout()
    os.makedirs("../assets", exist_ok=True)
    plt.savefig(f"../assets/{name}.png", dpi=150)
    plt.close()

# 1. Model Size Chart
labels = list(metrics["Model Size (MB)"].keys())
values = list(metrics["Model Size (MB)"].values())
colors = ['#444444', '#00ffcc']

plt.figure(figsize=(6, 4))
plt.bar(labels, values, color=colors)
plt.title("Model Size Comparison")
plt.ylabel("Size (MB)")
for i, v in enumerate(values):
    plt.text(i, v + 0.1, f"{v} MB", ha='center', color='white')
save_plot("chart_model_size")

# 2. Speed Chart
labels = list(metrics["Inference Speed (TPS)"].keys())
values = list(metrics["Inference Speed (TPS)"].values())
colors = ['#444444', '#00ffcc', '#009977']

plt.figure(figsize=(6, 4))
plt.bar(labels, values, color=colors)
plt.title("Inference Speed (Desktop CPU)")
plt.ylabel("Tokens / Sec")
for i, v in enumerate(values):
    plt.text(i, v + 10, f"{v}", ha='center', color='white')
save_plot("chart_speed")

# 3. Summary Table
data = [
    ["Metric", "FP16 Baseline", "Atomic-1Bit"],
    ["Params", "1.33 M", "1.33 M"],
    ["Precision", "FP16/FP32", "Ternary (1.58 bit)"],
    ["Size", "5.3 MB", "2.0 MB"],
    ["TPS (CPU)", "~555", "~60 (C++) / ~125 (Py)"],
    ["Energy Proxy", "High (F-MAC)", "Low (Int-Add)"]
]

fig, ax = plt.subplots(figsize=(8, 3))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=data, loc='center', cellLoc='left')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.5)

# Styling table
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor('#333333')
        cell.set_text_props(color='white', weight='bold')
    else:
        cell.set_facecolor('#1e1e1e')
        cell.set_text_props(color='#cccccc')

save_plot("table_summary")

# 4. Text Samples Comparison
atomic_text = "trying jumped hard should sighedma jumped We hurts key float clock\nshining explore sorry bigger danced fold flowers poured gl jumped\ndoes folding kind eye enormous Charlie pouredSarahA vine reading\njumped birds owners pencil Charlie stir folding..."
fp16_text = "a big................................................\n.....................................................\n.....................................................\n....................................................."

plt.figure(figsize=(10, 4))
plt.axis('off')
plt.title("Generated Text Samples (200 Steps Training)", color='white', pad=20)

# Left: Atomic
plt.text(0.05, 0.8, "Atomic-1Bit (1.58b)", color='#00ffcc', fontsize=12, weight='bold')
plt.text(0.05, 0.4, atomic_text, color='#cccccc', fontsize=10, fontfamily='monospace', va='top')

# Right: FP16
plt.text(0.55, 0.8, "FP16 Baseline (16b)", color='#ffcc00', fontsize=12, weight='bold')
plt.text(0.55, 0.4, fp16_text, color='#cccccc', fontsize=10, fontfamily='monospace', va='top')

# Divider line
plt.axvline(x=0.5, color='#444444', linestyle='--')

save_plot("text_samples_comparison")

print("Plots generated.")
