import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Create output directory
output_dir = "media/pinpad/director-results"
os.makedirs(output_dir, exist_ok=True)

# File paths
files = {
    'pinpad-3': 'director-result/pinpad-3.jsonl',
    'pinpad-dense-3': 'director-result/pinpad-dense-3.jsonl'
}

# Read and parse JSONL files
data = {}
for name, filepath in files.items():
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    data[name] = pd.DataFrame(records)
    print(f"Loaded {name}: {len(data[name])} records")

# =============================================================================
# Episode/Score comparison
# =============================================================================

fig, ax = plt.subplots(figsize=(6, 3.5), dpi=300)

colors = ['#1f77b4', '#ff7f0e']  # blue, orange

for (name, df), color in zip(data.items(), colors):
    # Filter for episode/score entries
    df_scores = df[['step', 'episode/score']].dropna()
    
    if df_scores.empty:
        print(f"No episode/score data for {name}")
        continue
    
    # Sort by step
    df_scores = df_scores.sort_values('step')
    
    x = df_scores['step'] / 1000  # thousands of steps
    y = df_scores['episode/score']
    
    # Smooth the curve
    window = 20
    y_smooth = y.rolling(window=window, min_periods=1).mean()
    
    # Plot raw data (light)
    ax.plot(x, y, linewidth=0.5, alpha=0.3, color=color)
    
    # Plot smoothed data (bold)
    ax.plot(x, y_smooth, linewidth=1.4, label=name, alpha=0.8, color=color)

ax.set_xlabel("Env. Steps (×10³)", fontsize=9)
ax.set_ylabel("Episode Return", fontsize=9)
ax.legend(fontsize=8, loc="best")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="both", labelsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(f"{output_dir}/director-episode-scores.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"✓ Saved: {output_dir}/director-episode-scores.png")

print("\n✓ Director results visualization complete!")
