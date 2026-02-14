import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Create output directory
output_dir = "media/pinpad/rssm-sweep"
os.makedirs(output_dir, exist_ok=True)

# Initialize the API
api = wandb.Api()

# Fetch the sweep
sweep = api.sweep("rm2278-university-of-cambridge/Hieros-hieros/uhfc6bh3")

print(f"Sweep: {sweep.name}")
print(f"Found {len(sweep.runs)} runs")

# =============================================================================
# Episode/Score for all runs, labeled by sweep parameter
# =============================================================================

# First, print config of first run to see what parameters exist
if sweep.runs:
    print(f"\nChecking config parameters...")
    for idx, run in enumerate(sweep.runs):
        print(f"\nRun {idx+1}: {run.name} (ID: {run.id})")
        # Fetch history to see max step
        history = run.history(keys=["episode/score", "_step"])
        if not history.empty:
            max_step = history["_step"].max()
            print(f"  Max step: {max_step}")
        # Check for common parameters that might differ
        for key in ['dynamics_model', 'steps', 'seed', 'dyn_cell', 'dyn_stoch', 'dyn_deter', 'max_hierarchy']:
            if key in run.config:
                print(f"  {key}: {run.config[key]}")

fig, ax = plt.subplots(figsize=(6, 3.5), dpi=300)

for idx, run in enumerate(sweep.runs):
    # Fetch history first to determine label
    history = run.history(keys=["episode/score", "_step"])
    
    if history.empty or "episode/score" not in history.columns:
        print(f"Skipping run {run.name} (no episode/score data)")
        continue
    
    # Clean and sort data
    df = history.dropna(subset=["episode/score"]).sort_values("_step")
    
    if df.empty:
        print(f"Skipping run {run.name} (all NaN)")
        continue
    
    # Get max_hierarchy parameter for labeling
    if 'max_hierarchy' in run.config:
        label = f"RSSM (max_hierarchy={run.config['max_hierarchy']})"
    else:
        max_step = df["_step"].max()
        label = f"RSSM ({int(max_step/1000)}k steps)"
    
    x = df["_step"] / 1000  # thousands of steps
    y = df["episode/score"]
    
    # Smooth the curve
    window = 20
    y_smooth = y.rolling(window=window, min_periods=1).mean()
    
    # Plot with label
    ax.plot(x, y_smooth, linewidth=1.4, label=label, alpha=0.8)

ax.set_xlabel("Env. Steps (×10³)", fontsize=9)
ax.set_ylabel("Episode Return", fontsize=9)
ax.legend(fontsize=7, loc="best")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="both", labelsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(f"{output_dir}/sweep-episode-scores.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"✓ Saved: {output_dir}/sweep-episode-scores.png")

print("\n✓ Episode scores visualization complete!")
