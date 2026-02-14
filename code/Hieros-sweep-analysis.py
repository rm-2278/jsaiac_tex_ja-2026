import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
import os

# Create output directory
output_dir = "media/pinpad/subactor-update-sweep"
os.makedirs(output_dir, exist_ok=True)

# Initialize the API
api = wandb.Api()

# Fetch the sweep
sweep = api.sweep("rm2278-university-of-cambridge/Hieros-hieros/w3isl3qy")

print(f"Sweep: {sweep.name}")
print(f"Found {len(sweep.runs)} runs")

# =============================================================================
# 1. Episode/Score for all runs, labeled by subactor-update-every
# =============================================================================

fig, ax = plt.subplots(figsize=(6, 3.5), dpi=300)

for run in sweep.runs:
    # Get the config parameter
    subactor_update_every = run.config.get("subactor_update_every", "unknown")
    
    # Fetch history
    history = run.history(keys=["episode/score", "_step"])
    
    if history.empty or "episode/score" not in history.columns:
        print(f"Skipping run {run.name} (no episode/score data)")
        continue
    
    # Clean and sort data
    df = history.dropna(subset=["episode/score"]).sort_values("_step")
    
    if df.empty:
        print(f"Skipping run {run.name} (all NaN)")
        continue
    
    x = df["_step"] / 1000  # thousands of steps
    y = df["episode/score"]
    
    # Smooth the curve
    window = 20
    y_smooth = y.rolling(window=window, min_periods=1).mean()
    
    # Plot with label
    label = f"update-every={subactor_update_every}"
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

# =============================================================================
# 2. Subgoal Visualization (temporal progression from left to right)
# =============================================================================

# Pick the first run that has report/subgoal_visualization data
selected_run = None
for run in sweep.runs:
    # Check if this run has any report/subgoal_visualization media
    try:
        history = run.history(keys=["report/subgoal_visualization"])
        if not history.empty and "report/subgoal_visualization" in history.columns:
            if history["report/subgoal_visualization"].notna().any():
                selected_run = run
                break
    except:
        continue

if selected_run is None:
    print("⚠ No runs found with report/subgoal_visualization data")
else:
    print(f"Using run {selected_run.name} for report/subgoal_visualization")
    
    # Fetch the full history with report/subgoal_visualization
    history = selected_run.history(keys=["report/subgoal_visualization", "_step"])
    
    # Filter out NaN rows and sort
    df = history.dropna(subset=["report/subgoal_visualization"]).sort_values("_step")
    
    # Skip the first step (uniform initialization)
    if len(df) > 1:
        df = df.iloc[1:]
    
    if not df.empty:
        # Select specific steps: 15k and 300k only (typical representative examples)
        target_steps = [15000, 300000]
        sampled_rows = []
        
        for target in target_steps:
            # Find the closest available step
            closest_idx = (df["_step"] - target).abs().idxmin()
            sampled_rows.append(df.loc[closest_idx])
        
        sampled = pd.DataFrame(sampled_rows)
        n_samples = len(sampled)
        
        # Create a figure with 2 rows × 1 column (vertical layout)
        fig, axes = plt.subplots(2, 1, figsize=(6, 4), dpi=300)
        axes = axes.flatten()
        
        for idx, (ax, (_, row)) in enumerate(zip(axes, sampled.iterrows())):
            # W&B stores media as wandb.data_types.Image objects
            # We need to download the actual image
            media_obj = row["report/subgoal_visualization"]
            
            if hasattr(media_obj, "_image"):
                # It's a wandb.Image, get the PIL image
                img = media_obj._image
            elif isinstance(media_obj, dict) and "path" in media_obj:
                # Fetch from wandb file
                file_path = media_obj["path"]
                # Download the file to a temporary location
                file_obj = selected_run.file(file_path)
                downloaded_path = file_obj.download(replace=True).name
                img = Image.open(downloaded_path)
            else:
                # Try to render as-is
                img = media_obj
            
            ax.imshow(img)
            step_thousands = row['_step'] / 1000
            ax.set_title(f"Step {step_thousands:.0f}k", fontsize=7)
            ax.axis("off")
        
        # Hide unused subplots
        for idx in range(n_samples, len(axes)):
            axes[idx].axis("off")
        
        plt.tight_layout()
        fig.savefig(f"{output_dir}/sweep-subgoal-temporal.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ Saved: {output_dir}/sweep-subgoal-temporal.png")
    else:
        print("⚠ No valid report/subgoal_visualization data found")

# =============================================================================
# 3. Position Heatmap (temporal progression)
# =============================================================================

selected_run = None
for run in sweep.runs:
    try:
        history = run.history(keys=["exploration/position_heatmap"])
        if not history.empty and "exploration/position_heatmap" in history.columns:
            if history["exploration/position_heatmap"].notna().any():
                selected_run = run
                break
    except:
        continue

if selected_run is None:
    print("⚠ No runs found with exploration/position_heatmap data")
else:
    print(f"Using run {selected_run.name} for position_heatmap")
    
    history = selected_run.history(keys=["exploration/position_heatmap", "_step"])
    df = history.dropna(subset=["exploration/position_heatmap"]).sort_values("_step")
    
    # Skip the first step (uniform initialization)
    if len(df) > 1:
        df = df.iloc[1:]
    
    if not df.empty:
        # Select specific steps: 1k, 100k, 200k, 300k, 400k
        target_steps = [1000, 100000, 200000, 300000, 400000]
        sampled_rows = []
        
        for target in target_steps:
            # Find the closest available step
            closest_idx = (df["_step"] - target).abs().idxmin()
            sampled_rows.append(df.loc[closest_idx])
        
        sampled = pd.DataFrame(sampled_rows)
        n_samples = len(sampled)
        
        fig, axes = plt.subplots(2, 3, figsize=(9, 6), dpi=300)
        axes = axes.flatten()
        
        for idx, (ax, (_, row)) in enumerate(zip(axes, sampled.iterrows())):
            media_obj = row["exploration/position_heatmap"]
            
            if hasattr(media_obj, "_image"):
                img = media_obj._image
            elif isinstance(media_obj, dict) and "path" in media_obj:
                file_path = media_obj["path"]
                file_obj = selected_run.file(file_path)
                downloaded_path = file_obj.download(replace=True).name
                img = Image.open(downloaded_path)
            else:
                img = media_obj
            
            ax.imshow(img)
            step_thousands = row['_step'] / 1000
            ax.set_title(f"Step {step_thousands:.0f}k", fontsize=7)
            ax.axis("off")
        
        # Hide unused subplots
        for idx in range(n_samples, len(axes)):
            axes[idx].axis("off")
        
        plt.tight_layout()
        fig.savefig(f"{output_dir}/sweep-heatmap-temporal.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ Saved: {output_dir}/sweep-heatmap-temporal.png")
    else:
        print("⚠ No valid position_heatmap data found")

print("\n✓ All visualizations complete!")
