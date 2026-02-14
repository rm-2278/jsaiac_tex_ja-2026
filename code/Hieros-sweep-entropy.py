import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
import os

# Create output directory
output_dir = "media/pinpad/entropy-sweep"
os.makedirs(output_dir, exist_ok=True)

# Initialize the API
api = wandb.Api()

# Fetch the sweep for actor entropy experiments
sweep = api.sweep("rm2278-university-of-cambridge/Hieros-hieros/zd4mp7ve")

print(f"Sweep: {sweep.name}")
print(f"Found {len(sweep.runs)} runs")

# =============================================================================
# 1. Episode/Score for all runs, labeled by actor_entropy
# =============================================================================

fig, ax = plt.subplots(figsize=(6, 3.5), dpi=300)

for run in sweep.runs:
    # Get the config parameter
    actor_entropy = run.config.get("actor_entropy", "unknown")
    
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
    label = f"entropy={actor_entropy}"
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
# 2. Subgoal Visualization (all seeds at 400k step)
# =============================================================================

# Collect all runs with subgoal_visualization
runs_with_subgoal = []
for run in sweep.runs:
    try:
        history = run.history(keys=["report/subgoal_visualization", "_step"])
        if not history.empty and "report/subgoal_visualization" in history.columns:
            if history["report/subgoal_visualization"].notna().any():
                runs_with_subgoal.append(run)
    except:
        continue

if not runs_with_subgoal:
    print("⚠ No runs found with report/subgoal_visualization data")
else:
    print(f"Found {len(runs_with_subgoal)} runs with subgoal_visualization")
    
    # Collect images from all runs at 400k step
    target_step = 400000
    images_data = []
    
    for run in runs_with_subgoal:
        history = run.history(keys=["report/subgoal_visualization", "_step"])
        df = history.dropna(subset=["report/subgoal_visualization"]).sort_values("_step")
        
        if df.empty:
            continue
        
        # Find closest to 400k
        closest_idx = (df["_step"] - target_step).abs().idxmin()
        row = df.loc[closest_idx]
        
        # Get actor_entropy for labeling
        label = run.config.get("actor_entropy", "unknown")
        images_data.append((run, row, label))
    
    if not images_data:
        print("⚠ No images found at 400k")
    else:
        n_samples = len(images_data)
        
        # Create a figure with max 2 images per row
        n_cols = min(2, n_samples)
        n_rows = (n_samples + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows), dpi=300)
        if n_samples == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (ax, (run, row, label)) in enumerate(zip(axes, images_data)):
            media_obj = row["report/subgoal_visualization"]
            
            if hasattr(media_obj, "_image"):
                img = media_obj._image
            elif isinstance(media_obj, dict) and "path" in media_obj:
                file_path = media_obj["path"]
                file_obj = run.file(file_path)
                downloaded_path = file_obj.download(replace=True).name
                img = Image.open(downloaded_path)
            else:
                img = media_obj
            
            ax.imshow(img)
            ax.set_title(f"actor_entropy={label}", fontsize=12)
            ax.axis("off")
        
        # Hide unused subplots
        for idx in range(n_samples, len(axes)):
            axes[idx].axis("off")
        
        plt.suptitle("Subgoal Visualization @ 400k steps", fontsize=11, fontweight='bold')
        plt.subplots_adjust(wspace=0.1, hspace=0.2)
        fig.savefig(f"{output_dir}/sweep-subgoal-temporal.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ Saved: {output_dir}/sweep-subgoal-temporal.png")

# =============================================================================
# 3. Position Heatmap (all seeds at 400k step)
# =============================================================================

# Collect all runs with position_heatmap
runs_with_heatmap = []
for run in sweep.runs:
    try:
        history = run.history(keys=["exploration/position_heatmap", "_step"])
        if not history.empty and "exploration/position_heatmap" in history.columns:
            if history["exploration/position_heatmap"].notna().any():
                runs_with_heatmap.append(run)
    except:
        continue

if not runs_with_heatmap:
    print("⚠ No runs found with exploration/position_heatmap data")
else:
    print(f"Found {len(runs_with_heatmap)} runs with position_heatmap")
    
    # Collect images from all runs at 400k step
    target_step = 400000
    images_data = []
    
    for run in runs_with_heatmap:
        history = run.history(keys=["exploration/position_heatmap", "_step"])
        df = history.dropna(subset=["exploration/position_heatmap"]).sort_values("_step")
        
        if df.empty:
            continue
        
        # Find closest to 400k
        closest_idx = (df["_step"] - target_step).abs().idxmin()
        row = df.loc[closest_idx]
        
        # Get actor_entropy for labeling
        label = run.config.get("actor_entropy", "unknown")
        images_data.append((run, row, label))
    
    if not images_data:
        print("⚠ No images found at 400k")
    else:
        n_samples = len(images_data)
        
        # Create a figure with max 3 images per row
        n_cols = min(3, n_samples)
        n_rows = (n_samples + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), dpi=300)
        if n_samples == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (ax, (run, row, label)) in enumerate(zip(axes, images_data)):
            media_obj = row["exploration/position_heatmap"]
            
            if hasattr(media_obj, "_image"):
                img = media_obj._image
            elif isinstance(media_obj, dict) and "path" in media_obj:
                file_path = media_obj["path"]
                file_obj = run.file(file_path)
                downloaded_path = file_obj.download(replace=True).name
                img = Image.open(downloaded_path)
            else:
                img = media_obj
            
            ax.imshow(img)
            ax.set_title(f"actor_entropy={label}", fontsize=12)
            ax.axis("off")
        
        plt.suptitle("Position Heatmap @ 400k steps", fontsize=11, fontweight='bold')
        plt.tight_layout()
        fig.savefig(f"{output_dir}/sweep-heatmap-temporal.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ Saved: {output_dir}/sweep-heatmap-temporal.png")

print("\n✓ All visualizations complete!")
