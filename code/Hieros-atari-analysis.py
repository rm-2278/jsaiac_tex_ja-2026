import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
import os

# Create output directory
output_dir = "media/atari"
os.makedirs(output_dir, exist_ok=True)

# Initialize the API
api = wandb.Api()

# Fetch the sweep
sweep = api.sweep("rm2278-university-of-cambridge/Hieros-hieros/llr4r8er")

print(f"Sweep: {sweep.name}")
print(f"Found {len(sweep.runs)} runs")

# For freeway, use specific run from different sweep
freeway_run = api.run("rm2278-university-of-cambridge/Hieros-hieros/19ymhh01")
print(f"\nSpecial freeway run: {freeway_run.name}")

# Group runs by task
runs_by_task = {}
for run in sweep.runs:
    if 'task' in run.config:
        task = run.config['task']
        if task not in runs_by_task:
            runs_by_task[task] = []
        runs_by_task[task].append(run)
        print(f"Run {run.name}: task={task}, seed={run.config.get('seed', 'unknown')}")

# Add freeway run to the group
if 'atari_freeway' not in runs_by_task:
    runs_by_task['atari_freeway'] = []
runs_by_task['atari_freeway'].append(freeway_run)
print(f"Added freeway run: {freeway_run.name}")

print(f"\nFound {len(runs_by_task)} different tasks")

# =============================================================================
# 1. Episode/Score for each task (averaged over seeds)
# =============================================================================

for task, task_runs in runs_by_task.items():
    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=300)
    
    # Check max steps for each run to decide threshold
    run_max_steps = []
    for run in task_runs:
        try:
            history = run.history(keys=["episode/score", "_step"])
            df = history.dropna(subset=["episode/score"])
            if not df.empty:
                max_step = df["_step"].max()
                run_max_steps.append((run, max_step))
                print(f"  {run.name}: max_step={max_step}")
        except Exception as e:
            print(f"Error checking run {run.name}: {e}")
    
    if not run_max_steps:
        print(f"⚠ No valid runs for task {task}")
        continue
    
    # Find the maximum step across all runs
    overall_max = max(step for _, step in run_max_steps)
    print(f"Task {task}: overall_max={overall_max}")
    
    # Use all available runs to get meaningful std (no filtering)
    filtered_runs = [run for run, _ in run_max_steps]
    
    if not filtered_runs:
        print(f"⚠ No runs for task {task}")
        continue
    
    print(f"Using {len(filtered_runs)} runs for task {task}")
    
    # Collect all scores for this task
    all_scores = []
    
    for run in filtered_runs:
        history = run.history(keys=["episode/score", "_step"])
        
        if history.empty or "episode/score" not in history.columns:
            continue
        
        df = history.dropna(subset=["episode/score"]).sort_values("_step")
        if df.empty:
            continue
        
        all_scores.append(df)
    
    if not all_scores:
        print(f"No data for task {task}")
        continue
    
    # Combine all runs and compute mean/std
    # First, align all dataframes to common steps
    all_steps = sorted(set().union(*[set(df["_step"].values) for df in all_scores]))
    
    # For each run, interpolate to common steps
    interpolated_scores = []
    for df in all_scores:
        # Interpolate to common steps
        interp_values = np.interp(all_steps, df["_step"].values, df["episode/score"].values)
        interpolated_scores.append(interp_values)
    
    # Convert to array and compute statistics
    score_array = np.array(interpolated_scores)
    mean_scores = np.mean(score_array, axis=0)
    min_scores = np.min(score_array, axis=0)
    max_scores = np.max(score_array, axis=0)
    
    x = np.array(all_steps) / 1000  # thousands of steps
    
    # Smooth mean, min, and max curves
    df_mean = pd.DataFrame({'score': mean_scores, 'min': min_scores, 'max': max_scores})
    window = 20
    y_smooth = df_mean['score'].rolling(window=window, min_periods=1).mean()
    min_smooth = df_mean['min'].rolling(window=window, min_periods=1).mean()
    max_smooth = df_mean['max'].rolling(window=window, min_periods=1).mean()
    
    # Plot with shaded min/max
    ax.plot(x, y_smooth, linewidth=1.4, label=f"{task} (n={len(filtered_runs)})", alpha=0.8)
    ax.fill_between(x, min_smooth, max_smooth, alpha=0.2)
    
    ax.set_xlabel("Env. Steps (×10³)", fontsize=9)
    ax.set_ylabel("Episode Return", fontsize=9)
    ax.set_title(f"Task: {task}", fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # Sanitize filename
    safe_task_name = task.replace('/', '_').replace(' ', '_')
    fig.savefig(f"{output_dir}/{safe_task_name}-scores.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {output_dir}/{safe_task_name}-scores.png")

# =============================================================================
# 2. Policy Image Visualization (temporal progression)
# =============================================================================

# First, check what image keys are available
print("\nChecking available media keys...")
sample_run = sweep.runs[0]
history = sample_run.history()
image_keys = [col for col in history.columns if 'image' in col.lower() or 'policy' in col.lower() or 'video' in col.lower() or 'report' in col.lower()]
print(f"Available media keys: {image_keys}")

for task, task_runs in runs_by_task.items():
    # Try different possible keys for policy visualization
    possible_keys = ['train_stats/policy_image', 'eval/policy', 'report/policy', 'report/openl_image', 'report/openl', 'eval/openl_image']
    
    selected_run = None
    selected_key = None
    max_steps = 0
    
    # Prefer run with most steps and valid policy images
    for key in possible_keys:
        for run in task_runs:
            try:
                history = run.history(keys=[key, "_step"])
                if not history.empty and key in history.columns:
                    valid_data = history[key].notna()
                    if valid_data.any():
                        # Get the max step for this run
                        run_max_step = history[valid_data]["_step"].max()
                        # Select run with most steps
                        if run_max_step > max_steps:
                            selected_run = run
                            selected_key = key
                            max_steps = run_max_step
            except:
                continue
        if selected_run:
            break
    
    if selected_run is None or selected_key is None:
        print(f"⚠ No runs found with policy/image data for task {task}")
        continue
    
    print(f"Using run {selected_run.name} with key '{selected_key}' for visualization (task: {task})")
    
    # Fetch the full history
    history = selected_run.history(keys=[selected_key, "_step"])
    
    # Filter out NaN rows and sort
    df = history.dropna(subset=[selected_key]).sort_values("_step")
    
    if df.empty:
        print(f"⚠ No valid {selected_key} data found for task {task}")
        continue
    
    # Select specific steps: exactly 100k, 200k, 300k, 400k (4 images)
    target_steps = [100000, 200000, 300000, 400000]
    sampled_rows = []
    
    for target in target_steps:
        # Find the exact step or closest available step
        if target in df["_step"].values:
            # Exact match
            sampled_rows.append(df[df["_step"] == target].iloc[0])
        else:
            # Find closest step
            closest_idx = (df["_step"] - target).abs().idxmin()
            sampled_rows.append(df.loc[closest_idx])
    
    if not sampled_rows:
        print(f"⚠ No sampled rows for task {task}")
        continue
    
    sampled = pd.DataFrame(sampled_rows)
    n_samples = len(sampled)
    
    # Create a figure with 1 row × n_samples columns (horizontal layout)
    fig, axes = plt.subplots(1, n_samples, figsize=(3*n_samples, 3), dpi=300)
    if n_samples == 1:
        axes = [axes]
    
    for idx, (ax, (_, row)) in enumerate(zip(axes, sampled.iterrows())):
        media_obj = row[selected_key]
        
        try:
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
            ax.set_title(f"{step_thousands:.0f}k", fontsize=9)
            ax.axis("off")
        except Exception as e:
            print(f"Error loading image for step {row['_step']}: {e}")
            ax.axis("off")
    
    plt.suptitle(f"Policy Evolution - {task}", fontsize=11, fontweight='bold')
    plt.tight_layout()
    safe_task_name = task.replace('/', '_').replace(' ', '_')
    fig.savefig(f"{output_dir}/{safe_task_name}-policy-temporal.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {output_dir}/{safe_task_name}-policy-temporal.png")

print("\n✓ All Atari visualizations complete!")
