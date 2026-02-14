import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
import os
import cv2
import tempfile

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
    # Look for policy_image data at 400k steps
    policy_key = 'train_stats/policy_image'
    
    selected_run = None
    max_steps = 0
    
    # Find run with policy_image data closest to 400k
    for run in task_runs:
        try:
            history = run.history(keys=[policy_key, "_step"])
            if not history.empty and policy_key in history.columns:
                valid_data = history[policy_key].notna()
                if valid_data.any():
                    run_max_step = history[valid_data]["_step"].max()
                    if run_max_step > max_steps:
                        selected_run = run
                        max_steps = run_max_step
        except:
            continue
    
    if selected_run is None:
        print(f"⚠ No runs found with policy_image data for task {task}")
        continue
    
    print(f"Using run {selected_run.name} with policy_image data for visualization (task: {task})")
    
    # Fetch the full history with policy_image
    history = selected_run.history(keys=[policy_key, "_step"])
    
    # Filter out NaN rows and sort
    df = history.dropna(subset=[policy_key]).sort_values("_step")
    
    if df.empty:
        print(f"⚠ No valid policy_image data found for task {task}")
        continue
    
    # Find the policy_image closest to 400k steps
    target_step = 400000
    closest_idx = (df["_step"] - target_step).abs().idxmin()
    policy_row = df.loc[closest_idx]
    
    print(f"Using policy_image at step {policy_row['_step']}")
    
    # Extract policy_image object
    media_obj = policy_row[policy_key]
    
    # Download and process the policy_image (GIF/video)
    try:
        if isinstance(media_obj, dict) and "path" in media_obj:
            file_path = media_obj["path"]
            file_obj = selected_run.file(file_path)
            downloaded_path = file_obj.download(replace=True).name
        elif hasattr(media_obj, "_image"):
            # If it's a static image, convert to temporary file for cv2
            img = media_obj._image
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                img.save(tmp.name)
                downloaded_path = tmp.name
        else:
            print(f"⚠ Unexpected policy_image format for task {task}")
            continue
        
        # Try to open as video/GIF with cv2
        cap = cv2.VideoCapture(downloaded_path)
        
        if not cap.isOpened():
            print(f"⚠ Could not open policy_image as video for task {task}")
            continue
        
        # Get total frame count and frame dimensions
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Total frames in policy_image: {total_frames}, Frame size: {frame_width}x{frame_height}")
        
        # Extract 6 frames with 5-frame intervals for better motion visibility
        n_frames = 6
        frame_interval = 5
        
        if total_frames >= (n_frames - 1) * frame_interval + 1:
            # Take frames with 5-frame intervals from the end
            last_frame = total_frames - 1
            frame_indices = []
            for i in range(n_frames):
                frame_idx = last_frame - (n_frames - 1 - i) * frame_interval
                frame_indices.append(frame_idx)
        else:
            # If not enough frames, use evenly spaced frames
            frame_indices = np.linspace(0, total_frames - 1, min(n_frames, total_frames), dtype=int).tolist()
        
        print(f"Extracting frames {frame_indices} with {frame_interval}-frame intervals from {total_frames} total frames")
        
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        
        cap.release()
        
        if len(frames) == 0:
            print(f"⚠ Could not extract frames from policy_image for task {task}")
            continue
        
        n_samples = len(frames)
        
        # Create a figure for 6 frames (2x3 grid)
        if n_samples <= 3:
            fig, axes = plt.subplots(1, n_samples, figsize=(6*n_samples, 6), dpi=300)
        else:
            n_cols = 3
            n_rows = (n_samples + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows), dpi=300)
            axes = axes.flatten()
        
        if n_samples == 1:
            axes = [axes]
        
        for idx, (ax, frame) in enumerate(zip(axes, frames)):
            # Use nearest neighbor interpolation for pixelated games to maintain sharpness
            ax.imshow(frame, interpolation='nearest')
            ax.set_title(f"Frame {idx+1}", fontsize=14)
            ax.axis("off")
        
        # Hide unused subplots if any
        for idx in range(n_samples, len(axes)):
            axes[idx].axis("off")
        
        plt.suptitle(f"Single Environment Policy - {task}", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.tight_layout()
        safe_task_name = task.replace('/', '_').replace(' ', '_')
        fig.savefig(f"{output_dir}/{safe_task_name}-policy-temporal.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ Saved: {output_dir}/{safe_task_name}-policy-temporal.png")
        
    except Exception as e:
        print(f"⚠ Error processing video for task {task}: {e}")
        continue

print("\n✓ All Atari visualizations complete!")
