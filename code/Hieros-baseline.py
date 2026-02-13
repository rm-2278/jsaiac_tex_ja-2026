import wandb
import pandas as pd
import matplotlib.pyplot as plt

# 1. Initialize the API
api = wandb.Api()

# 2. Fetch the specific run
# Your URL: https://wandb.ai/rm2278-university-of-cambridge/dreamerv3/runs/fltyjyib
run = api.run("rm2278-university-of-cambridge/dreamerv3/fltyjyib")

# 3. Download the full history (all metrics logged over time)
# Note: This might take a few seconds depending on the run length
history = run.history()

# 4. Define metric groups (each subactor plotted together)
metric_groups = [
	("episode/score", "Episode Return", [("episode/score", None)]),
	("extrinsic_reward", "Extrinsic Reward", [
		("train/Subactor-0/imag_extrinsic_reward_mean", "Sub 0"),
		("train/Subactor-1/imag_extrinsic_reward_mean", "Sub 1"),
		("train/Subactor-2/imag_extrinsic_reward_mean", "Sub 2"),
	]),
	("subgoal_reward", "Subgoal Reward", [
		("train/Subactor-0/imag_subgoal_reward_mean", "Sub 0"),
		("train/Subactor-1/imag_subgoal_reward_mean", "Sub 1"),
		("train/Subactor-2/imag_subgoal_reward_mean", "Sub 2"),
	]),
	("novelty_reward", "Novelty Reward", [
		("train/Subactor-0/imag_novelty_reward_mean", "Sub 0"),
		("train/Subactor-1/imag_novelty_reward_mean", "Sub 1"),
		("train/Subactor-2/imag_novelty_reward_mean", "Sub 2"),
	]),
	("actor_entropy", "Actor Entropy", [
		("train/Subactor-0/actor_entropy", "Sub 0"),
		("train/Subactor-1/actor_entropy", "Sub 1"),
		("train/Subactor-2/actor_entropy", "Sub 2"),
	]),
	("model_loss", "Model Loss", [
		("train/Subactor-0/model_loss", "Sub 0"),
		("train/Subactor-1/model_loss", "Sub 1"),
		("train/Subactor-2/model_loss", "Sub 2"),
	]),
]

# 5. Create a multi-panel figure with 2 columns per row (3x2 grid)
n_plots = len(metric_groups)
n_cols = 2
n_rows = 3
fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.0, 6.0), dpi=300)
axes = axes.flatten()

window = 20  # moving-average window for smoothing
colors = ['tab:blue', 'tab:orange', 'tab:green']

for plot_idx, (group_key, ylabel, metrics_list) in enumerate(metric_groups):
	ax = axes[plot_idx]
	
	# Plot each metric in the group
	for metric_idx, (key, label) in enumerate(metrics_list):
		if key not in history.columns:
			print(f"Warning: Metric '{key}' not found, skipping")
			continue
		
		df = history.dropna(subset=[key]).sort_values("_step")
		if df.empty:
			continue

		x = df["_step"] / 1e6  # environment steps in millions
		smooth = df[key].rolling(window=window, min_periods=1).mean()

		color = colors[metric_idx % len(colors)]
		
		# light raw curve
		ax.plot(x, df[key], color=color, linewidth=0.6, alpha=0.3)
		# smoothed main curve
		if label:
			ax.plot(x, smooth, color=color, linewidth=1.4, label=label)
		else:
			ax.plot(x, smooth, color=color, linewidth=1.4)

	ax.set_xlabel("Env. Steps (×10⁶)", fontsize=7)
	ax.set_ylabel(ylabel, fontsize=7)

	# Add legend if there are multiple lines
	if len(metrics_list) > 1 and any(label for _, label in metrics_list):
		ax.legend(fontsize=6, loc="best")

	# clean style
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	ax.tick_params(axis="both", labelsize=6)

# Hide unused subplots
for idx in range(n_plots, len(axes)):
	axes[idx].axis("off")

fig.tight_layout(w_pad=0.8, h_pad=0.8)

# Save the plot for your paper (tight, high-res)
fig.savefig("media/pinpad/Hieros-baseline.png", dpi=300, bbox_inches="tight")
plt.close(fig)