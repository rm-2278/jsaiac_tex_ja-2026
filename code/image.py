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

# 4. Inspect available metrics and pick one to plot
print("Available columns:")
print(list(history.columns))

metric_name = 'episode/score'  # <--- CHANGE THIS if you want a different metric

if metric_name not in history.columns:
	raise SystemExit(f"Metric '{metric_name}' not found. Pick one from the printed columns above.")

# Drop NaNs so we only plot valid points
df = history.dropna(subset=[metric_name])
if df.empty:
	raise SystemExit(f"Metric '{metric_name}' has only NaN values. Check which metric actually has data.")

# 5. Create the Matplotlib plot
plt.figure(figsize=(10, 6))

plt.plot(df['_step'], df[metric_name], label=metric_name, alpha=0.7, color='blue')

plt.title(f"Hieros: {metric_name}")
plt.xlabel("Steps")
plt.ylabel("Reward")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Save the plot for your paper
plt.savefig("../media/pinpad/Hieros-baseline.png", dpi=300)
plt.show()