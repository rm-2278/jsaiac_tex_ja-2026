import os
from PIL import Image
import numpy as np

# Analyze the structure of subgoal visualization
img_path = "media/pinpad/subactor-update-sweep/sweep-subgoal-temporal.png"

if os.path.exists(img_path):
    img = Image.open(img_path)
    img_array = np.array(img)
    
    print(f"Image shape: {img_array.shape}")
    print(f"Height: {img_array.shape[0]}, Width: {img_array.shape[1]}")
    
    # The image should have 5 time steps horizontally
    # Each time step should show a grid of subgoal visualizations
    # The grid has rows (hierarchy levels) and columns (different subgoals)
    
    # Let's try to understand the structure
    # If there are 5 time steps, each one takes up width/5
    width = img_array.shape[1]
    height = img_array.shape[0]
    
    time_step_width = width // 5
    print(f"\nEstimated width per time step: {time_step_width}")
    
    # Let's look at one time step (e.g., the first one)
    first_step = img_array[:, 0:time_step_width]
    
    # Save it for inspection
    Image.fromarray(first_step).save("media/pinpad/debug_first_step.png")
    print(f"Saved first step to media/pinpad/debug_first_step.png")
    print(f"First step shape: {first_step.shape}")
else:
    print(f"File not found: {img_path}")
