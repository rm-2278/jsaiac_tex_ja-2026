import os
from PIL import Image
import numpy as np

# Process subgoal visualization images
image_paths = [
    "media/pinpad/subactor-update-sweep/sweep-subgoal-temporal.png",
    "media/pinpad/reward-ratio-sweep/sweep-subgoal-temporal.png",
    "media/pinpad/entropy-sweep/sweep-subgoal-temporal.png",
    "media/pinpad/reward-design-sweep/sweep-subgoal-temporal.png"
]

for img_path in image_paths:
    if not os.path.exists(img_path):
        print(f"⚠ File not found: {img_path}")
        continue
    
    print(f"Processing: {img_path}")
    
    # Load image
    img = Image.open(img_path)
    img_array = np.array(img)
    
    # Get dimensions
    height, width = img_array.shape[:2]
    
    # The image shows 5 time steps horizontally (1k, 100k, 200k, 300k, 400k)
    # Each time step section contains a grid showing different hierarchy levels (rows)
    # and different subgoal positions (columns within each row)
    
    # We want to extract only the leftmost 2 columns from WITHIN each time step
    # This means we need to:
    # 1. Divide the image into 5 time step sections
    # 2. Within each section, identify the grid structure
    # 3. Extract only the leftmost 2 columns of grid cells
    
    n_time_steps = 5
    time_step_width = width // n_time_steps
    
    # Looking at the structure, each time step has multiple hierarchy levels (rows)
    # and multiple positions (columns). The hierarchy levels are stacked vertically.
    # Let's assume there are 3 hierarchy levels
    n_hierarchy = 3
    level_height = height // n_hierarchy
    
    # Within each level, there are multiple grid cell columns
    # Estimate: about 5-6 columns per level per time step
    n_cols_per_level = 5
    col_width = time_step_width // n_cols_per_level
    
    # Extract leftmost 2 columns from each hierarchy level within each time step
    extracted_sections = []
    
    for step_idx in range(n_time_steps):
        step_x_start = step_idx * time_step_width
        
        # For this time step, extract from all hierarchy levels
        level_sections = []
        for level_idx in range(n_hierarchy):
            level_y_start = level_idx * level_height
            level_y_end = (level_idx + 1) * level_height
            
            # Extract leftmost 2 columns
            extract_width = 2 * col_width
            step_x_end = step_x_start + extract_width
            
            level_section = img_array[level_y_start:level_y_end, step_x_start:step_x_end]
            level_sections.append(level_section)
        
        # Stack levels vertically for this time step
        step_combined = np.concatenate(level_sections, axis=0)
        extracted_sections.append(step_combined)
    
    # Concatenate time steps horizontally
    compressed_img = np.concatenate(extracted_sections, axis=1)
    
    # Convert back to PIL Image
    compressed_pil = Image.fromarray(compressed_img)
    
    # Save with -compressed suffix
    output_path = img_path.replace(".png", "-compressed.png")
    compressed_pil.save(output_path, dpi=(300, 300))
    
    print(f"  Original size: {img_array.shape}")
    print(f"  Compressed size: {compressed_img.shape}")
    print(f"  Reduction: width={compressed_img.shape[1]/img_array.shape[1]:.1%}, height={compressed_img.shape[0]/img_array.shape[0]:.1%}")
    print(f"  ✓ Saved: {output_path}")

print("\n✓ All subgoal visualizations compressed!")
