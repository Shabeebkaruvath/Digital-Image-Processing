import cv2
import numpy as np
import matplotlib.pyplot as plt

def region_growing(image, seed_points, tolerance=20):
    """
    Grows regions from a list of seed points.
    Each distinct region is assigned a unique random color.
    """
    h, w = image.shape
    # Create an empty color image to store the colored regions
    segmented_colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Global mask to keep track of pixels already assigned to any region
    global_visited = np.zeros((h, w), dtype=bool)
    
    # 8-connected neighborhood directions
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    for seed in seed_points:
        # Generate a distinct random color for the current seed's region
        region_color = np.random.randint(50, 255, size=3).tolist()
        
        # If the seed is already processed, skip it
        if global_visited[seed]:
            continue
            
        queue = [seed]
        seed_intensity = float(image[seed])
        
        # Local mask for the current region
        local_mask = np.zeros((h, w), dtype=bool)
        local_mask[seed] = True
        
        while queue:
            x, y = queue.pop(0)
            
            for dx, dy in neighbors:
                nx, ny = x + dx, y + dy
                
                # Check image boundaries
                if 0 <= nx < h and 0 <= ny < w:
                    if not global_visited[nx, ny] and not local_mask[nx, ny]:
                        # Homogeneity criterion: Compare pixel to the original seed intensity
                        if abs(float(image[nx, ny]) - seed_intensity) <= tolerance:
                            local_mask[nx, ny] = True
                            queue.append((nx, ny))
        
        # Apply the color to the segmented region
        segmented_colored[local_mask] = region_color
        # Update the global visited mask
        global_visited[local_mask] = True
        
    return segmented_colored

# 1. Load the grayscale image and resize it for faster computation
# A digital wallpaper with distinct contrasting elements works best here.
image = cv2.imread('image1.png', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (256, 256)) # Resizing speeds up the custom BFS loop

# 2. Define Seed Points (y, x coordinates)
# You will need to adjust these coordinates based on your specific image.
# For example, one seed in the foreground object, one in the background.
seeds = [(128, 128), (50, 50), (200, 200)] 

# 3. Apply Region Growing
# Tolerance defines how much the pixel intensity can deviate from the seed
tolerance_value = 15
segmented_result = region_growing(image, seeds, tolerance=tolerance_value)

# 4. Display the results
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
# Plot the initial seed points for visualization
for pt in seeds:
    plt.plot(pt[1], pt[0], 'rx', markersize=8) 
plt.title('Original Image with Seeds (Red X)')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(segmented_result, cv2.COLOR_BGR2RGB))
plt.title('Segmented Colored Regions')
plt.axis('off')

plt.tight_layout()
plt.show()