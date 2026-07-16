import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.feature import graycomatrix, graycoprops

# 1. Load the grayscale image
# Ensure 'image.png' is in your directory and is large enough (e.g., > 200x200 pixels)
image = cv2.imread('image1.png', cv2.IMREAD_GRAYSCALE)

# 2. Define two different regions (patches) for comparison
# Adjust these coordinates based on your specific image to capture different textures
# Format: image[y_start:y_end, x_start:x_end]
patch_size = 50
y1, x1 = 20, 20      # Coordinates for Region 1 (e.g., a smooth background area)
y2, x2 = 100, 100    # Coordinates for Region 2 (e.g., a highly textured object area)

patch1 = image[y1:y1+patch_size, x1:x1+patch_size]
patch2 = image[y2:y2+patch_size, x2:x2+patch_size]

# 3. Define a helper function to compute GLCM and extract features
def get_glcm_features(patch):
    # Compute the GLCM
    # distances=[1]: looks at the immediate neighbor
    # angles=[0]: looks in the horizontal direction (0 radians)
    # levels=256: standard grayscale depth
    glcm = graycomatrix(patch, distances=[1], angles=[0], levels=256, 
                        symmetric=True, normed=True)
    
    # Extract the 4 required features
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    
    return contrast, correlation, energy, homogeneity

# 4. Compute features for both regions
features1 = get_glcm_features(patch1)
features2 = get_glcm_features(patch2)

# 5. Print the comparison
print("=" * 60)
print(f"{'Feature':<15} | {'Region 1 (Patch 1)':<20} | {'Region 2 (Patch 2)'}")
print("=" * 60)
feature_names = ['Contrast', 'Correlation', 'Energy', 'Homogeneity']
for i, name in enumerate(feature_names):
    print(f"{name:<15} | {features1[i]:<20.4f} | {features2[i]:.4f}")
print("=" * 60)

# 6. Visualize the image and the selected regions
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# Show original image with bounding boxes
ax[0].imshow(image, cmap='gray')
rect1 = patches.Rectangle((x1, y1), patch_size, patch_size, linewidth=2, edgecolor='red', facecolor='none')
rect2 = patches.Rectangle((x2, y2), patch_size, patch_size, linewidth=2, edgecolor='blue', facecolor='none')
ax[0].add_patch(rect1)
ax[0].add_patch(rect2)
ax[0].set_title('Original Image with Regions')
ax[0].axis('off')

# Show Patch 1
ax[1].imshow(patch1, cmap='gray')
ax[1].set_title('Region 1 (Red Box)')
ax[1].axis('off')

# Show Patch 2
ax[2].imshow(patch2, cmap='gray')
ax[2].set_title('Region 2 (Blue Box)')
ax[2].axis('off')

plt.tight_layout()
plt.show()