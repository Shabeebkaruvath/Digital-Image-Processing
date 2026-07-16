import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

# 1. Load the image in grayscale
# Make sure 'image.png' is in your working directory
image = cv2.imread('image1.png', cv2.IMREAD_GRAYSCALE)

# 2. Define LBP parameters
# Radius of the circle and number of points to consider around the central pixel
radius = 1
n_points = 8 * radius

# 3. Compute the LBP image
# Using the 'uniform' method reduces the number of patterns and improves histogram stability
lbp_image = local_binary_pattern(image, n_points, radius, method='uniform')

# 4. Compute the LBP histogram
# For uniform LBP, the maximum value is n_points + 1. 
# We calculate the histogram over the flattened LBP image.
n_bins = int(lbp_image.max() + 1)
hist, bins = np.histogram(lbp_image.ravel(), bins=n_bins, range=(0, n_bins), density=True)

# 5. Visualize the original image, LBP image, and the histogram
plt.figure(figsize=(15, 5))

# Plot Original Grayscale Image
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Grayscale Image')
plt.axis('off')

# Plot LBP Image
plt.subplot(1, 3, 2)
plt.imshow(lbp_image, cmap='gray')
plt.title('LBP Image (Texture Features)')
plt.axis('off')

# Plot LBP Histogram
plt.subplot(1, 3, 3)
# Create a bar chart for the histogram
plt.bar(bins[:-1], hist, width=0.8, color='teal', edgecolor='black')
plt.title('LBP Histogram')
plt.xlabel('LBP Pattern (Uniform)')
plt.ylabel('Frequency (Normalized)')
plt.xticks(range(n_bins))

plt.tight_layout()
plt.show()