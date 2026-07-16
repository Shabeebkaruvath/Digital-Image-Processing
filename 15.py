import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the original image
# Note: For best results, use an image with overlapping objects (e.g., overlapping coins, cells, or pills).
img = cv2.imread('image1.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # For correct color display in Matplotlib
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Compute the Gradient Magnitude to detect object boundaries
# Using Sobel operators in X and Y directions
grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
grad_mag = cv2.magnitude(grad_x, grad_y)
# Normalize to 8-bit for visualization and processing
grad_mag = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# 3. Create a binary image to separate foreground from background
# OTSU's thresholding automatically finds the best threshold value
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Remove noise with morphological opening
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

# Identify the "sure background" by dilating the objects
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# 4. Handle Overlapping Objects using Distance Transform
# The distance transform calculates the distance from every foreground pixel to the nearest background pixel.
# Peaks in this map represent the deep centers of overlapping objects.
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

# Threshold the distance transform to get "sure foreground" (markers for distinct objects)
_, sure_fg = cv2.threshold(dist_transform, 0.6 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)

# Identify the unknown region (edges where objects overlap)
unknown = cv2.subtract(sure_bg, sure_fg)

# 5. Marker Labelling for the Watershed Algorithm
_, markers = cv2.connectedComponents(sure_fg)

# Add 1 to all labels so that the sure background is 1, not 0
markers = markers + 1

# Mark the unknown region with 0 (this is where watershed will figure out the boundaries)
markers[unknown == 255] = 0

# 6. Apply the Watershed Algorithm
# Modifies the markers array in-place. Boundary pixels will be set to -1.
markers = cv2.watershed(img, markers)

# 7. Display contours and regions on the original image
result_img = img_rgb.copy()
# Draw contours in Red (RGB: 255, 0, 0) where markers == -1
result_img[markers == -1] = [255, 0, 0]

# Display the pipeline
plt.figure(figsize=(16, 10))

plt.subplot(2, 3, 1)
plt.imshow(img_rgb)
plt.title('Original Overlapping Objects')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(grad_mag, cmap='gray')
plt.title('Gradient Magnitude (Boundaries)')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(binary, cmap='gray')
plt.title('Thresholded Binary')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(dist_transform, cmap='jet')
plt.title('Distance Transform Map')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(markers, cmap='tab20b')
plt.title('Watershed Segments (Regions)')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(result_img)
plt.title('Final Result: Contours Drawn')
plt.axis('off')

plt.tight_layout()
plt.show()