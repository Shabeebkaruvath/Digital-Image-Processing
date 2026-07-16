import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the image in grayscale
# Ensure your downloaded image is named 'image.png' and in the same folder
image = cv2.imread('image1.png', cv2.IMREAD_GRAYSCALE)

# 2. Convert to binary 
# Using THRESH_BINARY_INV assumes your objects are dark on a light background.
# If your objects are light on a dark background, use cv2.THRESH_BINARY instead.
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

# 3. Perform Connected Component Analysis
# num_labels: Total number of objects + 1 (background is label 0)
# labels: A matrix the same size as the image where each pixel is tagged with its object ID
# stats: Contains statistics like area, bounding boxes, etc.
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

# 4. Assign unique colors to each connected component
# Create an empty RGB image to hold the colored objects
colored_labels = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

# 5. Count objects and compute area
print(f"Total objects detected: {num_labels - 1}\n")
print(f"{'Object ID':<10} | {'Area (Pixels)':<15}")
print("-" * 28)

# Loop through each label (starting at 1 to ignore the background)
for i in range(1, num_labels):
    # Generate a random color for the object (R, G, B)
    color = np.random.randint(50, 255, size=3).tolist()
    
    # Apply the color to all pixels belonging to this specific label
    colored_labels[labels == i] = color
    
    # Extract the area from the stats array
    area = stats[i, cv2.CC_STAT_AREA]
    print(f"{i:<10} | {area:<15}")

# 6. Display the original, binary, and colored component images
plt.figure(figsize=(14, 5))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(binary, cmap='gray')
plt.title('Binary Image (Thresholded)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(colored_labels)
plt.title(f'Colored Components (Count: {num_labels - 1})')
plt.axis('off')

plt.tight_layout()
plt.show()