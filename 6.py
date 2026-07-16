import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = cv2.imread('image1.png', cv2.IMREAD_GRAYSCALE)

# Apply Laplacian filter
laplacian = cv2.Laplacian(image, cv2.CV_64F)
laplacian_abs = cv2.convertScaleAbs(laplacian)  # Convert to absolute values

# Edge enhanced image = Original + Edges
enhanced = cv2.addWeighted(image, 1, laplacian_abs, 0.5, 0)

# Display
titles = ['Original', 'Laplacian Edges', 'Edge Enhanced']
images = [image, laplacian_abs, enhanced]

plt.figure(figsize=(14, 5))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i], fontsize=13, fontweight='bold')
    plt.axis('off')

plt.suptitle('Laplacian Filter for Edge Detection', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('laplacian_edges.png', dpi=100)
plt.show()

 