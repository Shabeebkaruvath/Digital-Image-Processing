import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

# Unsharp Masking function
def unsharp_mask(img, sigma=1.0, weight=1.0):
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    mask = img - blurred          # Edge/detail mask
    sharpened = img + weight * mask
    return np.clip(sharpened, 0, 255).astype(np.uint8)

# Apply with different weights
weights = [0.5, 1.0, 1.5]
results = [unsharp_mask(image, sigma=1.0, ) for w in weights]

# Display
titles = ['Original'] + [f'Weight = {w}' for w in weights]
images = [image] + results

plt.figure(figsize=(12, 8))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i], fontsize=13, fontweight='bold')
    plt.axis('off')

plt.suptitle('Unsharp Masking with Different Weights', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('unsharp_masking.png', dpi=100)
plt.show()

 