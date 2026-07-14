import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

# ============ SOBEL OPERATOR ============
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobel_x, sobel_y)
sobel = cv2.convertScaleAbs(sobel)

# ============ PREWITT OPERATOR (Manual) ============
prewitt_x = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]])

prewitt_y = np.array([[-1, -1, -1],
                      [ 0,  0,  0],
                      [ 1,  1,  1]])

px = cv2.filter2D(image, cv2.CV_64F, prewitt_x)
py = cv2.filter2D(image, cv2.CV_64F, prewitt_y)
prewitt = cv2.magnitude(px, py)
prewitt = cv2.convertScaleAbs(prewitt)

# Display
titles = ['Original', 'Sobel', 'Prewitt']
images = [image, sobel, prewitt]

plt.figure(figsize=(14, 5))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i], fontsize=13, fontweight='bold')
    plt.axis('off')

plt.suptitle('Sobel vs Prewitt Edge Detection', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('sobel_vs_prewitt.png', dpi=100)
plt.show()

# Print kernels
print("SOBEL KERNEL (X-direction):     PREWITT KERNEL (X-direction):")
print("[-1  0  1]                      [-1  0  1]")
print("[-2  0  2]   ← center weighted  [-1  0  1]   ← equal weight")
print("[-1  0  1]                      [-1  0  1]")
print()
print("✅ Sobel: Better noise suppression (center weight = 2)")
print("✅ Prewitt: Simpler, equal weights")
print("✅ Both compute gradient: G = √(Gx² + Gy²)")