import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread("image.png")

if img is None:
    print("ERROR: Image not found.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Add salt & pepper noise (so smoothing effect is visible)
noise = np.zeros(gray.shape, np.uint8)
cv2.randu(noise, 0, 255)
noisy = cv2.add(gray, noise // 4)

# Apply average filter with different kernel sizes
k3 = cv2.blur(noisy, (3, 3))
k5 = cv2.blur(noisy, (5, 5))
k7 = cv2.blur(noisy, (7, 7))

# Plot
fig, axes = plt.subplots(1, 5, figsize=(18, 4))

images = [gray, noisy, k3, k5, k7]
titles = ["Original", "Noisy", "3×3 Avg", "5×5 Avg", "7×7 Avg"]

for ax, image, title in zip(axes, images, titles):
    ax.imshow(image, cmap="gray")
    ax.set_title(title, fontsize=13)
    ax.axis("off")

plt.suptitle("Average Filtering — Kernel Size Comparison", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("avg_filter_output.png", dpi=150)
plt.show()

print("Saved: avg_filter_output.png")