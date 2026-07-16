import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = cv2.imread('image1.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Add salt-and-pepper noise
def add_sp_noise(img, prob=0.05):
    noisy = img.copy()
    salt = np.random.random(img.shape) < prob/2
    pepper = np.random.random(img.shape) < prob/2
    noisy[salt] = 255
    noisy[pepper] = 0
    return noisy

noisy = add_sp_noise(gray)

# Apply filters
median = cv2.medianBlur(noisy, 3)
average = cv2.blur(noisy, (3, 3))

# Calculate PSNR
def psnr(orig, den):
    mse = np.mean((orig - den) ** 2)
    return 20 * np.log10(255 / np.sqrt(mse)) if mse != 0 else float('inf')

print(f"{'Filter':<20} {'PSNR (dB)':<10}")
print("-" * 30)
print(f"{'Noisy Image':<20} {psnr(gray, noisy):<10.2f}")
print(f"{'Median Filter (3x3)':<20} {psnr(gray, median):<10.2f}")
print(f"{'Average Filter (3x3)':<20} {psnr(gray, average):<10.2f}")

# Display results
titles = ['Original', 'Noisy (S&P)', 'Median Filter', 'Average Filter']
images = [gray, noisy, median, average]

plt.figure(figsize=(12, 8))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i], fontweight='bold')
    plt.axis('off')

plt.suptitle('Median vs Average Filter on Salt-and-Pepper Noise', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('comparison.png', dpi=100)
plt.show()
 