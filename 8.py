import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = cv2.imread('image1.png', cv2.IMREAD_GRAYSCALE)

# Convert to float32
float_img = np.float32(image)

# Compute 2D Fourier Transform
dft = cv2.dft(float_img, flags=cv2.DFT_COMPLEX_OUTPUT)

# Shift zero-frequency to center
dft_shift = np.fft.fftshift(dft)

# Calculate magnitude spectrum
magnitude = np.sqrt(dft_shift[:, :, 0]**2 + dft_shift[:, :, 1]**2)

# Log scaling for better visualization
magnitude_log = np.log1p(magnitude)

# Normalize to 0-255 for display
mag_normalized = cv2.normalize(magnitude_log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Display
plt.figure(figsize=(16, 4))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image', fontsize=13, fontweight='bold')
plt.axis('off')


plt.subplot(1, 3, 3)
plt.imshow(mag_normalized, cmap='gray')
plt.title('Magnitude (Log Scaled)', fontsize=13, fontweight='bold')
plt.axis('off')

plt.suptitle('2D Fourier Transform - Magnitude Spectrum', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('fourier_spectrum.png', dpi=100)
plt.show()

 