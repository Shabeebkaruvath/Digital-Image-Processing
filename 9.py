import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# Load image
image = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)

# ============ HELPER FUNCTIONS ============
def dft(img):
    return np.fft.fftshift(np.fft.fft2(img))

def idft(f):
    return np.real(np.fft.ifft2(np.fft.ifftshift(f)))

def freq_lowpass(img, size=30):
    """Low-pass filter in frequency domain"""
    f = dft(img)
    rows, cols = img.shape
    mask = np.zeros((rows, cols))
    mask[rows//2-size:rows//2+size, cols//2-size:cols//2+size] = 1
    return idft(f * mask)

def freq_highpass(img, size=30):
    """High-pass filter in frequency domain"""
    return img - freq_lowpass(img, size) + np.mean(img)

# ============ APPLY FILTERS WITH TIMING ============
# Spatial Smoothing
t1 = time.time()
spatial_smooth = cv2.GaussianBlur(image, (15, 15), 0)
time_s_smooth = (time.time() - t1) * 1000

# Frequency Smoothing
t2 = time.time()
freq_smooth = freq_lowpass(image, size=30)
time_f_smooth = (time.time() - t2) * 1000

# Spatial Sharpening
t3 = time.time()
blurred = cv2.GaussianBlur(image, (5, 5), 0)
spatial_sharp = np.clip(image + 1.5 * (image - blurred), 0, 255)
time_s_sharp = (time.time() - t3) * 1000

# Frequency Sharpening
t4 = time.time()
freq_sharp = np.clip(freq_highpass(image, size=30), 0, 255)
time_f_sharp = (time.time() - t4) * 1000

# ============ DISPLAY RESULTS ============
titles = ['Original',
          f'Spatial Smooth\n{time_s_smooth:.1f}ms',
          f'Freq Smooth\n{time_f_smooth:.1f}ms',
          f'Spatial Sharp\n{time_s_sharp:.1f}ms',
          f'Freq Sharp\n{time_f_sharp:.1f}ms']

images = [image, spatial_smooth, freq_smooth, spatial_sharp, freq_sharp]

plt.figure(figsize=(14, 6))
for i in range(5):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    color = 'green' if i in [0,1,3] else 'blue'
    plt.title(titles[i], fontsize=12, fontweight='bold', color=color)
    plt.axis('off')

plt.suptitle('Spatial vs Frequency Domain Filtering', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('spatial_vs_freq.png', dpi=100)
plt.show()

 