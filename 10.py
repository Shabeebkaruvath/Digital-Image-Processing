import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

# ============ GLOBAL THRESHOLDING (Manual) ============
# Try different manual thresholds
_, th50 = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)
_, th127 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
_, th200 = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)

# ============ OTSU'S AUTOMATIC THRESHOLDING ============
otsu_thresh, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# ============ DISPLAY RESULTS ============
titles = ['Original',
          f'Global (T=50)',
          f'Global (T=127)',
          f'Global (T=200)',
          f"Otsu (T={otsu_thresh})"]

images = [image, th50, th127, th200, otsu]
colors = ['black'] * 4 + ['green']

plt.figure(figsize=(14, 6))
for i in range(5):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i], fontsize=12, fontweight='bold', color=colors[i])
    plt.axis('off')

plt.suptitle('Global vs Otsu Thresholding', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('thresholding_comparison.png', dpi=100)
plt.show()

# ============ HISTOGRAM WITH THRESHOLDS ============
plt.figure(figsize=(10, 4))
plt.hist(image.ravel(), 256, [0, 256], color='gray', alpha=0.7)
plt.axvline(x=50, color='red', linestyle='--', label='T=50')
plt.axvline(x=127, color='orange', linestyle='--', label='T=127')
plt.axvline(x=200, color='purple', linestyle='--', label='T=200')
plt.axvline(x=otsu_thresh, color='green', linewidth=2, label=f'Otsu={otsu_thresh}')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.title('Histogram with Threshold Lines')
plt.legend()
plt.tight_layout()
plt.savefig('threshold_histogram.png', dpi=100)
plt.show()

# ============ PRINT COMPARISON ============
print(f"\n{'Method':<25} {'Threshold':<12} {'Type':<15}")
print("-" * 52)
print(f"{'Global Threshold':<25} {'50':<12} {'Manual':<15}")
print(f"{'Global Threshold':<25} {'127':<12} {'Manual (mid)':<15}")
print(f"{'Global Threshold':<25} {'200':<12} {'Manual':<15}")
print(f"{'Otsu Threshold':<25} {otsu_thresh:<12} {'Automatic ✅':<15}")

print("\n" + "="*55)
print("COMPARISON SUMMARY")
print("="*55)
print("""
┌──────────────────┬──────────────────────────────────┐
│ Global (Manual)  │ - Fixed threshold value          │
│                  │ - Need to guess T value          │
│                  │ - Fails if lighting changes      │
├──────────────────┼──────────────────────────────────┤
│ Otsu (Automatic) │ ✅ Finds optimal T automatically │
│                  │ ✅ Maximizes between-class variance│
│                  │ ✅ Works for bimodal histograms  │
│                  │ ❌ Fails for single-peak images  │
└──────────────────┴──────────────────────────────────┘

HOW OTSU WORKS:
1. Compute histogram of image
2. Try all thresholds (0-255)
3. For each T: calculate variance between classes
4. Pick T that MAXIMIZES between-class variance
5. Result: Best separation of foreground/background
""")