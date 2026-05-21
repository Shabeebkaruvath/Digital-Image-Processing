import cv2
import matplotlib.pyplot as plt

# Load image
img = cv2.imread("image1.png")

if img is None:
    print("ERROR: Image not found. Check filename/path.")
    exit()

# 1. Convert to Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Calculate Histogram
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

# 3. Display everything
plt.figure(figsize=(12, 4))

# Original
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off")

# Grayscale
plt.subplot(1, 3, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

# Histogram
plt.subplot(1, 3, 3)
plt.plot(hist, color="black")
plt.fill_between(range(256), hist.flatten(), alpha=0.4, color="gray")
plt.title("Grayscale Histogram")
plt.xlabel("Pixel Intensity (0–255)")
plt.ylabel("Frequency")
plt.xlim([0, 256])

plt.tight_layout()
plt.savefig("histogram_output.png")  # saves to same folder
plt.show()