import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the image
image = cv2.imread('image1.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 2. Convert to binary image
# Use THRESH_BINARY_INV if your objects are dark against a light background
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 3. Detect objects by finding contours
# RETR_EXTERNAL retrieves only the extreme outer contours (ignores holes inside objects)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 4. Count the number of detected objects
# We filter out very small contours to avoid counting noise
valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]
object_count = len(valid_contours)

print(f"Total detected objects: {object_count}")

# 5. Draw Bounding Boxes
output_image = image.copy()

for contour in valid_contours:
    # Option A: Standard Upright Bounding Box (Drawn in Green)
    # This creates a straight rectangle regardless of object rotation
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Option B: Minimum Enclosing Rectangle (Drawn in Red)
    # This creates a rotated rectangle that tightly fits the object
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box) # Convert coordinates to integers
    cv2.drawContours(output_image, [box], 0, (255, 0, 0), 2)

# 6. Display the results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(binary, cmap='gray')
plt.title('Binary Image')
plt.axis('off')

plt.subplot(1, 2, 2)
# Convert BGR to RGB for correct matplotlib color display
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.title(f'Detected Objects: {object_count}\n(Green: Upright, Red: Minimum Enclosing)')
plt.axis('off')

plt.tight_layout()
plt.show()