import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Create a synthetic binary image (a white rectangle on a black background)
img = np.zeros((300, 300), dtype=np.uint8)
cv2.rectangle(img, (50, 50), (250, 250), 255, -1)

# 2. Add synthetic noise
noisy_img = img.copy()
# Add external noise (salt noise / white dots in background)
for _ in range(500):
    x, y = np.random.randint(0, 300, 2)
    noisy_img[y, x] = 255
# Add internal noise (pepper noise / black holes in the object)
for _ in range(500):
    x, y = np.random.randint(50, 250, 2)
    noisy_img[y, x] = 0

# 3. Define the structuring element (kernel)
# A 5x5 rectangular kernel is used here
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# 4. Apply Opening (Erosion followed by Dilation)
# Useful for removing background noise
opening = cv2.morphologyEx(noisy_img, cv2.MORPH_OPEN, kernel)

# 5. Apply Closing (Dilation followed by Erosion)
# Useful for closing small holes inside the foreground objects
closing = cv2.morphologyEx(noisy_img, cv2.MORPH_CLOSE, kernel)

# 6. Apply Opening THEN Closing (to remove both types of noise)
opened_then_closed = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

# Display the results
titles = ['Original Noisy Image', 'Opening (Removes outer noise)', 
          'Closing (Removes inner holes)', 'Opening + Closing']
images = [noisy_img, opening, closing, opened_then_closed]

plt.figure(figsize=(16, 4))
for i in range(4):
    plt.subplot(1, 4, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()