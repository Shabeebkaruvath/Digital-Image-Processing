import cv2
import matplotlib.pyplot as plt

# 1. Load image and convert to binary
image = cv2.imread('image1.png', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# 2. Define Structuring Elements (SEs) in a dictionary for easy looping
shapes = {
    'Square 3x3': cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    'Square 5x5': cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
    'Circle 3x3': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    'Circle 5x5': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
}

# 3. Process and Plot in a single concise loop
fig, axes = plt.subplots(3, 5, figsize=(16, 9))

# Show Original Image in the first column
axes[0, 0].imshow(binary, cmap='gray'); axes[0, 0].set_title('Original Binary')

for i, (name, se) in enumerate(shapes.items()):
    col = i + 1
    
    # Show Structuring Element
    axes[0, col].imshow(se, cmap='gray')
    axes[0, col].set_title(f'SE: {name}')
    
    # Apply and Show Dilation (Boundaries Grow)
    dilated = cv2.dilate(binary, se, iterations=1)
    axes[1, col].imshow(dilated, cmap='gray')
    axes[1, col].set_title(f'Dilate (Grows)\n{name}')
    
    # Apply and Show Erosion (Boundaries Shrink)
    eroded = cv2.erode(binary, se, iterations=1)
    axes[2, col].imshow(eroded, cmap='gray')
    axes[2, col].set_title(f'Erode (Shrinks)\n{name}')

# Turn off axis labels for all subplots
for ax in axes.flat:
    ax.axis('off')

plt.suptitle('Morphological Operations: Dilation vs Erosion', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('morphology_summary.png', dpi=100)
plt.show()