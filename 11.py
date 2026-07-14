import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load and convert to binary
image = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# ============ STRUCTURING ELEMENTS ============
se_square_3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
se_square_5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
se_circle_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
se_circle_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# ============ DILATION (Boundaries GROW) ============
dil_sq3 = cv2.dilate(binary, se_square_3, iterations=1)
dil_sq5 = cv2.dilate(binary, se_square_5, iterations=1)
dil_ci3 = cv2.dilate(binary, se_circle_3, iterations=1)
dil_ci5 = cv2.dilate(binary, se_circle_5, iterations=1)

# ============ EROSION (Boundaries SHRINK) ============
ero_sq3 = cv2.erode(binary, se_square_3, iterations=1)
ero_sq5 = cv2.erode(binary, se_square_5, iterations=1)
ero_ci3 = cv2.erode(binary, se_circle_3, iterations=1)
ero_ci5 = cv2.erode(binary, se_circle_5, iterations=1)

# ============ DISPLAY DILATION ============
plt.figure(figsize=(14, 10))

plt.subplot(3, 3, 1)
plt.imshow(binary, cmap='gray')
plt.title('Original Binary', fontsize=12, fontweight='bold')
plt.axis('off')

plt.subplot(3, 3, 2)
plt.imshow(dil_sq3, cmap='gray')
plt.title('Dilate: Square 3x3\n↑ Grows', fontsize=11, fontweight='bold', color='green')
plt.axis('off')

plt.subplot(3, 3, 3)
plt.imshow(dil_sq5, cmap='gray')
plt.title('Dilate: Square 5x5\n↑↑ Grows more', fontsize=11, fontweight='bold', color='green')
plt.axis('off')

plt.subplot(3, 3, 5)
plt.imshow(dil_ci3, cmap='gray')
plt.title('Dilate: Circle 3x3\n↑ Grows (smooth)', fontsize=11, fontweight='bold', color='green')
plt.axis('off')

plt.subplot(3, 3, 6)
plt.imshow(dil_ci5, cmap='gray')
plt.title('Dilate: Circle 5x5\n↑↑ Grows more (smooth)', fontsize=11, fontweight='bold', color='green')
plt.axis('off')

plt.subplot(3, 3, 4)
plt.text(0.5, 0.5, 'DILATION\n(Boundaries Grow)', ha='center', va='center',
         fontsize=16, fontweight='bold', color='green',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
plt.axis('off')

plt.suptitle('Dilation: Object Boundaries GROW', fontsize=15, fontweight='bold', color='green')
plt.tight_layout()
plt.savefig('dilation_results.png', dpi=100)
plt.show()

# ============ DISPLAY EROSION ============
plt.figure(figsize=(14, 10))

plt.subplot(3, 3, 1)
plt.imshow(binary, cmap='gray')
plt.title('Original Binary', fontsize=12, fontweight='bold')
plt.axis('off')

plt.subplot(3, 3, 2)
plt.imshow(ero_sq3, cmap='gray')
plt.title('Erode: Square 3x3\n↓ Shrinks', fontsize=11, fontweight='bold', color='red')
plt.axis('off')

plt.subplot(3, 3, 3)
plt.imshow(ero_sq5, cmap='gray')
plt.title('Erode: Square 5x5\n↓↓ Shrinks more', fontsize=11, fontweight='bold', color='red')
plt.axis('off')

plt.subplot(3, 3, 4)
plt.text(0.5, 0.5, 'EROSION\n(Boundaries Shrink)', ha='center', va='center',
         fontsize=16, fontweight='bold', color='red',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
plt.axis('off')

plt.subplot(3, 3, 5)
plt.imshow(ero_ci3, cmap='gray')
plt.title('Erode: Circle 3x3\n↓ Shrinks (smooth)', fontsize=11, fontweight='bold', color='red')
plt.axis('off')

plt.subplot(3, 3, 6)
plt.imshow(ero_ci5, cmap='gray')
plt.title('Erode: Circle 5x5\n↓↓ Shrinks more (smooth)', fontsize=11, fontweight='bold', color='red')
plt.axis('off')

plt.suptitle('Erosion: Object Boundaries SHRINK', fontsize=15, fontweight='bold', color='red')
plt.tight_layout()
plt.savefig('erosion_results.png', dpi=100)
plt.show()

# ============ DISPLAY STRUCTURING ELEMENTS ============
plt.figure(figsize=(10, 3))
ses = [se_square_3, se_square_5, se_circle_3, se_circle_5]
titles_se = ['Square 3x3', 'Square 5x5', 'Circle 3x3', 'Circle 5x5']

for i in range(4):
    plt.subplot(1, 4, i+1)
    plt.imshow(ses[i], cmap='gray')
    plt.title(titles_se[i], fontsize=11, fontweight='bold')
    plt.axis('off')

plt.suptitle('Structuring Elements Used', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('structuring_elements.png', dpi=100)
plt.show()

# ============ SUMMARY ============
print("\n" + "="*60)
print("DILATION vs EROSION SUMMARY")
print("="*60)
print(f"""
┌─────────────┬────────────────────┬────────────────────┐
│  Operation  │  Effect            │  Rule              │
├─────────────┼────────────────────┼────────────────────┤
│ Dilation ✅  │ Boundaries GROW   │ If ANY neighbor=255│
│             │ Fills small holes │ → pixel becomes 255│
├─────────────┼────────────────────┼────────────────────┤
│ Erosion ❌   │ Boundaries SHRINK │ If ANY neighbor=0  │
│             │ Removes small obj │ → pixel becomes 0  │
└─────────────┴────────────────────┴────────────────────┘

STRUCTURING ELEMENT COMPARISON:
┌──────────────┬─────────────────┬─────────────────┐
│ SE Type      │ Effect on edges │ Best for        │
├──────────────┼─────────────────┼─────────────────┤
│ Square 3x3   │ Sharp growth    │ General purpose │
│ Square 5x5   │ More growth     │ Larger gaps     │
│ Circle 3x3   │ Smooth growth   │ Natural shapes  │
│ Circle 5x5   │ More smooth     │ Round objects   │
└──────────────┴─────────────────┴─────────────────┘

KEY RULES:
• Larger SE = More effect (more growth/shrink)
• Circle SE = Smoother edges than square
• Square SE = Faster computation
• Dilation + Erosion = Closing (fill holes)
• Erosion + Dilation = Opening (remove noise)
""")