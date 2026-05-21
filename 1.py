import cv2

# Load image from file
img = cv2.imread("image1.png")

# 1. Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Histogram Equalization
equalized = cv2.equalizeHist(gray)

# Display
cv2.imshow("Original", img)
cv2.imshow("Grayscale", gray)
cv2.imshow("Histogram Equalized", equalized)

# Save
cv2.imwrite("gray.jpg", gray)
cv2.imwrite("equalized.jpg", equalized)

cv2.waitKey(0)
cv2.destroyAllWindows()