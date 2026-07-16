import cv2
import matplotlib.pyplot as plt

# 1. Load two images in grayscale
# Ensure you have two related images in your directory
img1 = cv2.imread('image1.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)

# 2. Initialize the SIFT detector
sift = cv2.SIFT_create()

# 3. Detect keypoints and compute descriptors for both images
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

# 4. Match features using FLANN
# FLANN (Fast Library for Approximate Nearest Neighbors) is highly optimized for SIFT
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50) # Number of times the trees should be recursively traversed
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Find the 2 best matches for each descriptor (k=2)
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# 5. Apply Lowe's Ratio Test
# This filters out weak or ambiguous matches. 
# A match is considered "good" if the distance to the best match is significantly 
# shorter than the distance to the second-best match.
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# 6. Visualize matched keypoints with lines connecting correspondences
# DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS ensures we only draw keypoints that found a match
matched_img = cv2.drawMatches(
    img1, keypoints1, 
    img2, keypoints2, 
    good_matches, None, 
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    matchColor=(0, 255, 0) # Draw matching lines in green
)

# 7. Display the result
plt.figure(figsize=(16, 8))
plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
plt.title(f'SIFT Feature Matching with FLANN ({len(good_matches)} good matches)')
plt.axis('off')
plt.tight_layout()
plt.show()