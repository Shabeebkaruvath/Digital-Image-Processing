import cv2
import sys
import matplotlib.pyplot as plt

# 1. Load the same two images in grayscale
img1 = cv2.imread('image1.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)

# Safety check (as implemented previously)
if img1 is None:
    print("Error: Could not load image1. Check the file path.")
    sys.exit()
if img2 is None:
    print("Error: Could not load image2. Check the file path.")
    sys.exit()

# 2. Initialize the SURF detector
# The Hessian threshold determines how many keypoints are detected. 
# A higher threshold (e.g., 400-1000) yields fewer, but stronger features.
try:
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400)
except AttributeError:
    print("Error: SURF is not available in your OpenCV build. It requires the 'opencv-contrib-python' package and non-free algorithms enabled.")
    print("Falling back to ORB (a free alternative to SURF)...")
    surf = cv2.ORB_create()

# 3. Detect keypoints and compute descriptors
keypoints1, descriptors1 = surf.detectAndCompute(img1, None)
keypoints2, descriptors2 = surf.detectAndCompute(img2, None)

# 4. Match features using Brute-Force Matcher
# If the fallback ORB was used, we must use NORM_HAMMING. For SURF, NORM_L2 is used.
if surf.__class__.__name__ == 'ORB':
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    # Sort them in the order of their distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:50] # Keep the top 50 matches
else:
    # FLANN matcher for SURF (similar to SIFT)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    
    # Lowe's Ratio Test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

# 5. Visualize matched keypoints
matched_img = cv2.drawMatches(
    img1, keypoints1, 
    img2, keypoints2, 
    good_matches, None, 
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    matchColor=(0, 255, 0)
)

# 6. Display the result
plt.figure(figsize=(16, 8))
plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
plt.title(f'Feature Matching ({len(good_matches)} good matches)')
plt.axis('off')
plt.tight_layout()
plt.show()