import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from scipy.spatial import distance

# ==========================================
# 1. Feature Extraction (Color Histogram)
# ==========================================
def extract_color_histogram(image_path):
    """Extracts a normalized 3D color histogram from an image."""
    image = cv2.imread(image_path)
    if image is None:
        return None, None
    
    # Convert BGR to RGB for correct matplotlib color display
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Extract a 3D color histogram (8 bins per channel to reduce dimensionality)
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    
    # Normalize the histogram so image size doesn't affect the feature vector
    cv2.normalize(hist, hist)
    
    # Flatten into a 1D feature vector
    return hist.flatten(), image

# ==========================================
# 2. Build the Dataset Feature Vectors
# ==========================================
# Updated to point to the YOLO coco8 validation images from your workspace
dataset_dir = 'datasets/coco8/images/val' 

# Fallback in case the exact path varies slightly
if not os.path.exists(dataset_dir):
    dataset_dir = 'datasets'
    if not os.path.exists(dataset_dir):
        print("Error: Could not find the 'datasets' directory.")
        sys.exit()

feature_db = {}
image_db = {}

print(f"Extracting features from {dataset_dir}...")
for filename in os.listdir(dataset_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        filepath = os.path.join(dataset_dir, filename)
        features, img_rgb = extract_color_histogram(filepath)
        
        if features is not None:
            feature_db[filename] = features
            image_db[filename] = img_rgb

# Safety check: Ensure the dataset isn't empty
if not feature_db:
    print(f"Error: No valid images found in the '{dataset_dir}' directory.")
    sys.exit()

# ==========================================
# 3. Process the Query Image
# ==========================================
# Updated to match the image file present in your root folder
query_path = 'image1.png' 
query_features, query_img = extract_color_histogram(query_path)

# Safety check: Ensure query image loaded successfully
if query_features is None:
    print(f"Error: Could not read the query image at '{query_path}'. Check if the file exists.")
    sys.exit()

# ==========================================
# 4. Image Retrieval Based on Similarity
# ==========================================
distances = {}

# Compute Euclidean Distance 
for filename, db_features in feature_db.items():
    dist = distance.euclidean(query_features, db_features)
    distances[filename] = dist

# Sort the results by distance (Lowest distance = Highest similarity)
sorted_results = sorted(distances.items(), key=lambda x: x[1])

# Retrieve the top 3 most similar images (or fewer if the dataset is small)
num_results = min(3, len(sorted_results))
top_k_results = sorted_results[:num_results]

# ==========================================
# 5. Display the Results
# ==========================================
# Dynamically adjust columns depending on how many matches were found
plt.figure(figsize=(4 * (num_results + 1), 5))

# Display Query Image
plt.subplot(1, num_results + 1, 1)
plt.imshow(query_img)
plt.title('Query Image\n')
plt.axis('off')

# Display Top Retrievals
for i, (filename, dist) in enumerate(top_k_results):
    plt.subplot(1, num_results + 1, i + 2)
    plt.imshow(image_db[filename])
    plt.title(f'Rank {i+1}: {filename}\nDistance: {dist:.4f}')
    plt.axis('off')

plt.tight_layout()
plt.show()