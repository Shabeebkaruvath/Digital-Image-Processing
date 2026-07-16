import cv2
from ultralytics import YOLO

# ==========================================
# 1. Load a pre-trained YOLO model
# ==========================================
# This will automatically download the YOLOv8 nano model (yolov8n.pt) if not present
model = YOLO('yolov8n.pt') 

# ==========================================
# 2 & 3. Detect Objects and Draw Bounding Boxes
# ==========================================
# Load your image (ensure 'image.png' is in your directory)
image_path = 'image1.png'
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Could not load {image_path}. Please check the path.")
else:
    # Run inference on the loaded image
    results = model(image)
    
    # Iterate through the results and draw boxes manually
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Extract bounding box coordinates (x_min, y_min, x_max, y_max)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Extract confidence score and class ID
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            # Get the string label corresponding to the class ID
            label = f"{model.names[cls]}: {conf:.2f}"
            
            # Draw the bounding box (Green)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw the background for text to make it readable
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
            
            # Put the text label and confidence score
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Display the final image
    cv2.imshow('YOLOv8 Object Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ==========================================
# 4. Compute Precision, Recall, and mAP
# ==========================================
# Object detection metrics require a labeled dataset to compare predictions against ground truths.
# We use the built-in 'coco8.yaml' mini-dataset which automatically downloads for quick validation.
print("\nEvaluating model metrics over the dataset...")
metrics = model.val(data='coco8.yaml', imgsz=640, split='val', verbose=False)

# Extract and print the computed metrics
precision = metrics.results_dict['metrics/precision(B)']
recall = metrics.results_dict['metrics/recall(B)']
map50 = metrics.results_dict['metrics/mAP50(B)']
map50_95 = metrics.results_dict['metrics/mAP50-95(B)']

print("-" * 40)
print("YOLOv8 Model Performance Metrics")
print("-" * 40)
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"mAP@50:    {map50:.4f}")
print(f"mAP@50-95: {map50_95:.4f}")
print("-" * 40)