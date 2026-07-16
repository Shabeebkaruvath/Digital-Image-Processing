import cv2
import numpy as np

# 1. Initialize video capture
# Replace with the path to your downloaded MPEG/MP4 video
cap = cv2.VideoCapture('video1.mp4')

if not cap.isOpened():
    print("Error: Could not open video file.")

# 2. Initialize Background Subtractor
# MOG2 is a robust algorithm for background subtraction that also detects shadows
backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# Define a structuring element (kernel) for morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video stream.")
        break
        
    # Resize for better visibility (optional)
    frame = cv2.resize(frame, (800, 450))
    
    # Convert to grayscale for processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 3. Apply Background Subtraction
    # This isolates moving objects (foreground) from the static background
    fg_mask = backSub.apply(gray)
    
    # Threshold the mask to remove gray shadows (leaving only pure white foreground)
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    
    # 4. Apply Morphological Operations to clean the segmented objects
    # Opening (Erosion -> Dilation) removes small random noise pixels in the background
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    # Closing (Dilation -> Erosion) fills in small black holes inside the moving objects
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    
    # 5. Detect contours of the cleaned moving objects
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 6. Draw Bounding Boxes in real-time
    for contour in contours:
        # Filter out tiny contours to avoid drawing boxes around residual noise
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Draw a green rectangle around the moving object
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Motion Detected', (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
    # Display the original frame with bounding boxes and the cleaned binary mask
    cv2.imshow('Real-Time Motion Tracking', frame)
    cv2.imshow('Cleaned Foreground Mask', fg_mask)
    
    # Exit loop if 'q' is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Clean up resources
cap.release()
cv2.destroyAllWindows()