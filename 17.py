import cv2

# 1. Read a sample MPEG/MP4 video
# Replace 'sample_video.mp4' with the name of your downloaded video file
cap = cv2.VideoCapture('video.mp4')

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video file. Please check the file path.")

while cap.isOpened():
    # 2. Extract frames sequentially
    ret, frame = cap.read()
    
    # If frame is read correctly, ret is True. If we reach the end, it becomes False.
    if not ret:
        print("End of video stream reached.")
        break
        
    # Resize frame for better viewing on screen (optional)
    frame = cv2.resize(frame, (640, 360))

    # 3. Convert frames to grayscale for analysis
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 4. Convert grayscale frames to binary images
    # Pixels above 127 become 255 (white), others become 0 (black)
    _, binary_frame = cv2.threshold(gray_frame, 127, 255, cv2.THRESH_BINARY)
    
    # Display the frames sequentially in separate windows
    cv2.imshow('Original Color Video', frame)
    cv2.imshow('Grayscale Analysis', gray_frame)
    cv2.imshow('Binary Analysis', binary_frame)
    
    # Wait for 30 milliseconds before moving to the next frame
    # Press 'q' on your keyboard to exit the playback early
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release the video capture object and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()