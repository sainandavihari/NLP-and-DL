import cv2
from tracker import EuclideanDistTracker

# Create tracker object
tracker = EuclideanDistTracker()

# Use raw string for path and make sure the file exists
video_path = r"/Users/sainandaviharim/Desktop/object_tracking/highway.mp4"

# Try to open video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file: {video_path}")
    exit()

# Object detection from stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while True:
    ret, frame = cap.read()

    # If frame not read successfully, end of video or problem reading file
    if not ret:
        print("End of video or cannot read the frame.")
        break

    # Get frame dimensions safely
    height, width, _ = frame.shape

    # Extract Region of Interest (ROI)
    # Check that the slicing bounds are valid for this video
    roi = frame[340:720, 500:800]

    # 1. Object Detection
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append([x, y, w, h])

    # 2. Object Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Show windows
    cv2.imshow("ROI", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(30)
    if key == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
