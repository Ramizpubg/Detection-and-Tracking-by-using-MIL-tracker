import cv2
from ultralytics import YOLO
import torch

# Load your YOLO model
model_path = "C:/Users/rehma/Desktop/Ramizz/mobile.pt"
model = YOLO(model_path)

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Open a video capture (0 for the default webcam)
cap = cv2.VideoCapture(0)

# Initialize the tracker
tracker = cv2.TrackerMIL_create()
tracker_initialized = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Create a copy of the frame for displaying detection results
    detection_frame = frame.copy()

    if not tracker_initialized:
        # Run detection to find the object initially
        results = model(frame)

        # Assume the first detection is the object of interest
        for result in results:
            boxes = result.boxes
            if len(boxes) > 0:
                # Get the first detected box
                box = boxes[0]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf.item()

                # Draw the detection bounding box with confidence value
                cv2.rectangle(detection_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(detection_frame, f"Detected: {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Initialize the tracker with a scaled-down bounding box
                center_x = x1 + (x2 - x1) // 2
                center_y = y1 + (y2 - y1) // 2
                width = int((x2 - x1) * 0.5)  # Reduce width by 50%
                height = int((y2 - y1) * 0.5)  # Reduce height by 50%

                # Calculate new bounding box coordinates
                new_x1 = max(center_x - width // 2, 0)
                new_y1 = max(center_y - height // 2, 0)
                tracker.init(frame, (new_x1, new_y1, width, height))
                tracker_initialized = True
                break  # Exit the loop after initializing the tracker

        # Display detection results
        cv2.imshow("Detection", detection_frame)

    if tracker_initialized:
        # Update the tracker with the current frame
        success, bbox = tracker.update(frame)
        if success:
            # If tracking is successful, draw the bounding box on the frame
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            # If tracking fails, reset the tracker and run detection again
            tracker_initialized = False
            cv2.putText(frame, "Lost Track - Detecting Again", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display the tracking results in a separate window
        cv2.imshow("Tracking", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
