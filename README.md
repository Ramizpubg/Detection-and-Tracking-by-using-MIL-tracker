Here is a detailed guide on setting up and implementing object detection and tracking using YOLOv8 and the MIL tracker in Python. This guide includes system requirements, environment setup, project structure, labeling your dataset, and running the detection and tracking system.

## System Requirements

- **Operating System:** Windows, macOS, or Linux
- **Python Version:** 3.8 or later
- **Hardware:** 
  - **CPU:** Any modern multi-core processor
  - **GPU:** NVIDIA GPU with CUDA support (recommended for faster inference)
  - **RAM:** At least 8GB

## Environment Setup

1. **Install Python:** Make sure you have Python installed. You can download it from the [official Python website](https://www.python.org/).

2. **Create a Virtual Environment:** It's a good practice to create a virtual environment to manage your dependencies. You can create one using `venv`:

   ```bash
   python -m venv yolov8-env
   ```

3. **Activate the Virtual Environment:**

   - **Windows:**

     ```bash
     yolov8-env\Scripts\activate
     ```

   - **macOS/Linux:**

     ```bash
     source yolov8-env/bin/activate
     ```

4. **Install Required Libraries:**

   ```bash
   pip install ultralytics opencv-python opencv-contrib-python torch
   ```

   - `ultralytics`: This is the package that includes YOLOv8.
   - `opencv-python` and `opencv-contrib-python`: These packages are required for image processing and using OpenCV trackers.
   - `torch`: This is the PyTorch library, which is required for YOLOv8.

## Project Structure

Organize your project directory as follows:

```
your_project/
│
├── yolov8-env/                  # Virtual environment
├── data/                        # Directory for datasets
│   ├── images/                  # Images for training and testing
│   └── labels/                  # Labels in YOLO format
├── models/                      # Directory for YOLO models
│   └── mobile.pt                # YOLOv8 model file
├── output/                      # Directory for saving outputs
└── main.py                      # Main Python script for detection and tracking
```

## Using LabelImg to Label Your Dataset

**LabelImg** is a graphical image annotation tool that allows you to label images and save the annotations in YOLO format.

### Installing LabelImg

1. **Installation:**

   - You can install LabelImg using pip:

     ```bash
     pip install labelImg
     ```

2. **Running LabelImg:**

   - Start LabelImg by running the following command:

     ```bash
     labelImg
     ```

3. **Labeling Images:**

   - Open the images you want to label.
   - Select the YOLO format from the `View` menu.
   - Use the tools provided to draw bounding boxes around the objects of interest.
   - Save the annotations in the `labels` directory.

## Implementing Object Detection and Tracking with YOLOv8 and MIL Tracker

### Importing Necessary Libraries

```python
import cv2
from ultralytics import YOLO
import torch
```

### Loading the YOLO Model

```python
# Load your YOLO model
model_path = "models/mobile.pt"
model = YOLO(model_path)

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
```

### Initializing Video Capture

```python
# Open a video capture (0 for the default webcam)
cap = cv2.VideoCapture(0)
```

### Initializing the Tracker

```python
# Initialize the tracker
tracker = cv2.TrackerMIL_create()
tracker_initialized = False
```

### Running Detection and Tracking

```python
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
```

## Running the Detection and Tracking System

1. **Make sure your virtual environment is activated.**

2. **Run the script:**

   ```bash
   python main.py
   ```

3. **Press 'q' to quit the application.**

## Flowchart

Here is a simplified flowchart of the detection and tracking process:

```
+-------------------------+
|  Start                  |
+-------------------------+
            |
            v
+-------------------------+
|  Initialize Video       |
|  Capture and Tracker    |
+-------------------------+
            |
            v
+-------------------------+
|  Read Frame from Camera |
+-------------------------+
            |
            v
+-------------------------+
|  Detection with YOLOv8  |
+-------------------------+
            |
            v
+-------------------------+
|  Initialize Tracker     |
+-------------------------+
            |
            v
+-------------------------+
|  Tracking with MIL      |
+-------------------------+
            |
            v
+-------------------------+
|  Display Results        |
+-------------------------+
            |
            v
+-------------------------+
|  Continue/Exit Loop     |
+-------------------------+
```

## Conclusion

This guide provides a complete setup for object detection and tracking using YOLOv8 and the MIL tracker. By labeling your dataset and setting up your environment, you can achieve accurate and efficient tracking of objects in real-time. This system is extensible and can be further customized for various applications, such as surveillance, robotics, or interactive systems.
