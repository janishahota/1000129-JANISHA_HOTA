import cv2
from ultralytics import YOLO

# Load YOLOv8 model trained on your custom Roboflow dataset
model = YOLO('/Users/hota/PycharmProjects/IB 12/.venv/YoloSA/Bike Helmet Detection.v1i.yolov8/runs/detect/train/weights/best.pt')  # Update with your model path

# Define class names based on your dataset
HELMET_CLASS = 'helmet'
WITHOUT_HELMET_CLASS = 'without helmet'

# List to store frames where a violation occurred
violation_frames = []

# Parameters to track detection statistics
total_frames = 0

# Open video file
video_path = 'Driving without helmet.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video at {video_path}.")
    exit()

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot fetch the frame.")
        break

    frame_count += 1
    total_frames += 1
    print(f"Processing frame {frame_count}")

    # Perform YOLOv8 inference with lowered confidence threshold
    results = model(frame, conf=0.25)  # Adjust conf for sensitivity

    helmets = []
    without_helmets = []

    # Process detections for helmet and without helmet
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls)
            obj_class = model.names[class_id]
            confidence = box.conf.item() if hasattr(box, 'conf') else 0.0

            # Separate detected objects based on class
            if obj_class.lower() == HELMET_CLASS:
                helmets.append((box.xyxy[0], confidence))  # Add helmet bounding box with confidence
            elif obj_class.lower() == WITHOUT_HELMET_CLASS:
                without_helmets.append((box.xyxy[0], confidence))  # Add without helmet bounding box with confidence
                print(f"Without helmet detected: {box.xyxy[0]} with confidence {confidence:.2f}")

    # Check for helmet violations based on without helmet detections
    for without_helmet_box, confidence in without_helmets:
        # Draw bounding box for without helmet detection
        x1_nh, y1_nh, x2_nh, y2_nh = map(int, without_helmet_box)
        cv2.rectangle(frame, (x1_nh, y1_nh), (x2_nh, y2_nh), (0, 0, 255), 2)
        cv2.putText(frame, f"Without Helmet: {confidence:.2f}", (x1_nh, y1_nh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Save frame with bounding box to violation frames list
        violation_frames.append((frame.copy(), confidence))  # Store frame with bounding box and confidence
        print(f"Violation detected at frame {frame_count}: Without helmet with confidence {confidence:.2f}!")

    # Draw bounding boxes for helmets with confidence score
    for helmet_box, confidence in helmets:
        x1_h, y1_h, x2_h, y2_h = map(int, helmet_box)
        cv2.rectangle(frame, (x1_h, y1_h), (x2_h, y2_h), (0, 255, 0), 2)
        cv2.putText(frame, f"Helmet: {confidence:.2f}", (x1_h, y1_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the current frame with bounding boxes and confidence scores
    cv2.imshow('Video', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Video processing terminated by user.")
        break

# After processing the video, sort violation frames by confidence and take top 3
violation_frames = sorted(violation_frames, key=lambda x: x[1], reverse=True)[:3]

# Calculate the average confidence of the top 3 frames
if violation_frames:
    average_detection_rate = sum(conf for _, conf in violation_frames) / len(violation_frames)
    print(f"Average Detection Rate : {average_detection_rate:.2f}")

    # Show the top 3 violation frames
    for idx, (violation_frame, confidence) in enumerate(violation_frames, 1):
        cv2.imshow(f'Top Violation {idx}', violation_frame)
        print(f"Displaying top violation {idx} with confidence {confidence:.2f}. Press any key to continue.")
        cv2.waitKey(0)
        cv2.destroyWindow(f'Top Violation {idx}')
else:
    print("No violations detected.")

# Release resources
cap.release()
cv2.destroyAllWindows()
