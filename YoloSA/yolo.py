import cv2
from ultralytics import YOLO

# Load the regular YOLO model for cell phone detection.
phone_model = YOLO('yolov8m.pt')
# Load the custom YOLO model for steering wheel detection.
wheel_model = YOLO('/Users/hota/PycharmProjects/IB 12/.venv/YoloSA/cellphone detector drivers.v2i.yolov8/runs/detect/train5/weights/best.pt')

cap = cv2.VideoCapture('Roadsafetyviolation_9.mp4')
highest_confidence = 0.0
highest_confidence2 = 0.0
best_frame = None

total_frames = 0
frames_with_both_detected = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    total_frames += 1  # Count total number of frames

    # Run the phone model
    phone_results = phone_model(frame)

    # Run the custom wheel model
    wheel_results = wheel_model(frame)

    phone_detected = False
    wheel_detected = False

    # Process the phone detection results.
    for result in phone_results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls)
            obj_class = phone_model.names[class_id]

            # Check if detected object is a phone.
            if obj_class == 'cell phone':
                confidence = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{obj_class} {confidence:.2f}"
                color = (0, 255, 0)  # Green for phone
                phone_detected = True

                # Draw bounding box and label.
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Process the steering wheel detection results.
    for result in wheel_results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls)
            obj_class = wheel_model.names[class_id]

            # Check if detected object is a wheel.
            if obj_class == 'wheel':
                confidence2 = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{obj_class} {confidence2:.2f}"
                color = (255, 0, 0)  # Blue for wheel
                wheel_detected = True

                # Draw bounding box and label.
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Update best frame if both phone and wheel are detected and confidence is highest.
    if phone_detected and wheel_detected:
        frames_with_both_detected += 1  # Count frames where both are detected

        if confidence > highest_confidence:
            highest_confidence = confidence
            best_frame = frame.copy()

        if confidence2 > highest_confidence2:
            highest_confidence2 = confidence2
            best_frame = frame.copy()

    # Display the frame with bounding boxes.
    cv2.imshow('Violation Detection - Live', frame)

    # Break on 'q' key press.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Calculate accuracy as the ratio of frames with both objects detected to the total frames.
if total_frames > 0:
    accuracy = (frames_with_both_detected / total_frames) * 100
else:
    accuracy = 0.0

print(f"Overall Model Accuracy: {accuracy:.2f}%")

# Show the best frame if a violation is detected.
if best_frame is not None:
    print(f"Displaying frame with highest confidence: {(highest_confidence+highest_confidence2)/2:.2f}")
    cv2.imshow('Best Frame - Violation Detected', best_frame)
    cv2.waitKey(0)
else:
    print("No traffic violation detected in the video.")

cap.release()
cv2.destroyAllWindows()
