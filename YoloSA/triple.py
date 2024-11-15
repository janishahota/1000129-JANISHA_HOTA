import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load YOLOv8 model
model = YOLO('yolov8m.pt')

# Load the video
video_path = 'triples4.mov'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
else:
    # Lists for storing detection and frame data
    violation_frames = []
    total_frames = 0
    violation_count = 0
    total_detections = 0

    # Define class IDs and proximity threshold
    two_wheeler_class_id = 3  # Replace with actual class ID for two-wheeler
    person_class_id = 0       # Replace with actual class ID for person
    some_threshold = 400  # Increase this value if detections appear too strict

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO model on the frame
        results = model(frame)
        total_frames += 1

        # Separate detections
        two_wheelers = []
        people = []

        # Count unique objects detected in this frame
        frame_detections = 0

        for result in results[0].boxes:
            cls = int(result.cls)  # Class ID of the detected object
            xyxy = result.xyxy[0].cpu().numpy()  # Bounding box in (x1, y1, x2, y2)
            confidence = result.conf.cpu().numpy()[0]  # Confidence score

            # Count each unique detection in this frame
            frame_detections += 1

            # Append detections to respective lists
            if cls == two_wheeler_class_id:
                two_wheelers.append(xyxy)
                print(f"Two-Wheeler Detected - Frame {total_frames}")
            elif cls == person_class_id:
                people.append(xyxy)
                print(f"Person Detected - Frame {total_frames}")

            # Draw bounding boxes
            color = (0, 255, 0) if cls == two_wheeler_class_id else (255, 0, 0)
            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 2)
            label = "Two-Wheeler" if cls == two_wheeler_class_id else "Person"
            cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Add frame-level detection count to overall total
        total_detections += frame_detections

        # Check proximity for violations
        violation_detected = False
        for two_wheeler in two_wheelers:
            person_count = 0
            two_wheeler_center = ((two_wheeler[0] + two_wheeler[2]) / 2, (two_wheeler[1] + two_wheeler[3]) / 2)

            # Check distance of each person to this two-wheeler
            for person in people:
                person_center = ((person[0] + person[2]) / 2, (person[1] + person[3]) / 2)
                distance = ((two_wheeler_center[0] - person_center[0]) ** 2 + (two_wheeler_center[1] - person_center[1]) ** 2) ** 0.5

                # Debug each distance calculation
                print(f"Distance between two-wheeler and person on Frame {total_frames}: {distance:.2f} (Threshold: {some_threshold})")

                if distance < some_threshold:
                    person_count += 1
                    print(f"Person {person_count} is within range of a two-wheeler on Frame {total_frames}")

            # Flag a violation if three or more people are near the two-wheeler
            if person_count >= 3:
                violation_detected = True
                violation_count += 1
                print(f"Violation detected on Frame {total_frames} with {person_count} people near a two-wheeler.")
                break

        # Calculate overall detection accuracy as the average detection count per frame
        if total_frames > 0:
            overall_accuracy = (total_detections / total_frames) * 10

        # Show frame with accuracy info
        cv2.putText(frame, f"Overall Detection Accuracy: {overall_accuracy:.2f}%", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow('Real-time Detection', frame)
        if violation_detected:
            violation_frames.append(frame)  # Store frames with violations

        # Press 'q' to stop the video early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture
    cap.release()
    cv2.destroyAllWindows()

    # Display violation frames at the end
    for i, v_frame in enumerate(violation_frames):
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(v_frame, cv2.COLOR_BGR2RGB))
        plt.title(f'Violation Frame {i+1}')
        plt.axis('off')
        plt.show()

    # Final summary
    print(f"Total Frames: {total_frames}")
    print(f"Violation Count: {violation_count}")
    print(f"Final Overall Detection Accuracy: {overall_accuracy:.2f}%")
