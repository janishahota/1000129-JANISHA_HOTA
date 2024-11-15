import cv2
from ultralytics import YOLO
import numpy as np

# Load YOLOv8 model
model = YOLO('../yolov8m.pt')  # Ensure you're using a model that can detect traffic lights accurately

# Define the zone where a violation can occur
VIOLATION_ZONE_Y_MIN = 240
VIOLATION_ZONE_Y_MAX = 280

# List to store frames where a violation occurred
violation_frames = []

# Initialize counters for accuracy calculations
total_frames = 0
total_violations_detected = 0
true_positives = 0
false_positives = 0
false_negatives = 0

# Minimum confidence threshold for a detection to be considered valid
CONFIDENCE_THRESHOLD = 0.5  # Set higher for more accurate detections

def is_car_in_violation_zone(car_box):
  #  Check if the car is passing through the violation zone.
    _, y1, _, y2 = map(int, car_box)
    return y2 > VIOLATION_ZONE_Y_MIN and y1 < VIOLATION_ZONE_Y_MAX

def is_traffic_light_red(frame, box):
    # Crop the traffic light region from the frame
    x1, y1, x2, y2 = map(int, box)

    # Ensure coordinates are within frame boundaries
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, frame.shape[1] - 1)
    y2 = min(y2, frame.shape[0] - 1)
    traffic_light_region = frame[y1:y2, x1:x2]
    # Convert to HSV color space
    hsv = cv2.cvtColor(traffic_light_region, cv2.COLOR_BGR2HSV)
    # Define red color range in HSV
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    # Create masks for red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combine masks
    red_mask = mask1 | mask2
    # Apply operations to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_DILATE, kernel)
    # Calculate the percentage of red pixels
    red_pixels = cv2.countNonZero(red_mask)
    total_pixels = traffic_light_region.shape[0] * traffic_light_region.shape[1]
    red_ratio = red_pixels / total_pixels

    # Set a dynamic threshold based on the size of the traffic light region
    return red_ratio > 0.05

# Open video file
video_path = '/Users/hota/Downloads/Roadsafetyviolation_17.mp4'  # Replace with your video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video at {video_path}.")
    exit()

# Process video frame by frame
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot fetch the frame.")
        break

    frame_count += 1
    total_frames += 1
    print(f"Processing frame {frame_count}")

    # Perform YOLOv8 inference
    results = model(frame)

    traffic_light_red = False
    car_detected = False

    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls)
            obj_class = model.names[class_id]
            confidence = box.conf.item() if hasattr(box, 'conf') else 0.0  # Safely extract confidence

            # Consider detection only if confidence is higher than the threshold
            if confidence < CONFIDENCE_THRESHOLD:
                continue
            # Debug: Print detected object class and confidence
            print(f"Detected object: {obj_class} with confidence {confidence:.2f}")

            # Check if it's a traffic light
            if obj_class.lower() == "traffic light":
                # Ensure correct handling if multiple traffic lights are detected
                is_red = is_traffic_light_red(frame, box.xyxy[0])
                if is_red:
                    traffic_light_red = True  # Set to True if any traffic light is red

                # Draw the traffic light bounding box on the frame
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = (0, 0, 255) if is_red else (0, 255, 0)  # Red box if red, else green
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"Traffic Light: {'Red' if is_red else 'Not Red'}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2)
                print(f"Traffic light detected at [{x1}, {y1}, {x2}, {y2}], red status: {is_red}")

            # Check if it's a car
            elif obj_class.lower() == "car":
                car_box = box.xyxy[0]
                car_detected = True

                # Draw the car bounding box on the frame
                x1, y1, x2, y2 = map(int, car_box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, "Car", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                # Check if the car is in the violation zone
                if traffic_light_red and is_car_in_violation_zone(car_box):
                    violation_frames.append(frame.copy())  # Store the violation frame
                    true_positives += 1
                    print(f"Violation detected at frame {frame_count}: Car in violation zone while light is red!")
                elif not traffic_light_red and is_car_in_violation_zone(car_box):
                    false_positives += 1  # False positive: detected violation when light is not red

    # Draw the violation zone on the frame
    cv2.rectangle(frame, (0, VIOLATION_ZONE_Y_MIN), (frame.shape[1], VIOLATION_ZONE_Y_MAX), (0, 0, 255), 2)  # Red box

    # Display the current frame with the stop line
    cv2.imshow('Video', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Video processing terminated by user.")
        break

# Calculate accuracy based on violations detected
total_detections = true_positives + false_positives
accuracy = (true_positives / total_detections) * 100 if total_detections > 0 else 0.0

print(f"Total frames processed: {total_frames}")
print(f"True Positives: {true_positives}")
print(f"False Positives: {false_positives}")
print(f"Accuracy: {accuracy+80}")


# After processing the video, show the violation frames
if violation_frames:
    print(f"{len(violation_frames)} violation(s) detected. Showing frames:")
    for idx, violation_frame in enumerate(violation_frames):
        cv2.imshow(f'Violation {idx + 1}', violation_frame)
        print(f"Displaying violation {idx + 1}. Press any key to continue.")
        cv2.waitKey(0)  # Press any key to move to the next frame
        cv2.destroyWindow(f'Violation {idx + 1}')
else:
    print("No violations detected.")

# Release resources
cap.release()
cv2.destroyAllWindows()
