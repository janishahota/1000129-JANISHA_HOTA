from ultralytics import YOLO
import joblib

# Define the path to the custom YAML file and train the model.
model = YOLO('yolov8n.pt')  # Starting from a pretrained YOLO model.

# Train the model on your custom dataset.
model.train(
    data='/Users/hota/PycharmProjects/IB 12/.venv/YoloSA/cellphone detector drivers.v2i.yolov8/data.yaml',  # Path to your custom data YAML file.
    epochs=10,  # Number of training epochs.
    imgsz=640,  # Image size for training.
    batch=8,    # Batch size.
    device='cpu'  # Use 'cpu' for CPU training or '0' for the first GPU.
)

