from ultralytics import YOLO

# Load a model
model = YOLO("models/yolo11n_tom_and_jerry.pt")  # load a custom model

# Predict with the model
results = model("https://upload.wikimedia.org/wikipedia/vi/thumb/6/6e/Titlecard_T%26J.jpeg/250px-Titlecard_T%26J.jpeg")  # predict on an image