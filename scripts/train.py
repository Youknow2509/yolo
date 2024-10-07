from ultralytics import YOLO

# Load a model
model = YOLO("models/yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model with MPS
results = model.train(
                    data="datasets/data.yaml", 
                    epochs=100, 
                    imgsz=640, 
                    device="mps",
                    name="tom_and_jerry")

# Save the model
model.save("models/yolo11n_tom_and_jerry.pt")