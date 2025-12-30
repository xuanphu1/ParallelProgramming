from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("bestDetect.pt")

# Export the model to ONNX format
model.export(format="onnx")  # creates 'yolo11n.onnx'

# Load the exported ONNX model
onnx_model = YOLO("bestDetect.onnx")
