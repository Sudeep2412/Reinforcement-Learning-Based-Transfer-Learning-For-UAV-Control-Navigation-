from ultralytics import YOLO
import torch

print("PyTorch CUDA Available:", torch.cuda.is_available())  
print("YOLO Detects CUDA:", YOLO("yolov5s.pt").device)  
