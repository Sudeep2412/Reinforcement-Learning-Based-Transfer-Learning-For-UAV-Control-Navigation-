# %%
import torch
from ultralytics import YOLO
import control as ctrl
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import numpy as np
import pandas as pd
import os

# %%
# Ensure GPU is used if available
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)

# %%
annotation_file = r"/mnt/c/Users/sudee/OneDrive/Desktop/Research/train/images/train/_annotations.csv"


# Load annotation data if available
if os.path.exists(annotation_file):
    annotations = pd.read_csv(annotation_file)
else:
    print(f"Warning: Annotation file not found at {annotation_file}. Proceeding without annotations.")
    annotations = None

# %%
def train_object_detection():
    """Train YOLOv5 model on UAV dataset using folder of images."""
    model = YOLO("yolov5su.pt")  # Load pretrained model
    model.train(data="/mnt/c/Users/sudee/OneDrive/Desktop/Research/train/dataset.yaml", epochs=50, imgsz=640, device=0)
    return model

# %%
def detect_objects(model, image_path):
    """Run inference using trained YOLO model."""
    results = model(image_path)
    results.show()


# %%
def compute_hinf_transfer():
    """Compute H-infinity transfer function for UAV dynamics."""
    s = ctrl.TransferFunction.s
    G1 = 4**2 / (s**2 + 2*0.7*4*s + 4**2)  # Source UAV
    G2 = 1**2 / (s**2 + 2*0.8*1*s + 1**2)  # Target UAV
    M = G1 * ctrl.minreal(ctrl.feedback(1, G2))
    print("Computed H∞ Transfer Function:", M)
    return M


# %%
def a_star_path_planning(start, goal, obstacles, grid_size=(100, 100)):
    """Plan UAV path using A* algorithm."""
    G = nx.grid_2d_graph(*grid_size)
    for obs in obstacles:
        if obs in G:
            G.remove_node(obs)
    path = nx.astar_path(G, start, goal)
    return path


# %%
def plot_uav_path(path, obstacles, start, goal):
    """Visualize UAV path."""
    plt.figure()
    for obs in obstacles:
        plt.scatter(*obs, color='red')
    path_x, path_y = zip(*path)
    plt.plot(path_x, path_y, linestyle='--', marker='o', color='blue')
    plt.scatter(*start, color='green', label='Start')
    plt.scatter(*goal, color='black', label='Goal')
    plt.legend()
    plt.title("UAV Path Planning with A*")
    plt.show()

# %%
def detect_objects(model, image_folder):
    """Run inference using trained YOLO model on all images in a folder."""
    if not os.path.exists(image_folder):
        print(f"Error: Folder '{image_folder}' not found.")
        return

    for img_file in os.listdir(image_folder):
        if img_file.lower().endswith((".jpg", ".png", ".jpeg")):  # Ensure it's an image
            img_path = os.path.join(image_folder, img_file)
            
            # Run inference
            results = model(img_path)
            
            # Display the results
        for result in results:
            result.show()


# %%
def main():
    """Main function to execute UAV detection and path planning."""
    # Train Object Detection Model
    model = train_object_detection()
    detect_objects(model, "/mnt/c/Users/sudee/OneDrive/Desktop/Research/train/images/test")
    
    # Compute H∞ Transfer Function
    compute_hinf_transfer()
    
    # Plan UAV Path
    obstacles = [(30, 30), (40, 50), (70, 80)]
    start, goal = (10, 10), (90, 90)
    path = a_star_path_planning(start, goal, obstacles)
    print("Computed Path:", path)
    plot_uav_path(path, obstacles, start, goal)

if __name__ == "__main__":
    main()


# %%



