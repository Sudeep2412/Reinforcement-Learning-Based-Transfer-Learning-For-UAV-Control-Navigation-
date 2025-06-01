from ultralytics import YOLO

# Load the trained model
model = YOLO("runs/detect/train22/weights/best.pt")

# Perform inference on test images
results = model.predict(source="/mnt/c/Users/sudee/OneDrive/Desktop/Research/train/images/test", 
                        save=True, 
                        conf=0.5)

# Display results
for i, result in enumerate(results):
    print(f"Image {i + 1}:")
    result.show()  # Show each image with detections
    result.save(filename=f"output_{i}.jpg")  # Save the output images
