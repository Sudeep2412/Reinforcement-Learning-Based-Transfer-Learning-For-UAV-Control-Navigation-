import os
import pandas as pd

# Set WSL-compatible paths
csv_files = {
    "train": "/mnt/c/Users/sudee/OneDrive/Desktop/Research/train/images/train/_annotations.csv",
    "val": "/mnt/c/Users/sudee/OneDrive/Desktop/Research/train/images/val/_annotations.csv"
}
labels_folder = "/mnt/c/Users/sudee/OneDrive/Desktop/Research/train/labels"

# Ensure labels folders exist
for split in ['train', 'val']:
    os.makedirs(os.path.join(labels_folder, split), exist_ok=True)

# Column mapping dictionary to handle variations
column_mapping = {
    'x': 'x_center',
    'y': 'y_center',
    'w': 'width',
    'h': 'height',
    'label': 'class',
    'filename': 'filename'  # Keep filename consistent
}

# Convert CSV to YOLO format
for split, csv_path in csv_files.items():
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        
        # Rename columns dynamically if needed
        df.rename(columns={v: k for k, v in column_mapping.items() if v in df.columns}, inplace=True)

        for _, row in df.iterrows():
            img_name = row.get('filename', '').split('.')[0]  # Fallback to empty string if missing
            if not img_name:
                continue  # Skip if filename is missing or invalid

            label_file = os.path.join(labels_folder, split, f"{img_name}.txt")

            # Handle column variations dynamically
            class_id = row.get('class', row.get('label', 0))
            x_center = row.get('x_center', row.get('x', 0))
            y_center = row.get('y_center', row.get('y', 0))
            width = row.get('width', row.get('w', 0))
            height = row.get('height', row.get('h', 0))

            # Write YOLO format label file
            with open(label_file, 'w') as f:
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

        print(f"✅ {split} annotations converted to YOLO format.")
    else:
        print(f"⚠️ CSV file not found: {csv_path}")
