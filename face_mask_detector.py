import os
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array

# Absolute paths to dataset folders
with_mask_path = r'C:\Users\karth\OneDrive\Desktop\mla_executions\face_mask_detection\dataset\with_mask'
without_mask_path = r'C:\Users\karth\OneDrive\Desktop\mla_executions\face_mask_detection\dataset\without_mask'

# Initialize data and labels lists
data = []
labels = []

# Function to load images
def load_images_from_folder(folder, label):
    for file in os.listdir(folder):
        img_path = os.path.join(folder, file)
        image = cv2.imread(img_path)
        if image is None:
            continue  # skip unreadable images
        image = cv2.resize(image, (224, 224))  # Resize to MobileNetV2 expected input
        image = img_to_array(image)
        image = preprocess_input(image)

        data.append(image)
        labels.append(label)

# Load data from both folders
load_images_from_folder(with_mask_path, 1)      # 1 = With Mask
load_images_from_folder(without_mask_path, 0)   # 0 = Without Mask

# Convert to NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# One-hot encode the labels
labels = to_categorical(labels, num_classes=2)

# Split into train/test sets
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

print(f"✅ Loaded {len(data)} images: {len(trainX)} for training and {len(testX)} for testing.")
