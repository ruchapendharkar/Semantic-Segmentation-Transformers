"""
labels.py
This file dynamically generates class mappings from RGB masks and splits the data into training and testing sets.
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Function to dynamically map RGB values to class indices
def RGB2Label_dynamic(label, color_map):
    """
    Converts an RGB mask to class indices based on a dynamic color map.
    Args:
        label: RGB mask (H x W x 3).
        color_map: Dictionary mapping RGB tuples to class indices.
    Returns:
        Single-channel mask with class indices (H x W).
    """
    label_segment = np.zeros(label.shape[:2], dtype=np.uint8)  # Single-channel mask
    for class_idx, rgb in enumerate(color_map):
        label_segment[np.all(label == rgb, axis=-1)] = class_idx
    return label_segment

def generate_color_map(masks):
    """
    Generates a color map (unique RGB values to class indices) from a list of masks.
    Args:
        masks: List or array of RGB masks (N x H x W x 3).
    Returns:
        List of unique RGB colors.
    """
    unique_colors = set()
    for mask in masks:
        unique_colors.update(map(tuple, np.unique(mask.reshape(-1, 3), axis=0)))
    return list(unique_colors)

# Main function
def main():
    print("...Loading datasets!...")
    imageDataset = np.load(r'C:\Data\Northeastern\Advanced Computer Vision\Final Project\Semantic-Segmentation-Transformers\processed_kitti_images.npy')  # Load preprocessed images
    maskDataset = np.load(r'C:\Data\Northeastern\Advanced Computer Vision\Final Project\Semantic-Segmentation-Transformers\processed_kitti_masks.npy')    # Load preprocessed masks
    print(f"Mask dataset shape: {maskDataset.shape}")
    maskDataset = np.squeeze(maskDataset)
    print("...Loaded datasets!...")

    print("Generating color map...")
    color_map = generate_color_map(maskDataset)
    print(f"Unique RGB colors detected: {len(color_map)}")
    print(f"Color Map: {color_map}")

    labels = []

    # Convert RGB masks to class labels
    for i in range(maskDataset.shape[0]):
        print(f"Processing mask {i + 1}/{maskDataset.shape[0]}...")
        label = RGB2Label_dynamic(maskDataset[i], color_map)
        labels.append(label)

    labels = np.array(labels)
    labels = np.expand_dims(labels, axis=3)  # Expand dimensions for compatibility

    print(f"Total unique values in labels: {np.unique(labels)}")

    # Get number of classes dynamically
    total_classes = len(color_map)

    # Convert to categorical (one-hot encoding)
    labelsDataset = to_categorical(labels, num_classes=total_classes)

    # Split the data into training and testing sets
    train_imgs, test_imgs, train_masks, test_masks = train_test_split(
        imageDataset, labelsDataset, test_size=0.4, shuffle=True, random_state=3
    )

    # Verify shapes of training and testing images
    print(f"Training images: {len(train_imgs)}")
    print(f"Testing images: {len(test_imgs)}")
    print(f"Training masks: {len(train_masks)}")
    print(f"Testing masks: {len(test_masks)}")

    # Save the training and testing data
    np.save(r'C:\Data\Northeastern\Advanced Computer Vision\Final Project\Semantic-Segmentation-Transformers\kitti_train_images.npy', train_imgs)
    np.save(r'C:\Data\Northeastern\Advanced Computer Vision\Final Project\Semantic-Segmentation-Transformers\kitti_train_masks.npy', train_masks)
    np.save(r'C:\Data\Northeastern\Advanced Computer Vision\Final Project\Semantic-Segmentation-Transformers\kitti_test_images.npy', test_imgs)
    np.save(r'C:\Data\Northeastern\Advanced Computer Vision\Final Project\Semantic-Segmentation-Transformers\kitti_test_masks.npy', test_masks)

    print("Saved training and testing data successfully!")

if __name__ == "__main__":
    main()
