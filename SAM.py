import torch
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2
from torch.utils.data import Dataset
import random

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class KittiNPYDataset(Dataset):
    def __init__(self, images_path, masks_path):
        self.images = np.load(images_path)
        self.masks = np.load(masks_path)
        print(f"Image array shape: {self.images.shape}")
        print(f"Mask array shape: {self.masks.shape}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].copy()
        mask = self.masks[idx].copy()
        
        # Convert image to uint8 and ensure correct range
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.clip(0, 255).astype(np.uint8)

        return {
            'image': image,
            'mask': mask
        }

def initialize_sam():
    """Initialize SAM with automatic mask generation."""
    sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
    sam.to(DEVICE)
    
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100
    )
    
    return mask_generator

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_anns(anns, ax):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax.set_autoscale_on(False)
    
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    
    ax.imshow(img)
def compute_pixel_accuracy(pred_mask, gt_mask):
    """Compute pixel accuracy between prediction and ground truth."""
    if len(gt_mask.shape) > 2:
        # Convert multi-channel ground truth to binary mask
        gt_binary = np.any(gt_mask > 0, axis=-1)
    else:
        gt_binary = gt_mask > 0
    
    # Convert prediction to binary mask
    pred_binary = pred_mask > 0
    
    # Calculate accuracy
    correct = (pred_binary == gt_binary).sum()
    total = gt_binary.size
    return correct / total

def compute_iou(pred_mask, gt_mask):
    """Compute IoU between prediction and ground truth."""
    if len(gt_mask.shape) > 2:
        # Convert multi-channel ground truth to binary mask
        gt_binary = np.any(gt_mask > 0, axis=-1)
    else:
        gt_binary = gt_mask > 0
    
    # Convert prediction to binary mask
    pred_binary = pred_mask > 0
    
    # Calculate IoU
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    
    return intersection / union if union > 0 else 0

def process_dataset(dataset, mask_generator, num_samples=None):
    """Process multiple images from the dataset."""
    if num_samples is None:
        num_samples = len(dataset)
    else:
        num_samples = min(num_samples, len(dataset))
    
    total_iou = 0
    total_accuracy = 0
    
    for i in range(num_samples):
        print(f"\nProcessing image {i+1}/{num_samples}")
        
        # Get sample from dataset
        sample = dataset[i]
        image = sample['image']
        gt_mask = sample['mask']
        
        # Process with SAM
        masks = mask_generator.generate(image)
        
        # Create combined prediction mask
        pred_mask = np.zeros(image.shape[:2], dtype=bool)
        for mask_data in masks:
            pred_mask = np.logical_or(pred_mask, mask_data['segmentation'])
        
        # Compute metrics
        iou = compute_iou(pred_mask, gt_mask)
        pixel_acc = compute_pixel_accuracy(pred_mask, gt_mask)
        
        total_iou += iou
        total_accuracy += pixel_acc
        
        # Visualize results
        plt.figure(figsize=(15, 5))
        
        # Original image
        ax = plt.subplot(131)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        
        # SAM prediction
        ax = plt.subplot(132)
        plt.imshow(image)
        show_anns(masks, ax)
        plt.title(f'SAM Prediction ({len(masks)} segments)')
        plt.axis('off')
        
        # Ground truth
        ax = plt.subplot(133)
        if len(gt_mask.shape) > 2:
            gt_visible = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)
            for i in range(gt_mask.shape[-1]):
                color = np.random.randint(0, 255, 3)
                gt_visible[gt_mask[..., i] > 0] = color
        else:
            gt_visible = gt_mask
        
        plt.imshow(image)
        plt.imshow(gt_visible, alpha=0.5)
        plt.title('Ground Truth')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Print metrics
        print(f"Number of segments: {len(masks)}")
        print(f"IoU: {iou:.4f}")
        print(f"Pixel Accuracy: {pixel_acc:.4f}")
    
    # Print average metrics
    avg_iou = total_iou / num_samples
    avg_accuracy = total_accuracy / num_samples
    print("\nAverage Metrics:")
    print(f"Mean IoU: {avg_iou:.4f}")
    print(f"Mean Pixel Accuracy: {avg_accuracy:.4f}")



if __name__ == "__main__":
    # Initialize SAM
    print("Initializing SAM...")
    mask_generator = initialize_sam()
    
    # Load dataset
    print("\nLoading dataset...")
    val_dataset = KittiNPYDataset(
        images_path="kitti_test_images.npy",
        masks_path="kitti_test_masks.npy"
    )
    
    # Process images
    print("\nProcessing images...")
    process_dataset(val_dataset, mask_generator, num_samples=5)