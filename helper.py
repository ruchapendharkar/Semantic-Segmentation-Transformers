'''
helper.py
This file contains helper functions
'''
import torch
from torch.utils.data import  Dataset
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torchmetrics.functional import accuracy
import pandas as pd

# ==== Dataset Class ====
class KittiNPYDataset(Dataset):
    def __init__(self, images_path, masks_path, processor, transform=None):
        self.images = np.load(images_path)
        self.masks = np.load(masks_path)
        print(f"Image array shape: {self.images.shape}")
        print(f"Mask array shape: {self.masks.shape}")
        self.processor = processor
        self.transform = transform

    def convert_mask_to_single_channel(self, mask):
        """Convert multi-channel mask to single channel with class indices."""
        return np.argmax(mask, axis=-1)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx].copy()

        # Convert multi-channel mask to single channel
        mask = self.convert_mask_to_single_channel(mask)

        # Ensure image is in range [0, 255] and uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Convert image for processor
        if isinstance(image, torch.Tensor):
            image = image.numpy().transpose(1, 2, 0)

        # Process image
        inputs = self.processor(images=image, return_tensors="pt")

        # Remove batch dimension
        for k, v in inputs.items():
            inputs[k] = v.squeeze(0)

        # Convert mask to tensor
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        mask = mask.long()

        inputs['labels'] = mask
        return inputs

    def __len__(self):
        return len(self.images)

def plot_metrics(csv_path='metrics.csv'):
    # Read the CSV file
    metrics_df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(20, 5))
    
    # Loss Plot
    plt.subplot(141)
    plt.plot(metrics_df['epoch'], metrics_df['train_loss'], 'b-', label='Training')
    plt.plot(metrics_df['epoch'], metrics_df['val_loss'], 'r-', label='Validation')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Learning Rate Plot
    plt.subplot(142)
    plt.plot(metrics_df['epoch'], metrics_df['learning_rate'], 'g-')
    plt.title('Learning Rate Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    
    # IoU Plot
    plt.subplot(143)
    plt.plot(metrics_df['epoch'], metrics_df['val_iou'], 'g-o')
    plt.title('Validation IoU Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.grid(True)
    
    # Pixel Accuracy Plot
    plt.subplot(144)
    plt.plot(metrics_df['epoch'], metrics_df['val_pixel_acc'], 'm-o')
    plt.title('Validation Pixel Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Pixel Accuracy')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('segformer_training_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

# ==== Metrics ====
def compute_iou(preds, labels, num_classes):
    """Compute IoU (Intersection over Union) for predictions and ground truth."""
    preds = F.interpolate(preds.unsqueeze(1).float(), size=labels.shape[-2:], mode="nearest").squeeze(1)

    iou_per_class = []
    for cls in range(num_classes):
        pred_mask = (preds == cls)
        label_mask = (labels == cls)

        intersection = (pred_mask & label_mask).sum().item()
        union = (pred_mask | label_mask).sum().item()

        if union == 0:
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append(intersection / union)

    iou_per_class = [iou for iou in iou_per_class if not np.isnan(iou)]
    return sum(iou_per_class) / len(iou_per_class) if iou_per_class else 0.0

def compute_metrics(preds, labels, num_classes):
    """Compute IoU and Pixel Accuracy for the predictions and labels."""
    preds = F.interpolate(preds.unsqueeze(1).float(), size=labels.shape[-2:], mode="nearest").squeeze(1)

    preds_flat = preds.flatten()
    labels_flat = labels.flatten()

    iou = compute_iou(preds, labels, num_classes)
    pixel_acc = accuracy(preds_flat, labels_flat, task="multiclass", num_classes=num_classes)

    return {"iou": iou, "pixel_acc": pixel_acc.item()}

def visualize_results(image, prediction, ground_truth):
    plt.figure(figsize=(12, 4))

    mean = torch.tensor(processor.image_mean).view(3, 1, 1)
    std = torch.tensor(processor.image_std).view(3, 1, 1)
    image = image.cpu() * std + mean
    image = image.permute(1, 2, 0).clip(0, 1)

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(prediction.cpu(), cmap="tab20")
    plt.title("Prediction")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(ground_truth.cpu(), cmap="tab20")
    plt.title("Ground Truth")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

