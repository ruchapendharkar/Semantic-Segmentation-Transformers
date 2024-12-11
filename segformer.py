'''
segformer.py
This file loads the model, and adapts it to KITTI and trains it
'''
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
import matplotlib.pyplot as plt
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
from torchmetrics.functional import accuracy
import pandas as pd
from helper import *

# ==== Constants ====
BATCH_SIZE = 16
NUM_CLASSES = 29
EPOCHS = 150
LEARNING_RATE = 5e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 512

def visualize_predictions(model, val_dataset, num_samples=3):
    """Visualize predictions for a few samples from the validation dataset."""
    model.eval()
    with torch.no_grad():
        for i in range(num_samples):
            sample = val_dataset[i]
            pixel_values = sample["pixel_values"].unsqueeze(0).to(DEVICE)
            labels = sample["labels"]

            outputs = model(pixel_values=pixel_values)
            preds = torch.argmax(outputs.logits, dim=1).squeeze(0)

            visualize_results(pixel_values.squeeze(0).cpu(), preds.cpu(), labels)

def train_model(model, train_loader, val_loader, epochs):
    metrics = {
        'epoch': [], 
        'train_loss': [], 
        'val_loss': [], 
        'val_iou': [], 
        'val_pixel_acc': []
    }
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            pixel_values = batch['pixel_values'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Average Training Loss: {avg_train_loss:.4f}")

        # Validation phase
        model.eval()
        val_total_loss = 0
        all_metrics = {"iou": [], "pixel_acc": []}

        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch['pixel_values'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)

                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                val_total_loss += loss.item()

                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)

                batch_metrics = compute_metrics(preds, labels, num_classes=NUM_CLASSES)
                all_metrics["iou"].append(batch_metrics["iou"])
                all_metrics["pixel_acc"].append(batch_metrics["pixel_acc"])

        # Calculate average validation metrics
        avg_val_loss = val_total_loss / len(val_loader)
        avg_val_iou = sum(all_metrics["iou"]) / len(all_metrics["iou"])
        avg_val_acc = sum(all_metrics["pixel_acc"]) / len(all_metrics["pixel_acc"])

        # Store metrics
        metrics['epoch'].append(epoch + 1)
        metrics['train_loss'].append(avg_train_loss)
        metrics['val_loss'].append(avg_val_loss)
        metrics['val_iou'].append(avg_val_iou)
        metrics['val_pixel_acc'].append(avg_val_acc)

        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Mean IoU: {avg_val_iou:.4f}, Pixel Accuracy: {avg_val_acc:.4f}")

    # Save final model and metrics
    torch.save(model.state_dict(), "segformer_model_final.pth")
    print("Model saved as 'segformer_model_final.pth'")
    
    # Plot final metrics
    metrics_df = pd.DataFrame(metrics)
    plot_metrics(metrics_df)
    
    # Save metrics to CSV
    metrics_df.to_csv('Segformer-Training_metrics.csv', index=False)
    print("Training metrics saved to 'Segformer-Training_metrics.csv'")
    
    return metrics_df


if __name__ == "__main__":
    # Data Augmentation & Loader
    transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        ToTensorV2()
    ])

    # Initialize the processor
    processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

    # Initialize datasets
    print("\nInitializing training dataset...")
    train_dataset = KittiNPYDataset(
        images_path="kitti_train_images.npy",
        masks_path="kitti_train_masks.npy",
        processor=processor,
        transform=transform
    )

    print("\nInitializing validation dataset...")
    val_dataset = KittiNPYDataset(
        images_path="kitti_test_images.npy",
        masks_path="kitti_test_masks.npy",
        processor=processor,
        transform=transform
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=NUM_CLASSES,
        id2label={str(i): str(i) for i in range(NUM_CLASSES)},
        label2id={str(i): i for i in range(NUM_CLASSES)},
        ignore_mismatched_sizes=True
    )
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Train model and get metrics
    metrics_df = train_model(model, train_loader, val_loader, EPOCHS)

    # Load the trained model
    model.load_state_dict(torch.load("segformer_model_final.pth"))
    model.to(DEVICE)
    print("Trained model loaded.")

    # Visualize results for validation samples
    visualize_predictions(model, val_dataset, num_samples=3)
