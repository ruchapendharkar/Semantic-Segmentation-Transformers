import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
import matplotlib.pyplot as plt
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ==== Constants ====
BATCH_SIZE = 8
NUM_CLASSES = 29 
EPOCHS = 50
LEARNING_RATE = 5e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 512

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
        # mask shape: (H, W, num_classes)
        # Returns: (H, W) with values being class indices
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

# ==== Model Initialization ====
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512",
    num_labels=NUM_CLASSES,  # Updated to match your actual number of classes
    id2label={str(i): str(i) for i in range(NUM_CLASSES)},
    label2id={str(i): i for i in range(NUM_CLASSES)},
    ignore_mismatched_sizes=True
)
model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

def train_model(model, train_loader, val_loader, epochs):
    for epoch in range(epochs):
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
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}")
        
        if (epoch + 1) % 2 == 0:
            validate_model(model, val_loader)

def validate_model(model, val_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch['pixel_values'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            if total_loss == loss.item():
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                visualize_results(pixel_values[0], preds[0], labels[0])
                
        avg_loss = total_loss / len(val_loader)
        print(f"Validation Loss: {avg_loss:.4f}")

# ==== Visualization ====
def visualize_results(image, prediction, ground_truth):
    plt.figure(figsize=(12, 4))
    
    # Denormalize using the processor's normalization values
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

# Main function
if __name__ == "__main__":
    train_model(model, train_loader, val_loader, EPOCHS)