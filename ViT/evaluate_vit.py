import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from train import KittiDataset, ViTForSegmentation  # Assuming train.py contains these classes
import os

# Paths
train_image_dir = "/Users/shalakapadalkar/Desktop/ACV/data_semantics/training/image_2"  # Update this to your validation images directory
train_mask_dir = "/Users/shalakapadalkar/Desktop/ACV/data_semantics/training/semantic_rgb"  # Update this to your validation masks directory
saved_model_path = "/Users/shalakapadalkar/Desktop/ACV/ViT/vit_segmentation_model_complete.pth"  # Path to the saved model

# Define image and mask transformations
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

mask_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
])

# Load validation dataset
validation_dataset = KittiDataset(
    train_image_dir,
    train_mask_dir,
    transform=image_transform,
    target_transform=mask_transform
)

validation_dataloader = DataLoader(validation_dataset, batch_size=8, shuffle=False)

# Load model and metadata
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load saved checkpoint
checkpoint = torch.load(saved_model_path, map_location=device)
color_to_class = checkpoint['color_to_class']  # Extract the color-to-class mapping
num_classes = len(color_to_class)

# Initialize model and load weights
model = ViTForSegmentation(num_classes).to(device)
model.load_state_dict(checkpoint['model_state_dict'])

# Evaluation function
def evaluate(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()  # Same loss function as training

    with torch.no_grad():  # Disable gradient computation
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, masks)
            total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {average_loss:.4f}")
    return average_loss

# Run evaluation
if __name__ == "__main__":
    if not os.path.exists(train_image_dir) or not os.path.exists(train_mask_dir):
        raise FileNotFoundError("Please update the validation image/mask paths in the script.")
    
    print("Starting evaluation...")
    evaluate(model, validation_dataloader, device)
