import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from train import KittiDataset, ViTForSegmentation  # Assuming train.py contains these classes

# Paths
train_image_dir = "/Users/shalakapadalkar/Desktop/ACV/data_semantics/training/image_2"  # Path to training images
train_mask_dir = "/Users/shalakapadalkar/Desktop/ACV/data_semantics/training/semantic_rgb"  # Path to training masks
saved_model_path = "/Users/shalakapadalkar/Desktop/ACV/ViT/vit_segmentation_model_batch16.pth"  # Path to the saved model

# Define image and mask transformations
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

mask_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
])

# Load training dataset
train_dataset = KittiDataset(
    train_image_dir,
    train_mask_dir,
    transform=image_transform,
    target_transform=mask_transform
)

# Load model and metadata
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load saved checkpoint
checkpoint = torch.load(saved_model_path, map_location=device)
color_to_class = checkpoint['color_to_class']  # Extract the color-to-class mapping
num_classes = len(color_to_class)

# Initialize model and load weights
model = ViTForSegmentation(num_classes).to(device)
model.load_state_dict(checkpoint['model_state_dict'])

# Function to calculate pixel accuracy
def calculate_pixel_accuracy(predicted_mask, ground_truth_mask):
    """
    Calculate pixel accuracy between predicted and ground truth masks.
    """
    # Ensure both masks are NumPy arrays
    predicted_mask = np.array(predicted_mask)
    ground_truth_mask = np.array(ground_truth_mask)

    # Calculate the number of correctly classified pixels
    correct = np.sum(predicted_mask == ground_truth_mask)

    # Calculate the total number of pixels
    total = ground_truth_mask.size

    # Calculate accuracy
    accuracy = correct / total
    return accuracy

# Visualization function
def visualize_predictions(model, dataset, device, num_samples=10):
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient computation
        for i in range(num_samples):
            image, mask = dataset[i]
            image_tensor = image.to(device).unsqueeze(0)  # Add batch dimension
            output = model(image_tensor)  # Get model prediction
            predicted_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

            # Calculate pixel accuracy
            accuracy = calculate_pixel_accuracy(predicted_mask, mask.numpy())
            print(f"Sample {i + 1}: Pixel Accuracy = {accuracy:.4f}")

            # Plot the input image, ground truth, and prediction
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 3, 1)
            plt.title("Input Image")
            plt.imshow(image.permute(1, 2, 0).cpu())  # Convert CHW to HWC
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.title("Ground Truth Mask")
            plt.imshow(mask.numpy(), cmap="tab20")  # Visualize the ground truth
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.title(f"Predicted Mask (Accuracy: {accuracy:.2%})")
            plt.imshow(predicted_mask, cmap="tab20")  # Visualize the prediction
            plt.axis("off")

            plt.show()

# Run visualization
if __name__ == "__main__":
    print("Visualizing predictions and calculating pixel accuracy...")
    visualize_predictions(model, train_dataset, device)
