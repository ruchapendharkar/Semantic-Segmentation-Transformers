import os
import glob
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from transformers import ViTModel
import torch.nn.functional as F

# Dataset class for KITTI
class KittiDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, target_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        self.transform = transform
        self.target_transform = target_transform
        self.color_to_class = self.generate_color_to_class_mapping()

    def generate_color_to_class_mapping(self):
        unique_colors = set()
        for mask_path in self.mask_paths:
            mask = np.array(Image.open(mask_path))
            unique_colors.update([tuple(color) for color in np.unique(mask.reshape(-1, 3), axis=0)])
        return {color: idx for idx, color in enumerate(sorted(unique_colors))}

    def encode_mask(self, mask):
        mask_array = np.array(mask)
        encoded_mask = np.zeros(mask_array.shape[:2], dtype=np.int64)
        for color, class_idx in self.color_to_class.items():
            encoded_mask[(mask_array == color).all(axis=-1)] = class_idx
        return encoded_mask

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("RGB")
        mask = self.encode_mask(mask).astype(np.uint8)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(Image.fromarray(mask))
            mask = np.array(mask)
        mask = torch.tensor(mask, dtype=torch.long)
        return image, mask


# Model class
class ViTForSegmentation(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(768, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        outputs = self.vit(pixel_values=x).last_hidden_state
        outputs = outputs[:, 1:, :]
        b, n, c = outputs.size()
        h = w = int(n ** 0.5)
        if h * w != n:
            raise ValueError(f"Invalid sequence length {n}. Ensure input size is compatible with ViT patches.")
        outputs = outputs.transpose(1, 2).reshape(b, c, h, w)
        outputs = self.decoder(outputs)
        outputs = F.interpolate(outputs, size=(224, 224), mode="bilinear", align_corners=False)
        return outputs


# Training Code (wrapped in `if __name__ == "__main__":`)
if __name__ == "__main__":
    # Define paths
    image_dir = "/Users/shalakapadalkar/Desktop/ACV/data_semantics/training/image_2"
    mask_dir = "/Users/shalakapadalkar/Desktop/ACV/data_semantics/training/semantic_rgb"

    # Define transformations
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=Image.NEAREST),
    ])

    # Load dataset and dataloader
    dataset = KittiDataset(image_dir, mask_dir, transform=image_transform, target_transform=mask_transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(dataset.color_to_class)
    model = ViTForSegmentation(num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader)}")

    # Save the model in proper format
    torch.save({
        'model_state_dict': model.state_dict(),
        'color_to_class': dataset.color_to_class
    }, "vit_segmentation_model_batch16.pth")
