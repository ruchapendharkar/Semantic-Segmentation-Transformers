import torch
from torchvision.transforms import ToTensor, Normalize, Resize
from transformers import SegformerForSemanticSegmentation, AutoImageProcessor
from segment_anything import SamPredictor, sam_model_registry
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ==== Constants ====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UNET_IMAGE_SIZE = (256, 256)  # UNet input size
SEGFORMER_SAM_IMAGE_SIZE = (512, 512)  # SegFormer and SAM input size
NUM_CLASSES = 29  # Update according to your dataset

# ==== Dataset Class ====
class KittiNPYDataset:
    def __init__(self, images_path, masks_path):
        self.images = np.load(images_path)
        self.masks = np.load(masks_path) if masks_path else None
        print(f"Loaded dataset: {len(self.images)} images.")
        if masks_path:
            print(f"Masks shape: {self.masks.shape}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx] if self.masks is not None else None
        return image, mask

# ==== Load Models ====
# Load UNet model
unet_model = tf.keras.models.load_model("Segmentation-Model-UNET.h5")

# Load SegFormer model
segformer_model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512",
    num_labels=NUM_CLASSES,
    id2label={str(i): str(i) for i in range(NUM_CLASSES)},
    label2id={str(i): i for i in range(NUM_CLASSES)},
    ignore_mismatched_sizes=True
)
segformer_model.load_state_dict(torch.load(r"C:\Users\yutik\Semantic-segmentation\segformer_model.pth", map_location=DEVICE))
segformer_model.to(DEVICE)
segformer_processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

# Load SAM model
sam_model = sam_model_registry["vit_b"](checkpoint=r"C:\Users\yutik\Semantic-segmentation\sam_vit_b_01ec64.pth")
sam_model.to(DEVICE)
sam_predictor = SamPredictor(sam_model)

# ==== Helper Functions ====
def preprocess_image(image):
    """
    Preprocess the image for all models, considering different input sizes.
    Args:
        image (np.ndarray): Input image array.
    Returns:
        dict: Preprocessed inputs for UNet, SegFormer, and SAM.
    """
    # Ensure image is in range [0, 255] and has 3 channels
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    # Resize image for UNet (256x256)
    unet_image = np.array(Image.fromarray(image).resize(UNET_IMAGE_SIZE))

    # Preprocessing for UNet (TensorFlow)
    unet_input = unet_image / 255.0  # Normalize to [0, 1]
    unet_input = np.expand_dims(unet_input, axis=0)  # Add batch dimension

    # Resize image for SegFormer and SAM (512x512)
    segformer_sam_image = np.array(Image.fromarray(image).resize(SEGFORMER_SAM_IMAGE_SIZE))

    # Preprocessing for SegFormer (PyTorch)
    segformer_input = segformer_processor(images=segformer_sam_image, return_tensors="pt")
    segformer_input = {k: v.to(DEVICE) for k, v in segformer_input.items()}

    # Preprocessing for SAM
    sam_predictor.set_image(segformer_sam_image)  # Set image for SAM predictor

    return {
        "unet": unet_input,
        "segformer": segformer_input,
        "sam": segformer_sam_image,
        "original": image  # Original for visualization
    }

def get_segformer_prediction(model, inputs):
    """
    Get predictions from SegFormer and reduce them to 2D.
    """
    outputs = model(pixel_values=inputs["pixel_values"])
    preds = torch.argmax(outputs.logits, dim=1).squeeze(0).cpu().numpy()  # Shape: (H, W)
    return preds

def get_sam_prediction(predictor, point_coords=None, point_labels=None):
    """
    Get predictions from SAM.
    """
    if point_coords is None:
        h, w = SEGFORMER_SAM_IMAGE_SIZE
        point_coords = np.array([[h // 2, w // 2]])  # Center point
        point_labels = np.array([1])  # Assume foreground label

    masks, _, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=False
    )
    return masks[0]  # First mask (binary mask)

def visualize_results(image, unet_mask, segformer_mask, sam_mask, ground_truth=None):
    """
    Visualize input image and predictions from all models.
    """
    plt.figure(figsize=(15, 10))

    # Input image
    plt.subplot(2, 3, 1)
    plt.imshow(image.astype(np.uint8))
    plt.title("Input Image")
    plt.axis("off")

    # Ground truth
    if ground_truth is not None:
        plt.subplot(2, 3, 2)
        plt.imshow(ground_truth, cmap="viridis")
        plt.title("Ground Truth")
        plt.axis("off")

    # UNet Prediction
    plt.subplot(2, 3, 3)
    plt.imshow(unet_mask, cmap="viridis")
    plt.title("UNet Prediction")
    plt.axis("off")

    # SegFormer Prediction
    plt.subplot(2, 3, 4)
    plt.imshow(segformer_mask, cmap="viridis")
    plt.title("SegFormer Prediction")
    plt.axis("off")

    # SAM Prediction
    plt.subplot(2, 3, 5)
    plt.imshow(sam_mask, cmap="gray")
    plt.title("SAM Prediction")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# ==== Main Code ====
print("\nLoading dataset...")
val_dataset = KittiNPYDataset(
    images_path=r"C:\Users\yutik\Semantic-segmentation\kitti_test_images.npy",
    masks_path=r"C:\Users\yutik\Semantic-segmentation\kitti_test_masks.npy"
)

for idx in range(len(val_dataset)):
    image, ground_truth = val_dataset[idx]

    # Preprocess the image
    inputs = preprocess_image(image)

    # UNet Prediction
    unet_prediction = unet_model.predict(inputs["unet"])[0].argmax(axis=-1)  # Shape: (256, 256)

    # SegFormer Prediction
    segformer_prediction = get_segformer_prediction(segformer_model, inputs["segformer"])  # Shape: (512, 512)

    # SAM Prediction
    sam_prediction = get_sam_prediction(sam_predictor)  # Shape: (512, 512)

    # Visualize Results
    visualize_results(inputs["original"], unet_prediction, segformer_prediction, sam_prediction, ground_truth)
