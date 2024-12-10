import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix

def compute_metrics(y_true, y_pred, num_classes):
    """
    Compute mIoU and Pixel Accuracy.
    Args:
        y_true (numpy.ndarray): Ground truth masks.
        y_pred (numpy.ndarray): Predicted masks.
        num_classes (int): Number of classes.
    Returns:
        dict: mIoU and pixel accuracy.
    """
    y_pred = np.argmax(y_pred, axis=-1)
    y_true = np.argmax(y_true, axis=-1)

    y_pred_flat = y_pred.flatten()
    y_true_flat = y_true.flatten()

    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=list(range(num_classes)))

    iou_per_class = []
    for i in range(num_classes):
        intersection = cm[i, i]
        union = cm[i, :].sum() + cm[:, i].sum() - cm[i, i]
        if union == 0:
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append(intersection / union)

    mean_iou = np.nanmean(iou_per_class)
    pixel_accuracy = np.sum(np.diag(cm)) / np.sum(cm)

    return {"mIoU": mean_iou, "pixel_accuracy": pixel_accuracy}

def plot_metrics(metrics, history=None):
    """
    Plot metrics and loss.
    Args:
        metrics (dict): mIoU and pixel accuracy values.
        history (dict): Training history containing loss and accuracy values.
    """
    plt.figure(figsize=(15, 5))

    if history:
        # Plot loss
        loss = history['loss']
        val_loss = history['val_loss']
        epochs = range(1, len(loss) + 1)

        plt.subplot(1, 2, 1)
        plt.plot(epochs, loss, 'b', label="Training Loss")
        plt.plot(epochs, val_loss, 'r', label="Validation Loss")
        plt.title("Training vs Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()

    # Plot mIoU and Pixel Accuracy
    plt.subplot(1, 2, 2)
    plt.plot([metrics["mIoU"]], 'b-', label="mIoU")
    plt.plot([metrics["pixel_accuracy"]], 'g-', label="Pixel Accuracy")
    plt.title("Validation Metrics")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig("Metrics_and_Loss_Plot.png")
    plt.show()

def main():
    # File paths
    test_imgs_path = r'C:\Users\yutik\Semantic-segmentation\kitti_test_images.npy'
    test_masks_path = r'C:\Users\yutik\Semantic-segmentation\kitti_test_masks.npy'
    model_path = "Segmentation-Model-UNET.h5"
    num_classes = 29

    # Load test data
    print("Loading test data...")
    test_imgs = np.load(test_imgs_path)
    test_masks = np.load(test_masks_path)
    print(f"Test data shape: {test_imgs.shape}, {test_masks.shape}")

    # Load trained model
    print("Loading trained model...")
    model = load_model(model_path)
    print("Model loaded successfully!")

    # If you have history stored (optional)
    try:
        history = np.load("training_history.npy", allow_pickle=True).item()
    except FileNotFoundError:
        history = None
        print("Training history file not found. Skipping loss plot.")

    # Make predictions
    print("Making predictions on test data...")
    predictions = model.predict(test_imgs, verbose=1)

    # Compute metrics
    print("Computing metrics...")
    metrics = compute_metrics(test_masks, predictions, num_classes)
    print(f"mIoU: {metrics['mIoU']:.4f}")
    print(f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")

    # Plot metrics and results
    plot_metrics(metrics, history)

if __name__ == "__main__":
    main()
