import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
from model import getNetwork

def compute_metrics_epoch(y_true, y_pred, num_classes):
    """
    Compute mIoU and Pixel Accuracy for each epoch.
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

    return mean_iou, pixel_accuracy

def train_network_with_metrics(train_imgs_path, train_masks_path, test_imgs_path, test_masks_path, num_classes=29):
    model = getNetwork()

    train_imgs = np.load(train_imgs_path)
    train_masks = np.load(train_masks_path)
    test_imgs = np.load(test_imgs_path)
    test_masks = np.load(test_masks_path)

    model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])

    history = {
        "loss": [],
        "val_loss": [],
        "accuracy": [],
        "val_accuracy": [],
        "val_miou": [],
        "val_pixel_acc": []
    }

    epochs = 100
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.fit(train_imgs, train_masks, batch_size=16, verbose=0, epochs=1, shuffle=False)

        train_loss, train_acc = model.evaluate(train_imgs, train_masks, verbose=0)
        val_loss, val_acc = model.evaluate(test_imgs, test_masks, verbose=0)
        predictions = model.predict(test_imgs, verbose=0)
        miou, pixel_acc = compute_metrics_epoch(test_masks, predictions, num_classes)

        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_miou"].append(miou)
        history["val_pixel_acc"].append(pixel_acc)

    return history

def plot_metrics(history):
    """
    Plot accuracy, loss, mIoU, and pixel accuracy over epochs.
    """
    epochs = range(1, len(history["loss"]) + 1)

    plt.figure(figsize=(20, 5))

    # Loss plot
    plt.subplot(1, 4, 1)
    plt.plot(epochs, history["loss"], label="Training Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.title("Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 4, 2)
    plt.plot(epochs, history["accuracy"], label="Training Accuracy")
    plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy")
    plt.title("Accuracy Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # mIoU plot
    plt.subplot(1, 4, 3)
    plt.plot(epochs, history["val_miou"], label="Validation mIoU")
    plt.title("mIoU Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("mIoU")
    plt.legend()

    # Pixel Accuracy plot
    plt.subplot(1, 4, 4)
    plt.plot(epochs, history["val_pixel_acc"], label="Validation Pixel Accuracy")
    plt.title("Pixel Accuracy Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Pixel Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig("Training_Validation_Metrics.png")
    plt.show()

def main():
    train_imgs_path = r'C:\Data\Northeastern\Advanced Computer Vision\Final Project\Semantic-Segmentation-Transformers\kitti_test_images.npy'
    train_masks_path = r'C:\Data\Northeastern\Advanced Computer Vision\Final Project\Semantic-Segmentation-Transformers\kitti_train_masks.npy'
    test_imgs_path = r'C:\Data\Northeastern\Advanced Computer Vision\Final Project\Semantic-Segmentation-Transformers\kitti_test_images.npy'
    test_masks_path = r'C:\Data\Northeastern\Advanced Computer Vision\Final Project\Semantic-Segmentation-Transformers\kitti_train_masks.npy'

    history = train_network_with_metrics(train_imgs_path, train_masks_path, test_imgs_path, test_masks_path)
    plot_metrics(history)

if __name__ == "__main__":
    main()
