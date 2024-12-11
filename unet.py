"""
unet.py
This file trains the U-Net model and tracks metrics every 10 epochs
"""
from model import getNetwork
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def compute_metrics(y_true, y_pred, num_classes):
   """
   Compute mIoU and Pixel Accuracy.
   Args:
       y_true: Ground truth masks
       y_pred: Predicted masks
       num_classes: Number of segmentation classes
   Returns:
       dict: mIoU and pixel accuracy values
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

def train_network(train_imgs_path, train_masks_path, test_imgs_path, test_masks_path, num_classes=29):
   """
   Train network and track metrics every 10 epochs.
   """
   model = getNetwork()
   print("Loading data...")
   train_imgs = np.load(train_imgs_path)
   train_masks = np.load(train_masks_path) 
   test_imgs = np.load(test_imgs_path)
   test_masks = np.load(test_masks_path)
   
   model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])
   
   history = {
       'loss': [],
       'accuracy': [],
       'val_loss': [],
       'val_accuracy': [],
       'mIoU': [],
       'pixel_accuracy': [],
       'epochs': []
   }
   
   epochs = 100
   for epoch in range(epochs):
       print(f"Epoch {epoch+1}/{epochs}")
       
       metrics = model.fit(train_imgs, train_masks,
                         batch_size=16,
                         verbose=1,
                         epochs=1,
                         validation_data=(test_imgs, test_masks),
                         shuffle=False)
       
       history['loss'].append(metrics.history['loss'][0])
       history['accuracy'].append(metrics.history['accuracy'][0])
       history['val_loss'].append(metrics.history['val_loss'][0])
       history['val_accuracy'].append(metrics.history['val_accuracy'][0])
       
       # Compute IoU metrics every 10 epochs
       if (epoch + 1) % 10 == 0:
           print(f"\nComputing metrics for epoch {epoch+1}...")
           predictions = model.predict(test_imgs)
           epoch_metrics = compute_metrics(test_masks, predictions, num_classes)
           history['mIoU'].append(epoch_metrics['mIoU'])
           history['pixel_accuracy'].append(epoch_metrics['pixel_accuracy'])
           history['epochs'].append(epoch + 1)
           
           print(f"Epoch {epoch+1} Metrics:")
           print(f"mIoU: {epoch_metrics['mIoU']:.4f}")
           print(f"Pixel Accuracy: {epoch_metrics['pixel_accuracy']:.4f}\n")
   
   model.save('unet_model.h5')
   print("Model saved!")
   
   return history

def plot_metrics(history):
   """
   Plot training metrics.
   """
   plt.figure(figsize=(20, 5))
   
   # Accuracy Plot
   plt.subplot(141)
   plt.plot(range(1, len(history['accuracy']) + 1), history['accuracy'], 'b-', label='Training')
   plt.plot(range(1, len(history['val_accuracy']) + 1), history['val_accuracy'], 'r-', label='Validation')
   plt.title('Accuracy Over Epochs')
   plt.xlabel('Epochs')
   plt.ylabel('Accuracy')
   plt.legend()
   plt.grid(True)
   
   # Loss Plot
   plt.subplot(142)
   plt.plot(range(1, len(history['loss']) + 1), history['loss'], 'b-', label='Training')
   plt.plot(range(1, len(history['val_loss']) + 1), history['val_loss'], 'r-', label='Validation')
   plt.title('Loss Over Epochs')
   plt.xlabel('Epochs')
   plt.ylabel('Loss')
   plt.legend()
   plt.grid(True)
   
   # mIoU Plot
   plt.subplot(143)
   plt.plot(history['epochs'], history['mIoU'], 'g-o')
   plt.title('mIoU (Every 10 Epochs)')
   plt.xlabel('Epochs')
   plt.ylabel('mIoU')
   plt.grid(True)
   
   # Pixel Accuracy Plot
   plt.subplot(144)
   plt.plot(history['epochs'], history['pixel_accuracy'], 'm-o')
   plt.title('Pixel Accuracy (Every 10 Epochs)')
   plt.xlabel('Epochs')
   plt.ylabel('Pixel Accuracy')
   plt.grid(True)
   
   plt.tight_layout()
   plt.savefig('UNET-training_metrics.png', dpi=300, bbox_inches='tight')
   plt.close()

def save_metrics_to_csv(history):
   """
   Save metrics to CSV file.
   """
   metrics_dict = {
       'epoch': history['epochs'],
       'mIoU': history['mIoU'],
       'pixel_accuracy': history['pixel_accuracy']
   }
   np.savetxt('UNET-metrics.csv', 
              np.column_stack([metrics_dict[key] for key in metrics_dict.keys()]), 
              delimiter=',',
              header='epoch,mIoU,pixel_accuracy',
              comments='')
   print("Metrics saved to metrics.csv")

def main():
   train_imgs_path = "kitti_train_images.npy"
   train_masks_path = "kitti_train_masks.npy"
   test_imgs_path = "kitti_test_images.npy"
   test_masks_path = "kitti_test_masks.npy"
   
   print("Starting training...")
   history = train_network(train_imgs_path, train_masks_path, 
                         test_imgs_path, test_masks_path)
   
   print("Plotting metrics...")
   plot_metrics(history)
   
   print("Saving metrics...")
   save_metrics_to_csv(history)
   
   print("Done!")

if __name__ == "__main__":
   main()