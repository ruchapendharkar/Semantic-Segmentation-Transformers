'''
data-preprocess.py
This file preprocesses the images
'''
import cv2
from PIL import Image
import numpy as np
from patchify import patchify 
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import random
import os

# Calculates the values for cropping the image
def calculateCropSize(image, patchSize):
    sizeX = (image.shape[1] // patchSize) * patchSize
    sizeY = (image.shape[0] // patchSize) * patchSize
    return sizeX, sizeY

# Converts the images into patches
def patchImages(image, patchSize):
    # Convert image to numpy array
    image = np.array(image)
    # Split the image into small patches specified by patch size
    patchedImage = patchify(image, (patchSize, patchSize, 3), step=patchSize)
    return patchedImage

# Apply Min-Max scaling and normalize the image
def normalizeImage(patchedImage):
    scaler = MinMaxScaler()
    scaled_patch = scaler.fit_transform(patchedImage.reshape(-1, 3)).reshape(patchedImage.shape)
    scaled_patch = scaled_patch[0] 
    # Ensure pixel values are within [0, 1] range
    scaled_patch = np.clip(scaled_patch, 0, 1)
    return scaled_patch

# Preprocess Images and Masks  
def processImages(imageFolder, maskFolder, patchSize=256):
    dataset = []
    for filename in os.listdir(imageFolder):
        if filename.endswith(".png"):
            image_path = os.path.join(imageFolder, filename)
            mask_path = os.path.join(maskFolder, filename)

            # Read image and mask
            testImage = cv2.imread(image_path)
            testMask = cv2.imread(mask_path)

            if testImage is not None and testMask is not None:
                # Convert masks to RGB
                testMask = cv2.cvtColor(testMask, cv2.COLOR_BGR2RGB)

                # Calculate crop size
                sizeX, sizeY = calculateCropSize(testImage, patchSize)

                # Crop and patch images
                image = Image.fromarray(testImage)
                cropped_image = image.crop((0, 0, sizeX, sizeY))
                patched_image = patchImages(cropped_image, patchSize)

                mask = Image.fromarray(testMask)
                cropped_mask = mask.crop((0, 0, sizeX, sizeY))
                patched_mask = patchImages(cropped_mask, patchSize)

                for i in range(patched_image.shape[0]):
                    for j in range(patched_image.shape[1]):
                        patchedImage = patched_image[i, j, :, :]
                        patchedMask = patched_mask[i, j, :, :]

                        scaledImage = normalizeImage(patchedImage)
                        dataset.append((scaledImage, patchedMask))
    
    return dataset

# Main function
def main():
    # Paths to KITTI dataset folders
    datasetFolder = r'C:\Data\Northeastern\Advanced Computer Vision\Final Project\data_semantics\training'
    imageFolder = os.path.join(datasetFolder, 'image_2')
    maskFolder = os.path.join(datasetFolder, 'semantic_rgb')

    # Preprocess images and masks
    processedDataset = processImages(imageFolder, maskFolder)

    # Separate images and masks
    images, masks = zip(*processedDataset)

    # Save processed images and masks separately
    np.save(r'C:\Data\Northeastern\Advanced Computer Vision\Final Project\Semantic-Segmentation-Transformers\processed_kitti_images.npy', images)
    np.save(r'C:\Data\Northeastern\Advanced Computer Vision\Final Project\Semantic-Segmentation-Transformers\processed_kitti_masks.npy', masks)

    print("Done!")
    # Test plot
    random_idx = random.randint(0, len(images) - 1)

    # Squeeze any extra dimensions from the image and mask
    image_to_plot = np.squeeze(images[random_idx])
    mask_to_plot = np.squeeze(masks[random_idx])

    plt.figure(figsize=(14, 8))
    plt.subplot(121)
    plt.title("Image")
    plt.imshow(image_to_plot)
    plt.subplot(122)
    plt.title("Mask")
    plt.imshow(mask_to_plot)
    plt.show()

if __name__ == "__main__":
    main()
