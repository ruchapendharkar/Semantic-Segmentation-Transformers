'''
This file segments and displays the results on the images based on the model
'''
import numpy as np
import random
import matplotlib.pyplot as plt
from keras.models import load_model

#Generates predictions based on the test images and model 
def displayPredictions(model_path, test_images_path, test_masks_path, n=1):
    # Load the trained model
    model = load_model(model_path)
    
    # Load test images and masks
    print("...Loading data....")
    test_imgs = np.load(test_images_path) 
    test_masks = np.load(test_masks_path)
    print("Data Loaded!")
    
    # Get the indices of masks
    y_test_argmax = np.argmax(test_masks, axis=3)
    
    # Generate and display images
    for i in range(n):
        
        index = random.randint(0, len(test_imgs) - 1)

        test_image = test_imgs[index]
        mask = y_test_argmax[index]
        test_image_input = np.expand_dims(test_image, 0)

        prediction = model.predict(test_image_input)
        predicted_image = np.argmax(prediction, axis=3)
        predicted_image = predicted_image[0,:,:]

        #display
        plt.figure(figsize=(10,8))
        plt.subplot(131)
        plt.title("Image")
        plt.imshow(test_image)
        plt.subplot(132)
        plt.title("Mask")
        plt.imshow(mask)
        plt.subplot(133)
        plt.title("Prediction")
        plt.imshow(predicted_image)
        plt.show()

# Compares results of the three models
def compareResults(model_path, model2,model3, test_images_path, test_masks_path):
    model1 = load_model(model_path)
    model2 = load_model(model2)
    model3 = load_model(model3)

    # Load test images and masks
    print("...Loading data....")
    test_imgs = np.load(test_images_path) 
    test_masks = np.load(test_masks_path)
    print("Data Loaded!")

    y_test_argmax = np.argmax(test_masks, axis=3)
    

    for i in range(4):
    
        index = random.randint(0, len(test_imgs) - 1)

        test_image = test_imgs[index]
        mask = y_test_argmax[index]
        test_image_input = np.expand_dims(test_image, 0)

        prediction1 = model1.predict(test_image_input)
        predicted_image1 = np.argmax(prediction1, axis=3)
        predicted_image1 = predicted_image1[0,:,:]

        prediction2 = model2.predict(test_image_input)
        predicted_image2 = np.argmax(prediction2, axis=3)
        predicted_image2 = predicted_image2[0,:,:]

        prediction3 = model3.predict(test_image_input)
        predicted_image3 = np.argmax(prediction3, axis=3)
        predicted_image3 = predicted_image3[0,:,:]

        plt.figure(figsize=(10,8))
        plt.subplot(151)
        plt.title("Image")
        plt.imshow(test_image)
        plt.subplot(152)
        plt.title("Mask")
        plt.imshow(mask)
        plt.subplot(153)
        plt.title("UNET")
        plt.imshow(predicted_image1)
        plt.subplot(154)
        plt.title("UNet with Attention")
        plt.imshow(predicted_image2)
        plt.subplot(155)
        plt.title("Pretrained RESNET")
        plt.imshow(predicted_image3)
        plt.show()
    


# main function
def main():

    #Update file paths as needed
    model_path = r'C:\Data\Northeastern\Advanced Computer Vision\Final Project\Semantic-Segmentation-Transformers\Segmentation-Model-UNET.h5'

    test_images_path = r'C:\Data\Northeastern\Advanced Computer Vision\Final Project\Semantic-Segmentation-Transformers\kitti_test_images.npy'
    test_masks_path = r'C:\Data\Northeastern\Advanced Computer Vision\Final Project\Semantic-Segmentation-Transformers\kitti_test_masks.npy'
    displayPredictions(model_path, test_images_path, test_masks_path, n=10)
    #compareResults(model_path, model2, model3, test_images_path, test_masks_path)


if __name__ == "__main__":
    main()