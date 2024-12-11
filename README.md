# **Comparison of U-Net and SegFormer for Semantic Segmentation on KITTI Dataset**

## **Introduction**
This project compares the performance of two semantic segmentation models, **U-Net** and **SegFormer**, on the KITTI dataset. The objective is to evaluate their accuracy, efficiency, and scalability for real-world autonomous driving scenarios.

## **Dataset**
- **Dataset:** KITTI Semantic Segmentation Dataset
- **Preprocessed Format:** Stored as `.npy` files for efficient loading
- **Classes:** 29 classes
- **Image Resolution:** 256x256 for U-Net, 512x512 for SegFormer
- Download the KITTI dataset from the official website: [KITTI Dataset](https://www.cvlibs.net/datasets/kitti/).

---

## **Requirements**
Install the required dependencies for the project:
```bash
pip install -r requirements.txt
```
---
## Data Preprocessing
The following scripts preprocess the KITTI dataset for semantic segmentation tasks and dynamically generate class labels from RGB masks. The preprocessing workflow includes cropping and patching images, normalizing pixel values, dynamically mapping RGB values to class indices, and splitting the dataset into training and testing sets and saves them in .npy format.

To preprocess the data, run the following commands:

1. ``` python data-preprocess.py```
2. ```python labels.py```
---
## U-Net
To train U-Net based on the model defined in ```model.py``` run the following script:
```bash
python unet.py
```
---
## Segformer
To trained Segformer run the following script:
```bash
python segformer.py
```
---
## Results and Comparison
To view the results and compare the trained models, run the following script:
```bash
python comparison.py
```
---
## Conclusion
This project provides insights into the strengths and weaknesses of U-Net and SegFormer for semantic segmentation tasks. Detailed results and visulizations and are presented in the ```plots/``` folder.