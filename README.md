# Iris Segmentation using DINOv3 for Iris Image Analysis

## Project Overview
This project demonstrates how to perform iris segmentation using DINOv3 (a self-supervised vision transformer).  
It covers dataset preparation, model training, evaluation, and visualization for segmenting the iris region in eye images.  
The workflow is useful for applications in biometrics, medical imaging, and computer vision research.

---

## What You Will Learn
By exploring this project, you will learn how to:
- Organize datasets for iris segmentation (images and masks).
- Train a segmentation model using DINOv3 features.
- Apply preprocessing techniques for image-mask pairs.
- Evaluate segmentation quality with accuracy and visualization.
- Implement few-shot learning to train models effectively with limited labeled data.
- Save and load trained models for testing.
- Extend this workflow for other medical or biometric segmentation tasks.

---

## Tech Stack
- Python 3.9+ – [Python](https://www.python.org/) programming language  
- PyTorch – [PyTorch](https://pytorch.org/) deep learning framework  
- DINOv3 – [DINOv3](https://github.com/facebookresearch/dinov3) self-supervised vision transformer backbone  
- OpenCV – [OpenCV](https://opencv.org/) for image preprocessing and visualization  
- Pillow (PIL) – [Pillow](https://python-pillow.org/) for image loading and manipulation  
- NumPy & Pandas – [NumPy](https://numpy.org/) and [Pandas](https://pandas.pydata.org/) for numerical computations and data handling  
- Matplotlib & SciPy – [Matplotlib](https://matplotlib.org/) for plotting, [SciPy](https://scipy.org/) for signal processing  
- Scikit-learn – [scikit-learn](https://scikit-learn.org/) for evaluation metrics and baseline models (e.g., Logistic Regression)  
- tqdm – [tqdm](https://github.com/tqdm/tqdm) for progress bars  
- Requests & Subprocess – [Requests](https://docs.python-requests.org/) for downloading datasets, subprocess for running shell commands  


---

## Example Workflow
1. Prepare dataset with eye images and segmentation masks.
2. Extract DINOv3 features and train segmentation model.
3. Evaluate model on test data.
4. Save predictions, plots, and trained model checkpoints.

---

## Outputs
- Segmentation masks for iris regions.
- Training plots for loss and accuracy.
- Saved trained model checkpoints.
- Example predictions comparing ground-truth vs. predicted masks.

---

## Installation
Clone the repo and install dependencies:

```bash
git clone https://github.com/bhuiyanra/iris-segmentation-dinov3.git
cd iris-segmentation-dinov3
conda env create -f environment.yml
