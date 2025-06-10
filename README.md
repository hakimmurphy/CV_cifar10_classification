# CIFAR-10 Image Classification

This repository contains a deep learning project for classifying images in the CIFAR-10 dataset using a Convolutional Neural Network (CNN) implemented in TensorFlow/Keras.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Dataset](#dataset)
* [Model Architecture](#model-architecture)
* [Installation](#installation)
* [Usage](#usage)
* [Training Results](#training-results)
* [Key Findings](#key-findings)
* [Next Steps](#next-steps)
* [Folder Structure](#folder-structure)

---

## Project Overview

The goal of this project is to build and train a CNN that can accurately classify 32×32 color images into one of 10 categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).

The workflow includes:

1. Loading and preprocessing the CIFAR-10 dataset.
2. Defining a CNN architecture.
3. Training the model with real-time data augmentation.
4. Evaluating performance on validation data.
5. Visualizing loss and accuracy curves.

## Dataset

* **CIFAR-10**: 60,000 images (50K train / 10K test)
* 10 classes, 6,000 images per class
* Images are 32×32 pixels with three color channels (RGB)

Built-in Keras utility is used to download and preprocess the data.

## Model Architecture

The CNN consists of:

* **Conv Blocks**: Multiple convolutional layers with ReLU activation, followed by max pooling.
* **Batch Normalization**: To accelerate convergence and improve stability.
* **Dropout Layers**: For regularization.
* **Fully Connected Layers**: To map to the 10 output classes.

A summary of key layers:

```plaintext
Input: 32×32×3 image
Pretrained ResNet50 (include
_
top=False) backbone
Flatten output maps
Dense(512, relu) → BatchNorm → Dropout(0.5)
Dense(256, relu) → BatchNorm → Dropout(0.2)
Dense(10, softmax) output
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/cifar10-classification.git
   cd cifar10-classification
   ```

2. Create a virtual environment and activate it:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\\Scripts\\activate   # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Open the Jupyter notebook:

   ```bash
   jupyter notebook Hakim_Murphy_CV_cifar10.ipynb
   ```

2. Run all cells to:

   * Load and preprocess data
   * Build and compile the model
   * Train with data augmentation
   * Plot loss and accuracy curves

3. Adjust hyperparameters (learning rate, batch size, epochs) in the notebook as needed.

## Training Results

After training for 10 epochs, the model achieved:

* **Training Loss**: \~0.32
* **Validation Loss**: \~0.83
* **Training Accuracy**: \~90%
* **Validation Accuracy**: \~75%

## Key Findings

* **Fast initial learning**: Rapid drop in loss and rise in accuracy in the first 5–6 epochs.
* **Healthy generalization** through epoch 6: Training and validation metrics closely track.
* **Mild overfitting** after epoch 6: Validation plateaus while training improves.

## Next Steps

* Implement **EarlyStopping** callback to halt training when validation stops improving.
* Apply stronger **regularization** (dropout, weight decay) or **augmentations** to reduce overfitting.
* Introduce a **learning rate scheduler** (e.g., ReduceLROnPlateau).
* Save the best model weights using **ModelCheckpoint**.

## Folder Structure

```plaintext
├── figures/                 # Generated plots (loss_accuracy.png)
├── Hakim_Murphy_CV_cifar10.ipynb  # Jupyter notebook
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── .gitignore               # Git ignore rules
```
