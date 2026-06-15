# Comparative Study of ANN and CNN on CIFAR-10 Dataset

## Project Description

This project explores the effectiveness of deep learning models for image classification using the CIFAR-10 benchmark dataset.

## Objectives

- Build an ANN-based image classifier
- Build a CNN-based image classifier
- Compare model performance
- Analyze training strategies

## About CIFAR-10

- 60,000 RGB images
- 10 classes
- 50,000 training images
- 10,000 testing images

## Methodology

### Data Preprocessing
- Dataset loading
- Pixel normalization
- Reshaping images for ANN

### ANN Model
Fully connected neural network with dropout regularization.

### CNN Model
- Convolution Layers
- ReLU Activation
- Batch Normalization
- Max Pooling
- Dropout

## Training Improvements

1. Increased ANN depth
2. CNN filter progression (32 → 64 → 128)
3. Training for 20 epochs
4. Early Stopping callback
5. Data Augmentation

## Performance Comparison

| Model | Accuracy |
|---------|---------|
| ANN | 43.23% |
| CNN | 72.42% |
| Augmented CNN | 46.32% |

## Key Findings

- CNN outperformed ANN.
- Data augmentation improved generalization.
- Early stopping reduced overfitting.

## Future Scope

- Transfer Learning
- Hyperparameter Tuning
- ResNet Architectures
