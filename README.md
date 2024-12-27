# Logistic and Support Vector Machine Regression for MNIST Classification

This project implements logistic regression, multi-class logistic regression, and support vector machines (SVMs) for classifying handwritten digits from the MNIST dataset. It includes preprocessing steps, gradient-based optimization for training models, and visualization of accuracy.

## Overview
The project uses logistic regression and SVMs to classify images of digits in the MNIST dataset, focusing on:
- Binary and multi-class logistic regression using gradient descent.
- SVM models with linear and RBF kernels.
- Parameter tuning for SVMs, including hyperparameters like `C` and `gamma`.
- Performance evaluation on training, validation, and test datasets.

## Key Features

### 1. Data Preprocessing
- Loads and splits the MNIST dataset into training, validation, and test sets.
- Normalizes pixel values to the range [0, 1].
- Removes features with minimal variance for better model performance.

### 2. Logistic Regression
- **Binary Logistic Regression**:
  - Implements gradient-based optimization using `scipy.optimize.minimize`.
  - Computes the error function and gradient for binary classification.
- **Multi-class Logistic Regression**:
  - Extends binary logistic regression to multi-class classification using the softmax function.

### 3. Support Vector Machines (SVM)
- Trains SVM models with linear and RBF kernels using `scikit-learn`.
- Evaluates model accuracy on training, validation, and test datasets.
- Explores the effect of varying `C` values for RBF kernels on classification accuracy.

### 4. Visualization
- Plots accuracy for different `C` values in RBF kernels.
- Provides training, validation, and test accuracies for comparative analysis.

## Results
- **Logistic Regression**:
  - Achieved high accuracy on training and test datasets using both binary and multi-class approaches.
- **SVM**:
  - Linear kernel SVM achieved competitive accuracy.
  - RBF kernel SVM showed improved performance with tuned hyperparameters.

## Getting Started

### Prerequisites
Install the required libraries:
```bash
pip install numpy scipy matplotlib scikit-learn
