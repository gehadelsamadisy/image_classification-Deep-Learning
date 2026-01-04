# CIFAR-10 Image Classification with Convolutional Neural Networks

A deep learning project implementing and comparing CNN architectures for image classification on the CIFAR-10 dataset.

## 📋 Overview

This project demonstrates the development and comparison of two CNN models:

- **Baseline Model**: A simple 2-layer CNN for establishing performance benchmarks
- **Improved Model**: An enhanced architecture with batch normalization, dropout, and deeper convolutional blocks

## 🎯 Results

| Model           | Test Accuracy |
| --------------- | ------------- |
| Baseline CNN    | 67.66%        |
| Improved CNN    | **83.65%**    |
| **Improvement** | +15.99%       |

## 📊 Dataset

**CIFAR-10** consists of 60,000 32×32 color images across 10 classes:

| Class      | Examples |
| ---------- | -------- |
| Airplane   | ✈️       |
| Automobile | 🚗       |
| Bird       | 🐦       |
| Cat        | 🐱       |
| Deer       | 🦌       |
| Dog        | 🐕       |
| Frog       | 🐸       |
| Horse      | 🐴       |
| Ship       | 🚢       |
| Truck      | 🚚       |

**Split:**

- Training: 32,000 images
- Validation: 8,000 images
- Test: 10,000 images

## 🏗️ Model Architectures

### Baseline Model

```
Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Flatten → Dense(64) → Dense(10)
```

### Improved Model

```
[Conv Block 1] → [Conv Block 2] → [Conv Block 3] → Dense Layers → Output

Each Conv Block:
├── Conv2D + BatchNorm + ReLU
├── Conv2D + BatchNorm + ReLU
├── MaxPooling2D
└── Dropout(0.25)

Dense Layers:
├── Dense(256) + BatchNorm + Dropout(0.5)
├── Dense(128) + BatchNorm + Dropout(0.5)
└── Dense(10, softmax)
```

## 🛠️ Key Techniques

- **Batch Normalization**: Stabilizes training and allows higher learning rates
- **Dropout**: Prevents overfitting (0.25 after conv blocks, 0.5 after dense layers)
- **Padding='same'**: Preserves spatial dimensions through convolutions
- **Learning Rate Scheduling**: ReduceLROnPlateau reduces LR when validation loss plateaus
- **Early Stopping**: Stops training when validation loss stops improving

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

### Running the Notebook

1. Open `img-class.ipynb` in Jupyter Notebook or Google Colab
2. Run all cells sequentially
3. Training takes approximately 5-10 minutes on GPU

## 📈 Training Callbacks

| Callback          | Configuration                          |
| ----------------- | -------------------------------------- |
| ReduceLROnPlateau | factor=0.5, patience=3, min_lr=1e-6    |
| EarlyStopping     | patience=10, restore_best_weights=True |


```

## 🔍 Visualizations

The notebook includes:

- Training/validation accuracy and loss curves
- Sample predictions with true vs predicted labels
- Color-coded results (green = correct, red = incorrect)

## 💡 Key Findings

1. **Deeper architectures** with more convolutional blocks capture hierarchical features better
2. **Batch normalization** significantly speeds up training convergence
3. **Dropout regularization** effectively reduces overfitting
4. **Learning rate scheduling** helps fine-tune the model in later epochs

## 🛡️ Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Matplotlib
- scikit-learn

## 📝 License

This project is for educational purposes.
