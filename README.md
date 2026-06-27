# 🧠 Transfer Learning & CNN Comparison — CIFAR-10

**Comparing ResNet18, AlexNet, and MobileNetV2 on CIFAR-10 using PyTorch transfer learning**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10-green.svg)](https://www.cs.toronto.edu/~kriz/cifar.html)

---

## 📌 Overview

This project compares three **pre-trained deep learning architectures** fine-tuned on the CIFAR-10 image classification dataset using PyTorch. It demonstrates transfer learning from ImageNet weights and systematic evaluation across multiple metrics.

---

## 🏗️ Models Compared

| Model | Pre-trained On | Final Layer Modified |
|---|---|---|
| **ResNet-18** | ImageNet | `fc` → Linear(512, 10) |
| **AlexNet** | ImageNet | `classifier[6]` → Linear(4096, 10) |
| **MobileNetV2** | ImageNet | `classifier[1]` → Linear(1280, 10) |

---

## 📊 Dataset — CIFAR-10

- **Classes:** 10 (airplane, car, bird, cat, deer, dog, frog, horse, ship, truck)
- **Split:** 80% train / 10% validation / 10% test (from 60K images)
- **Augmentation:** Random crop (32×32, padding=4) + horizontal flip
- **Normalisation:** Mean (0.491, 0.472, 0.457) / Std (0.202, 0.199, 0.201)

---

## ⚙️ Training Setup

| Config | Value |
|---|---|
| Optimizer | Adam (default) or SGD with momentum=0.9 |
| Loss | CrossEntropyLoss |
| Epochs | 10 (configurable) |
| Batch size | 64–128 (configurable) |
| Learning rate | 0.001 (configurable) |

---

## 📈 Metrics Tracked

- Training & validation **loss** and **accuracy** per epoch
- **F1-score** (weighted)
- **Precision** and **Recall** (weighted)
- **Confusion matrix** on test set

---

## 🗂️ Key Code Structure

```python
# Load data with optional augmentation
trainloader, valloader, testloader = load_data(batch_size=64, data_augmentation=True)

# Choose model: 'resnet18', 'alexnet', or 'mobilenet'
model = get_model(model_name="resnet18")

# Train with metrics tracking
train_model(model, trainloader, valloader, optimizer, criterion, num_epochs=10)

# Evaluate on test set with confusion matrix
test_confusion_matrix(model, testloader)
```

---

## 🚀 How to Run

```bash
git clone https://github.com/chandr-csgp/CNN-Transfer-Learning-CIFAR10.git
cd CNN-Transfer-Learning-CIFAR10

pip install torch torchvision scikit-learn matplotlib seaborn

# Run with default settings (ResNet18, Adam, 10 epochs)
python Keras.py
```

---

## 🔧 Tech Stack

`Python` · `PyTorch` · `torchvision` · `scikit-learn` · `Matplotlib` · `Seaborn`

---

## 👤 Author

**Chandra Sekar Putta** — University of Adelaide, MDS 2024–2026  
[LinkedIn](https://linkedin.com/in/chandra-sekar-p-b4b033402) · [GitHub](https://github.com/chandr-csgp)
