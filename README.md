# Transfer Learning on CIFAR-10 — ResNet-18, AlexNet, MobileNetV2

Fine-tunes ImageNet-pretrained CNNs from `torchvision` on CIFAR-10 and
tracks loss, accuracy, F1, precision and recall per epoch. Written in PyTorch
for a University of Adelaide deep learning course.

Everything is in one script: [`Test.py`](Test.py).

---

## What the script does

1. **Data** — downloads CIFAR-10 to `./data` via `torchvision.datasets.CIFAR10`.
   - The official 50,000-image training set is split 80/20 into
     **40,000 train / 10,000 validation** (`random_split`, unseeded).
   - The official **10,000-image test set** is used for the final confusion matrix.
   - Optional augmentation: `RandomCrop(32, padding=4)` + `RandomHorizontalFlip`.
   - Normalised with mean `(0.4911, 0.4722, 0.4565)`, std `(0.2023, 0.1994, 0.2010)`.
   - Images are resized to 224×224 for AlexNet; ResNet-18 and MobileNetV2 run at 32×32.
2. **Model** — `get_model(model_name)` loads ImageNet weights and replaces the
   final layer with a 10-class `Linear`:

   | `model_name` | Base weights | Layer replaced |
   |---|---|---|
   | `resnet18` | `ResNet18_Weights.IMAGENET1K_V1` | `fc` → `Linear(512, 10)` |
   | `alexnet` | `AlexNet_Weights.IMAGENET1K_V1` | `classifier[6]` → `Linear(4096, 10)` |
   | `mobilenet` | `MobileNet_V2_Weights.IMAGENET1K_V1` | `classifier[1]` → `Linear(1280, 10)` |

   All layers are trained (full fine-tuning, nothing frozen).
3. **Training** — `train_model()` trains with `CrossEntropyLoss`, prints
   per-epoch train/val loss and accuracy plus weighted F1, precision and
   recall on the validation set, then plots those curves.
4. **Test** — `test_confusion_matrix()` plots a confusion matrix on the test set.

## Default run

`python Test.py` calls:

```python
main(model_name="resnet18", batch_size=64, learning_rate=0.001,
     optimizer_type="adam", num_epochs=10, data_augmentation=True)
```

| Setting | Default | Options |
|---|---|---|
| Model | `resnet18` | `resnet18`, `alexnet`, `mobilenet` |
| Optimizer | Adam, lr 0.001 | `optimizer_type="sgd"` → SGD, momentum 0.9, weight decay 5e-4 |
| Epochs | 10 | any |
| Batch size | 64 | any |
| Augmentation | on | `data_augmentation=False` |

To compare models, edit the `main(...)` call at the bottom of `Test.py`.

## How to run

```bash
git clone https://github.com/chandr-csgp/CNNs-for-Image-Classification.git
cd CNNs-for-Image-Classification
pip install torch torchvision scikit-learn matplotlib seaborn numpy
python Test.py
```

CIFAR-10 (~170 MB) downloads automatically on first run. A GPU is strongly
recommended, especially for AlexNet at 224×224.

## Known limitations

- **No saved results.** The script prints metrics and shows plots but writes
  nothing to disk, so no accuracy numbers are committed here — run it to
  reproduce them.
- **Test accuracy isn't printed** — only the confusion matrix is shown for the
  test set.
- **Validation images are augmented.** The train/val split is taken after the
  training transform is applied, so validation images also get random
  crops/flips when augmentation is on.
- **Unseeded split and training**, so results vary slightly between runs.
- ResNet-18 and MobileNetV2 were pretrained on 224×224 images; running them at
  32×32 works but under-uses the pretrained features.

## Tech

Python · PyTorch · torchvision · scikit-learn · Matplotlib · Seaborn

---

**Chandra Sekar Putta** — Master of Data Science, University of Adelaide ·
[LinkedIn](https://linkedin.com/in/chandra-sekar-p-b4b033402) · [GitHub](https://github.com/chandr-csgp)
