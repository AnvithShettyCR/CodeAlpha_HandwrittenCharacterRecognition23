# CodeAlpha_HandwrittenCharacterRecognition23

##  Overview
A Convolutional Neural Network (CNN) that classifies handwritten letters (A-Z) from grayscale images. Built as part of the CodeAlpha Machine Learning Internship (Task 3).

##  Dataset
- **Source:** EMNIST Letters split (loaded via torchvision)
- **Size:** 124,800 training images, 20,800 test images
- **Classes:** 26 (A-Z)
- **Image format:** 28x28 grayscale

##  Data Preparation
- Fixed EMNIST's known image orientation quirk (images ship rotated/mirrored relative to standard format) via flip + rotation transform, verified visually before training
- Normalized pixel values to [0, 1]
- Reshaped to CNN-compatible 4D tensors (samples, 28, 28, 1)
- One-hot encoded labels for multi-class classification

##  Model Architecture
A 3-block CNN:
- 3× Conv2D blocks (32 → 64 → 128 filters) with BatchNormalization and MaxPooling
- Dense layer (128 units) with Dropout (0.5) for regularization
- Softmax output layer (26 classes)
- **Total parameters:** 244,506

##  Results
- **Training accuracy:** 95.64%
- **Validation accuracy:** 94.20%
- **Test set accuracy:** 94%

Training curves show validation accuracy plateauing around epoch 9-10 while training accuracy continues climbing — an early sign of mild overfitting, which could be addressed with early stopping in a production setting.

##  Key Insight — Confusion Analysis
Rather than reporting a single accuracy number, per-class analysis reveals *which* letters the model struggles with and why:

| Confusion Pair | Misclassified Count | Likely Cause |
|---|---|---|
| L → I | 195 | Nearly identical vertical-stroke shapes in handwriting |
| I → L | 146 | Same as above (bidirectional confusion) |
| G → Q | 111 | Similar loop-with-tail structure |
| V → U | ~29 | Rounded vs. pointed base, subtle in handwriting |

These pairs also show the lowest per-class F1-scores (L: 0.76, I: 0.76, G: 0.86, Q: 0.87), consistent with the confusion matrix — confirming the model's difficulty aligns with genuine visual ambiguity in handwritten characters, not a data or training issue.

##  Tech Stack
Python, TensorFlow/Keras, torchvision (data loading), NumPy, matplotlib, seaborn, scikit-learn

##  Files
- `CodeAlpha_HandwrittenCharacterRecognition.ipynb` — Full notebook
- `emnist_cnn_model.h5` — Saved trained CNN model
- `requirements.txt` — Dependencies

##  How to Run
```bash
pip install -r requirements.txt
jupyter notebook CodeAlpha_HandwrittenCharacterRecognition.ipynb
```

##  Author
Anvith CR— CodeAlpha Machine Learning Intern
