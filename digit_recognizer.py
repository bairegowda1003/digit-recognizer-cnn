"""
digit_recognizer.py
---------------------
Handwritten Digit Recognizer using a Convolutional Neural Network (CNN)
trained on the MNIST dataset (60,000 training / 10,000 test images of
handwritten digits 0-9).

Pipeline:
1. Load MNIST (auto-downloads via Keras; falls back to sklearn's digits
   dataset if there's no internet access in the environment)
2. Preprocess (normalize, reshape, one-hot encode labels)
3. Build a CNN (Conv2D -> MaxPool -> Conv2D -> MaxPool -> Dense)
4. Train and validate
5. Evaluate on test set (accuracy, confusion matrix)
6. Visualize predictions on sample images
7. Save the trained model
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# ---------------------------------------------------------------------
# 1. Load data (MNIST, with an offline-safe fallback)
# ---------------------------------------------------------------------
USING_FULL_MNIST = True
try:
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    img_size = 28
    print(f"Loaded full MNIST: {X_train.shape[0]} train / {X_test.shape[0]} test images (28x28)")
except Exception as e:
    print(f"Could not download full MNIST ({e}).")
    print("Falling back to sklearn's built-in digits dataset (8x8 images, 1797 samples) "
          "so the pipeline still runs end-to-end offline.")
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split as tts
    digits = load_digits()
    X, y = digits.images, digits.target
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=42, stratify=y)
    img_size = 8
    USING_FULL_MNIST = False

# ---------------------------------------------------------------------
# 2. Preprocess
# ---------------------------------------------------------------------
X_train = X_train.astype("float32") / (16.0 if not USING_FULL_MNIST else 255.0)
X_test = X_test.astype("float32") / (16.0 if not USING_FULL_MNIST else 255.0)

X_train = X_train.reshape(-1, img_size, img_size, 1)
X_test = X_test.reshape(-1, img_size, img_size, 1)

num_classes = 10
y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# ---------------------------------------------------------------------
# 3. Build CNN
# ---------------------------------------------------------------------
cnn_layers = [
    keras.Input(shape=(img_size, img_size, 1)),
    layers.Conv2D(32, kernel_size=3, padding="same", activation="relu"),
    layers.MaxPooling2D(pool_size=2),
    layers.Conv2D(64, kernel_size=3, padding="same", activation="relu"),
]
if img_size >= 28:
    # Only add a second pooling step when the image is large enough
    # (keeps the architecture valid for both full 28x28 MNIST and the
    # smaller 8x8 offline fallback dataset, with no unsafe Lambda layers)
    cnn_layers.append(layers.MaxPooling2D(pool_size=2))
cnn_layers += [
    layers.Flatten(),
    layers.Dropout(0.4),
    layers.Dense(128, activation="relu"),
    layers.Dense(num_classes, activation="softmax"),
]
model = keras.Sequential(cnn_layers)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# ---------------------------------------------------------------------
# 4. Train
# ---------------------------------------------------------------------
history = model.fit(
    X_train, y_train_cat,
    validation_split=0.1,
    epochs=8,
    batch_size=64,
    verbose=2,
)

# ---------------------------------------------------------------------
# 5. Evaluate
# ---------------------------------------------------------------------
test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f}")

y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
print("\nClassification report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("Confusion Matrix - Digit Recognizer")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("Saved confusion_matrix.png")

# ---------------------------------------------------------------------
# 6. Training curves
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(history.history["accuracy"], label="train")
axes[0].plot(history.history["val_accuracy"], label="val")
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(history.history["loss"], label="train")
axes[1].plot(history.history["val_loss"], label="val")
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].legend()

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
print("Saved training_curves.png")

# ---------------------------------------------------------------------
# 7. Sample predictions grid
# ---------------------------------------------------------------------
fig, axes = plt.subplots(3, 5, figsize=(10, 6))
idxs = np.random.choice(len(X_test), 15, replace=False)
for ax, idx in zip(axes.flat, idxs):
    ax.imshow(X_test[idx].reshape(img_size, img_size), cmap="gray")
    correct = y_pred[idx] == y_test[idx]
    ax.set_title(f"Pred: {y_pred[idx]} (True: {y_test[idx]})",
                 color="green" if correct else "red", fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig("sample_predictions.png", dpi=150)
print("Saved sample_predictions.png")

# ---------------------------------------------------------------------
# 8. Save model
# ---------------------------------------------------------------------
model.save("digit_recognizer_model.keras")
print("Saved digit_recognizer_model.keras")
