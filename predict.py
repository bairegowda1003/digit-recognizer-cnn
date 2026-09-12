"""
predict.py
-----------
Standalone inference script — load the trained CNN model and run
predictions without retraining. Shows a random test-set digit, the
model's prediction, and its confidence.

Usage:
    python predict.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras


def load_test_data():
    try:
        (_, _), (X_test, y_test) = keras.datasets.mnist.load_data()
        img_size = 28
        X_test = X_test.astype("float32") / 255.0
    except Exception:
        from sklearn.datasets import load_digits
        digits = load_digits()
        X_test, y_test = digits.images, digits.target
        img_size = 8
        X_test = X_test.astype("float32") / 16.0
    return X_test.reshape(-1, img_size, img_size, 1), y_test, img_size


if __name__ == "__main__":
    model = keras.models.load_model("digit_recognizer_model.keras")
    X_test, y_test, img_size = load_test_data()

    idx = np.random.randint(0, len(X_test))
    sample = X_test[idx:idx + 1]
    true_label = y_test[idx]

    probs = model.predict(sample, verbose=0)[0]
    pred_label = np.argmax(probs)
    confidence = probs[pred_label] * 100

    print(f"True label:      {true_label}")
    print(f"Predicted label: {pred_label}")
    print(f"Confidence:      {confidence:.2f}%")
    print("\nAll class probabilities:")
    for digit, p in enumerate(probs):
        bar = "#" * int(p * 40)
        print(f"  {digit}: {p*100:5.1f}% {bar}")

    plt.imshow(sample.reshape(img_size, img_size), cmap="gray")
    plt.title(f"Predicted: {pred_label} (True: {true_label})")
    plt.axis("off")
    plt.savefig("single_prediction.png", dpi=150)
    print("\nSaved single_prediction.png")
