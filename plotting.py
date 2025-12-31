# examples/plotting.py
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt


def plot_history(history: Dict[str, List[float]], filename: str = "training_curves.png"):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_iteration_loss"])
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss per Iteration")

    epochs = np.arange(1, len(history["train_accuracy"]) + 1)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_accuracy"], label="Train Acc")

    if len(history["val_accuracy"]) > 0:
        plt.plot(epochs, history["val_accuracy"], label="Val Acc")

    if len(history["test_accuracy"]) > 0:
        test_epochs = np.linspace(1, len(epochs), len(history["test_accuracy"]))
        plt.plot(test_epochs, history["test_accuracy"], "ro", label="Test Acc")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    print(f"Saved {filename}")
