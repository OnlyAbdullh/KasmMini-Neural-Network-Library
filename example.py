import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import fetch_openml

from KasmMiniNN import (
    Dense,
    Sigmoid,
    Relu,
    BatchNormalization,
    SoftmaxWithLoss,
    NeuralNetwork,
    SGD,
    Trainer, HyperparameterTuner,
)


def build_required_network(input_dim: int, hidden1: int, hidden2: int, num_classes: int) -> NeuralNetwork:
    layers = [
        Dense(input_dim, hidden1),
        Sigmoid(),
        BatchNormalization(hidden1),
        Dense(hidden1, hidden2),
        Relu(),
        Dense(hidden2, num_classes),
    ]
    return NeuralNetwork(layers, SoftmaxWithLoss())


def prepare_iris():
    iris = load_iris()
    X = iris.data.astype(np.float32)
    y = iris.target.astype(int)
    x_train, x_test, t_train_raw, t_test_raw = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    ohe = OneHotEncoder(sparse_output=False)
    t_train = ohe.fit_transform(t_train_raw.reshape(-1, 1))
    t_test = ohe.transform(t_test_raw.reshape(-1, 1))
    return x_train, x_test, t_train, t_test

def prepare_mnist(test_size=10000, random_state=0):
    X, y = fetch_openml(
        'mnist_784',
        version=1,
        return_X_y=True,
        as_frame=False
    )

    X = X.astype(np.float32) / 255.0   # normalization
    y = y.astype(np.int64)

    x_train, x_test, t_train_raw, t_test_raw = train_test_split(
        X,
        y,
        test_size=test_size,
        shuffle=True,
        stratify=y,
        random_state=random_state
    )

    ohe = OneHotEncoder(sparse_output=False)
    t_train = ohe.fit_transform(t_train_raw.reshape(-1, 1))
    t_test = ohe.transform(t_test_raw.reshape(-1, 1))

    return x_train, x_test, t_train, t_test


def plot_history(history):

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history["iteration_loss"])
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss per Iteration")

    epochs = np.arange(1, len(history["train_accuracy"]) + 1)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_accuracy"], label="Train Acc")
    plt.plot(epochs, history["val_accuracy"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per Epoch")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=200)
    print("Saved training_curves.png")


def main(mode: str = "train"):
    x_train, x_test, t_train, t_test = prepare_iris()

    if mode == "tune":
        print("Searching for best hyperparameters...")
        print("=" * 60)

        split_idx = int(len(x_train) * 0.85)
        x_train_split, x_val_split = x_train[:split_idx], x_train[split_idx:]
        t_train_split, t_val_split = t_train[:split_idx], t_train[split_idx:]
        tuner = HyperparameterTuner(
            build_network=lambda config: build_required_network(
                input_dim=x_train.shape[1],
                hidden1=32,
                hidden2=16,
                num_classes=t_train.shape[1],
            ),
            x_train=x_train_split,
            t_train=t_train_split,
            x_val=x_val_split,
            t_val=t_val_split,
        )

        results = tuner.grid_search(
            learning_rates=[0.01, 0.001, 0.1],
            batch_sizes=[64, 100, 128],
            hidden_sizes=[32, 64, 128],
            optimizer_types=["sgd", "momentum", "adam"],
            dropout_rates=[0.0, 0.2, 0.3],
            epochs_list=[1,2],
        )

        print("\n" + "=" * 60)
        print(f"Best Validation Accuracy: {results['best_score']:.4f}")
        print("\nBest Parameters:")
        for param, value in results["best_params"].items():
            print(f"  {param}: {value}")
        return
    else:
        print("=" * 60)
        network = build_required_network(
            input_dim=x_train.shape[1],
            hidden1=32,
            hidden2=16,
            num_classes=t_train.shape[1],
        )

        optimizer = SGD(lr=0.1)
        trainer = Trainer(
            network,
            optimizer,
            x_train,
            t_train,
            x_test,
            t_test,
            epochs=5,
            batch_size=100,
            eval_interval=10,
        )

        history = trainer.fit()

        final_train_acc = history["train_accuracy"][-1]
        final_val_acc = history["val_accuracy"][-1]
        print(f"\nfinal Result - Train Acc: {final_train_acc:.4f} | Val Acc: {final_val_acc:.4f}")

        plot_history(history)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "tune"],
    )

    args = parser.parse_args()
    main(mode=args.mode)
