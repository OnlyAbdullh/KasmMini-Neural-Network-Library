from typing import Dict, Any
import numpy as np
from datasets import prepare_dataset
from plotting import plot_history
from KasmMiniNN import (
    Dense,
    Sigmoid,
    Relu,
    BatchNormalization,
    SoftmaxCrossEntropy,
    NeuralNetwork,
    SGD,
    Adam,
    Trainer,
    HyperparameterTuner,
    Tanh,
    Dropout,
    Convolution,
    MaxPooling,
    Flatten,
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
    return NeuralNetwork(layers, SoftmaxCrossEntropy())


def build_cnn_network(input_shape: tuple, num_classes: int) -> NeuralNetwork:
    """
    بناء شبكة CNN بسيطة
    input_shape: (C, H, W) مثل (1, 28, 28) للـ MNIST
    """
    C, H, W = input_shape

    layers = [
        # Conv1: (1, 28, 28) -> (32, 24, 24)
        Convolution(in_channels=C, out_channels=32, kernel_size=5, stride=1, pad=0),
        Relu(),
        # Pool1: (32, 24, 24) -> (32, 12, 12)
        MaxPooling(pool_size=2, stride=2),

        # Conv2: (32, 12, 12) -> (64, 8, 8)
        Convolution(in_channels=32, out_channels=64, kernel_size=5, stride=1, pad=0),
        Relu(),
        # Pool2: (64, 8, 8) -> (64, 4, 4)
        MaxPooling(pool_size=2, stride=2),

        # Flatten: (64, 4, 4) -> (1024,)
        Flatten(),

        # FC layers
        Dense(64 * 4 * 4, 128),
        Relu(),
        Dropout(0.5),
        Dense(128, num_classes),
    ]

    return NeuralNetwork(layers, SoftmaxCrossEntropy())


def build_simple_cnn(input_shape: tuple, num_classes: int) -> NeuralNetwork:
    """
    بناء شبكة CNN أبسط للتجربة السريعة
    """
    C, H, W = input_shape

    layers = [
        # Conv: (1, 28, 28) -> (16, 24, 24)
        Convolution(in_channels=C, out_channels=16, kernel_size=5, stride=1, pad=0),
        Relu(),
        # Pool: (16, 24, 24) -> (16, 12, 12)
        MaxPooling(pool_size=2, stride=2),

        Flatten(),

        Dense(16 * 12 * 12, 64),
        Relu(),
        Dense(64, num_classes),
    ]

    return NeuralNetwork(layers, SoftmaxCrossEntropy())


def build_network_from_config(input_dim: int, num_classes: int, config: Dict[str, Any]) -> NeuralNetwork:
    hidden_size = config["hidden_size"]
    num_layers = config["num_layers"]
    dropout_rate = config["dropout_rate"]

    activations = {"relu": Relu, "sigmoid": Sigmoid, "tanh": Tanh}
    Act = activations.get(config["activation"], Relu)

    layers = []
    in_dim = input_dim

    for _ in range(num_layers):
        layers.append(Dense(in_dim, hidden_size))
        layers.append(BatchNormalization(hidden_size))
        layers.append(Act())
        if dropout_rate > 0.0:
            layers.append(Dropout(dropout_rate))
        in_dim = hidden_size

    layers.append(Dense(in_dim, num_classes))
    return NeuralNetwork(layers, SoftmaxCrossEntropy())


def reshape_for_cnn(x: np.ndarray, dataset_name: str) -> np.ndarray:
    if dataset_name == 'mnist':
        # MNIST: (N, 784) -> (N, 1, 28, 28)
        return x.reshape(-1, 1, 28, 28)
    elif dataset_name == 'iris':
        raise ValueError("CNN is not suitable for Iris dataset. Please use MLP instead.")
    return x

def main():
    print("=" * 60)
    print("KasmMiniNN - Neural Network Training")
    print("=" * 60)

    print("\nChoose the dataset:")
    print(" 1 - Iris (for MLP only)")
    print(" 2 - MNIST (for both MLP and CNN)")
    dataset_choice = input("Enter 1 or 2: ").strip()

    if dataset_choice == "1":
        dataset_name = 'iris'
        x_train, x_val, x_test, t_train, t_val, t_test = prepare_dataset('iris')
        can_use_cnn = False
    else:
        dataset_name = 'mnist'
        x_train, x_val, x_test, t_train, t_val, t_test = prepare_dataset('mnist')
        can_use_cnn = True

    num_classes = t_train.shape[1]

    print("\n" + "=" * 60)
    print("Choose the network type:")
    print(" 1 - MLP (Multi-Layer Perceptron)")
    if can_use_cnn:
        print(" 2 - Simple CNN (Convolutional Neural Network)")
        print(" 3 - Deep CNN (More layers)")

    network_choice = input("Enter your choice: ").strip()

    use_cnn = False
    if can_use_cnn and network_choice in ["2", "3"]:
        use_cnn = True
        x_train = reshape_for_cnn(x_train, dataset_name)
        x_val = reshape_for_cnn(x_val, dataset_name)
        x_test = reshape_for_cnn(x_test, dataset_name)

    print("\n" + "=" * 60)
    print("Choose the mode:")
    print(" 1 - Train the model")
    print(" 2 - Grid Search (hyperparameter tuning - MLP only)")
    print(" 3 - Random Search (hyperparameter tuning - MLP only)")
    print(" 4 - K-Fold Grid Search (hyperparameter tuning - MLP only)")

    mode_choice = input("Enter 1, 2, 3, or 4: ").strip()

    if mode_choice == "1":
        print("\n" + "=" * 60)
        print("Training the network...")
        print("=" * 60)

        if use_cnn:
            if network_choice == "2":
                network = build_simple_cnn(input_shape=(1, 28, 28), num_classes=num_classes)
                print("Using Simple CNN architecture")
            else:
                network = build_cnn_network(input_shape=(1, 28, 28), num_classes=num_classes)
                print("Using Deep CNN architecture")

            optimizer = Adam(lr=0.001)
            epochs = 10
            batch_size = 64
        else:
            network = build_required_network(
                input_dim=x_train.shape[1],
                hidden1=32,
                hidden2=16,
                num_classes=num_classes,
            )
            print("Using MLP architecture")
            optimizer = SGD(lr=0.1)
            epochs = 20
            batch_size = 100

        trainer = Trainer(
            network,
            optimizer,
            x_train=x_train,
            t_train=t_train,
            x_val=x_val,
            t_val=t_val,
            x_test=x_test,
            t_test=t_test,
            epochs=epochs,
            batch_size=batch_size,
            eval_interval=1 if use_cnn else 10,
        )

        history = trainer.fit()

        final_train_acc = history["train_accuracy"][-1]
        final_val_acc = (
            history["val_accuracy"][-1]
            if len(history["val_accuracy"]) > 0
            else float("nan")
        )
        final_test_acc = (
            history["test_accuracy"][-1]
            if len(history["test_accuracy"]) > 0
            else float("nan")
        )

        print("\n" + "=" * 60)
        print("Final Results:")
        print(f"  Train Accuracy: {final_train_acc:.4f}")
        print(f"  Val Accuracy:   {final_val_acc:.4f}")
        print(f"  Test Accuracy:  {final_test_acc:.4f}")
        print("=" * 60)

        plot_history(history)

    elif mode_choice in {"2", "3", "4"}:
        if use_cnn:
            print("\n⚠️  Hyperparameter tuning is only available for MLP networks.")
            return

        search_names = {
            "2": "Grid Search",
            "3": "Random Search",
            "4": "K-Fold Grid Search",
        }
        print(f"\nSearching for best hyperparameters using {search_names[mode_choice]}...")
        print("=" * 60)

        tuner = HyperparameterTuner(
            build_network=lambda config: build_network_from_config(
                input_dim=x_train.shape[1],
                num_classes=num_classes,
                config=config,
            ),
            x_train=x_train,
            t_train=t_train,
            x_val=x_val,
            t_val=t_val,
        )

        if mode_choice == "2":
            results = tuner.grid_search(
                learning_rates=[0.01, 0.001, 0.1],
                batch_sizes=[64, 100],
                hidden_sizes=[32, 128],
                optimizer_types=["adam"],
                dropout_rates=[0.0, 0.3],
                epochs_list=[10, 15, 20],
                num_layers_list=[2],
                activation_types=["relu"],
            )
        elif mode_choice == "3":
            results = tuner.random_search(
                learning_rates=[0.01, 0.001, 0.1, 0.0001],
                batch_sizes=[32, 64, 100, 128],
                hidden_sizes=[32, 64, 128, 256],
                optimizer_types=["adam", "sgd"],
                dropout_rates=[0.0, 0.1, 0.3, 0.5],
                n_iter=20,
                epochs_list=[10, 15],
                num_layers_list=[1, 2, 3],
                activation_types=["relu", "tanh"],
                random_state=42,
            )
        else:
            results = tuner.kfold_search(
                learning_rates=[0.01, 0.001],
                batch_sizes=[64, 100],
                hidden_sizes=[32, 128],
                optimizer_types=["adam"],
                dropout_rates=[0.0, 0.3],
                epochs_list=[10],
                num_layers_list=[1, 2],
                activation_types=["relu"],
                n_splits=5,
                random_state=42,
            )

        best_params = results["best_params"]
        print("\n" + "=" * 60)
        print(f"Best Validation Accuracy: {results['best_val_accuracy']:.4f}")
        print("\nBest Parameters:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        print("\n" + "=" * 60)
        print("Training final model with best hyperparameters on TRAIN+VAL...")
        print("=" * 60)

        x_train_full = np.concatenate([x_train, x_val], axis=0)
        t_train_full = np.concatenate([t_train, t_val], axis=0)

        final_network = build_network_from_config(
            input_dim=x_train_full.shape[1],
            num_classes=t_train_full.shape[1],
            config=best_params,
        )

        final_optimizer = HyperparameterTuner._create_optimizer(
            best_params["optimizer_type"],
            best_params["learning_rate"],
        )

        final_trainer = Trainer(
            network=final_network,
            optimizer=final_optimizer,
            x_train=x_train_full,
            t_train=t_train_full,
            x_val=None,
            t_val=None,
            x_test=x_test,
            t_test=t_test,
            epochs=best_params["epochs"],
            batch_size=best_params["batch_size"],
            eval_interval=1,
        )

        final_history = final_trainer.fit()
        final_train_acc = final_history["train_accuracy"][-1]
        final_test_acc = (
            final_history["test_accuracy"][-1]
            if len(final_history["test_accuracy"]) > 0
            else float("nan")
        )

        print(
            f"\nFinal Result (TRAIN+VAL) "
            f"| Train Acc: {final_train_acc:.4f} "
            f"| Test Acc: {final_test_acc:.4f}"
        )
        plot_history(final_history)

    else:
        print("\nInvalid choice. Please run the program again.")


if __name__ == "__main__":
    main()