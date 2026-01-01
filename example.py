from typing import Dict, Any, Tuple
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
    Trainer,
    HyperparameterTuner,
    Tanh,
    Dropout,
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


def main():
    print("Choose the dataset:")
    print(" 1 - Iris")
    print(" 2 - mnist")
    choice = input("Enter 1, 2: ").strip()
    if choice == "1":
        x_train, x_val, x_test, t_train, t_val, t_test = prepare_dataset('iris')
    else:
        x_train, x_val, x_test, t_train, t_val, t_test = prepare_dataset('mnist')
    print("Choose the mode:")
    print(" 1 - Train the model (train)")
    print(" 2 - Grid Search (tune)")
    print(" 3 - Random Search (random)")
    print(" 4 - K-Fold Grid Search (kfold)")

    choice = input("Enter 1, 2, 3, or 4: ").strip()

    if choice in {"2", "3", "4"}:
        search_names = {
            "2": "Grid Search",
            "3": "Random Search",
            "4": "K-Fold Grid Search",
        }
        print(f"\nSearching for best hyperparameters using {search_names[choice]}...")
        print("=" * 60)

        tuner = HyperparameterTuner(
            build_network=lambda config: build_network_from_config(
                input_dim=x_train.shape[1],
                num_classes=t_train.shape[1],
                config=config,
            ),
            x_train=x_train,
            t_train=t_train,
            x_val=x_val,
            t_val=t_val,
        )

        if choice == "2":
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
        elif choice == "3":
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
            x_train=x_train,
            t_train=t_train,
            x_val=x_val,
            t_val=t_val,
            x_test=x_test,
            t_test=t_test,
            epochs=20,
            batch_size=100,
            eval_interval=10,
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

        print(
            f"\nFinal Result - "
            f"Train Acc: {final_train_acc:.4f} "
            f"| Val Acc: {final_val_acc:.4f} "
            f"| Test Acc: {final_test_acc:.4f}"
        )
        plot_history(history)


if __name__ == "__main__":
    main()
