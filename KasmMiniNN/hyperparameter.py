import itertools
from typing import Callable, Dict, List, Any
import numpy as np
from sklearn.model_selection import KFold

from .trainer import Trainer
from .optimizers import SGD, Momentum, AdaGrad, Adam, Optimizer
from .network import NeuralNetwork


class HyperparameterTuner:
    def __init__(
            self,
            build_network: Callable[[Dict[str, Any]], NeuralNetwork],
            x_train: np.ndarray,
            t_train: np.ndarray,
            x_val: np.ndarray,
            t_val: np.ndarray,
    ):
        self.build_network = build_network
        self.x_train = x_train
        self.t_train = t_train
        self.x_val = x_val
        self.t_val = t_val

    @staticmethod
    def _create_optimizer(name: str, lr: float) -> Optimizer:
        name = name.lower()
        if name == "sgd": return SGD(lr=lr)
        if name == "momentum": return Momentum(lr=lr)
        if name == "adagrad": return AdaGrad(lr=lr)
        if name == "adam": return Adam(lr=lr)
        raise ValueError(f"Unknown optimizer_type: {name}")

    def grid_search(
            self,
            learning_rates: List[float],
            batch_sizes: List[int],
            hidden_sizes: List[int],
            optimizer_types: List[str],
            dropout_rates: List[float],
            epochs_list: List[int] | None = None,
            num_layers_list: List[int] | None = None,
            activation_types: List[str] | None = None,
    ) -> Dict[str, Any]:
        if epochs_list is None:
            epochs_list = [5]
        if num_layers_list is None:
            num_layers_list = [1]
        if activation_types is None:
            activation_types = ["relu"]

        best_val_accuracy = float('-inf')
        best_params: Dict[str, Any] = {}
        all_results: List[Dict[str, Any]] = []

        for lr, batch, hidden, opt_name, drop, epoch_num, num_layers, activation in itertools.product(
                learning_rates,
                batch_sizes,
                hidden_sizes,
                optimizer_types,
                dropout_rates,
                epochs_list,
                num_layers_list,
                activation_types
        ):
            config: Dict[str, Any] = {
                "learning_rate": lr,
                "batch_size": batch,
                "hidden_size": hidden,
                "optimizer_type": opt_name,
                "dropout_rate": drop,
                "epochs": epoch_num,
                "num_layers": num_layers,
                "activation": activation,
            }

            network = self.build_network(config)
            optimizer = self._create_optimizer(opt_name, lr)
            trainer = Trainer(
                network=network,
                optimizer=optimizer,
                x_train=self.x_train,
                t_train=self.t_train,
                x_val=self.x_val,
                t_val=self.t_val,
                x_test=None,
                t_test=None,
                epochs=epoch_num,
                batch_size=batch,
                eval_interval=epoch_num,
            )
            history = trainer.fit(verbose=False)
            val_acc = history["val_accuracy"][-1] if len(history["val_accuracy"]) > 0 else float('-inf')
            record = {**config, "val_accuracy": val_acc}
            all_results.append(record)

            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                best_params = config

        return {
            "best_val_accuracy": best_val_accuracy,
            "best_params": best_params,
            "results": all_results,
        }

    # Random search with fixed number of trials
    def random_search(
            self,
            learning_rates: List[float],
            batch_sizes: List[int],
            hidden_sizes: List[int],
            optimizer_types: List[str],
            dropout_rates: List[float],
            n_iter: int = 10,
            epochs_list: List[int] | None = None,
            num_layers_list: List[int] | None = None,
            activation_types: List[str] | None = None,
            random_state: int | None = None,
    ) -> Dict[str, Any]:
        if n_iter <= 0:
            raise ValueError("n_iter must be positive")
        if epochs_list is None:
            epochs_list = [5]
        if num_layers_list is None:
            num_layers_list = [1]
        if activation_types is None:
            activation_types = ["relu"]

        if random_state is not None:
            np.random.seed(random_state)

        best_val_accuracy = float('-inf')
        best_params: Dict[str, Any] = {}
        all_results: List[Dict[str, Any]] = []

        for _ in range(n_iter):
            config: Dict[str, Any] = {
                "learning_rate": np.random.choice(learning_rates),
                "batch_size": np.random.choice(batch_sizes),
                "hidden_size": np.random.choice(hidden_sizes),
                "optimizer_type": np.random.choice(optimizer_types),
                "dropout_rate": np.random.choice(dropout_rates),
                "epochs": np.random.choice(epochs_list),
                "num_layers": np.random.choice(num_layers_list),
                "activation": np.random.choice(activation_types),
            }

            network = self.build_network(config)
            optimizer = self._create_optimizer(config["optimizer_type"], config["learning_rate"])
            trainer = Trainer(
                network=network,
                optimizer=optimizer,
                x_train=self.x_train,
                t_train=self.t_train,
                x_val=self.x_val,
                t_val=self.t_val,
                x_test=None,
                t_test=None,
                epochs=config["epochs"],
                batch_size=config["batch_size"],
                eval_interval=config["epochs"],
            )
            history = trainer.fit(verbose=False)
            val_acc = history["val_accuracy"][-1] if len(history["val_accuracy"]) > 0 else float('-inf')
            record = {**config, "val_accuracy": val_acc}
            all_results.append(record)

            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                best_params = config

        return {
            "best_val_accuracy": best_val_accuracy,
            "best_params": best_params,
            "results": all_results,
        }
