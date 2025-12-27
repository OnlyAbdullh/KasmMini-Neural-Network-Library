import itertools
from typing import Callable, Dict, List, Any
import numpy as np

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
        if name == "sgd":
            return SGD(lr=lr)
        if name == "momentum":
            return Momentum(lr=lr)
        if name == "adagrad":
            return AdaGrad(lr=lr)
        if name == "adam":
            return Adam(lr=lr)
        raise ValueError(f"Unknown optimizer_type: {name}")

    def grid_search(
            self,
            learning_rates: List[float],
            batch_sizes: List[int],
            hidden_sizes: List[int],
            optimizer_types: List[str],
            dropout_rates: List[float],
            epochs_list: List[int] = None,
            num_layers_list: List[int] = None,
            activation_types: List[str] = None,
    ) -> Dict[str, Any]:
        if epochs_list is None:
            epochs_list = [5]
        if num_layers_list is None:
            num_layers_list = [1]
        if activation_types is None:
            activation_types = ["relu"]

        best_score = -1.0
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
            config = {
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
                network,
                optimizer,
                self.x_train,
                self.t_train,
                self.x_val,
                self.t_val,
                epochs=epoch_num,
                batch_size=batch,
                eval_interval=epoch_num,
            )
            history = trainer.fit(verbose=False)
            val_acc = history["val_accuracy"][-1]
            record = {**config, "val_accuracy": val_acc}
            all_results.append(record)

            if val_acc > best_score:
                best_score = val_acc
                best_params = config

        return {
            "best_score": best_score,
            "best_params": best_params,
            "results": all_results,
        }