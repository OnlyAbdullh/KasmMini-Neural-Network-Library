import numpy as np
from typing import Dict, List, Tuple
from .network import NeuralNetwork
from .optimizers import Optimizer


class Trainer:
    def __init__(
        self,
        network: NeuralNetwork,
        optimizer: Optimizer,
        x_train: np.ndarray,
        t_train: np.ndarray,
        x_val: np.ndarray,
        t_val: np.ndarray,
        epochs: int = 20,
        batch_size: int = 32,
        eval_interval: int = 1,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.network = network
        self.optimizer = optimizer
        self.x_train = x_train
        self.t_train = t_train
        self.x_val = x_val
        self.t_val = t_val
        self.epochs = epochs
        self.batch_size = batch_size
        self.eval_interval = eval_interval
        self.train_size = x_train.shape[0]

        self.history: Dict[str, List[float]] = {
            "iteration_loss": [],
            "train_accuracy": [],
            "val_accuracy": []
        }

    def _iterate_minibatches(self) -> Tuple[np.ndarray, np.ndarray]:
        idx = np.random.permutation(self.train_size)
        for start in range(0, self.train_size, self.batch_size):
            batch_idx = idx[start:start + self.batch_size]
            yield self.x_train[batch_idx], self.t_train[batch_idx]

    def train_step(self, x_batch: np.ndarray, t_batch: np.ndarray) -> float:
        grads = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)
        batch_loss = self.network.loss(x_batch, t_batch, train=True)
        return batch_loss

    def fit(self, verbose: bool = True) -> Dict[str, List[float]]:
        if verbose:
            print(f"{'Epoch':>6} | {'Train Acc':>10} | {'Val Acc':>10}")
            print("-" * 40)

        for epoch in range(1, self.epochs + 1):
            for x_batch, t_batch in self._iterate_minibatches():
                batch_loss = self.train_step(x_batch, t_batch)
                self.history["iteration_loss"].append(batch_loss)

            train_acc = self.network.accuracy(self.x_train, self.t_train)
            val_acc = self.network.accuracy(self.x_val, self.t_val)

            self.history["train_accuracy"].append(train_acc)
            self.history["val_accuracy"].append(val_acc)

            if verbose and epoch % self.eval_interval == 0:
                print(f"{epoch:6d} | {train_acc:10.4f} | {val_acc:10.4f}")

        return self.history