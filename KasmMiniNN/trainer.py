import numpy as np
from typing import Dict, List, Tuple, Generator, Optional
from .network import NeuralNetwork
from .optimizers import Optimizer


class Trainer:
    def __init__(
        self,
        network: NeuralNetwork,
        optimizer: Optimizer,
        x_train: np.ndarray,
        t_train: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        t_val: Optional[np.ndarray] = None,
        x_test: Optional[np.ndarray] = None,
        t_test: Optional[np.ndarray] = None,
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
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = batch_size
        self.eval_interval = eval_interval
        self.train_size = x_train.shape[0]

        self.history: Dict[str, List[float]] = {
            "train_iteration_loss": [],
            "train_epoch_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "test_accuracy": [],
        }

    def _iterate_minibatches(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        idx = np.random.permutation(self.train_size)
        for start in range(0, self.train_size, self.batch_size):
            batch_idx = idx[start:start + self.batch_size]
            yield self.x_train[batch_idx], self.t_train[batch_idx]

    def train_step(self, x_batch: np.ndarray, t_batch: np.ndarray) -> float:
        self.network.train()
        grads = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)
        batch_loss = self.network.loss(x_batch, t_batch, train=True)
        return batch_loss

    def fit(self, verbose: bool = True) -> Dict[str, List[float]]:
        if verbose:
            header = f"{'Epoch':>6} | {'Train Acc':>10} | {'Val Acc':>10} | {'Test Acc':>10}"
            print(header)
            print("-" * len(header))

        for epoch in range(1, self.epochs + 1):
            epoch_losses: List[float] = []
            for x_batch, t_batch in self._iterate_minibatches():
                batch_loss = self.train_step(x_batch, t_batch)
                self.history["train_iteration_loss"].append(batch_loss)
                epoch_losses.append(batch_loss)

            self.history["train_epoch_loss"].append(float(np.mean(epoch_losses)))

            train_acc = self.network.accuracy(self.x_train, self.t_train)
            self.history["train_accuracy"].append(train_acc)

            val_acc = None
            if self.x_val is not None and self.t_val is not None:
                val_acc = self.network.accuracy(self.x_val, self.t_val)
                self.history["val_accuracy"].append(val_acc)

            test_acc = None
            if self.x_test is not None and self.t_test is not None:
                if epoch == self.epochs:
                    test_acc = self.network.accuracy(self.x_test, self.t_test)
                    self.history["test_accuracy"].append(test_acc)

            if verbose and epoch % self.eval_interval == 0:
                va = val_acc if val_acc is not None else float("nan")
                ta = test_acc if test_acc is not None else float("nan")
                print(f"{epoch:6d} | {train_acc:10.4f} | {va:10.4f} | {ta:10.4f}")

        return self.history
