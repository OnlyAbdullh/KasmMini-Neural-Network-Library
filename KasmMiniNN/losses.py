import numpy as np
from typing import Optional


class SoftmaxWithLoss:
    def __init__(self) -> None:
        self.loss: Optional[float] = None
        self.y: Optional[np.ndarray] = None
        self.t: Optional[np.ndarray] = None

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, x: np.ndarray, t: np.ndarray) -> float:
        if t.ndim == 1:
            one_hot = np.zeros((t.size, x.shape[1]), dtype=float)
            one_hot[np.arange(t.size), t] = 1.0
            t = one_hot
        if x.shape[0] != t.shape[0]:
            raise ValueError("x and t must have the same batch size")

        self.t = t
        self.y = self._softmax(x)
        batch_size = x.shape[0]
        self.loss = -np.sum(
            np.log(self.y[np.arange(batch_size), t.argmax(axis=1)] + 1e-7)
        ) / batch_size
        return self.loss

    def backward(self, dout: float = 1.0) -> np.ndarray:
        if self.t is None or self.y is None:
            raise RuntimeError("forward must be called before backward")
        batch_size = self.t.shape[0]
        return dout * (self.y - self.t) / batch_size


class MeanSquaredError:

    def __init__(self) -> None:
        self.y: Optional[np.ndarray] = None
        self.t: Optional[np.ndarray] = None
        self.loss: Optional[float] = None

    def forward(self, y: np.ndarray, t: np.ndarray) -> float:
        if y.shape != t.shape:
            raise ValueError("y and t must have the same shape")
        self.y = y
        self.t = t
        batch_size = y.shape[0]
        self.loss = np.sum((y - t) ** 2) / (2 * batch_size)
        return self.loss

    def backward(self, dout: float = 1.0) -> np.ndarray:
        if self.t is None or self.y is None:
            raise RuntimeError("forward must be called before backward")
        batch_size = self.t.shape[0]
        return dout * (self.y - self.t) / batch_size