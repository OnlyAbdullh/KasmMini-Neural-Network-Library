import numpy as np
from typing import Optional


class SoftmaxCrossEntropy:
    def __init__(self) -> None:
        self.loss: Optional[float] = None
        self.y: Optional[np.ndarray] = None
        self.t: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, t: np.ndarray) -> float:
        if t.ndim == 1:
            one_hot = np.zeros((t.size, x.shape[1]), dtype=float)
            one_hot[np.arange(t.size), t] = 1.0
            t = one_hot
        if x.shape[0] != t.shape[0]:
            raise ValueError("x and t must have the same batch size")

        batch_size = x.shape[0]

        x_max = np.max(x, axis=1, keepdims=True)
        x_shifted = x - x_max
        log_sum_exp = np.log(np.sum(np.exp(x_shifted), axis=1, keepdims=True))

        log_softmax = x_shifted - log_sum_exp
        self.loss = -np.sum(t * log_softmax) / batch_size

        self.y = np.exp(log_softmax)
        self.t = t
        return self.loss

    def backward(self, dout: float = 1.0) -> np.ndarray:
        if self.t is None or self.y is None:
            raise RuntimeError("forward must be called before backward")
        batch_size = self.t.shape[0]
        dx = dout * (self.y - self.t) / batch_size

        return dx


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


class BinaryCrossEntropy:
    def __init__(self):
        self.y: Optional[np.ndarray] = None
        self.t: Optional[np.ndarray] = None
        self.loss: Optional[float] = None

    def forward(self, y: np.ndarray, t: np.ndarray) -> float:
        self.y = np.clip(y, 1e-7, 1 - 1e-7)
        self.t = t
        batch_size = y.shape[0]
        self.loss = -np.sum(t * np.log(self.y) + (1 - t) * np.log(1 - self.y)) / batch_size
        return self.loss

    def backward(self, dout: float = 1.0) -> np.ndarray:
        if self.y is None or self.t is None:
            raise RuntimeError("forward must be called before backward")
        batch_size = self.t.shape[0]
        return dout * (self.y - self.t) / (self.y * (1 - self.y) + 1e-7) / batch_size
