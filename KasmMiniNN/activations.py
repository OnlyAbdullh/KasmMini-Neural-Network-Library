import numpy as np
from typing import Optional
from .layers import Layer


class Relu(Layer):

    def __init__(self) -> None:
        self.mask: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        self.mask = x <= 0
        return np.maximum(0, x)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.mask is None:
            raise RuntimeError("forward must be called before backward")
        dout = dout.copy()
        dout[self.mask] = 0
        return dout

class LeakyReLU(Layer):

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
        self.mask: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] *= self.alpha
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.mask is None:
            raise RuntimeError("forward must be called before backward")
        dx = dout.copy()
        dx[self.mask] *= self.alpha
        return dx

class Sigmoid(Layer):

    def __init__(self) -> None:
        self.out: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        self.out = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        return self.out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.out is None:
            raise RuntimeError("forward must be called before backward")
        return dout * self.out * (1.0 - self.out)


class Tanh(Layer):
    def __init__(self) -> None:
        self.out: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        self.out = np.tanh(x)
        return self.out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.out is None:
            raise RuntimeError("forward must be called before backward")
        return dout * (1.0 - self.out ** 2)


class Linear(Layer):
    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        return x

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout
