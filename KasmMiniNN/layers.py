import numpy as np
from typing import Dict


class Layer:

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        raise NotImplementedError("forward must be implemented in subclasses")

    def backward(self, dout: np.ndarray) -> np.ndarray:
        raise NotImplementedError("backward must be implemented in subclasses")

    @property
    def params(self) -> Dict[str, np.ndarray]:
        return {}

    @property
    def grads(self) -> Dict[str, np.ndarray]:
        return {}


class Dense(Layer):

    def __init__(
            self,
            input_size: int,
            output_size: int,
            weight_init: str = "he",
            bias_init: float = 0.
    ):
        if input_size <= 0 or output_size <= 0:
            raise ValueError("input_size and output_size must be positive")
        if weight_init == "he":
            scale = np.sqrt(2.0 / input_size)
        elif weight_init == "xavier":
            scale = np.sqrt(1.0 / input_size)
        else:
            scale = 0.01

        self.W: np.ndarray = np.random.randn(input_size, output_size) * scale
        self.b: np.ndarray = np.full(output_size, bias_init, dtype=float)
        self.x: np.ndarray | None = None
        self.dW: np.ndarray | None = None
        self.db: np.ndarray | None = None

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.x is None:
            raise RuntimeError("forward must be called before backward")
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx

    @property
    def params(self) -> Dict[str, np.ndarray]:
        return {"W": self.W, "b": self.b}

    @property
    def grads(self) -> Dict[str, np.ndarray]:
        if self.dW is None or self.db is None:
            return {"W": np.zeros_like(self.W), "b": np.zeros_like(self.b)}
        return {"W": self.dW, "b": self.db}
