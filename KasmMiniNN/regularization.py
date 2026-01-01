import numpy as np
from typing import Dict, Optional
from .layers import Layer


class Dropout(Layer):
    def __init__(self, dropout_ratio: float = 0.5):
        if not 0.0 <= dropout_ratio < 1.0:
            raise ValueError("dropout_ratio must be in [0,1)")
        self.dropout_ratio = dropout_ratio
        self.mask: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        if train:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.mask is None:
            raise RuntimeError("forward must be called before backward")
        return dout * self.mask


class BatchNormalization(Layer):

    def __init__(
            self,
            feature_size: int,
            momentum: float = 0.9,
            running_mean: Optional[np.ndarray] = None,
            running_var: Optional[np.ndarray] = None,
    ):
        if feature_size <= 0:
            raise ValueError("feature_size must be positive")
        self.gamma: np.ndarray = np.ones(feature_size, dtype=float)
        self.beta: np.ndarray = np.zeros(feature_size, dtype=float)
        self.momentum = momentum
        self.running_mean = running_mean
        self.running_var = running_var

        self.batch_size: Optional[int] = None
        self.xc: Optional[np.ndarray] = None
        self.xn: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.dgamma: Optional[np.ndarray] = None
        self.dbeta: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        if self.running_mean is None or self.running_var is None:
            _, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc ** 2, axis=0)
            std = np.sqrt(var + 1e-7)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std

            self.running_mean = (
                    self.momentum * self.running_mean +
                    (1 - self.momentum) * mu
            )
            self.running_var = (
                    self.momentum * self.running_var +
                    (1 - self.momentum) * var
            )
        else:
            xc = x - self.running_mean
            xn = xc / (np.sqrt(self.running_var + 1e-7))

        return self.gamma * xn + self.beta

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.xn is None or self.std is None or self.xc is None or self.batch_size is None:
            raise RuntimeError("forward must be called before backward")

        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)

        dxn = self.gamma * dout
        dxc = dxn / self.std

        dstd = -np.sum((dxn * self.xc) / (self.std ** 2), axis=0)
        dvar = 0.5 * dstd / self.std

        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)

        dx = dxc - (dmu / self.batch_size)

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx

    @property
    def params(self) -> Dict[str, np.ndarray]:
        return {"gamma": self.gamma, "beta": self.beta}

    @property
    def grads(self) -> Dict[str, np.ndarray]:
        if self.dgamma is None or self.dbeta is None:
            return {"gamma": np.zeros_like(self.gamma), "beta": np.zeros_like(self.beta)}
        return {"gamma": self.dgamma, "beta": self.dbeta}
