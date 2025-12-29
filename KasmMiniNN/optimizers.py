import numpy as np
from typing import Dict, Mapping


class Optimizer:
    def update(self, params: Dict[str, np.ndarray], grads: Mapping[str, np.ndarray]) -> None:
        raise NotImplementedError("update must be implemented in subclasses")


class SGD(Optimizer):

    def __init__(self, lr: float = 0.01):
        if lr <= 0:
            raise ValueError("learning rate must be positive")
        self.lr = lr

    def update(self, params: Dict[str, np.ndarray], grads: Mapping[str, np.ndarray]) -> None:
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class Momentum(Optimizer):

    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        if lr <= 0:
            raise ValueError("learning rate must be positive")
        self.lr = lr
        self.momentum = momentum
        self.v: Dict[str, np.ndarray] | None = None

    def update(self, params: Dict[str, np.ndarray], grads: Mapping[str, np.ndarray]) -> None:
        if self.v is None:
            self.v = {key: np.zeros_like(val) for key, val in params.items()}

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


class AdaGrad(Optimizer):
    def __init__(self, lr: float = 0.01):
        if lr <= 0:
            raise ValueError("learning rate must be positive")
        self.lr = lr
        self.h: Dict[str, np.ndarray] | None = None

    def update(self, params: Dict[str, np.ndarray], grads: Mapping[str, np.ndarray]) -> None:
        if self.h is None:
            self.h = {key: np.zeros_like(val) for key, val in params.items()}

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class Adam(Optimizer):

    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999):
        if lr <= 0:
            raise ValueError("learning rate must be positive")
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m: Dict[str, np.ndarray] | None = None
        self.v: Dict[str, np.ndarray] | None = None

    def update(self, params: Dict[str, np.ndarray], grads: Mapping[str, np.ndarray]) -> None:
        if self.m is None or self.v is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1

        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (
                1.0 - self.beta1 ** self.iter
        )

        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key] ** 2 - self.v[key])
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
