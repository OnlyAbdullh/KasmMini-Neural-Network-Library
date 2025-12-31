import numpy as np
from typing import Dict, List, Union, Optional
from .layers import Layer, Dense
from .losses import SoftmaxCrossEntropy, MeanSquaredError, BinaryCrossEntropy


class NeuralNetwork:
    def __init__(
            self,
            layers: List[Layer],
            loss_layer: Union[SoftmaxCrossEntropy, MeanSquaredError, BinaryCrossEntropy],
    ):
        if not layers:
            raise ValueError("at least one layer is required")
        self.layers = layers
        self.loss_layer = loss_layer
        self.params: Dict[str, np.ndarray] = {}
        self.grads: Dict[str, np.ndarray] = {}
        self.training: bool = True
        self._aggregate_params()

    def train(self, mode: bool = True) -> None:
        self.training = mode

    def eval(self) -> None:
        self.train(False)

    def init_weight(self, method: str = "he") -> None:
        for layer in self.layers:
            if isinstance(layer, Dense):
                input_size, output_size = layer.W.shape
                if method == "he":
                    scale = np.sqrt(2.0 / input_size)
                elif method == "xavier":
                    scale = np.sqrt(1.0 / input_size)
                else:
                    scale = 0.01
                layer.W = np.random.randn(input_size, output_size) * scale
                layer.b = np.zeros(output_size, dtype=float)
        self._aggregate_params()

    def _aggregate_params(self) -> None:
        params: Dict[str, np.ndarray] = {}
        for idx, layer in enumerate(self.layers):
            for name, value in layer.params.items():
                params[f"{layer.__class__.__name__}{idx}_{name}"] = value
        self.params = params

    def _aggregate_grads(self) -> None:
        grads: Dict[str, np.ndarray] = {}
        for idx, layer in enumerate(self.layers):
            for name, grad in layer.grads.items():
                grads[f"{layer.__class__.__name__}{idx}_{name}"] = grad
        self.grads = grads

    def forward(self, x: np.ndarray, train: Optional[bool] = None) -> np.ndarray:
        if train is None:
            train = self.training
        for layer in self.layers:
            x = layer.forward(x, train=train)
        return x

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.eval()
        scores = self.forward(x, train=False)
        return np.argmax(scores, axis=1)

    def loss(self, x: np.ndarray, t: np.ndarray, train: Optional[bool] = None) -> float:
        if train is None:
            train = self.training
        scores = self.forward(x, train=train)
        return self.loss_layer.forward(scores, t)

    def accuracy(self, x: np.ndarray, t: np.ndarray) -> float:
        self.eval()
        scores = self.forward(x)  # eval mode
        preds = np.argmax(scores, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        return np.sum(preds == t).astype(float) / float(x.shape[0])

    def gradient(self, x: np.ndarray, t: np.ndarray) -> Dict[str, np.ndarray]:
        self.train(True)
        self.loss(x, t, train=True)
        dout = self.loss_layer.backward(1.0)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        self._aggregate_params()
        self._aggregate_grads()
        return self.grads
