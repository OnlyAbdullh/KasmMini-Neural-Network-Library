import numpy as np
from typing import Dict, List, Union
from .layers import Layer, Dense
from .losses import SoftmaxWithLoss, MeanSquaredError


class NeuralNetwork:
    def __init__(self, layers: List[Layer], loss_layer: Union[SoftmaxWithLoss, MeanSquaredError]):
        if not layers:
            raise ValueError("at least one layer is required")
        self.layers = layers
        self.loss_layer = loss_layer
        self.params: Dict[str, np.ndarray] = {}
        self.grads: Dict[str, np.ndarray] = {}
        self._aggregate_params()

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

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x, train=train)
        return x

    def predict(self, x: np.ndarray) -> np.ndarray:
        scores = self.forward(x, train=False)
        return np.argmax(scores, axis=1)

    def loss(self, x: np.ndarray, t: np.ndarray, train: bool = True) -> float:
        scores = self.forward(x, train=train)
        return self.loss_layer.forward(scores, t)

    def accuracy(self, x: np.ndarray, t: np.ndarray) -> float:
        scores = self.forward(x, train=False)
        preds = np.argmax(scores, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        return float(np.sum(preds == t) / x.shape[0])

    def gradient(self, x: np.ndarray, t: np.ndarray) -> Dict[str, np.ndarray]:
        _ = self.loss(x, t, train=True)
        dout = self.loss_layer.backward()
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        self._aggregate_params()
        self._aggregate_grads()
        return self.grads


def build_mlp(
    input_size: int,
    hidden_sizes: List[int],
    output_size: int,
    use_batchnorm: bool = False,
    dropout_rate: float | None = None,
) -> NeuralNetwork:
    from .activations import Relu
    from .regularization import BatchNormalization, Dropout

    layers: List[Layer] = []
    layer_sizes = [input_size] + hidden_sizes + [output_size]
    for i in range(len(layer_sizes) - 1):
        layers.append(Dense(layer_sizes[i], layer_sizes[i + 1]))
        if i < len(hidden_sizes):
            layers.append(Relu())
            if use_batchnorm:
                layers.append(BatchNormalization(layer_sizes[i + 1]))
            if dropout_rate is not None:
                layers.append(Dropout(dropout_rate))
    loss_layer = SoftmaxWithLoss()
    return NeuralNetwork(layers, loss_layer)