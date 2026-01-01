from typing import Tuple
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import fetch_openml


def _load_iris() -> Tuple[np.ndarray, np.ndarray]:
    iris = load_iris()
    X = iris.data.astype(np.float32)
    y = iris.target.astype(np.int64)
    return X, y


def _load_mnist() -> Tuple[np.ndarray, np.ndarray]:
    X, y = fetch_openml(
        "mnist_784",
        version=1,
        return_X_y=True,
        as_frame=False
    )
    X = X.astype(np.float32) / 255.0
    y = y.astype(np.int64)
    return X, y


def prepare_dataset(
        dataset: str,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: int = 42,
) -> Tuple[np.ndarray, ...]:
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must sum to 1.0")

    if dataset.lower() == "iris":
        X, y = _load_iris()
    elif dataset.lower() == "mnist":
        X, y = _load_mnist()
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    x_train, x_temp, y_train_raw, y_temp_raw = train_test_split(
        X,
        y,
        test_size=(1.0 - train_size),
        stratify=y,
        shuffle=True,
        random_state=random_state,
    )
    val_ratio_in_temp = val_size / (val_size + test_size)

    x_val, x_test, y_val_raw, y_test_raw = train_test_split(
        x_temp,
        y_temp_raw,
        test_size=(1.0 - val_ratio_in_temp),
        stratify=y_temp_raw,
        shuffle=True,
        random_state=random_state,
    )

    ohe = OneHotEncoder(sparse_output=False)
    t_train = ohe.fit_transform(y_train_raw.reshape(-1, 1))
    t_val = ohe.transform(y_val_raw.reshape(-1, 1))
    t_test = ohe.transform(y_test_raw.reshape(-1, 1))

    return x_train, x_val, x_test, t_train, t_val, t_test
