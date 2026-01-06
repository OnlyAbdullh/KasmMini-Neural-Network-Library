import numpy as np
from typing import Dict, Optional
from .layers import Layer



def im2col(input_data: np.ndarray, filter_h: int, filter_w: int,
           stride: int = 1, pad: int = 0) -> np.ndarray:

    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data,
                 [(0, 0), (0, 0), (pad, pad), (pad, pad)],
                 'constant')

    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3)
    col = col.reshape(N * out_h * out_w, -1)
    return col


def col2im(col: np.ndarray, input_shape: tuple, filter_h: int,
           filter_w: int, stride: int = 1, pad: int = 0) -> np.ndarray:

    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w)
    col = col.transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

class Convolution(Layer):

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1, pad: int = 0,
                 weight_init: str = "he"):
        """
        Args:
            in_channels: عدد القنوات في المدخل
            out_channels: عدد الفلاتر (القنوات في المخرج)
            kernel_size: حجم الفلتر (مربع)
            stride: خطوة التحرك
            pad: حجم الـ padding
            weight_init: طريقة تهيئة الأوزان
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

        if weight_init == "he":
            scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        elif weight_init == "xavier":
            scale = np.sqrt(1.0 / (in_channels * kernel_size * kernel_size))
        else:
            scale = 0.01

        # W: (out_channels, in_channels, kernel_size, kernel_size)
        self.W = np.random.randn(out_channels, in_channels,
                                 kernel_size, kernel_size) * scale
        self.b = np.zeros(out_channels)

        self.x: Optional[np.ndarray] = None
        self.col: Optional[np.ndarray] = None
        self.col_W: Optional[np.ndarray] = None
        self.dW: Optional[np.ndarray] = None
        self.db: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:

        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape

        out_h = 1 + int((H + 2 * self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2 * self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T  # (C*FH*FW, FN)

        out = np.dot(col, col_W) + self.b  # (N*out_h*out_w, FN)
        out = out.reshape(N, out_h, out_w, FN).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.x is None or self.col is None or self.col_W is None:
            raise RuntimeError("forward must be called before backward")

        FN, C, FH, FW = self.W.shape

        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)

        dcol_W = np.dot(self.col.T, dout)
        self.dW = dcol_W.T.reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW,
                    stride=self.stride, pad=self.pad)

        return dx

    @property
    def params(self) -> Dict[str, np.ndarray]:
        return {"W": self.W, "b": self.b}

    @property
    def grads(self) -> Dict[str, np.ndarray]:
        if self.dW is None or self.db is None:
            return {"W": np.zeros_like(self.W), "b": np.zeros_like(self.b)}
        return {"W": self.dW, "b": self.db}


class MaxPooling(Layer):


    def __init__(self, pool_size: int = 2, stride: int = 2, pad: int = 0):
        self.pool_h = pool_size
        self.pool_w = pool_size
        self.stride = stride
        self.pad = pad

        self.x: Optional[np.ndarray] = None
        self.arg_max: Optional[np.ndarray] = None
        self.out_h: Optional[int] = None
        self.out_w: Optional[int] = None

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:

        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max
        self.out_h = out_h
        self.out_w = out_w

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.x is None or self.arg_max is None:
            raise RuntimeError("forward must be called before backward")

        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2] * dmax.shape[3], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w,
                    stride=self.stride, pad=self.pad)

        return dx

class Flatten(Layer):

    def __init__(self):
        self.original_shape: Optional[tuple] = None

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        """
        x: (N, C, H, W)
        output: (N, C*H*W)
        """
        self.original_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.original_shape is None:
            raise RuntimeError("forward must be called before backward")
        return dout.reshape(self.original_shape)