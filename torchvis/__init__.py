__version__ = '0.1.0'

from typing import Union

import numpy as np
import functools
import operator
import math

import torch
import torch.nn.functional as F
import torch.nn as nn

from . import stftlib
from . import selfish as G
from .selflib import util as RT
from numpy import fft as fftlib

Tensorlike = Union[np.ndarray, torch.Tensor]


# lol at this function.
# https://stackoverflow.com/questions/2166818/how-to-check-if-an-object-is-an-instance-of-a-namedtuple
def is_namedtuple(x):
    t = type(x)
    b = t.__bases__
    if len(b) != 1 or b[0] != tuple: return False
    f = getattr(t, '_fields', None)
    if not isinstance(f, tuple): return False
    return all(type(n)==str for n in f)


def dictlike(x):
    return isinstance(x, dict)


def listlike(x):
    return isinstance(x, (list, tuple))


def stringlike(x):
    return isinstance(x, (bytes, str))


def intlike(x):
    return isinstance(x, int)


def allint(x):
    return all([intlike(v) for v in x])


def length(x):
    try:
        return len(x)
    except TypeError:
        return 1


def keys(x):
    if is_namedtuple(x):
        return list(x._fields)
    if dictlike(x):
        return list(x)
    n = length(x)
    return [i for i in range(n)]


def vals(x):
    if dictlike(x):
        return list(x.values())
    try:
        return list(x)
    except TypeError:
        return [x]


def items(x):
    return list(zip(keys(x), vals(x)))


def listify(x):
    k, v = keys(x), vals(x)
    if allint(k):
        return v
    return collections.OrderedDict(zip(k, v))


def flatten(x):
    if listlike(x):
        x = functools.reduce(operator.add, [flatten(v) for v in x])
    return vals(x)


def flatlist(x):
    if listlike(x):
        x = list(x)
    else:
        x = [x]
    return flatten(x)


def shapeof(x):
    if hasattr(x, "shape"):
        x = x.shape
    return flatlist(x)


def to_numpy(x, concat_axis=-1) -> np.ndarray:
    if is_torch_tensor(x):
        return x.numpy()
    if is_numpy_tensor(x):
        return x
    x = flatlist(x)
    x = [v.numpy() if hasattr(v, 'numpy') else v for v in x]
    # reshape to 2D.
    x = [v.reshape([-1, v.shape[-1]]) for v in x]
    x = np.concatenate(x, axis=concat_axis)
    return x


def is_torch_tensor(x):
    return isinstance(x, torch.Tensor)


def is_numpy_tensor(x):
    return isinstance(x, np.ndarray)


def to_tensor(x, concat_axis=-1) -> torch.Tensor:
    return torch.tensor(to_numpy(x, concat_axis=concat_axis))


def tt(x):
    if not is_torch_tensor(x):
        return torch.tensor(x)
    else:
        return x

to_tensor = tt


def checker(*sizes, invert=False) -> torch.Tensor:
    """Checkerboard pattern."""
    sizes = flatlist(sizes)
    if isinstance(sizes[-1], bool): # unsure about this
        invert = sizes[-1]
        sizes = sizes[0:-1]
    if len(sizes) <= 1:
        return (torch.arange(sizes[-1]) % 2 == 0) != invert
    if len(sizes) == 2:
        h, w = sizes
        return torch.stack([checker(w, invert=i % 2 != 0) for i in range(h)])
    else:
        [checker(sizes[-1], invert=i % 2 != 0) for i in range(sizes[-1])]
    t=(torch.arange(sizes[-1]) % 2 == 0) == invert
    #return t if len(sizes) < 2 else torch.stack([t, t.roll([1])] * ((sizes[-2])//2) + [t] * (sizes[-2] % 2))
    return t if len(sizes) < 2 else torch.stack([t, t.roll([1])] * ((sizes[-2])//2))# + [t] * (sizes[-2] % 2))


def checker(*sizes, invert=False, lo=0, hi=1) -> torch.Tensor:
    sizes = flatlist(sizes)
    t = torch.ones(sizes)
    for slices in [[slice(None,None,2 if i == j else None) for i in range(len(t.shape))] for j in range(len(t.shape))]:
        t[slices] *= -1
    t = t.maximum(torch.zeros_like(t))
    return t * hi + (1 - t) * lo


def roll(x, shifts=None, dims=None):
    """Roll along the specified dims. By default, roll by half along width and height."""
    x0 = x
    x = to_kernel(x)
    shifts = [x.shape[-2] // 2, x.shape[-1] // 2] if shifts is None else shifts
    if dims is None:
        dims = torch.arange(len(x.shape))[-2:].tolist()
    x = x.roll(dims=dims, shifts=shifts)
    return from_kernel(x, x0)

def shift(x, dim=-1, by=1):
    return roll(x, by, dim)


def rollor(t: Tensorlike, shift=1, dim=0) -> torch.Tensor:
    """Rolls every other slice along dim."""
    return torch.stack([x.roll(shift) if i % 2 == 0 else x for i, x in enumerate(t.unbind(dim))], dim=dim)


def rnorm(x: Tensorlike) -> torch.Tensor:
    """Remap a real signal to [-1 .. 1]."""
    if x.is_complex():
        raise ValueError("rnorm: x.is_complex() should be False")
    v = x - (x.max() + x.min()) / 2
    return v / v.max()

def rnorm0(x : Tensorlike) -> torch.Tensor:
    """Remap a real signal to [0 .. 1]."""
    return rnorm(x) * 0.5 + 0.5


def icircular(x: Tensorlike, *, radius=1.0) -> torch.Tensor:
    """Constrain complex numbers to a unit circle."""
    return torch.polar(torch.ones_like(x.abs()) * radius, x.angle())


def inormabs(x: Tensorlike) -> torch.Tensor:
    """Return the magnitude scaled to [-1 .. 1]."""
    assert x.is_complex()
    return rnorm(x.abs())


def inorm(x: Tensorlike) -> torch.Tensor:
    """Return the real component scaled to [-1 .. 1]."""
    assert x.is_complex()
    return rnorm(x.real)


def fftshift(x: Tensorlike, axes=None) -> torch.Tensor:
    return torch.tensor(fftlib.fftshift(to_numpy(x), axes=axes))


def ifftshift(x: Tensorlike, axes=None) -> torch.Tensor:
    return torch.tensor(fftlib.ifftshift(to_numpy(x), axes=axes))


def fftfreq(n, d=1.0) -> torch.Tensor:
    return torch.tensor(fftlib.fftfreq(n, d=d))


def rfftfreq(n, d=1.0) -> torch.Tensor:
    return torch.tensor(fftlib.rfftfreq(n, d=d))


def fft(x: Tensorlike, n=None, axis=-1) -> torch.Tensor:
    return torch.tensor(fftlib.fft(to_numpy(x), n=n, axis=axis))


def ifft(x: Tensorlike, n=None, axis=-1) -> torch.Tensor:
    return torch.tensor(fftlib.ifft(to_numpy(x), n=n, axis=axis))


def rfft(x: Tensorlike, *args) -> torch.Tensor:
    return torch.tensor(fftlib.rfft(to_numpy(x), *args))


def irfft(x: Tensorlike, *args) -> torch.Tensor:
    return torch.tensor(fftlib.irfft(to_numpy(x), *args))


def fft2(x: Tensorlike, *args) -> torch.Tensor:
    return torch.tensor(fftlib.fft2(to_numpy(x), *args))


def ifft2(x: Tensorlike, *args) -> torch.Tensor:
    return torch.tensor(fftlib.ifft2(to_numpy(x), *args))


def rfft2(x: Tensorlike, *args) -> torch.Tensor:
    return torch.tensor(fftlib.rfft2(to_numpy(x), *args))


def irfft2(x: Tensorlike, *args) -> torch.Tensor:
    return torch.tensor(fftlib.irfft2(to_numpy(x), *args))


def rstft(x: Tensorlike, Nwin, Nfft=None) -> torch.Tensor:
    return torch.tensor(stftlib.stft(to_numpy(x), Nwin, Nfft=Nfft, Ffft=np.fft.rfft))


def irstft(x: Tensorlike, Nwin) -> torch.Tensor:
    return torch.tensor(stftlib.istft(to_numpy(x), Nwin, Ffft=np.fft.irfft))


def stft(x: Tensorlike, Nwin, Nfft=None) -> torch.Tensor:
    return torch.tensor(stftlib.stft(to_numpy(x), Nwin, Nfft=Nfft, Ffft=np.fft.fft))


def istft(x: Tensorlike, Nwin) -> torch.Tensor:
    return torch.tensor(stftlib.istft(to_numpy(x), Nwin, Ffft=np.fft.ifft))

def unwrap(x: Tensorlike, dim=None, jump=np.pi) -> torch.Tensor:
    prev = None
    accum = 0
    def inner(v):
        nonlocal prev
        nonlocal accum
        if prev is None:
            prev = v
        r = v - prev
        if r > jump:
            while r > jump:
                accum -= 2*jump
                r -= 2*jump
        elif r < -jump:
            while r < -jump:
                accum += 2*jump
                r += 2*jump
        prev = v
        return r + accum
    return tensormap(inner, x, dim=dim)

def unwrap_acc(x: Tensorlike, dim=None, jump=np.pi) -> torch.Tensor:
    prev = None
    accum = 0
    def inner(v):
        nonlocal prev
        nonlocal accum
        if prev is None:
            prev = v
        r = v - prev
        if r > jump:
            while r > jump:
                accum -= 2*jump
                r -= 2*jump
        elif r < -jump:
            while r < -jump:
                accum += 2*jump
                r += 2*jump
        prev = v
        return accum * torch.ones_like(v)
    return tensormap(inner, x, dim=dim)


def unwrap1d(p, eps=1e-8):
    # N = length(p);
    N = len(p)
    # up = zeros(size(p));
    up = torch.zeros_like(p)
    # pm1 = p(1);
    pm1 = p[0]
    # up(1) = pm1;
    up[0] = pm1
    # po = 0;
    po = 0
    # thr = pi - eps;
    thr = np.pi - eps
    # pi2 = 2*pi;
    pi2 = 2*np.pi
    # for i=2:N
    for i in range(1, N):
        # cp = p(i) + po;
        cp = p[i] + po
        # dp = cp-pm1;
        dp = cp - pm1
        # pm1 = cp;
        pm1 = cp
        # if dp>thr
        if dp > thr:
            # while dp>thr
            while dp > thr:
                # po = po - pi2
                po = po - pi2
                # dp = dp - pi2;
                dp = dp - pi2
            # end
        # end
        # if dp<-thr
        if dp < -thr:
            # while dp<-thr
            while dp < -thr:
                # po = po + pi2
                po = po + pi2
                # dp = dp + pi2;
                dp = dp + pi2
            # end
        # end
        # cp = p(i) + po;
        cp = p[i] + po
        # pm1 = cp;
        pm1 = cp
        # up(i) = cp;
        up[i] = cp
    return up


def unwrap2d(x, dim=None):
    if dim is None:
        dim = [-1, -2]
    if is_complex(x):
        x = x.angle()
    return tensormap(unwrap1d, x, dim=dim)


def sample(kernel, y, x):
    if y is None:
        y = slice(None, None, None)
    else:
        y = tt(y).long()
        y = y % height(kernel)
    if x is None:
        x = slice(None, None, None)
    else:
        x = tt(x).long()
        x = x % width(kernel)
    return kernel[..., y, x]


def sampled(kernel, y, x):
    w = width(kernel)
    h = height(kernel)
    iy = tt(y*h).long()
    ix = tt(x*w).long()
    return kernel[..., iy % h, ix % w]


def second_difference(x, jump=None):
    x0 = x
    x = to_kernel(x)
    Nx, Cx, Hx, Wx = size(x)
    # algorithm: https://i.imgur.com/HwrmyvG.png
    def U(v):
        #prn(v)
        # unwrap
        #return 0 if v < 0 else v
        if jump is not None:
            if v < -jump:
                while v < -jump:
                    v += 2*jump
            if v > jump:
                while v > jump:
                    v -= 2*jump
        #return v.abs()
        return v
    def P(i, j):
        # if i < 0:
        #     i += Wx
        # if j < 0:
        #     j += Hx
        #result = sample(x, j % Hx, i % Wx)
        # i %= Wx
        # j %= Hx
        i = np.clip(i, 0, Wx-1)
        j = np.clip(j, 0, Hx-1)
        result = sample(x, j, i)
        #return result.abs()
        return result
    def H(i, j):
        return U(P(i - 1, j) - P(i, j)) \
               - U(P(i, j) - P(i + 1, j))
    def V(i, j):
        return U(P(i, j - 1) - P(i, j)) \
               - U(P(i, j) - P(i, j + 1))
    def D1(i, j):
        return U(P(i - 1, j - 1) - P(i, j)) \
               - U(P(i, j) - P(i + 1, j + 1))
    def D2(i, j):
        return U(P(i - 1, j + 1) - P(i, j)) \
               - U(P(i, j) - P(i + 1, j - 1))
    def D(i, j):
        h = H(i, j)
        v = V(i, j)
        d1 = D1(i, j)
        d2 = D2(i, j)
        #return (h * h + v * v + d1 * d1 + d2 * d2) ** 0.5
        return h.abs() + v.abs() + d1.abs() + d2.abs()
        #return v
    out = torch.zeros_like(x)
    for i in range(0, Wx):
        for j in range(0, Hx):
            out[..., j, i] = D(i, j)
    return from_kernel(out, x0)


def mkfilter(shift, I=None):
    I = impulse(1,1) if I is None else I
    filter = roll(I, shift)
    filter += roll(I, [-x for x in shift])
    filter -= 2 * I
    #filter += -1*roll(I, [-x for x in shift])
    #filter += roll(I, shift[::-1])
    return filter

def tap(img, shift, I=None):
    filter = mkfilter(shift, I)
    return conv2d(img, filter)

def smoothness(img):
    x = tap(img, [1,0]).abs()
    x += tap(img, [0,1]).abs()
    x += tap(img, [1,1]).abs()
    x += tap(img, [1,-1]).abs()
    return widen(x)


def dct1(x):
    """
    Discrete Cosine Transform, Type I
    :param x: the input signal
    :return: the DCT-I of the signal over the last dimension
    """
    x = tt(x)
    x_shape = x.shape
    x = x.view(-1, x_shape[-1])
    #return rfft(torch.cat([x, x.flip([1])[:, 1:-1]], dim=1))[:, :, 0].view(*x_shape)
    return rfft(torch.cat([x, x.flip([1])[:, 1:-1]], dim=1)).view(*x_shape)


def idct1(X):
    """
    The inverse of DCT-I, which is just a scaled DCT-I
    Our definition if idct1 is such that idct1(dct1(x)) == x
    :param X: the input signal
    :return: the inverse DCT-I of the signal over the last dimension
    """
    X = tt(X)
    n = X.shape[-1]
    return dct1(X) / (2 * (n - 1))



def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x = tt(x)
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    #Vc = rfft(v, 1, onesided=False)
    Vc = rfft(v, onesided=False)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct(dct(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    X = tt(X)
    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    #v = irfft(V, 1, onesided=False)
    v = irfft(V, onesided=False)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)


def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x = tt(x)
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(X, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct_2d(dct_2d(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X = tt(X)
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)

try:
    import scipy.fft
    from functools import wraps

    @wraps(scipy.fft.dctn)
    def dctn(x, type=2, s=None, axes=None, norm=None, overwrite_x=False):
        return tt(scipy.fft.dctn(to_numpy(x), type=type, s=s, axes=axes, norm=norm, overwrite_x=overwrite_x))

    @wraps(scipy.fft.idctn)
    def idctn(x, type=2, s=None, axes=None, norm=None, overwrite_x=False):
        return tt(scipy.fft.idctn(to_numpy(x), type=type, s=s, axes=axes, norm=norm, overwrite_x=overwrite_x))

except ImportError:
    pass

try:
    import matplotlib.pyplot as plt
    import imgcat

    def plot(*args, cat=True, figsize=None, dpi=300, linewidth=0.4, **kws):
        if figsize is None:
            # https://stackoverflow.com/questions/47633546/relationship-between-dpi-and-figure-size
            figsize = dpi / 72
            figsize = (2*figsize, figsize)
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.plot(*args, linewidth=linewidth, **kws)
        if cat:
            imgcat.imgcat(fig)
        return fig, ax
except ImportError:
    pass


def tensormap(fn, x: Tensorlike, *args, dim=-1, after=None, before=None, index=False) -> torch.Tensor:
    if dim is None:
        dim = np.arange(len(x.shape)).tolist()
    dims = flatlist(dim)
    if before is not None:
        x = before(x)
    shape = x.shape
    for dim in dims:
        if index:
            x = [fn(y.reshape(-1), i, *args) for i, y in enumerate(x.unbind(dim))]
        else:
            x = [fn(y.reshape(-1), *args) for y in x.unbind(dim)]
        #x = [fn(y.reshape(-1), *([i] if index else ([] + list(args)))).reshape(*y.shape) for i, y in enumerate(x.unbind(dim))]
        x = torch.stack(x, dim=dim)
        #x = x.reshape(*shape)
    if after is not None:
        x = after(x)
    return x


def impulse(*size, dtype=torch.float32):
    size = flatlist(size)
    t = torch.tensor([1], dtype=dtype)
    t = t.reshape([1] * len(size))
    t = F.pad(t, flatlist([(p, p) for p in size]))
    return t


# def to_kernel(x):
#     x = to_tensor(x)
#     while len(x.shape) < 4:
#         x = x.unsqueeze(0)
#     return x



def conv2d(w, filter):
    w0 = w
    w = to_kernel(w)
    f = to_kernel(filter)
    x = F.conv2d(w.float(), f.float())
    return from_kernel(x, w0)

# def conv2d(w, filter):
#     size = list(w.shape)
#     w = to_kernel(w)
#     filter = to_kernel(filter)
#     H, W = filter.shape[-2:]
#     w = F.pad(w, [H // 2, W // 2] * 2)
#     #padlist = flatlist([(value // 2, value // 2) for value in filter.shape])
#     #w = F.pad(w, padlist)
#     result = F.conv2d(w, filter)
#     result = result.reshape(size)
#     return result
#
# def conv2d2(w, filter):
#     size = list(w.shape)
#     w = to_kernel(w).reshape(size)
#     filter = to_kernel(filter).reshape(size)
#


def prn(x, *args):
    print(x, *args)
    return x


def ndim(x):
    return len(x.shape)


def pairify(x):
    x = flatlist(x)
    if len(x) < 2:
        x *= 2
    return x


def widen(x, along_h=1, along_w=None):
    x0 = x
    x = to_kernel(x)
    if along_w is None:
        along_w = along_h
    along_h = pairify(along_h)
    along_w = pairify(along_w)
    x = F.pad(x, along_w + along_h)
    return from_kernel(x, x0)


def narrow(x, along_h=1, along_w=None):
    x0 = x
    x = to_kernel(x)
    if along_w is None:
        along_w = along_h
    along_h = pairify(along_h)
    along_w = pairify(along_w)
    #x = x[..., along_h[0]:-along_h[1], along_w[0]:-along_w[1]]
    x = F.pad(x, [-x for x in along_w + along_h])
    return from_kernel(x, x0)


def widen_h(img, by=1):
    return widen(img, along_h=by, along_w=0)


def widen_w(img, by=1):
    return widen(img, along_h=0, along_w=by)


def narrow_h(img, by=1):
    return narrow(img, along_h=by, along_w=0)


def narrow_w(img, by=1):
    return narrow(img, along_h=0, along_w=by)


def reflect(x, padding=None):
    x0 = x
    x = to_kernel(x)
    if padding is None:
        padding = [i-1 for i in x.shape]
    x = torch.nn.ReflectionPad2d(padding)(x)
    # if len(x.shape) > len(shape):
    #     x = x[0]
    # return x
    return from_kernel(x, x0)


def width(x):
    return x.shape[-1] if hasattr(x, 'shape') else x


def height(x):
    if ndim(x) < 2:
        return 1
    return x.shape[-2]


def channel_count(x):
    if ndim(x) < 3:
        return 1
    return x.shape[-3]


def batch_size(x):
    if ndim(x) < 4:
        return 1
    return x.shape[-4]


def size(x):
    return batch_size(x), channel_count(x), height(x), width(x)


N_ = batch_size
C_ = channel_count
H_ = height
W_ = width


def dim_index(x, dim):
    if dim < 0:
        return ndim(x) - dim
    else:
        return dim


def unsqueeze(x, dim, count=1) -> torch.Tensor:
    for i in range(count):
        x = x.unsqueeze(dim)
    return x


def to_kernel(x) -> torch.Tensor:
    x = tt(x)
    N = ndim(x)
    if N == 1:
        return unsqueeze(x, 0, 3)
    elif N == 2:
        return unsqueeze(x, 0, 2)
    elif N == 3:
        return unsqueeze(x, 1)
    elif N > 4:
        return x.view(-1, *x.shape[-3:])
    else:
        return x


def from_kernel(x, orig) -> torch.Tensor:
    N = ndim(orig)
    if N <= 1:
        return x.view(-1)
    elif N <= 2:
        return x.view(-1, x.shape[-1])
    elif N <= 3:
        return x.view(-1, x.shape[-2], x.shape[-1])
    else:
        return x


def is_complex(x):
    return x.is_complex()


def to_float(x: np.ndarray):
    x = real(x)
    if isinstance(x, torch.Tensor):
        return x.float()
    return np.cast(to_numpy(x), np.float32)



def real(x):
    x = tt(x)
    if is_complex(x):
        x = x.real
    return x.float()

def imag(x):
    x = tt(x)
    if is_complex(x):
        x = x.imag
    else:
        x = x * 0.0
    return x.float(0)


def mag(x):
    if x.is_complex():
        return x.abs().float()
    return x


def conv(image, filter):
    shape = image.shape[:]
    image = to_kernel(image)
    filter = to_kernel(filter)
    F.conv2d(image, filter)
    # TODO


# equivalence = tensormap( lambda x: tt( np.fft.fft(x) ), images[0], dim=[-1, -2] ) ; vis( roll( t ).abs(), roll( fft2( images[0] ) ).abs() )

#tensormap(lambda x: tt(np.hamming(512))*x, tt(np.hamming(512)))

def hamming(*size):
    if len(size) == 1:
        img = size[0]
        return hamming(height(img), width(img))
    size = flatlist(size)
    x = tt(np.hamming(size.pop()))
    for k in size[::-1]:
        x = tensormap(lambda v: tt(np.hamming(k)) * v, x)
    return x



import time
from ansi_escapes import ansiEscapes


def prd(*messages, delay=0.1):
    x, *messages = flatlist(messages)
    print(x)
    for msg in messages:
        time.sleep(delay)
        print(ansiEscapes.cursorPrevLine + str(msg))


def hsl2rgb(hsl):
    h = hsl[0] / 360
    s = hsl[1] / 100
    l = hsl[2] / 100

    if s == 0:
        val = l * 255
        return [val, val, val]

    if (l < 0.5):
        t2 = l * (1 + s)
    else:
        t2 = l + s - l * s

    t1 = 2 * l - t2

    rgb = [0, 0, 0]
    for i in range(3):
        t3 = h + 1 / 3 * -(i - 1)
        if (t3 < 0):
            t3 += 1

        if (t3 > 1):
            t3 -= 1

        if (6 * t3 < 1):
            val = t1 + (t2 - t1) * 6 * t3
        elif (2 * t3 < 1):
            val = t2
        elif (3 * t3 < 2):
            val = t1 + (t2 - t1) * (2 / 3 - t3) * 6
        else:
            val = t1

        rgb[i] = val * 255

    return rgb

# import sparkvis
#
# angles = sparkvis.textures['angles'].copy()

angles = np.array(list(''.join('🕒🕑🕐🕛🕚🕙🕘🕗🕖🕕🕔🕓')))
#angles = np.array(list(''.join('→↗↑↖←↙↓↘')))
angles = np.array([' '+c for c in '→↗↑↖←↙↓↘'])
#angles = np.array([' '+c for c in '⇨⬀⇧⬁⇦⬃⇩⬂'])
#angles = np.array([c+k+l for c in '→↗↑↖←↙↓↘' for k in ['', '\u0307', '\u0308'] for l in ['', '\u0324', '\u0323']])
angles = np.array([c+k+l for c in '→↗↑↖←↙↓↘' for k in [''] for l in ['', '\u030D', '\u030E']])
angles = np.array([' '+c+k for c in '→↗↑↖←↙↓↘' for k in ['', '\u0307', '\u0308', '\u0308\u0323', '\u0308\u0324']])
#angles = np.array([' '+c+k for c in '→↗↑↖←↙↓↘' for k in ['', '\u0307', '\u0308', '\u0308\u0324']])
#angles = np.array([' '+c+k for c in '→↗↑↖←↙↓↘' for k in ['', '\u0307', '\u0308']])
angles = np.array([' '+c for c in '→↗↑↖←↙↓↘'])
empty = '  '

def angle(x, empty=empty):
    if not is_complex(x):
        return empty
    x = x.angle()
    return angles[((x/np.pi*0.5) * (len(angles)+1)).long()]


from ansi_styles import ansiStyles
from supports_color import supportsColor

def to_rgb(rgb):
    if isinstance(rgb, str):
        rgb = ansiStyles.hexToRgb(rgb)
    rgb = flatlist(rgb)
    if len(rgb) < 3:
        rgb += [None] * (3 - len(rgb))
    rgb = [rgb[0] if x is None else x for x in rgb]
    rgb = [int(max(0, min(255, x*255))) if not isinstance(x, int) else x for x in rgb]
    return rgb

def stylize(c, rgb=None, foreground=True):
    if not supportsColor.stdout or rgb is None:
        return c
    rgb = to_rgb(rgb)
    color = ansiStyles.color if foreground else ansiStyles.bgColor
    close = color.close
    #close = ansiStyles.modifier.reset.close
    if supportsColor.stdout.has16m:
        return color.ansi16m(*rgb) + str(c) + close
    if supportsColor.stdout.has256:
        return color.ansi256(ansiStyles.rgbToAnsi256(*rgb)) + str(c) + close
    else:
        return color.ansi(ansiStyles.rgbToAnsi(*rgb)) + str(c) + close


def colorize(c, fgColor=None, bgColor=None):
    if fgColor is not None:
        c = stylize(c, fgColor, foreground=True)
    if bgColor is not None:
        c = stylize(c, bgColor, foreground=False)
    return c


def div(a, b):
    a = tt(a)
    b = tt(b)
    x = a / b
    x[x != x] = a[x != x]
    return x

def sinc(x):
    x = tt(x)
    if is_complex(x):
        x = x.angle() / np.pi
    r = div((x * np.pi).sin(), x * np.pi)
    r[x == 0] = 1.0
    return r

class Vis:
    def __init__(self, value):
        self.value = tt(value)
    def to_string(self, *, join=True):
        x = to_kernel(self.value)
        if len(x.shape) <= 0:
            return ''.join(angle(x))
        x = x.view(-1, x.shape[-1])
        m = mag(x)
        lo = m.min()
        hi = m.max()
        #x = (x - m.min()) / (m.max() - m.min())
        lines = []
        for row in x:
            line = []
            for value in row:
                c = angle(value)
                if (hi - lo).abs() > 1e-5 or True:
                    v = (mag(value) - lo) / (hi - lo)
                    # if c == angles[0]: # or v < 0.01:
                    #     c = '  '
                    #v = float(to_numpy(v + 0.0001).log())
                    #c = colorize(c, None, v)
                    if c is empty:
                        c = colorize(c, None, v)
                    else:
                        v = v ** (1 / 2.2)
                        #hue = (value.angle() + np.pi) * 180 / np.pi
                        #u = [u/255.0 for u in hsl2rgb((hue, 100, 50))]
                        # u = 1.0 - v
                        # Vx = value.angle().cos()
                        # Vy = value.angle().sin()
                        # u = (v*Vx, v*1.0, v*Vy)
                        # #u = (Vx, 1.0, Vy)
                        #c = colorize(c, v, None)
                        c = colorize(c, 1.0 if v < 0.9 else 0.0, v)
                line.append(c)
            if join:
                line = ''.join(line)
            lines.append(line)
        if join:
            lines = '\n'.join(lines)
            # if supportsColor.stdout:
            #     lines = lines + colorize('|', 0, 0)
        return lines
        #return '\n'.join([''.join(row) for row in angle(x)])
    def __str__(self):
        return self.to_string()
    def __repr__(self):
        return '\n'+str(self)
    def __float__(self):
        if len(self.value.shape) > 0:
            raise ValueError("Can't convert multiple angles to float")
        return float(self.value.angle())


# A cosine wave of frequency f is sampled at times t = n * 1/Fs, where n is the
# sample number (an integer) and Fs is the sample rate.
#
# Verify that cos(2pi*f*t) = cos(2*pi*(k*Fs-f)t) = cos(2pi(k*Fs+f)t) for all integers k, positive or negative.


def complex(real, imag):
    return torch.complex(tt(real).float(), tt(imag).float())

def polar(angle, magnitude=1.0):
    #return torch.polar(tt(magnitude).float(), tt(angle).float())
    return complex(tt(magnitude) * tt(angle).cos(), tt(magnitude) * tt(angle).sin())

def lin(a, b, n):
    #return np.linspace(to_numpy(tt(a)), to_numpy(tt(b)), n + 1)
    return torch.linspace(tt(a), tt(b), tt(n + 1))


def til(a, b, n):
    return lin(a, b, n)[:-1]
    #return lin(a, b, n)[1:]


def steps(*args):
    if len(args) == 1:
        return torch.arange(*args)
    if len(args) == 2:
        return til(0, *args)
    return til(*args)


def upto(*args):
    if len(args) == 1:
        return torch.arange(args[0] + 1)
    if len(args) == 2:
        return lin(0, *args)
    return lin(*args)


def exp(x):
    return torch.exp(tt(x))

def e(x):
    return exp(1j * tt(x))


# def imgshift(img, dy, dx):
#     # https://stackoverflow.com/questions/25827916/matlab-shifting-an-image-using-fft
#     v = to_kernel(img)
#     n, c, h, w = size(v)
#     yF, xF = torch.meshgrid( torch.arange(h).float() - h//2, torch.arange(w).float() - w//2, )
#     v = ifft2(fft2(v) * exp(-1j * 2 * pi * (xF*dx / w + yF*dy / h)))
#     #v = v.abs()
#     return from_kernel(v, img)


def grid(h, w, sy=1, sx=1, dy=0, dx=0):
    return torch.meshgrid(
        til(dy, dy+sy, h),
        til(dx, dx+sx, w),
    )


@wraps(grid)
def sumgrid(*args, **kws):
    return sum(grid(*args, **kws))


# def imgshift(img, dy, dx):
#     v = to_kernel(img)
#     n, c, h, w = size(v)
#     yF, xF = torch.meshgrid( torch.arange(h).float(), torch.arange(w).float(), )
#     v = ifft2(fft2(v) * exp(-2 * pi * 1j * (xF * dx / w + yF * dy / h)))
#     return from_kernel(v, img)


PI = 1j * np.pi
TAU = 2j * np.pi


def rot(x):
    return exp(2j * np.pi * tt(x))


def imgshift(img, dy, dx):
    v = to_kernel(img)
    h = height(img)
    w = width(img)
    f = sumgrid(h, w, dy, dx)
    v = ifft2(fft2(v) * rot(-f))
    return from_kernel(v, img)


def imgshift(img, dy, dx):
    v = to_kernel(img)
    h = height(img)
    w = width(img)
    Ry, Rx = grid(h, w)
    v = fft2(v)
    v *= rot(-Ry) ** dy
    v *= rot(-Rx) ** dx
    v = ifft2(v)
    v = real(v)
    return from_kernel(v, img)



def stretch(img, h=None, w=None):
    if h is None:
        h = 2
    if w is None:
        w = 2
    img_h = height(img)
    img_w = width(img)
    return sample(img, *grid(int(h * img_h), int(w * img_w), img_h, img_w))



# def imgshifter(img, dy, dx):
#     h = height(img)
#     w = width(img)
#     yF, xF = torch.meshgrid( torch.arange(h).float(), torch.arange(w).float(), )
#     return exp(-2j * pi * (xF * dx / w + yF * dy / h))


def idx(n, i):
    return int(i % n)


def frac(img, dy, dx):
    h = height(img)
    w = width(img)
    Fy = torch.zeros(1, 1, h, w)
    Fy[..., idx(h, dy + 1), idx(w, dx + 0)] = dy % 1.0
    Fy[..., idx(h, dy + 0), idx(w, dx + 0)] = 1.0 - dy % 1.0
    Fx = torch.zeros(1, 1, h, w)
    Fx[..., idx(h, dy + 0), idx(w, dx + 1)] = dx % 1.0
    Fx[..., idx(h, dy + 0), idx(w, dx + 0)] = 1.0 - dx % 1.0
    return from_kernel(Fy, img), from_kernel(Fx, img)

pi = np.pi
tau = 2 * np.pi


def circ1d(signal, kernel):
    kernel = tt(kernel)
    s = to_kernel(signal)
    k = to_kernel(kernel)
    Sw = width(s)
    Kw = width(k)
    k = F.pad(k, [0, Sw - Kw])
    kw = width(k)
    i = sumgrid(Sw, kw, Sw, -kw).long() % kw
    #return k[..., i]
    #return i, k
    #return from_kernel(k, kernel)[..., i]
    result = from_kernel(k, kernel)[..., i]
    result = shift(result, by=Kw//2)
    return result



def window1d(signal, kernel):
    Aw = width(signal)
    Bw = width(kernel)
    return sumgrid(Aw, Bw, Aw, Bw)[:-Bw+1].long()


def sampler1d(signal, keepdim=False):
    if callable(signal):
        return signal
    signal = tt(signal)
    v = to_kernel(signal)
    w = width(v)
    def sample_signal(t=None):
        if t is None:
            return signal
        i = tt(t).long()
        o = v[..., i % w]
        if keepdim:
            return from_kernel(o, signal)
        else:
            return o
    return sample_signal


def conv1d(signal, kernel):
    s = sampler1d(signal)
    k = sampler1d(kernel)
    i = window1d(s(), k())
    r = (s(i) * k()).sum(-1)
    return from_kernel(r, s())


def corr1d(signal, kernel):
    return conv1d(signal, -sampler1d(kernel)())


def rdft(signal):
    s = sampler1d(signal)
    N = width(s())
    k = til(0, N, N)
    def sample_signal(f=None):
        if f is None:
            return s()
        return (s(k) * (-2 * pi * f * k / N).cos()).sum()
    return sample_signal


def tr(x):
    x = tt(x)
    v = to_kernel(x)
    v = v.permute(0,1,-1,-2)
    return from_kernel(v, x)


def dft_f(signal):
    s = sampler1d(signal)
    N = width(s())
    k = til(0, N, N)
    def sample_signal(f=None):
        if f is None:
            return s()
        return s(k) * exp(-2j * pi * f * k / N)
    return sample_signal


def dft(signal):
    s = dft_f(signal)
    N = width(s())
    k = til(0, N, N)
    return torch.stack([s(t).sum(-1) for t in k], -1)


def dft2(signal):
    #return dft(dft(tt(signal).t()).t())
    return dft(tr(dft(tr(signal))))


def idft_f(signal):
    s = sampler1d(signal)
    N = width(s())
    k = til(0, N, N)
    def sample_signal(f=None):
        if f is None:
            return s()
        return s(k) * exp(2j * pi * f * k / N) / N
    return sample_signal


def idft(signal):
    s = idft_f(signal)
    N = width(s())
    k = til(0, N, N)
    return torch.stack([s(t).sum(-1) for t in k], -1)


def idft2(signal):
    return idft(tr(idft(tr(signal))))


def edges(img):
    return widen(conv2d(img, tr([[0, 0.25, 0], [-0.25, 0, 0.25], [0, -0.25, 0]])))


def besselsum(v, x, n):
    den = 1/(math.factorial(n) * np.prod([1j*v + m for m in range(0,n+1)]))
    return (-1) ** n * den * (x/2) ** (2*n)

constgamma = np.array(
    [float('inf'), 1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0, 362880.0, 3628800.0, 39916800.0, 479001600.0,
     6227020800.0, 87178291200.0, 1307674368000.0, 20922789888000.0, 355687428096000.0, 6402373705728000.0])


def gamma(n):
    return math.factorial(int(n - 1))


def besselsum_n(v, x, n):
    #den = 1/(math.factorial(n) * gamma(1+v+n))
    #return (-1)**n * den * (x/2) ** (2*n)
    #den = 1/(math.factorial(n) * math.factorial(v+n))
    r = (-1)**n
    #r /= gamma(1+v+n)
    for i in range(int(v)):
        r /= n
    # for i in range(2, int(n+1)):
    #     r /= i
    # #r *= (x/2) ** (2*n)
    # for i in range(int(2*n)):
    #     r *= x/2
    # #r /= math.factorial(n)
    # for i in range(2, int(n+1)):
    #     r /= i
    for i in range(1, int(n+1)):
        r *= x/2
        r /= i
        r /= i
        r *= x/2

    # r *= (x/2) ** (n)
    # for i in range(2, int(n)+1):
    #     r /= i
    # for i in range(2, int(n+v)+1):
    #     r /= i
    # r *= (x/2) ** (n)

    # r /= math.factorial(n)**2
    # r /= math.factorial(n)
    # r /= math.factorial(n)
    # for i in range(int(v)):
    #     r /= n
    #r /= math.factorial(n)
    #r /= math.factorial(v+n)
    return r

def bessel(v, x, N=20):
    return (x/2)**v * sum([besselsum_n(v, x, n) for n in range(N)])
    #(tt(-1) ** n) * (1/([1j * v + m for m in torch.arange(n+1)]) * ((x/2) ** (2*n))


def Cf(v, x):
    v = tt(v)
    x = tt(x)
    lh = 1 - (1 / (1 + v*v)) * (x / 2) ** 2
    rh =     (v / (1 + v*v)) * (x / 2) ** 2
    t = (v * x.log())
    return lh * t.cos() + rh * t.sin()


def Sf(v, x):
    v = tt(v)
    x = tt(x)
    lh = 1 - (1 / (1 + v*v)) * (x / 2) ** 2
    rh =     (v / (1 + v*v)) * (x / 2) ** 2
    t = (v * x.log())
    return lh * t.sin() + rh * t.cos()



# static double polevl(double x, const double coef[], int N)
# {
#     double ans;
#     const double *p = coef;
#     int i = N;
#
#     ans = *p++;
#     do
#         ans = ans * x + *p++;
#     while(--i);
#
#     return ans ;
# }

def polevl(x, coef, N=None):
    # ans = 1.0
    # for i in range(N):
    #     ans = ans * x + coef[i]
    # return ans
    ans = coef[0]
    for i, p in enumerate(coef[1:]):
        ans = ans * x + p
    return ans


# double p1evl(double x, const double coef[], int N)
# {
#     double ans;
#     const double *p = coef;
#     int i = N - 1;
#
#     ans = x + *p++;
#     do
#         ans = ans * x + *p++;
#     while(--i);
#
#     return ans;
# }

def p1evl(x, coef, N=None):
    ans = x + coef[0]
    for i, p in enumerate(coef[1:]):
        ans = ans * x + p
    return ans

# #define THPIO4 2.35619449019234492885
# #define SQ2OPI .79788456080286535588
# #define Z1 1.46819706421238932572E1
# #define Z2 4.92184563216946036703E1
#
THPIO4 = 2.35619449019234492885
SQ2OPI = .79788456080286535588
Z1 = 1.46819706421238932572E1
Z2 = 4.92184563216946036703E1

#     const double RP[4] = {
RP = np.array([
    -8.99971225705559398224E8,
    4.52228297998194034323E11,
    -7.27494245221818276015E13,
    3.68295732863852883286E15,
])
#     };
#     const double RQ[8] = {
RQ = np.array([
    6.20836478118054335476E2,
    2.56987256757748830383E5,
    8.35146791431949253037E7,
    2.21511595479792499675E10,
    4.74914122079991414898E12,
    7.84369607876235854894E14,
    8.95222336184627338078E16,
    5.32278620332680085395E18,
])
#     };
#     const double PP[7] = {
PP = np.array([
    7.62125616208173112003E-4,
    7.31397056940917570436E-2,
    1.12719608129684925192E0,
    5.11207951146807644818E0,
    8.42404590141772420927E0,
    5.21451598682361504063E0,
    1.00000000000000000254E0,
])
#     };
#     const double PQ[7] = {
PQ = np.array([
    5.71323128072548699714E-4,
    6.88455908754495404082E-2,
    1.10514232634061696926E0,
    5.07386386128601488557E0,
    8.39985554327604159757E0,
    5.20982848682361821619E0,
    9.99999999999999997461E-1,
])
#     };
#     const double QP[8] = {
QP = np.array([
    5.10862594750176621635E-2,
    4.98213872951233449420E0,
    7.58238284132545283818E1,
    3.66779609360150777800E2,
    7.10856304998926107277E2,
    5.97489612400613639965E2,
    2.11688757100572135698E2,
    2.52070205858023719784E1,
])
#     };
#     const double QQ[7] = {
QQ = np.array([
    7.42373277035675149943E1,
    1.05644886038262816351E3,
    4.98641058337653607651E3,
    9.56231892404756170795E3,
    7.99704160447350683650E3,
    2.82619278517639096600E3,
    3.36093607810698293419E2,
])
#     };

# static double _bessel_j1(double x)
# {
#     double w, z, p, q, xn;
#
#     w = x;
#     if (x < 0)
#         w = -x;
#
#     if (w <= 5.0) {
#         z = x * x;
#         w = polevl(z, RP, 3) / p1evl(z, RQ, 8);
#         w = w * x * (z - Z1) * (z - Z2);
#         return w ;
#     }
#
#     w = 5.0 / x;
#     z = w * w;
#     p = polevl(z, PP, 6) / polevl(z, PQ, 6);
#     q = polevl(z, QP, 7) / p1evl(z, QQ, 7);
#     xn = x - THPIO4;
#     p = p * cos(xn) - w * q * sin(xn);
#     return p * SQ2OPI / sqrt(x);
# }


def bessel_j1(x):
    w = x
    if x < 0:
        w = -x

    if w <= 5.0:
        z = x * x
        w = polevl(z, RP, 3) / p1evl(z, RQ, 8)
        w = w * x * (z - Z1) * (z - Z2)
        return w

    w = 5.0 / x
    z = w * w
    p = polevl(z, PP, 6) / polevl(z, PQ, 6)
    q = polevl(z, QP, 7) / p1evl(z, QQ, 7)
    xn = x - THPIO4
    p = p * np.cos(xn) - w * q * np.sin(xn)
    return p * SQ2OPI / (x**0.5)


def j1(x):
    r = x * 0.0

    if True:
        i = np.abs(x) <= 5.0
        v = x[i]
        z = v * v
        w = polevl(z, RP, 3) / p1evl(z, RQ, 8)
        w = w * v * (z - Z1) * (z - Z2)
        r[i] = w

    i = i == False
    v = x[i]
    w = 5.0 / v
    z = w * w
    p = polevl(z, PP, 6) / polevl(z, PQ, 6)
    q = polevl(z, QP, 7) / p1evl(z, QQ, 7)
    xn = v - THPIO4
    p = p * np.cos(xn) - w * q * np.sin(xn)
    r[i] = p * SQ2OPI / (v**0.5)
    return r


PP0 = np.array([
    7.96936729297347051624E-4,
    8.28352392107440799803E-2,
    1.23953371646414299388E0,
    5.44725003058768775090E0,
    8.74716500199817011941E0,
    5.30324038235394892183E0,
    9.99999999999999997821E-1,
])

PQ0 = np.array([
    9.24408810558863637013E-4,
    8.56288474354474431428E-2,
    1.25352743901058953537E0,
    5.47097740330417105182E0,
    8.76190883237069594232E0,
    5.30605288235394617618E0,
    1.00000000000000000218E0,
])

QP0 = np.array([
    -1.13663838898469149931E-2,
    -1.28252718670509318512E0,
    -1.95539544257735972385E1,
    -9.32060152123768231369E1,
    -1.77681167980488050595E2,
    -1.47077505154951170175E2,
    -5.14105326766599330220E1,
    -6.05014350600728481186E0,
])

QQ0 = np.array([
    #  1.00000000000000000000E0,
    6.43178256118178023184E1,
    8.56430025976980587198E2,
    3.88240183605401609683E3,
    7.24046774195652478189E3,
    5.93072701187316984827E3,
    2.06209331660327847417E3,
    2.42005740240291393179E2,
])

YP0 = np.array([
    1.55924367855235737965E4,
    -1.46639295903971606143E7,
    5.43526477051876500413E9,
    -9.82136065717911466409E11,
    8.75906394395366999549E13,
    -3.46628303384729719441E15,
    4.42733268572569800351E16,
    -1.84950800436986690637E16,
])

YQ0 = np.array([
    # /* 1.00000000000000000000E0, */
    1.04128353664259848412E3,
    6.26107330137134956842E5,
    2.68919633393814121987E8,
    8.64002487103935000337E10,
    2.02979612750105546709E13,
    3.17157752842975028269E15,
    2.50596256172653059228E17,
])

# /*  5.783185962946784521175995758455807035071 */
DR10 = 5.78318596294678452118E0

# /* 30.47126234366208639907816317502275584842 */
DR20 = 3.04712623436620863991E1

RP0 = np.array([
    -4.79443220978201773821E9,
    1.95617491946556577543E12,
    -2.49248344360967716204E14,
    9.70862251047306323952E15,
])

RQ0 = np.array([
    # /* 1.00000000000000000000E0, */
    4.99563147152651017219E2,
    1.73785401676374683123E5,
    4.84409658339962045305E7,
    1.11855537045356834862E10,
    2.11277520115489217587E12,
    3.10518229857422583814E14,
    3.18121955943204943306E16,
    1.71086294081043136091E18,
])

NPY_PI_4 = .78539816339744830962


def bessel_j0(x):
    if x < 0:
        x = -x

    if x <= 5.0:
        z = x * x
        if x < 1.0e-5:
            return 1.0 - z / 4.0

        p = (z - DR10) * (z - DR20)
        p = p * polevl(z, RP0, 3) / p1evl(z, RQ0, 8)
        return p

    w = 5.0 / x
    q = 25.0 / (x * x)
    p = polevl(q, PP0, 6) / polevl(q, PQ0, 6)
    q = polevl(q, QP0, 7) / p1evl(q, QQ0, 7)
    xn = x - NPY_PI_4
    p = p * np.cos(xn) - w * q * np.sin(xn)
    return p * SQ2OPI / x**0.5


def j0(x):
    x = np.abs(x)
    r = x * 0.0

    if True:
        i = x <= 5.0
        v = x[i]
        z = v * v
        # if x < 1.0e-5:
        #     return 1.0 - z / 4.0

        p = (z - DR10) * (z - DR20)
        p = p * polevl(z, RP0, 3) / p1evl(z, RQ0, 8)
        r[i] = p

    i = x > 5.0
    v = x[i]
    w = 5.0 / v
    q = 25.0 / (v * v)
    p = polevl(q, PP0, 6) / polevl(q, PQ0, 6)
    q = polevl(q, QP0, 7) / p1evl(q, QQ0, 7)
    xn = v - NPY_PI_4
    p = p * np.cos(xn) - w * q * np.sin(xn)
    r[i] = (p * SQ2OPI / v**0.5)
    return r

# def j1(x):
#     x = tt(x).numpy()
#     if ndim(x) <= 0:
#         return bessel_j1(x)
#     return tt(np.array([bessel_j1(v) for v in x]))
#
#
# def j0(x):
#     x = tt(x).numpy()
#     if ndim(x) <= 0:
#         return bessel_j0(x)
#     return tt(np.array([bessel_j0(v) for v in x]))




# https://dsp.stackexchange.com/questions/58301/2-d-circularly-symmetric-low-pass-filter
def circularLowpassKernel(omega_c, N):  # omega = cutoff frequency in radians (pi is max), N = horizontal size of the kernel, also its vertical size, must be odd.
  kernel = np.fromfunction(lambda x, y: omega_c*j1(omega_c*np.sqrt((x - (N - 1)/2)**2 + (y - (N - 1)/2)**2))/(2*np.pi*np.sqrt((x - (N - 1)/2)**2 + (y - (N - 1)/2)**2)), [N, N])
  kernel[(N - 1)//2, (N - 1)//2] = omega_c**2/(4*np.pi)
  return kernel


# https://dsp.stackexchange.com/questions/58449/efficient-implementation-of-2-d-circularly-symmetric-low-pass-filter
def rotatedCosineWindow(N):  # N = horizontal size of the targeted kernel, also its vertical size, must be odd.
  return np.fromfunction(lambda y, x: np.maximum(np.cos(np.pi/2*np.sqrt(((x - (N - 1)/2)/((N - 1)/2 + 1))**2 + ((y - (N - 1)/2)/((N - 1)/2 + 1))**2)), 0), [N, N])


import PIL.Image
import io
import requests
import builtins


def url2img(url):
    return PIL.Image.open(io.BytesIO(requests.get(url).content))


def tensor(x):
    if isinstance(x, PIL.Image.Image):
        x = np.array(x.resize((min(x.width, 64), min(x.height, 64))).convert('L')) / 255.0
    return tt(x)


def visual_repr(x):
    return repr(Vis(tensor(x)))


def patch_repr():
    PIL.Image.Image.__repr__ = visual_repr
    torch.Tensor.__repr__ = visual_repr


def hcat(*args, sep=''):
    return '\n'.join(map(lambda *xs: sep.join(xs), *[x.splitlines() for x in args]))


def hrepr(*args, sep='', repr=builtins.repr):
    return '\n'.join(map(lambda *xs: sep.join(xs), *[repr(x).strip().splitlines() for x in args]))


def see(x):
    x = tensor(x)
    print(hrepr(x, fft2(x), repr=visual_repr))


def see(*args):
    #print(hcat(*[repr(Vis(tensor(x))) for x in flatlist(args)]))
    print(hcat(*[repr(Vis(x)) for x in args], sep=' '))


def updown(img):
    return torch.flipud(tensor(img))


def leftright(img):
    return torch.fliplr(tensor(img))


def flip(img):
    return leftright(updown(img))


def padx(img, by=1):
    #return widen(img, along_w=[0, by] if by > 0 else [abs(by), 0], along_h=0)
    return (widen if by > 0 else narrow)(img, along_w=[0, abs(by)], along_h=0)


def pady(img, by=1):
    return (widen if by > 0 else narrow)(img, along_h=[0, abs(by)], along_w=0)


def pad(img, by=1):
    img = pady(img, by=by)
    img = padx(img, by=by)
    return img


def repeat(n, f):
    @wraps(f)
    def repeater(x, *args, **kws):
        for i in range(n):
            x = f(x, *args, **kws)
        return x
    return repeater


def area(img):
    return width(img) * height(img)


def dc(x):
    v = to_kernel(x)
    r = torch.zeros_like(v)
    r[:, :, 0, 0] = v[:, :, 0, 0]
    return from_kernel(r, x)


def toprow(img):
    x = to_kernel(img)
    x = x[:, :, 0:1, :]
    return from_kernel(x, img)