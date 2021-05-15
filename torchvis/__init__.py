__version__ = '0.1.0'

from . import stft

import numpy as np

import torch
import torch.nn.functional as F
import torch.fft

from typing import Union


Tensorlike = Union[np.ndarray, torch.Tensor]


def checker(*sizes) -> torch.Tensor:
    """Checkerboard pattern."""
    t=torch.arange(sizes[-1]) % 2 == 0
    return t if len(sizes) < 2 else torch.stack([t, t.roll([1])] * ((sizes[-2])//2) + [t] * (sizes[-2] % 2))


def roll(x, shifts=None, dims=None):
    """Roll along the specified dims. By default, roll by half along width and height."""
    shifts = [x.shape[-2] // 2, x.shape[-1] // 2] if shifts is None else shifts
    if dims is None:
        dims = torch.arange(len(x.shape))[-2:].tolist()
    return x.roll(dims=dims, shifts=shifts)


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


def iabsnorm(x: Tensorlike) -> torch.Tensor:
    assert x.is_complex()
    return rnorm(x.abs())
