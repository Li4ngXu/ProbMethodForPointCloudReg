import scipy
import numpy as np
from typing import Callable
import scipy
from numba import jit, njit
import warnings
import torch


def gaussian_kernel_3d(x):
    '''
    a standard gaussian kernel
    '''

    try:
        return gaussian_kernel_3d.standard_normal.pdf(x)
    except:
        gaussian_kernel_3d.standard_normal = scipy.stats.multivariate_normal([0, 0, 0], np.eye(3))

    return gaussian_kernel_3d.standard_normal.pdf(x)


@njit(fastmath=True)
def gaussian_kernel_3d_speed(x):
    x = np.asarray(x)
    sqd = np.sum(x**2, axis=-1)
    return 1/((2*np.pi)**1.5) * np.exp(-0.5*sqd)


def gaussian_kernel_3d_torch(x):
    return 1.0/((2.0*torch.pi)**1.5) * torch.exp(-0.5*torch.sum(x**2, dim=-1))


def cubic_uniform_kernel_3d(x: np.ndarray):
    '''
    A uniform kernel: K(x) = 1/8 for -1<=x<=1
    '''

    if x.max() > 1 or x.min() < -1:
        return 0

    return 1.0/8


class KernelDensityEstimator:
    '''
    A class for KDE
    '''

    def __init__(self, h, kernel_func: Callable, samples: np.ndarray, use_pytorch: bool = False, torch_device="cpu", dtype = torch.float64) -> None:
        self.use_pytorch = use_pytorch
        try:
            import torch
        except:
            warnings.warn("Can not use pytorch")
            self.use_pytorch = False
        self.torch_device = torch_device
        if use_pytorch:
            self.samples = torch.as_tensor(samples, device=torch_device, dtype=dtype)
        else:
            self.samples = samples
        self.h = h
        self.K = kernel_func
        self.n = samples.shape[0]
        self.dtype = dtype

    def select_bandwidth(self):
        # Not yet finished, please pass bandwidth to init function.
        h = self.h
        self.h = h

    def __torchpdf(self, x):

        x_tensor = torch.asarray(x, device=self.torch_device, requires_grad=True, dtype=self.dtype, copy=True)
        components = self.K((x_tensor.unsqueeze(dim=-2)-self.samples)/self.h)
        summary = torch.sum(components, dim=-1)  # summary (n_x)
        result = 1.0/self.n/self.h * summary
        return result

    def pdf(self, x: np.ndarray):
        '''
        Given x, calculate the estimated probability density function: 1/(n*h) \sum_{i=1}^n K((x-x_i)/h)
        NOTE: x should be an array representing ONE point. x can not be a set of points.
        '''
        if self.use_pytorch:
            return self.__torchpdf(x)
        x = np.asarray(x).T

        x = np.expand_dims(x, axis=-2)
        components = self.K((x-self.samples)/self.h)

        # Calculate \sum_{i=1}^n K((x-x_i)/h)
        summary = np.sum(components, axis=-1)

        result = 1.0/self.n/self.h * summary

        return result
