from typing import Optional, Callable
import numpy as np

from numba import njit, prange

from nn import Parameter
from .layer import Layer


class ConvLayer(Layer):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, parent=None):
        super(ConvLayer, self).__init__(parent)
        self.weight = Parameter(np.zeros((input_channels, output_channels, kernel_size, kernel_size), dtype=np.float32))
        self.bias = Parameter(np.zeros(output_channels, dtype=np.float32))
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.stride = stride
        self.initialize()

    @staticmethod
    @njit(parallel=True, cache=True)
    def forward_numba(data, weights, bias):
        # TODO
        # data is N x C x H x W
        # kernel is COld x CNew x K x K
        s = data.shape
        n = s[0]
        d = s[1]
        h = s[2]
        w = s[3]

        o = weights.shape
        outsize = o[1]
        out = np.zeros((n,outsize,h,w),dtype = np.float32)
     
        for l in range(n):
            for o in range(outsize):
                for k in range(d):
                    for j in range(self.padding,self.stride,h-self.padding):
                        for i in range(self.padding,self.stride,w-self.padding):
                            for n in range(self.kernel_size):
                                for m in range(self.kernel_size):
                                    out[l,o,j,i] = out[l,o,j,i] + (data[l,k,j-(self.padding)+n,i-(self.padding)+m]*weights[k,o,n,m])+bias[o]
                                         
        return out

    def forward(self, data):
        padded = np.pad(data,((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)),'constant')
        output = self.forward_numba(padded, self.weight, self.bias)
        return output

    @staticmethod
    @njit(cache=True, parallel=True)
    def backward_numba(previous_grad, data, kernel, kernel_grad):
        # TODO
        # data is N x C x H x W
        # kernel is COld x CNew x K x K
        return None

    def backward(self, previous_partial_gradient):
        # TODO
        return None

    def selfstr(self):
        return "Kernel: (%s, %s) In Channels %s Out Channels %s Stride %s" % (
            self.weight.data.shape[2],
            self.weight.data.shape[3],
            self.weight.data.shape[0],
            self.weight.data.shape[1],
            self.stride,
        )

    def initialize(self, initializer: Optional[Callable[[Parameter], None]] = None):
        if initializer is None:
            self.weight.data = np.random.normal(0, 0.1, self.weight.data.shape)
            self.bias.data = 0
        else:
            for param in self.own_parameters():
                initializer(param)
        super(ConvLayer, self).initialize()
