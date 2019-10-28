import numbers

import numpy as np
from numba import njit, prange

from .layer import Layer


class MaxPoolLayer(Layer):
    def __init__(self, kernel_size: int = 2, stride: int = 2, parent=None):
        super(MaxPoolLayer, self).__init__(parent)
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.stride = stride
        self.data = None
        self.padded_data = None

    @staticmethod
    @njit(parallel=True, cache=True)
    def forward_numba(data):
        s = data.shape
        n = s[0]
        h = s[2]
        w = s[3]
        d = s[1]

        jlim = (h-kernel_size) + 1 
        ilim = (w-kernel_size) + 1 

                  
        for l in range(n):
            for o in range(d):
                for j in range(0,jlim,stride):
                    for i in range(0,ilim,stride):
                        
                        out[l,o,j//stride,i//stride] = np.max(data[l,:,j:j+(kernel_size),i:i+(kernel_size)])
                                                                        
        return out

    def forward(self, data):
        padded = np.pad(data,((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)),'constant')
        output_array = np.zeros((data.shape[0],self.weight.data.shape[1],(padded.shape[2]-self.kernel_size)//self.stride + 1,(padded.shape[3]-self.kernel_size)//self.stride + 1),dtype=np.float32)

        output = self.forward_numba(padded, self.stride, self.kernel_size,output_array)
        
        self.data = data
        self.padded_data = padded
        return output
        
    @staticmethod
    @njit(cache=True, parallel=True)
    def backward_numba(previous_grad, data):
        # data is N x C x H x W
        # TODO
        return None

    def backward(self, previous_partial_gradient):
        # TODO
        return None

    def selfstr(self):
        return str("kernel: " + str(self.kernel_size) + " stride: " + str(self.stride))
