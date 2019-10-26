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
        self.data = None
        self.padded_data = None

    @staticmethod
    @njit(parallel=True, cache=True)
    def forward_numba(data, weights, bias, stride, padding, kernel_size,input_shape,weight_shape,out):
        # TODO
        # data is N x C x H x W
        # kernel is COld x CNew x K x K
        
        s = input_shape
        n = s[0]
        d = s[1]
        h = s[2]
        w = s[3]

        ws = weight_shape
        c = ws[1]

        jlim = (h-kernel_size) + 1 + padding
        ilim = (w-kernel_size) + 1 + padding
             
        for l in prange(n):
            for o in prange(c):
                for k in range(d):
                    for j in range(padding,jlim,stride):
                        for i in range(padding,ilim,stride):
                            for n in range(kernel_size):
                                for m in range(kernel_size):
                                    out[l,o,(j-padding)//stride,(i-padding)//stride] += (data[l,k,j-(padding)+n,i-(padding)+m]*weights[k,o,n,m])
                                                                        
                            out[l,o,(j-padding)//stride,(i-padding)//stride] += bias[o]

        return out

    def forward(self, data):
        padded = np.pad(data,((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)),'constant')
        output_array = np.zeros((data.shape[0],self.weight.data.shape[1],(padded.shape[2]-self.kernel_size)//self.stride + 1,(padded.shape[3]-self.kernel_size)//self.stride + 1),dtype=np.float32)
        output = self.forward_numba(padded, self.weight.data, self.bias.data, self.stride, self.padding, self.kernel_size,padded.shape,self.weight.data.shape,output_array)
        self.data = data
        self.padded_data = padded
        return output

    @staticmethod
    @njit(cache=True, parallel=True)
    def backward_numba(padded_data, padded_data_transpose, output, prev_grad, padded_prev_grad, weights, weights_transpose, weights_grad, padding, stride, kernel_size, padded_data_shape, prev_shape, padding2):
        # TODO
        # data is N x C x H x W
        # kernel is COld x CNew x K x K
        # First I'll compute weights_grad
        
        s = padded_data_shape
        n = s[0]
        d = s[1]
        h = s[2]
        w = s[3]

        c = prev_shape[1]

        jlim = (h-prev_shape[2]) + 1 + padding
        ilim = (w-prev_shape[3]) + 1 + padding
             
        for l in prange(d):
            for o in prange(c):
                for k in range(n):
                    for j in range(padding,jlim,stride):
                        for i in range(padding,ilim,stride):
                            for n in range(prev_shape[2]):
                                for m in range(prev_shape[3]):
                                    weights_grad[l,o,(j-padding)//stride,(i-padding)//stride] += (padded_data_transpose[l,k,j-(padding)+n,i-(padding)+m])*prev_grad[k,o,n,m]

        jlim = ((prev_shape[2]+(2*padding2)) - kernel_size) + 1 + padding2
        ilim = ((prev_shape[3]+(2*padding2)) - kernel_size) + 1 + padding2
        
        for l in prange(n):
            for o in prange(d):
                for k in range(c):
                    for j in range(padding2,jlim,stride):
                        for i in range(padding2,ilim,stride):
                            for n in range(kernel_size):
                                for m in range(kernel_size):
                                    output[l,o,(j-padding2)//stride,(i-padding2)//stride] += (padded_prev_grad[l,k,j-(padding2)+n,i-(padding2)+m]*prev_grad[k,o,n,m])
                                      
        return output

        return None

    def backward(self, previous_partial_gradient):
        padding2 = self.padding
        padded_grad = np.pad(previous_partial_gradient,((0,0),(0,0),(padding2,padding2),(padding2,padding2)),'constant')
        output_array = np.zeros(np.shape(self.data), dtype = np.float32)
        weight_grad = np.zeros((np.shape(self.weight.data)),dtype = np.float32)
        back = backward_numba(self.padded_data, np.transpose(self.padded_data,(1,0,2,3)), output_array, previous_partial_gradient, padded_grad, self.weight.data,np.transpose(self.weight.data,(1,0,3,2)), self.weight.grad, self.padding, self.stride, self.kernel_size, self.padded_data.shape, previous_partial_gradient.shape, padding2)
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
