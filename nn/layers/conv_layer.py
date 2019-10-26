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
    def forward_numba(data, weights, bias, stride, kernel_size,out):
        # TODO
        # data is N x C x H x W
        # kernel is COld x CNew x K x K
        
        s = data.shape
        n = s[0]
        h = s[2]
        w = s[3]

        ws = weights.shape
        c = ws[1]

        jlim = (h-kernel_size) + 1 #+ padding
        ilim = (w-kernel_size) + 1 #+ padding

##        print("prev_shape: ",prev_shape)
##        print("padded data shape: ",s)
##        print("jlim: ",jlim)
##        print("ilim: ",ilim)
##        print("Weight shape: ",weights.shape)
##        print("padding: ",padding)
             
        for l in prange(n):
            for o in prange(c):
                for j in range(0,jlim,stride):
                    for i in range(0,ilim,stride):
                        out[l,o,j//stride,i//stride] = np.sum((data[l,:,j:j+(kernel_size),i:i+(kernel_size)]*weights[:,o,:,:]))+bias[o]
                                                                        
        return out

    def forward(self, data):
        padded = np.pad(data,((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)),'constant')
        output_array = np.zeros((data.shape[0],self.weight.data.shape[1],(padded.shape[2]-self.kernel_size)//self.stride + 1,(padded.shape[3]-self.kernel_size)//self.stride + 1),dtype=np.float32)

        output = self.forward_numba(padded, self.weight.data, self.bias.data, self.stride, self.kernel_size,output_array)
        
        self.data = data
        self.padded_data = padded
        return output

    @staticmethod
    @njit(cache=True, parallel=True)
    def backward_numba(padded_data, prev_grad, weight, weight_grad, bias_grad,output, padded_output, padding, stride):
        # TODO
        # data is N x C x H x W
        # kernel is COld x CNew x K x K
        [n,d,hin,win] =  padded_data.shape
        [n,c,hout,wout] = prev_grad.shape
        
        k = weight.shape[2]
        # First I'll compute weights_grad
        
##        jlim = (hin-hout) + 1
##        ilim = (win-wout) + 1

##        for l in range(d):
##            for o in range(c):
##                for j in range(0,jlim,stride):
##                    for i in range(0,ilim,stride):
##                        weight_grad[l,o,j//stride,i//stride] = np.sum(padded_data_transpose[l,:,j:j+k,i:i+k]*prev_grad[:,c,:,:])
                        
        # Now for gradient with respect to inputs and bias grad

##        jlim = hout
##        ilim = wout

        for l in range(n):
            padded_data_slice = padded_data[l]
            padded_output_slice = padded_output[l]
            for v in range(c):
                for j in range(hout):
                    for i in range(wout):
                        h1 = j*stride
                        h2 = j*stride+k
                        w1 = i*stride
                        w2 = i*stride+k
                        padded_data_slice_smaller = padded_data_slice[:,h1:h2,w1:w2]
                        padded_output_slice[:,h1:h2,w1:w1] += weight[:,v,:,:]*prev_grad[l,v,j,i]
                        #output[l,:,h1:h2,w1:w2] += (prev_grad[l,o,j,i]*weight[:,o,:,:])
                        bias_grad[v] += prev_grad[l,v,j,i]
                        #weight_grad[:,o,:,:] += padded_data_slice_smaller*prev_grad[l,o,j,i]
            if(padding == 0):
                output[l,:,:,:] = padded_output_slice[:,:,:]
            else:
                output[l,:,:,:] = padded_output_slice[:,padding:-padding,padding:-padding]
        return(output)
                 
    def backward(self, previous_partial_gradient):
        padded_output_array = np.zeros(self.padded_data.shape,dtype=np.float32)

        output_array = np.zeros(self.data.shape,dtype=np.float32)

           
        back = self.backward_numba(self.padded_data, previous_partial_gradient, self.weight.data, self.weight.grad, self.bias.grad,output_array,padded_output_array, self.padding, self.stride)

        
        return back

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
