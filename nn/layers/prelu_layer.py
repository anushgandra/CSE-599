import numpy as np
from numba import njit, prange

from nn import Parameter
from .layer import Layer


class PReLULayer(Layer):
    def __init__(self, size: int, initial_slope: float = 0.1, parent=None):
        super(PReLULayer, self).__init__(parent)
        self.slope = Parameter(np.full(size, initial_slope))
        self.data = None
        self.size = size

    def forward(self, data):
        
        out1 = np.minimum(0,data)
        out1 = out1*(np.reshape(self.slope.data,(1,self.size,1)))
                     
        out2 = np.maximum(0,data)
        out = out1+out2

        self.data = data
        return out
    

    def backward(self, previous_partial_gradient):
        out = np.copy(self.data)
        temp1 = np.argwhere(out>0)
        out[out<=0] = 1
        
        out = out*np.reshape(self.slope.data,(1,self.size,1))
        
        out[temp1[:,0],temp1[:,1]] = 1

        output = np.multiply(out,previous_partial_gradient)

        grad_mult = np.copy(self.data)
        grad_mult[grad_mult>0] = 0

        grad = np.multiply(previous_partial_gradient,grad_mult)

        self.slope.grad = np.sum(np.sum(grad,axis=0),axis=1)
        return output
