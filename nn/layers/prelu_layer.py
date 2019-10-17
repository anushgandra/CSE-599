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
        print(self.slope)
        return out
    

    def backward(self, previous_partial_gradient):
        out1 = self.data>0
        out2 = (self.data<=0)*np.reshape(self.slope.data,(1,self.size,1))

        output = np.multiply((out1+out2),previous_partial_gradient)

        grad_mult = np.copy(self.data)
        grad_mult[grad_mult>0] = 0

        grad = np.multiply(previous_partial_gradient,grad_mult)

        if (self.size<2):
            self.slope.grad = np.sum(grad)
        else:
            self.slope.grad = np.sum(np.sum(grad,axis=0),axis=1)
            
        return output
