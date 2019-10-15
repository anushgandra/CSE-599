import numpy as np
from numba import njit, prange

from nn import Parameter
from .layer import Layer


class PReLULayer(Layer):
    def __init__(self, size: int, initial_slope: float = 0.1, parent=None):
        super(PReLULayer, self).__init__(parent)
        self.slope = Parameter(np.full(size, initial_slope))
        self.data = None

    def forward(self, data):
        out1 = np.copy(data)
        out2 = np.copy(data)

        out1 = np.moveaxis(out1,1,2)

        out1[out1>0] = 0
        out1 = out1*self.slope.data

        out1 = np.moveaxis(out1,2,1)

        out2[out2<=0] = 0
        out = out1+out2

        self.data = data
        return out
    

    def backward(self, previous_partial_gradient):
        out3 = np.copy(self.data)
        out3[out3>0] = 1
        out3[out3<=0] = 0

        out4 = np.copy(self.data)
        temp1 = np.argwhere(out4>0)
        out4[out4<=0] = 1
        out4 = np.moveaxis(out4,1,2)
        out4 = out4*self.slope.data
        out4 = np.moveaxis(out4,2,1)
        out4[temp1[:,0],temp1[:,1]] = 0

        out = out3+out4

        grad_mult = np.copy(self.data)
        grad_mult[grad_mult>0] = 0

        grad = np.multiply(previous_partial_gradient,grad_mult)

        self.slope.grad = np.sum(np.sum(grad,axis=0),axis=1)
        return out
