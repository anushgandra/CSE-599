from collections.abc import Iterable
from typing import Tuple

import numpy as np

from .layer import Layer


class AddLayer(Layer):
    def __init__(self, parents):
        super(AddLayer, self).__init__(parents)
        self.data = None

    def forward(self, inputs: Iterable):
        # TODO: Add all the items in inputs. Hint, python's sum() function may be of use.
        output = np.sum(inputs,axis=0)
        self.data = inputs
        return output

    def backward(self, previous_partial_gradient) -> Tuple[np.ndarray, ...]:
        # TODO: You should return as many gradients as there were inputs.
        #   So for adding two tensors, you should return two gradient tensors corresponding to the
        #   order they were in the input.
        s = self.data.shape
        output = np.zeros(s,dtype=np.float32)
        for i in range(s.shape[0]):
            output[i] = previous_partial_gradient
        
        return output
