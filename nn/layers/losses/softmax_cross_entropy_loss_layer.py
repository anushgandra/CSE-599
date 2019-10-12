import numpy as np

from .loss_layer import LossLayer


class SoftmaxCrossEntropyLossLayer(LossLayer):
    def __init__(self, reduction="mean", parent=None):
        """

        :param reduction: mean reduction indicates the results should be summed and scaled by the size of the input (excluding the axis dimension).
            sum reduction means the results should be summed.
        """
        self.reduction = reduction
        self.input_softmax = None
        self.targets = None
        super(SoftmaxCrossEntropyLossLayer, self).__init__(parent)

    def forward(self, logits, targets, axis=-1) -> float:
        """

        :param logits: ND non-softmaxed outputs. All dimensions (after removing the "axis" dimension) should have the same length as targets
        :param targets: (N-1)D class id integers.
        :param axis: Dimension over which to run the Softmax and compare labels.
        :return: single float of the loss.
        """
        #assuming logits is 2D:
        logits = np.transpose(np.transpose(logits) - np.amax(logits,axis=axis))
        logits = np.exp(logits)
        temp = np.sum(logits,axis=axis,keepdims=True)
        logits = logits/temp
        self.input_softmax = logits
        self.targtes = targets
        log_logits = np.log(logits)
        H = np.zeros(targets.shape)
        for i in range(0,targets.size):
            H[i] =  -1*log_logits[i,targets[i]]

        if(self.reduction == "sum"):
            H = np.sum(H[i])
        else:
            H = np.sum(H[i])/np.size(logits,axis)
        return H

        

        


        

    def backward(self) -> np.ndarray:
        """
        Takes no inputs (should reuse computation from the forward pass)
        :return: gradients wrt the logits the same shape as the input logits
        """
        y = self.input_softmax - targets 
        return y
