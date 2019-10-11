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
        max_elements = np.amax(logits,axis=1)
        logits_subt = np.transpose(np.transpose(logits)-max_elements)

        exp_logits = np.exp(logits_subt) 
        exp_logits_sum = np.sum(exp_logits,axis=1)

        softmax = exp_logits/exp_logits_sum[:,None]
        
        self.input_softmax = softmax
        self.targets = targets

        log_softmax = np.log(softmax)
        
        losses = -1*np.dot(log_softmax,targets)

        if(self.reduction == "mean"):
            loss = np.average(losses)
        else:
            loss = np.sum(losses)

        return loss


        

    def backward(self) -> np.ndarray:
        """
        Takes no inputs (should reuse computation from the forward pass)
        :return: gradients wrt the logits the same shape as the input logits
        """
        y = self.input_softmax - targets 
        return y
