import os
import mxnet as mx
import numpy as np


class RangeLossOp(mx.operator.CustomOp):
    def __init__(self, margin, k, alpha, beta):
        self.margin = margin
        self.k = k
        self.alpha = alpha
        self.beta = beta
