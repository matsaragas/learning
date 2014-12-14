"""
Layers using cuda-convnet Theano wrappers that are part of pylearn2.
"""


import theano
import theano.tensor as T
import numpy as np

from theano.sandbox.cuda.basic_ops import gpu_contiguous
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from pylearn2.sandbox.cuda_convnet.pool import MaxPool
from pylearn2.sandbox.cuda_convnet.stochastic_pool import StochasticMaxPool, WeightedMaxPool
from pylearn2.sandbox.cuda_convnet.response_norm import CrossMapNorm
from theano.sandbox.cuda import host_from_gpu


class CudaConvnetInput2DLayer(layers.Input2DLayer):
    """
    Like Input2DLayer, but the data is expected to be in c01b order instead of bc01.
    """
    
    def get_output_shape(self):
        return (self.n_features. self.width, self.height, self.mb_size) # c01b instead of bc01
        
class CudaConvnetConv2DLayer(object):
  
    def __init__(self, input_layer, n_filters, filter_size, weights_std, init_bias_value, stride = 1, nonlinearity = layers.rectify, dropout = 0., partial_sum = None, pad = 0, untie_biases = False):
      
  
        """
        Only the valid border mode is supported.
     
        n_filters should be multiple of 16
        """
     
        self.input_layer = input_layer 
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.weights_std = weights_std
        self.init_bias_value = np.float(init_bias_value)
        self.stride = stride
        self.nonlinearity = nonlinearity
        self.dropout = dropout
        self.partial_sum = partial_sum
        self.pad = pad
        self.untie_biases = untie_biases
        self.mb_size = self.input_layer.mb_size
     
        self.input_shape = self.input_layer.get_output_shape()
     
        self.filter_shape = (self.input_shape[0], filter_size, filter_size, n_filters)
     
        self.W = layers.shared_single(4)
     
        if self.untie_biases:
           self.b = layers.shared_single(3)
        else:
           self.b = layers.shared_single(1)
        
        self.params = [self.W, self.b]
        self.bias_params = [self.b]
        self.reset_params()
     
        self.filter_acts_op = FilterActs(stride = self.stride, partial_sum = self.partial_sum, pad = self.pad)
     
     
     
    def reset_params(self): 
  
