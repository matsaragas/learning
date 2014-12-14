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
        
        self.W.set_value(np.random.randn(*self.filter_shape).astype(np.float32) * self.weights_std)
        
        if self.untie_biases:
           self.b.set_value(np.ones(self.get_output_shape()[:3]).astype(np.float32) * self.init_bias_value)
        else:
           self.b.set_value(np.ones(self.n_filters).astype(np.float32) * self.init_bias_value)
           
    def get_output_shape(self):
        output_width = (self.input_shape[1] + 2 * self.pad - self.filter_size + self.stride) // self.stride
        output_height = (self.input_shape[2] + 2 * self.pad - self.filter_size + self.stride) // self.stride
        output_shape = (self.n_filters, output_width, output_height, self.mb_size)
        return output_shape
        
    def output(self, input=None, dropout_active = True, *args, **kwargs):
        if input = None:
            input = self.input_layer.output(dropout_active = dropout_active, *args, **kwargs):
                
        if dropout_active and (self.dropour > 0.):
            retain_prob = 1 - self.dropout
            mask = layers.srng.binomial(input.shape, p = retain_prob, dtype = 'int32').astype('float32')
            
            input = input / retain_prob * mask
            
        contiguous_input = gpu_contiguous(input)
        contiguous_filter = gpu_contiguous(self.W)
        conved = self.filter_acts_op(contiguous_input, contiguous_filters)
        
        if self.untie_biases:
            conved += self.b.dimshuffle(0,1,2,'x')
        else:
            conved += self.b.dimshuffle(0,'x','x','x')
            
        return self.nonlinearity(conved)
        

class CudaConvnetPooling2DLayer(objects):
    def __init__(self, input_layer, pool_size, stride = None):
        """
        pool size is an Integer, not a tuple. We can only do square pooling windows.
        if the stride is none, it is taken to be the same as the pool size.
        borders are never ignored.
        """
        
        self.pool_size = pool_size
        self.stride = stride if stride is None else pool_size
        self.input_layer = input_layer
        self.params = []
        self.bias_params = []
        self.mb_bias = self.input_layer.mb_size
        
        self.pool_op = MaxPool(ds = self.pool_size, stride = self.stride)
        
  
