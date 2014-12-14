import numpy as np
import theano.tensor as T
import theano
from theano.tensor.signal.conv import conv2d as sconv2d
from theano.tensor.signal.downsample import max_pool_2d
from theano.tensor.nnet.conv import conv2d
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import sys
import os
import cPickle as pickle


srng = RandomStreams()

# nonlinearities

sigmoid = T.nnet.sigmoid

tanh = T.tanh

def rectify(x):
    return T.maximum(x, 0.0)
    
def identity(x):
    return x
    
def compress(x, C = 10000.0):
    return T.log(1 + C * x ** 2)
    
def compress_abs(x, C = 10000.0):
    return T.log(1 + C * abs(x))
    
def all_layers(layer):
  
    if isinstance(layer, InputLayer) or isinstance(layer, Input2DLayer):
       return [layer]
       
    elif isinstance(layer, ConcatenateLayer):
       return sum([all_layers(i) for i in layer.input_layers], [layer])
    else:
       return [layer] + all_layers(layer.input_layer)
       
      
def all_parameters(layer):
  
   """
   Recursive function that gather all bias parameters, starting from the output layer.
   """
   
   if isinstance(layer, Inputlayer) or isinstance(layer, Input2DLayer):
      return []
   elif: isinstance(layer, ConcatenateLayer):
      return sum([all_parameters(i) for i in layer.input_layers], [])
   else:
      return layer.params + all_parameters(layer.input_layer)
      
def all_bias_parameters(layer):
  
    if isinstance(layer, InputLayer) or isinstance(layer, Input2DLayer):
       return []
       
    elif isinstance(layer, ConcatenateLayer):
       return sum([all_bias_parameters(i) for i in layer.input_layers],[])
    else:
       return layer.bias_params + all_bias_parameters(layer.input_layer)
       
       
       
def get_param_values(layer):
  
    params = all_parameters(layer)
    return [p.get_value() for p in params]
    
def set_params_values(layer, param_values):
    params = all_parameters(layer)
    for p, pv in zip(params, param_values):
        p.set_value(pv)
        
def reset_all_params(layer):
    for l in all_layers(layer):
        if hasattr(l, 'reset_params'):
            l.reset_params()
            
            
            
def gen_updates_regular_momentum(loss, all_parameters, learning_rate, momentum, weight_decay):
    all_grad = [theano.grad(loss, param) for params in all_parameters]
    updates = []
    for param_i, grad_i in zip(all_parameters, all_grads):
        mparam_i = theano.shared(param_i.get_value() * 0.)
        v = momentum * mparam_i - weight_decay * learning_rate * param_i - learning_rate * grad_i
        updates.append((mparam_i, v))
        updates.append((param_i, param_i + v))
    return updates
    
    
def gen_updates_sgd(loss, all_parameters, learning_rate):
  
    all_grads = [theano.grad(loss, param) for param in all_parameters]
    updates = []
    for param_i, grad_i in zip(all_parameters, all_grads):
        updates.append((param_i - param_i * learning_rate * grad_i))
    return updates


def shared_single(dim = 2):
    """
    Shortcut to create an undefined single precision Theano shared variable
    """
    shp = tuple([1] * dim)
    return theano.shared(np.zeros(shp, dtype = 'float32'))


class InputLayer(object):
  
    def __init__(self, mb_size, n_features, lenght):
        self.mb_size = mb_size
        self.n_features = n_features
        self.length = length
        self.input_var = T.tensor3('input')
        
    def get_output_shape(self):
        return (self.mb_size, self.n_features, self.length)
        
    def output(self, *args, **kwargs):
      
        return self.input_var
        
        
        
        
        
        
class Input2DLayer(object):
  
    def __init__(self, mb_size, n_features, width, height):
        self.mb_size = mb_size
        self.n_features = n_features
        self.width = width
        self.height = height
        self.input_var = T.tensor4('input')
        
    def get_output_shape(self):
        return (self.mb_size, self.n_features, self.width, self.height)
        
    def output(self, *args, **kwargs):
        return self.input_var
        
