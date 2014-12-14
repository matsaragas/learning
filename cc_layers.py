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
  
