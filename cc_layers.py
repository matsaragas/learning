"""
Layers using cuda-convnet Theano wrappers that are part of pylearn2.
"""


import theano
import theano.tensor as T
import numpy as np

from theano.sandbox.cuda.basic_ops import gpu_contiguous
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
