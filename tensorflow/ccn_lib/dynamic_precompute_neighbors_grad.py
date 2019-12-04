import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

ccn1d_lib = tf.load_op_library("../ccn_lib/ccn1d_lib.so")

@ops.RegisterGradient("DynamicPrecomputeNeighbors")
def dynamic_precompute_neighbors_grad(op, *grad):
	return (None, None)