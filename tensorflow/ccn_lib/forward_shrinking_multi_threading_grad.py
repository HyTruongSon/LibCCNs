import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

ccn1d_lib = tf.load_op_library("../ccn_lib/ccn1d_lib.so")

@ops.RegisterGradient("ForwardShrinkingMultiThreading")
def forward_shrinking_multi_threading_grad(op, *grad):
	nThreads = op.get_attr("nThreads")
	start_index = op.inputs[0]
	output_grad = grad[0]

	input_grad = ccn1d_lib.backward_shrinking_multi_threading(start_index, output_grad, nThreads = nThreads)
	return (None, input_grad)