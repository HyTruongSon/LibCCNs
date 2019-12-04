import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

ccn1d_lib = tf.load_op_library("../ccn_lib/ccn1d_lib.so")

@ops.RegisterGradient("ForwardNormalizing")
def forward_shrinking_grad(op, *grad):
	start_index = op.inputs[0]
	output_grad = grad[0]

	input_grad = ccn1d_lib.backward_normalizing(start_index, output_grad)
	return (None, input_grad)
