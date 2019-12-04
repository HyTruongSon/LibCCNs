import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

ccn1d_lib = tf.load_op_library("../ccn_lib/ccn1d_lib.so")

@ops.RegisterGradient("ForwardContractions")
def forward_contractions_grad(op, *grad):
	input_receptive_field = op.inputs[0]
	input_start_index = op.inputs[1]
	output_receptive_field = op.inputs[2]
	output_start_index = op.inputs[3]
	output_grad = grad[0]

	input_grad = ccn1d_lib.backward_contractions(input_receptive_field, input_start_index, output_receptive_field, output_start_index, output_grad)
	return (None, None, None, None, input_grad)