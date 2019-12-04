import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

ccn1d_lib = tf.load_op_library("../ccn_lib/ccn1d_lib.so")

'''
@ops.RegisterGradient("DynamicPrecomputeNeighbors")
def dynamic_precompute_neighbors_grad(op, *grad):
	return (None, None)
'''

@ops.RegisterGradient("ForwardContractions")
def forward_contractions_grad(op, *grad):
	input_receptive_field = op.inputs[0]
	input_start_index = op.inputs[1]
	output_receptive_field = op.inputs[2]
	output_start_index = op.inputs[3]
	output_grad = grad[0]

	input_grad = ccn1d_lib.backward_contractions(input_receptive_field, input_start_index, output_receptive_field, output_start_index, output_grad)
	return (None, None, None, None, input_grad)

@ops.RegisterGradient("ForwardContractionsMultiThreading")
def forward_contractions_multi_threading_grad(op, *grad):
	nThreads = op.get_attr("nThreads")
	input_receptive_field = op.inputs[0]
	input_start_index = op.inputs[1]
	output_receptive_field = op.inputs[2]
	output_start_index = op.inputs[3]
	output_grad = grad[0]

	input_grad = ccn1d_lib.backward_contractions_multi_threading(input_receptive_field, input_start_index, output_receptive_field, output_start_index, output_grad, nThreads = nThreads)
	return (None, None, None, None, input_grad)

@ops.RegisterGradient("ForwardNormalizing")
def forward_shrinking_grad(op, *grad):
	start_index = op.inputs[0]
	output_grad = grad[0]

	input_grad = ccn1d_lib.backward_normalizing(start_index, output_grad)
	return (None, input_grad)

@ops.RegisterGradient("ForwardShrinking")
def forward_shrinking_grad(op, *grad):
	start_index = op.inputs[0]
	output_grad = grad[0]

	input_grad = ccn1d_lib.backward_shrinking(start_index, output_grad)
	return (None, input_grad)

@ops.RegisterGradient("ForwardShrinkingMultiThreading")
def forward_shrinking_multi_threading_grad(op, *grad):
	nThreads = op.get_attr("nThreads")
	start_index = op.inputs[0]
	output_grad = grad[0]

	input_grad = ccn1d_lib.backward_shrinking_multi_threading(start_index, output_grad, nThreads = nThreads)
	return (None, input_grad)