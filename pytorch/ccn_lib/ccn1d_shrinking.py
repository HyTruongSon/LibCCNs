import math
import torch

import ccn1d_lib

class ccn1d_shrinking(torch.autograd.Function):
	@staticmethod
	def forward(ctx, nVertices, start_index, input_tensor, nThreads):
		# Sizes
		nChannels = input_tensor.size(1)

		# Output tensor creation
		output = torch.zeros(nVertices, nChannels)

		# Compute the output tensor (in C++)
		if nThreads > 1:
			ccn1d_lib.forward_shrinking_multi_threading(start_index, input_tensor, output, nThreads)
		else:
			ccn1d_lib.forward_shrinking(start_index, input_tensor, output)

		# Save the tensors for backward pass
		saved_variables = [start_index, input_tensor]
		ctx.save_for_backward(*saved_variables)
		ctx.nThreads = nThreads

		# Return the output tensor
		return output

	@staticmethod
	def backward(ctx, output_grad):
		# Saved tensors from the forward pass
		start_index, input_tensor = ctx.saved_variables

		# Sizes
		input_nRows = input_tensor.size(0)
		nChannels = input_tensor.size(1)

		# Input gradient tensor creation
		input_grad = torch.zeros(input_nRows, nChannels)

		# Compute the input gradient tensor (in C++)
		if ctx.nThreads > 1:
			ccn1d_lib.backward_shrinking_multi_threading(start_index, input_grad, output_grad, ctx.nThreads)
		else:
			ccn1d_lib.backward_shrinking(start_index, input_grad, output_grad)

		# Return the input gradient tensor
		return (None, None, input_grad, None)