import math
import torch

import ccn1d_lib

class ccn1d_contractions(torch.autograd.Function):
	@staticmethod
	def forward(ctx, input_receptive_field, input_start_index, output_receptive_field, output_start_index, input_tensor, nThreads):
		# Sizes
		nContractions = ccn1d_lib.get_nContractions()
		input_nRows = input_tensor.size(0)
		input_nChannels = input_tensor.size(1)
		output_nRows = output_receptive_field.size(0)
		output_nChannels = input_nChannels * nContractions

		# Output tensor creation
		output = torch.zeros(output_nRows, output_nChannels)

		# Compute the output tensor (in C++)
		if nThreads > 1:
			ccn1d_lib.forward_contractions_multi_threading(input_receptive_field, input_start_index, output_receptive_field, output_start_index, input_tensor, output, nThreads)
		else:
			ccn1d_lib.forward_contractions(input_receptive_field, input_start_index, output_receptive_field, output_start_index, input_tensor, output)

		# Save the tensors for backward pass
		saved_variables = [input_receptive_field, input_start_index, output_receptive_field, output_start_index, input_tensor]
		ctx.save_for_backward(*saved_variables)
		ctx.nThreads = nThreads

		# Return the output tensor
		return output

	@staticmethod
	def backward(ctx, output_grad):
		# Saved tensors from the forward pass
		input_receptive_field, input_start_index, output_receptive_field, output_start_index, input_tensor = ctx.saved_variables

		# Sizes
		input_nRows = input_tensor.size(0)
		input_nChannels = input_tensor.size(1)

		# Input gradient tensor creation
		input_grad = torch.zeros(input_nRows, input_nChannels)

		# Compute the input gradient tensor (in C++)
		if ctx.nThreads > 1:
			ccn1d_lib.backward_contractions_multi_threading(input_receptive_field, input_start_index, output_receptive_field, output_start_index, input_grad, output_grad, ctx.nThreads)
		else:
			ccn1d_lib.backward_contractions(input_receptive_field, input_start_index, output_receptive_field, output_start_index, input_grad, output_grad)

		# Return the input gradient tensor
		return (None, None, None, None, input_grad, None)