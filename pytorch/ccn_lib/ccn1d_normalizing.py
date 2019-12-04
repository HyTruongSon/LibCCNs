import math
import torch

import ccn1d_lib

class ccn1d_normalizing(torch.autograd.Function):
	@staticmethod
	def forward(ctx, start_index, input_tensor):
		# Output tensor creation
		output = torch.zeros(input_tensor.size(0), input_tensor.size(1))

		# Compute the output tensor (in C++)
		ccn1d_lib.forward_normalizing(start_index, input_tensor, output)

		# Save the tensors for backward pass
		saved_variables = [start_index, input_tensor]
		ctx.save_for_backward(*saved_variables)

		# Return the output tensor
		return output

	@staticmethod
	def backward(ctx, output_grad):
		# Saved tensors from the forward pass
		start_index, input_tensor = ctx.saved_variables

		# Input gradient tensor creation
		input_grad = torch.zeros(input_tensor.size(0), input_tensor.size(1))

		# Compute the input gradient tensor (in C++)
		ccn1d_lib.backward_normalizing(start_index, input_grad, output_grad)

		# Return the input gradient tensor
		return (None, input_grad)