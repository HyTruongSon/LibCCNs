import math
import torch

import RisiContraction_18

NCONTRACTIONS = 18
NTHREADS = 1024

class RisiContraction_18_Op(torch.autograd.Function):
	@staticmethod
	def forward(ctx, tensor, adj):
		# Sizes
		N = tensor.size(0)
		nChannels = tensor.size(3)

		# Output tensor creation
		value = torch.zeros(N, N, nChannels * NCONTRACTIONS).to(device = 'cuda')

		# Call the CUDA extension
		RisiContraction_18.RisiContraction_18_forward(tensor, adj, value, nThreads = NTHREADS)

		# Save the tensors for backward pass
		saved_variables = [adj]
		ctx.save_for_backward(*saved_variables)

		# Return the output tensor
		return value

	@staticmethod
	def backward(ctx, value_gradient):
		# Saved tensors from the forward pass
		adj = ctx.saved_variables

		# Sizes
		N = adj.size(0)
		nChannels = value_gradient.size(2) // NCONTRACTIONS

		# Input gradient tensor creation
		tensor_gradient = torch.zeros(N, N, N, nChannels).to(device = 'cuda')

		# Compute the input gradient tensor
		RisiContraction_18.RisiContraction_18_backward(tensor_gradient, adj, value_gradient, nThreads = NTHREADS)

		# Return the input gradient tensor
		return (tensor_gradient, None)