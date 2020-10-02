import math
import torch

import RisiContraction_18

import sys
sys.path.append('pytorch_gpu/')

from RisiContraction_18_Op import RisiContraction_18_Op

NCONTRACTIONS = 18
NTHREADS = 1024

N = 10
nChannels = 20

tensor = torch.randn(N, N, N, nChannels).type(torch.DoubleTensor).to(device = 'cuda')
adj = torch.randn(N, N).type(torch.DoubleTensor).to(device = 'cuda')

# Forward -- Pythonic
value = RisiContraction_18_Op.apply(tensor, adj)
print("Done")

# Forward (CUDA extension)
value = torch.zeros(N, N, nChannels * NCONTRACTIONS).type(torch.DoubleTensor).to(device = 'cuda')
RisiContraction_18.RisiContraction_18_forward(tensor, adj, value, NTHREADS)
print("Done")

# Backward (CUDA extension)
tensor_gradient = torch.zeros(N, N, N, nChannels).type(torch.DoubleTensor).to(device = 'cuda')
value_gradient = torch.randn(N, N, nChannels * NCONTRACTIONS).type(torch.DoubleTensor).to(device = 'cuda')
RisiContraction_18.RisiContraction_18_backward(tensor_gradient, adj, value_gradient, NTHREADS)
print("Done")