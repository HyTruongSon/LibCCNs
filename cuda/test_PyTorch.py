import math
import torch

import datetime

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
print("Test 1")
start_time = datetime.datetime.now()

value = RisiContraction_18_Op.apply(tensor, adj) # Call the autograd operator

end_time = datetime.datetime.now()
time_diff = end_time - start_time
execution_time = time_diff.total_seconds() * 1000
print("Time ellapsed:", execution_time)

# Forward (CUDA extension)
print("Test 2")
start_time = datetime.datetime.now()

value = torch.zeros(N, N, nChannels * NCONTRACTIONS).type(torch.DoubleTensor).to(device = 'cuda')
RisiContraction_18.RisiContraction_18_forward(tensor, adj, value, NTHREADS)

end_time = datetime.datetime.now()
time_diff = end_time - start_time
execution_time = time_diff.total_seconds() * 1000
print("Time ellapsed:", execution_time)

# Backward (CUDA extension)
print("Test 3")
start_time = datetime.datetime.now()

tensor_gradient = torch.zeros(N, N, N, nChannels).type(torch.DoubleTensor).to(device = 'cuda')
value_gradient = torch.randn(N, N, nChannels * NCONTRACTIONS).type(torch.DoubleTensor).to(device = 'cuda')
RisiContraction_18.RisiContraction_18_backward(tensor_gradient, adj, value_gradient, NTHREADS)

end_time = datetime.datetime.now()
time_diff = end_time - start_time
execution_time = time_diff.total_seconds() * 1000
print("Time ellapsed:", execution_time)