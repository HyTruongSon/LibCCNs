# CCN1D -- Global Feature
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

import sys
sys.path.append('../ccn_lib/')

import ccn1d_lib
from ccn1d_contractions import ccn1d_contractions
from ccn1d_shrinking import ccn1d_shrinking
from ccn1d_normalizing import ccn1d_normalizing

import Dataset
import Graph
import Molecule
import Atom

torch.manual_seed(123456789)

dtype_float = torch.float
dtype_int = torch.int
device = torch.device("cpu")

class CCN1D_Encoder(nn.Module):
	def __init__(self, input_size, message_sizes, message_mlp_sizes, output_size, nThreads, activation):
		super(CCN1D_Encoder, self).__init__()

		# Hyper-parameters
		self.input_size = input_size
		self.message_sizes = message_sizes
		self.message_mlp_sizes = message_mlp_sizes
		self.output_size = output_size
		self.nThreads = nThreads	

		# Activation
		self.activation = F.relu # default
		if activation == 'relu':
			self.activation = F.relu
		if activation == 'sigmoid':
			self.activation = F.sigmoid

		# Number of layers/levels/iterations
		self.nLayers = len(self.message_sizes)

		# Number of contractions
		self.nContractions = ccn1d_lib.get_nContractions()

		# Mapping the sparse feature by a linear model
		assert len(self.input_size) == 2

		self.fully_connected_1 = nn.Linear(self.input_size[0], self.input_size[1])

		# Message Passing weights initialization
		self.message_weights = []

		for layer in range(self.nLayers):
			if layer == 0:
				d1 = self.input_size[1] * self.nContractions
			else:
				d1 = self.message_sizes[layer - 1] * self.nContractions
			d2 = self.message_sizes[layer]

			# Construct an MLP mapping from d1 dimensions into d2 dimensions
			weights = []

			for l in range(len(self.message_mlp_sizes) + 1):
				if l == 0:
					n1 = d1
				else:
					n1 = self.message_mlp_sizes[l - 1]
				
				if l == len(self.message_mlp_sizes):
					n2 = d2
				else:
					n2 = self.message_mlp_sizes[l]

				w = torch.nn.Parameter(torch.randn(n1, n2, device = device, dtype = dtype_float, requires_grad = True))
				weights.append(w)

			self.message_weights.append(weights)

		self.message_params = torch.nn.ParameterList([item for sublist in self.message_weights for item in sublist])

		self.num_final_features = self.input_size[1] + sum(self.message_sizes)
		self.fully_connected_2 = nn.Linear(self.num_final_features, self.output_size)

	def forward(self, graph):
		# Compute the receptive fields
		self.edges_rf = ccn1d_lib.precompute_neighbors(graph.edges_tensor, graph.nVertices, self.nLayers)

		# Mapping the sparse feature by a linear model
		dense_feature = self.activation(self.fully_connected_1(graph.feature))

		# Message Passing
		self.message = []
		self.shrinked_message = []

		for layer in range(self.nLayers):
			# print("Layer", layer)

			# Message Passing
			input_receptive_field = self.edges_rf[layer][0]
			input_start_index = self.edges_rf[layer][1] 
			output_receptive_field = self.edges_rf[layer + 1][0] 
			output_start_index = self.edges_rf[layer + 1][1]

			if layer == 0:
				m = ccn1d_contractions.apply(input_receptive_field, input_start_index, output_receptive_field, output_start_index, dense_feature, self.nThreads)
			else:
				m = ccn1d_contractions.apply(input_receptive_field, input_start_index, output_receptive_field, output_start_index, self.message[layer - 1], self.nThreads)
			
			m = ccn1d_normalizing.apply(output_start_index, m)

			for l in range(len(self.message_mlp_sizes) + 1):
				if l == 0:
					h = torch.matmul(m, self.message_weights[layer][l])
				else:
					h = torch.matmul(h, self.message_weights[layer][l])
				h = self.activation(h)

			self.message.append(h)
			self.shrinked_message.append(ccn1d_shrinking.apply(graph.nVertices, output_start_index, h, self.nThreads))

			# print("Message Passing")

		# Total representation
		self.total_representation = torch.cat([dense_feature] + self.shrinked_message, dim = 1)
		self.encoded = self.fully_connected_2(self.total_representation)

		return self.encoded