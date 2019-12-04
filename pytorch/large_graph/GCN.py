# GCN class
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

import Edge
import Vertex
import Dataset

torch.manual_seed(123456789)

dtype = torch.float
device = torch.device("cpu")

class GCN(nn.Module):
	def __init__(self, input_size, message_sizes, message_mlp_sizes, num_classes):
		super(GCN, self).__init__()

		self.input_size = input_size
		self.message_sizes = message_sizes
		self.message_mlp_sizes = message_mlp_sizes
		self.num_classes = num_classes		

		# Number of layers/levels/iterations
		self.nLayers = len(self.message_sizes)

		# Mapping the sparse feature by a linear model
		assert len(self.input_size) == 2

		self.fully_connected_1 = nn.Linear(self.input_size[0], self.input_size[1])

		# Message Passing weights initialization
		self.message_weights = []
		for layer in range(self.nLayers):
			if layer == 0:
				d1 = self.input_size[1]
			else:
				d1 = self.message_sizes[layer - 1]
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

				w = torch.nn.Parameter(torch.randn(n1, n2, device = device, dtype = dtype, requires_grad = True))
				weights.append(w)

			self.message_weights.append(weights)

		self.message_params = torch.nn.ParameterList([item for sublist in self.message_weights for item in sublist])

		self.fully_connected_2 = nn.Linear(self.message_sizes[self.nLayers - 1], self.num_classes)

	def forward(self, data, task, sparse):
		assert data.nClasses == self.num_classes

		# +----------------------+
		# | Dense representation |
		# +----------------------+
		if sparse == 'dense':
			# Dense adjacency matrix
			dense_adj = data.sparse_adj.to_dense()

			# Dense feature matrix
			dense_feature = data.sparse_feature.to_dense()

			# Normalize the adjacency matrix
			N = dense_adj.size(0)
			degree = torch.diag(1.0 / (dense_adj + torch.eye(N)).sum(1).view(N))
			adj = torch.matmul(dense_adj, degree)

			# Mapping the sparse feature by a linear model
			dense_feature = self.fully_connected_1(dense_feature)

			# Message Passing
			self.message = []
			for layer in range(self.nLayers):
				if layer == 0:
					m = torch.matmul(adj, dense_feature)
				else:
					m = torch.matmul(adj, self.message[layer - 1])
				
				for l in range(len(self.message_mlp_sizes) + 1):
					if l == 0:
						h = torch.matmul(m, self.message_weights[layer][l])
					else:
						h = torch.matmul(h, self.message_weights[layer][l])
					h = F.relu(h)
				self.message.append(h)

			# Prediction
			if task == 'train':
				split = torch.ByteTensor(data.train_mask)

			if task == 'val':
				split = torch.ByteTensor(data.val_mask)

			if task == 'test':
				split = torch.ByteTensor(data.test_mask)

			representation = self.message[self.nLayers - 1][split]
			representation = self.fully_connected_2(representation)
			return F.log_softmax(representation)

		# +-----------------------+
		# | Sparse representation |
		# +-----------------------+
		assert sparse == 'sparse'

		# Sparse adjacency matrix
		sparse_adj = data.sparse_adj

		# Sparse feature matrix
		sparse_feature = data.sparse_feature

		# Normalize the adjacency matrix
		N = sparse_adj.size(0)
		degree = torch.diag(1.0 / (sparse_adj.to_dense() + torch.eye(N)).sum(1).view(N))
		adj = torch.mm(sparse_adj, degree)

		# Mapping the sparse feature by a linear model
		dense_feature = self.fully_connected_1(sparse_feature.to_dense())

		# Message Passing
		self.message = []
		for layer in range(self.nLayers):
			if layer == 0:
				m = torch.mm(adj, dense_feature)
			else:
				m = torch.mm(adj, self.message[layer - 1])
				
			for l in range(len(self.message_mlp_sizes) + 1):
				if l == 0:
					h = torch.matmul(m, self.message_weights[layer][l])
				else:
					h = torch.matmul(h, self.message_weights[layer][l])
				h = F.relu(h)
			self.message.append(h)

		# Prediction
		if task == 'train':
			split = torch.ByteTensor(data.train_mask)

		if task == 'val':
			split = torch.ByteTensor(data.val_mask)

		if task == 'test':
			split = torch.ByteTensor(data.test_mask)

		representation = self.message[self.nLayers - 1][split]
		representation = self.fully_connected_2(representation)
		return F.log_softmax(representation)
