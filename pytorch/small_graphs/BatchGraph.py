# (Batching) Graph class
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

import numpy as np

cpu_device = 'cpu'
gpu_device = 'gpu'

class BatchGraph:
	def __init__(self, adjacencies, vertex_features):
		# CPU
		adjacencies = adjacencies.to(device = cpu_device)
		vertex_features = vertex_features.to(device = cpu_device)

		# Number of graphs
		self.nGraphs = adjacencies.size(0)
		self.N = adjacencies.size(1)
		self.nFeatures = vertex_features.size(2)
		self.nVertices = self.nGraphs * self.N
		self.nEdges = int(torch.sum(adjacencies))

		assert adjacencies.size(2) == self.N
		assert vertex_features.size(0) == self.nGraphs
		assert vertex_features.size(1) == self.N

		# Vertex indexing
		self.start_index = np.zeros((self.nGraphs + 1), dtype = np.int32)
		count = 0
		for mol in range(self.nGraphs):
			self.start_index[mol] = count
			count += self.N
		assert count == self.nVertices
		self.start_index[self.nGraphs] = self.nVertices
		self.start_index = torch.from_numpy(self.start_index)

		# Adjacency matrix (total of all smaller molecules)
		self.adj = np.zeros((self.nVertices, self.nVertices), dtype = np.float32)

		# Edge indexing
		self.edges_tensor = np.zeros((self.nEdges, 2), dtype = np.float32)
		count = 0
		for graph in range(self.nGraphs):
			for vertex1 in range(self.N):
				u = self.start_index[graph] + vertex1
				for vertex2 in range(self.N):
					if adjacencies[graph, vertex1, vertex2] > 0:
						v = self.start_index[graph] + vertex2
						self.edges_tensor[count, 0] = u
						self.edges_tensor[count, 1] = v
						count += 1

						# Adjacency matrix
						self.adj[u, v] = 1.0
						self.adj[v, u] = 1.0
		assert count == self.nEdges
		self.edges_tensor = torch.from_numpy(self.edges_tensor)
		self.adj = torch.from_numpy(self.adj)

		# Feature indexing
		x = []
		y = []
		v = []
		for graph in range(self.nGraphs):
			for vertex in range(self.N):
				for feature in range(self.nFeatures):
					x.append(self.start_index[graph] + vertex)
					y.append(feature)
					v.append(vertex_features[graph, vertex, feature])
		index_tensor = torch.LongTensor([x, y])
		value_tensor = torch.FloatTensor(v)
		self.feature = torch.sparse.FloatTensor(index_tensor, value_tensor, torch.Size([self.nVertices, self.nFeatures]))