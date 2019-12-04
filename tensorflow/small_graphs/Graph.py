# Graph class
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

import numpy as np
import Molecule
import Atom

dtype_float = tf.float32
dtype_int = tf.int32
dtype_string = tf.string
dtype_bool = tf.bool
device_cpu = tf.DeviceSpec(device_type = "CPU")
device_gpu = tf.DeviceSpec(device_type = "GPU")

class Graph:
	def __init__(self, molecules, all_atomic_type, nClasses):
		self.molecules = molecules

		self.nMolecules = len(self.molecules)
		self.nVertices = 0
		self.nEdges = 0
		for mol in range(self.nMolecules):
			self.nVertices += self.molecules[mol].nAtoms
			for atom in range(self.molecules[mol].nAtoms):
				self.nEdges += self.molecules[mol].atoms[atom].nNeighbors

		# Vertex indexing
		self.start_index = np.zeros((self.nMolecules + 1, 1), dtype = np.int32)
		count = 0
		for mol in range(self.nMolecules):
			self.start_index[mol] = count
			count += self.molecules[mol].nAtoms
		assert count == self.nVertices
		self.start_index[self.nMolecules] = self.nVertices

		# Edge indexing
		self.edges_tensor = np.zeros((self.nEdges, 2), dtype = np.int32)
		count = 0
		for mol in range(self.nMolecules):
			for atom in range(self.molecules[mol].nAtoms):
				u = self.start_index[mol] + atom
				for i in range(self.molecules[mol].atoms[atom].nNeighbors):
					v = self.start_index[mol] + self.molecules[mol].atoms[atom].neighbors[i]
					self.edges_tensor[count, 0] = u
					self.edges_tensor[count, 1] = v
					count += 1
		assert count == self.nEdges

		# Feature indexing - Sparse
		nAtomicTypes = len(all_atomic_type)
		x = []
		y = []
		v = []
		for mol in range(self.nMolecules):
			for i in range(self.molecules[mol].nAtoms):
				index = all_atomic_type.index(self.molecules[mol].atomic_type[i])
				x.append(self.start_index[mol] + i)
				y.append(index)
				v.append(1.0)
		x = np.reshape(np.array(x), (len(x), 1))
		y = np.reshape(np.array(y), (len(y), 1))
		indices_tensor = np.concatenate((x, y), axis = 1)
		values_tensor = np.array(v)
		self.sparse_feature = tf.SparseTensor(indices = indices_tensor, values = values_tensor, dense_shape = [self.nVertices, nAtomicTypes])

		# Feature indexing - Dense
		self.feature = np.zeros((self.nVertices, nAtomicTypes))
		assert len(x) == len(v)
		assert len(y) == len(v)
		for i in range(len(x)):
			self.feature[x[i], y[i]] = v[i]

		# Label indexing
		self.label = np.zeros((self.nMolecules, nClasses))
		for mol in range(self.nMolecules):
			self.label[mol, self.molecules[mol].class_] = 1.0

		# Receptive fields to TensorFlow tensors
		# self.start_index = tf.convert_to_tensor(value = self.start_index, dtype = dtype_int)
		# self.edges_tensor = tf.convert_to_tensor(value = self.edges_tensor, dtype = dtype_int)