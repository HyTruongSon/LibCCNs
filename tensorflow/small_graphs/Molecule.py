# Molecule class
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

import numpy as np
import Atom

dtype_float = tf.float32
dtype_int = tf.int32
dtype_string = tf.string
dtype_bool = tf.bool
device_cpu = tf.DeviceSpec(device_type = "CPU")
device_gpu = tf.DeviceSpec(device_type = "GPU")

class Molecule:
	def __init__(self, index, nAtoms, atomic_type):
		self.index = index
		self.nAtoms = nAtoms
		self.atomic_type = atomic_type
		assert self.nAtoms == len(self.atomic_type)

		self.atoms = []
		for i in range(self.nAtoms):
			self.atoms.append(Atom.Atom(index, self.atomic_type[i]))

	def set_neighbors(self, atom_index, nNeighbors, neighbors, weights):
		assert atom_index >= 0
		assert atom_index < self.nAtoms
		self.atoms[atom_index].set_neighbors(nNeighbors, neighbors, weights)

	def set_class(self, class_):
		self.class_ = class_

	def from_atomic_type_to_feature(self, all_atomic_type, sparse = True):
		nAtomicTypes = len(all_atomic_type)
		if sparse == True:
			x = []
			y = []
			v = []
			for i in range(self.nAtoms):
				index = all_atomic_type.index(self.atomic_type[i])
				assert index >= 0
				assert index < nAtomicTypes
				x.append(i)
				y.append(index)
				v.append(1.0)
			x = np.reshape(np.array(x), (len(x), 1))
			y = np.reshape(np.array(y), (len(y), 1))
			indices_tensor = np.concatenate((x, y), axis = 1)
			values_tensor = np.array(v)
			self.sparse_feature = tf.SparseTensor(indices = indices_tensor, values = values_tensor, dense_shape = [self.nAtoms, nAtomicTypes])
		else:
			self.feature = np.zeros((self.nAtoms, nAtomicTypes))
			for i in range(self.nAtoms):
				index = all_atomic_type.index(self.atomic_type[i])
				assert index >= 0
				assert index < nAtomicTypes
				self.feature[i, index] = 1.0