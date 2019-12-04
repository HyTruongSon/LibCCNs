# CCN1D class
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

import sys
sys.path.append('../ccn_lib/')

import ccn1d_grad
# import forward_contractions_grad
# import forward_contractions_multi_threading_grad
# import forward_normalizing_grad
# import forward_shrinking_grad
# import forward_shrinking_multi_threading_grad

# Load the library
ccn1d_lib = tf.load_op_library('../ccn_lib/ccn1d_lib.so')
print(dir(ccn1d_lib))

import Edge
import Vertex
import Dataset

import numpy as np

# tf.random.set_seed(123456789)

dtype_float = tf.float32
dtype_int = tf.int32
dtype_string = tf.string
dtype_bool = tf.bool
device_cpu = tf.DeviceSpec(device_type = "CPU")
device_gpu = tf.DeviceSpec(device_type = "GPU")

trainable = True
mean = 0.0
stddev = 1.0

class CCN1D:
	def Linear(self, input, input_size, output_size, sparse_input = False):
		W = tf.Variable(tf.random.normal([input_size, output_size], mean = mean, stddev = stddev), dtype = dtype_float, trainable = trainable)
		b = tf.Variable(tf.random.normal([output_size], mean = mean, stddev = stddev), dtype = dtype_float, trainable = trainable)
		if sparse_input == True:
			output = tf.add(tf.sparse.sparse_dense_matmul(input, W), b)
		else:
			output = tf.add(tf.linalg.matmul(input, W), b)
		return output

	def Activation(self, input):
		if self.activation == 'sigmoid':
			output = tf.sigmoid(input)
		else:
			output = tf.nn.relu(input)
		return output

	def to_list_of_tensors(self, tensors, prefix):
		result = []
		for i in range(len(tensors)):
			# print(tensors[i].shape)
			# result.append(tf.get_variable(prefix + str(i), initializer = tensors[i], trainable = False))
			result.append(tf.convert_to_tensor(tensors[i], dtype_int))
			# print(result[i])
		return result

	def __init__(self, data, input_size, message_sizes, message_mlp_sizes, nThreads, activation, learning_rate):
		# Interactive session
		self.sess = tf.InteractiveSession()

		# Hyper-parameters
		self.data = data
		self.input_size = input_size
		self.message_sizes = message_sizes
		self.message_mlp_sizes = message_mlp_sizes
		self.nThreads = nThreads	
		self.activation = activation
		self.learning_rate = learning_rate

		# Number of layers/levels/iterations
		self.nLayers = len(self.message_sizes)

		# Number of contractions
		self.nContractions = self.sess.run(ccn1d_lib.get_num_contractions())

		# Compute the receptive fields
		self.edges_rf = ccn1d_lib.precompute_neighbors(data.edges_tensor, nVertices = data.nPapers, nLayers = self.nLayers, nTensors = 2 * (self.nLayers + 1))
		self.reversed_edges_rf = ccn1d_lib.precompute_neighbors(data.reversed_edges_tensor, nVertices = data.nPapers, nLayers = self.nLayers, nTensors = 2 * (self.nLayers + 1))

		self.edges_rf = self.sess.run(self.edges_rf)
		self.reversed_edges_rf = self.sess.run(self.reversed_edges_rf)

		self.edges_rf = self.to_list_of_tensors(self.edges_rf, "edges")
		self.reversed_edges_rf = self.to_list_of_tensors(self.reversed_edges_rf, "reversed_edges")

		# Mapping the sparse feature by a linear model
		# self.dense_feature = self.Activation(self.Linear(tf.dtypes.cast(tf.sparse.to_dense(data.sparse_feature), dtype_float), self.data.nVocab, self.input_size))
		self.dense_feature = self.Activation(self.Linear(tf.dtypes.cast(data.sparse_feature, dtype_float), self.data.nVocab, self.input_size, sparse_input = True))

		# Message Passing weights initialization
		self.message_weights = []
		self.message_reversed_weights = []

		for layer in range(self.nLayers):
			if layer == 0:
				d1 = self.input_size * self.nContractions
			else:
				d1 = self.message_sizes[layer - 1] * self.nContractions
			d2 = self.message_sizes[layer]

			# Construct an MLP mapping from d1 dimensions into d2 dimensions
			weights = []
			reversed_weights = []

			for l in range(len(self.message_mlp_sizes) + 1):
				if l == 0:
					n1 = d1
				else:
					n1 = self.message_mlp_sizes[l - 1]
				
				if l == len(self.message_mlp_sizes):
					n2 = d2
				else:
					n2 = self.message_mlp_sizes[l]

				w = tf.Variable(tf.random.normal([n1, n2], mean = mean, stddev = stddev), dtype = dtype_float, trainable = trainable)
				reversed_w = tf.Variable(tf.random.normal([n1, n2], mean = mean, stddev = stddev), dtype = dtype_float, trainable = trainable)

				weights.append(w)
				reversed_weights.append(reversed_w)

			self.message_weights.append(weights)
			self.message_reversed_weights.append(reversed_weights)

		# Message Passing
		self.message = []
		self.message_reversed = []

		self.shrinked_message = []
		self.shrinked_message_reversed = []

		for layer in range(self.nLayers):
			# Message Passing
			input_receptive_field = self.edges_rf[2 * layer]
			input_start_index = self.edges_rf[2 * layer + 1] 
			output_receptive_field = self.edges_rf[2 * layer + 2] 
			output_start_index = self.edges_rf[2 * layer + 3]

			if layer == 0:
				m = ccn1d_lib.forward_contractions_multi_threading(
					input_receptive_field, 
					input_start_index, 
					output_receptive_field, 
					output_start_index, 
					self.dense_feature, 
					nThreads = self.nThreads
				)
			else:
				m = ccn1d_lib.forward_contractions_multi_threading(
					input_receptive_field, 
					input_start_index, 
					output_receptive_field, 
					output_start_index, 
					self.message[layer - 1], 
					nThreads = self.nThreads
				)
			
			m = ccn1d_lib.forward_normalizing(output_start_index, m)

			for l in range(len(self.message_mlp_sizes) + 1):
				if l == 0:
					h = tf.linalg.matmul(m, self.message_weights[layer][l])
				else:
					h = tf.linalg.matmul(h, self.message_weights[layer][l])
				h = self.Activation(h)

			self.message.append(h)

			'''
			self.shrinked_message.append(
				ccn1d_lib.forward_shrinking_multi_threading(
					output_start_index, 
					h, 
					nThreads = self.nThreads
				)
			)
			'''

			self.shrinked_message.append(
				ccn1d_lib.forward_shrinking(
					output_start_index, 
					h
				)
			)

			# (Reversed) Message Passing
			input_receptive_field = self.reversed_edges_rf[2 * layer]
			input_start_index = self.reversed_edges_rf[2 * layer + 1] 
			output_receptive_field = self.reversed_edges_rf[2 * layer + 2] 
			output_start_index = self.reversed_edges_rf[2 * layer + 3]

			if layer == 0:
				m = ccn1d_lib.forward_contractions_multi_threading(
					input_receptive_field, 
					input_start_index, 
					output_receptive_field, 
					output_start_index, 
					self.dense_feature, 
					nThreads = self.nThreads
				)
			else:
				m = ccn1d_lib.forward_contractions_multi_threading(
					input_receptive_field, 
					input_start_index, 
					output_receptive_field, 
					output_start_index, 
					self.message_reversed[layer - 1], 
					nThreads = self.nThreads
				)
			
			m = ccn1d_lib.forward_normalizing(output_start_index, m)

			for l in range(len(self.message_mlp_sizes) + 1):
				if l == 0:
					h = tf.linalg.matmul(m, self.message_reversed_weights[layer][l])
				else:
					h = tf.linalg.matmul(h, self.message_reversed_weights[layer][l])
				h = self.Activation(h)

			self.message_reversed.append(h)

			'''
			self.shrinked_message_reversed.append(
				ccn1d_lib.forward_shrinking_multi_threading(
					output_start_index, 
					h,
					nThreads = self.nThreads
				)
			)
			'''

			self.shrinked_message_reversed.append(
				ccn1d_lib.forward_shrinking(
					output_start_index, 
					h
				)
			)			

		# Total representation
		self.list_of_tensors = [self.dense_feature] + self.shrinked_message + self.shrinked_message_reversed
		self.total_representation = tf.concat(self.list_of_tensors, axis = 1)
		self.nFeatures_total = self.input_size + 2 * self.nLayers * self.message_mlp_sizes[len(self.message_mlp_sizes) - 1]

		# Task: train, val, test
		self.split = tf.placeholder("bool", [None])

		self.representation = self.Activation(self.Linear(self.total_representation, self.nFeatures_total, self.data.nClasses))
		self.representation = tf.boolean_mask(self.representation, self.split)

		self.predict = tf.nn.softmax(self.representation)

		# Label
		self.label = tf.placeholder(dtype_float, [None, data.nClasses])

		# Loss function
		self.cost = tf.reduce_mean(- tf.reduce_sum(self.label * tf.log(self.predict), reduction_indices = 1))

		# Optimizer
		self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.cost)
		# self.optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate).minimize(self.cost)

		# Correct prediction
		self.correct_prediction = tf.equal(tf.argmax(self.predict, 1), tf.argmax(self.label, 1))

		# Accuracy
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, dtype_float))

		# Initialize the variables 
		self.sess.run(tf.global_variables_initializer())

	def dictionary(self, task):
		if task == 'train':
			feed_dict = {self.split: self.data.train_mask, self.label: self.data.train_label}
			return feed_dict

		if task == 'val':
			feed_dict = {self.split: self.data.val_mask, self.label: self.data.val_label}
			return feed_dict

		assert task == 'test'
		feed_dict = {self.split: self.data.test_mask, self.label: self.data.test_label}
		return feed_dict

	def train(self, task = 'train', nIterations = 1):
		feed_dict = self.dictionary(task)
		costs = []
		for iter in range(nIterations):
			_, c = self.sess.run([self.optimizer, self.cost], feed_dict = feed_dict)
			costs.append(c)
		return costs

	def get_loss(self, task):
		feed_dict = self.dictionary(task)
		return self.sess.run(self.cost, feed_dict = feed_dict)

	def get_accuracy(self, task):
		feed_dict = self.dictionary(task)
		return self.sess.run(self.accuracy, feed_dict = feed_dict)

	def get_representation(self, task = 'train'):
		feed_dict = self.dictionary(task)
		return self.sess.run(self.total_representation, feed_dict = feed_dict)