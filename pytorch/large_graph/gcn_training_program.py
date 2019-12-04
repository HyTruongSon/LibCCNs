# Training program for Graph Convolution Neural Networks

from absl import flags
from absl import logging
from absl import app

import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

import Dataset
import GCN 

dtype = torch.float
device = torch.device("cpu")

FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', '', 'Dataset directory')
flags.DEFINE_string('data_name', '', 'Dataset name')
flags.DEFINE_integer('epochs', 1024, 'Number of epochs')
flags.DEFINE_float('learning_rate', 1e3, 'Learning rate')
flags.DEFINE_string('sparse', 'dense', 'Using sparse representation or not')
flags.DEFINE_integer('input_size', 16, 'Input size')
flags.DEFINE_string('message_sizes', '', 'Message sizes')
flags.DEFINE_string('message_mlp_sizes', '', 'Multi-layer perceptron sizes')

def main(argv):
	data_dir = FLAGS.data_dir
	data_name = FLAGS.data_name
	epochs = FLAGS.epochs
	learning_rate = FLAGS.learning_rate
	sparse = FLAGS.sparse
	input_size = FLAGS.input_size
	message_sizes = [int(element) for element in FLAGS.message_sizes.strip().split(',')]
	message_mlp_sizes = [int(element) for element in FLAGS.message_mlp_sizes.strip().split(',')]

	class_fn = data_dir + "/" + data_name + "/" + data_name + ".class"
	vertex_fn = data_dir + "/" + data_name + "/" + data_name + ".vertex"
	edge_fn = data_dir + "/" + data_name + "/" + data_name + ".edge"
	feature_fn = data_dir + "/" + data_name + "/" + data_name + ".feature"
	meta_fn = data_dir + "/" + data_name + "/" + data_name + ".meta"
	train_fn = data_dir + "/" + data_name + "/" + data_name + ".train"
	val_fn = data_dir + "/" + data_name + "/" + data_name + ".val"
	test_fn = data_dir + "/" + data_name + "/" + data_name + ".test"

	print('Data name:', data_name)

	data = Dataset.Dataset(class_fn, vertex_fn, edge_fn, feature_fn, meta_fn, train_fn, val_fn, test_fn)
	data.sparse_adj()
	data.sparse_feature()

	print('Number of vertices:', data.nVertices)
	print('Number of papers:', data.nPapers)
	print('Number of papers without features:', data.nNoFeatures)
	print('Number of edges:', data.nEdges)
	print('Vocabulary size:', data.nVocab)
	print('Number of classes:', data.nClasses)
	print('Classes:', data.classes)
	print('Density:', data.density)
	assert len(data.edges) == data.nEdges
	print('Training examples:', len(data.train))
	print('Validation examples:', len(data.val))
	print('Testing examples:', len(data.test))
	assert len(data.vertices) == data.nPapers
	
	print('Adjacency matrix:', data.sparse_adj)
	print('Feature matrix:', data.sparse_feature)

	# Model creation
	input_size = [data.nVocab, input_size]

	model = GCN.GCN(input_size, message_sizes, message_mlp_sizes, data.nClasses)
	optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, amsgrad = True)

	print('\n--- Training -------------------------------')
	for epoch in range(epochs):
		print('\nEpoch', epoch)

		# Training
		optimizer.zero_grad()
		output = model(data, 'train', sparse)
		target = torch.from_numpy(np.array(data.train_label)).to(torch.long)
		loss = F.nll_loss(output, target)
		loss.backward()
		optimizer.step()
		print('Training loss =', loss.item())

		# Validation
		with torch.no_grad():
			output = model(data, 'val', sparse)
			target = torch.from_numpy(np.array(data.val_label)).to(torch.long)
			predict = output.argmax(dim = 1, keepdim = True)
			correct = predict.eq(target.view_as(predict)).sum().item()
			accuracy = 100.0 * correct / len(data.val_label)
			print('Validation accuracy =', accuracy)

	# Validation
	print('\n--- Testing -------------------------------')
	with torch.no_grad():
		output = model(data, 'test', sparse)
		target = torch.from_numpy(np.array(data.test_label)).to(torch.long)
		predict = output.argmax(dim = 1, keepdim = True)
		correct = predict.eq(target.view_as(predict)).sum().item()
		accuracy = 100.0 * correct / len(data.test_label)
		print('Testing accuracy =', accuracy)

if __name__ == '__main__':
	app.run(main)