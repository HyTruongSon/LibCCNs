# Training program for Covariant Compositional Networks (CCN) 1D

from absl import flags
from absl import logging
from absl import app

import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import Dataset
import CCN1D

dtype = torch.float
device = torch.device("cpu")

FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', '', 'Dataset directory')
flags.DEFINE_string('data_name', '', 'Data name')
flags.DEFINE_integer('epochs', 1024, 'Number of epochs')
flags.DEFINE_float('learning_rate', 1e3, 'Learning rate')
flags.DEFINE_integer('input_size', 16, 'Input size')
flags.DEFINE_string('message_sizes', '', 'Message sizes')
flags.DEFINE_string('message_mlp_sizes', '', 'Multi-layer perceptron sizes')
flags.DEFINE_integer('nThreads', 14, 'Number of threads')
flags.DEFINE_string('activation', 'relu', 'Activation')
flags.DEFINE_integer('tsne', 0, 't-SNE visualization')

def visualization(data, X, y, figure_num, title):
	plt.figure(figure_num)
	for c in range(data.nClasses):
		label = data.classes[c]
		X_part = X[y == c, :]
		plt.plot(X_part[:, 0], X_part[:, 1], '.', label = label)
	plt.title(title)
	plt.legend()
	plt.show()

def main(argv):
	data_dir = FLAGS.data_dir
	data_name = FLAGS.data_name
	epochs = FLAGS.epochs
	learning_rate = FLAGS.learning_rate
	input_size = FLAGS.input_size
	message_sizes = [int(element) for element in FLAGS.message_sizes.strip().split(',')]
	message_mlp_sizes = [int(element) for element in FLAGS.message_mlp_sizes.strip().split(',')]
	nThreads = FLAGS.nThreads
	activation = FLAGS.activation
	tsne = FLAGS.tsne

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
	data.sparse_feature()
	data.edges_to_tensor()
	data.reversed_edges_to_tensor()

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
	print('Feature matrix:', data.sparse_feature)
	print('Edges tensor:', data.edges_tensor)
	print('Reversed edges tensor:', data.reversed_edges_tensor)

	# Model creation
	input_size = [data.nVocab, input_size]

	model = CCN1D.CCN1D(input_size, message_sizes, message_mlp_sizes, data.nClasses, nThreads, activation)
	optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, amsgrad = True)

	print('\n--- Training -------------------------------')
	for epoch in range(epochs):
		print('\nEpoch', epoch)

		start_time = time.time()
		# Training
		optimizer.zero_grad()
		output = model(data, 'train')
		target = torch.from_numpy(np.array(data.train_label)).to(torch.long)
		loss = F.nll_loss(output, target)
		loss.backward()
		optimizer.step()
		print('Training loss =', loss.item())
		end_time = time.time()
		print('Training time=', (end_time - start_time))

		# Validation
		with torch.no_grad():
			output = model(data, 'val')
			target = torch.from_numpy(np.array(data.val_label)).to(torch.long)
			predict = output.argmax(dim = 1, keepdim = True)
			correct = predict.eq(target.view_as(predict)).sum().item()
			accuracy = 100.0 * correct / len(data.val_label)
			print('Validation accuracy =', accuracy)

	# Validation
	print('\n--- Testing -------------------------------')
	with torch.no_grad():
		output = model(data, 'test')
		target = torch.from_numpy(np.array(data.test_label)).to(torch.long)
		predict = output.argmax(dim = 1, keepdim = True)
		correct = predict.eq(target.view_as(predict)).sum().item()
		accuracy = 100.0 * correct / len(data.test_label)
		print('Testing accuracy =', accuracy)

	# t-SNE visualization
	if tsne > 0:
		print('t-SNE visualization')
		X = model.total_representation.data.numpy()
		X_embedded = TSNE(n_components = 2).fit_transform(X)

		X_train = X_embedded[np.array(data.train_mask) == 1, :]
		y_train = np.array(data.train_label)
		
		X_val = X_embedded[np.array(data.val_mask) == 1, :]
		y_val = np.array(data.val_label)
		
		X_test = X_embedded[np.array(data.test_mask) == 1, :]
		y_test = np.array(data.test_label)

		visualization(data, X_train, y_train, 0, data_name + " - Training set")
		visualization(data, X_val, y_val, 1, data_name + " - Validation set")
		visualization(data, X_test, y_test, 2, data_name + " - Testing set")

if __name__ == '__main__':
	app.run(main)