# Training program for Covariant Compositional Networks (CCN) 1D Variational Autoencoder

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
import Graph
import Molecule
import Atom
import CCN1D_Encoder
import Dot_Decoder
import Variational_Autoencoder

dtype = torch.float
device = torch.device("cpu")

FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', '', 'Dataset directory')
flags.DEFINE_string('data_name', '', 'Data name')
flags.DEFINE_integer('epochs', 1024, 'Number of epochs')
flags.DEFINE_float('learning_rate', 1e3, 'Learning rate')
flags.DEFINE_integer('input_size', 16, 'Input size')
flags.DEFINE_integer('output_size', 32, 'Output size')
flags.DEFINE_string('message_sizes', '', 'Message sizes')
flags.DEFINE_string('message_mlp_sizes', '', 'Multi-layer perceptron sizes')
flags.DEFINE_integer('nThreads', 14, 'Number of threads')
flags.DEFINE_integer('batch_size', 10, 'Batch size')
flags.DEFINE_string('activation', 'relu', 'Activation')
flags.DEFINE_integer('tsne', 0, 't-SNE visualization')
flags.DEFINE_integer('fold', 0, 'Fold number')

def visualization(data, X, y, figure_num, title):
	plt.figure(figure_num)
	for c in range(data.nClasses):
		label = data.classes[c]
		X_part = X[y == c, :]
		plt.plot(X_part[:, 0], X_part[:, 1], '.', label = label)
	plt.title(title)
	plt.legend()
	plt.show()

def CLEAR_LOG(report_fn):
	report = open(report_fn, "w")
	report.close()

def LOG(report_fn, str):
	report = open(report_fn, "a")
	report.write(str + "\n")
	report.close()

def vae_loss(predict, z_mu, z_var, target):
	# Reconstruction loss
	predict_flat = predict.view(-1)
	target_flat = target.view(-1)
	reconstruction_loss =  F.binary_cross_entropy_with_logits(predict_flat, target_flat, reduction = 'mean')

	# KL loss
	N = z_mu.size(0)
	KL_loss = 0.5 / N * torch.sum(torch.exp(z_var) + z_mu**2 - 1.0 - z_var)

	# Total loss
	total_loss = reconstruction_loss + KL_loss

	return total_loss, reconstruction_loss, KL_loss

def train_batch(data, indices, model, optimizer):
	# Batch creation
	batch = []
	for i in range(len(indices)):
		batch.append(data.molecules[indices[i]])

	# Molecular graphs concatenation
	graph = Graph.Graph(batch, data.all_atomic_type)

	# Training
	optimizer.zero_grad()
	predict, z_mu, z_var = model(graph)
	target = graph.adj
	total_loss, reconstruction_loss, KL_loss = vae_loss(predict, z_mu, z_var, target) # F.nll_loss(predict, target)
	total_loss.backward()
	optimizer.step()

	return total_loss.item()

def predict_batch(data, indices, model):
	# Batch creation
	batch = []
	for i in range(len(indices)):
		batch.append(data.molecules[indices[i]])

	# Molecular graphs concatenation
	graph = Graph.Graph(batch, data.all_atomic_type)

	# Predicting
	with torch.no_grad():
		predict, z_mu, z_var = model(graph)
		target = graph.adj
		# predict = torch.sigmoid(predict)
		total_loss, reconstruction_loss, KL_loss = vae_loss(predict, z_mu, z_var, target)
	return total_loss.item()

def get_batch(data, indices, model):
	# Batch creation
	batch = []
	for i in range(len(indices)):
		batch.append(data.molecules[indices[i]])

	# Molecular graphs concatenation
	graph = Graph.Graph(batch, data.all_atomic_type)

	# Predicting
	with torch.no_grad():
		output = model(graph)
	return model.representation.data.numpy()	

def get_representation(data, model):
	X = np.zeros((data.nMolecules, model.total_representation.size(1)))
	start = 0
	while start < data.nMolecules:
		finish = min(start + FLAGS.batch_size, data.nMolecules)
		indices = [index for index in range(start, finish)]
		x = get_batch(data, indices, model)
		X[start:finish, :] = x[:, :]
		start = finish
	return X

def main(argv):
	data_dir = FLAGS.data_dir
	data_name = FLAGS.data_name
	epochs = FLAGS.epochs
	learning_rate = FLAGS.learning_rate
	input_size = FLAGS.input_size
	output_size = FLAGS.output_size
	message_sizes = [int(element) for element in FLAGS.message_sizes.strip().split(',')]
	message_mlp_sizes = [int(element) for element in FLAGS.message_mlp_sizes.strip().split(',')]
	nThreads = FLAGS.nThreads
	batch_size = FLAGS.batch_size
	activation = FLAGS.activation
	tsne = FLAGS.tsne
	fold = FLAGS.fold

	data_fn = data_dir + "/" + data_name + "/" + data_name + ".dat"
	meta_fn = data_dir + "/" + data_name + "/" + data_name + ".meta"
	train_fn = data_dir + "/" + data_name + "/" + data_name + ".train." + str(fold)
	test_fn = data_dir + "/" + data_name + "/" + data_name + ".test." + str(fold)
	report_fn = "reports/" + data_name + ".fold." + str(fold) + ".report" 

	print('Data name:\n' + data_name)
	print("Data file:\n" + data_fn)
	print("Meta file:\n" + meta_fn)
	print("Training file:\n" + train_fn)
	print("Testing file:\n" + test_fn)

	CLEAR_LOG(report_fn)
	LOG(report_fn, 'Data name:\n' + data_name)
	LOG(report_fn, "Data file:\n" + data_fn)
	LOG(report_fn, "Meta file:\n" + meta_fn)
	LOG(report_fn, "Training file:\n" + train_fn)
	LOG(report_fn, "Testing file:\n" + test_fn)

	data = Dataset.Dataset(data_fn, meta_fn, train_fn, test_fn)

	print("Number of molecules:\n" + str(data.nMolecules))
	print("Number of atomic types:\n" + str(data.nAtomicTypes))
	print("Atomic types:")
	print(data.all_atomic_type)
	print("Number of molecular labels:\n" + str(data.nClasses))
	print("Molecular labels:")
	print(data.classes)
	print("Training indices:")
	print(data.train_indices)
	print("Testing indices:")
	print(data.test_indices)

	# Model creation
	input_size = [len(data.all_atomic_type), input_size]

	encoder = CCN1D_Encoder.CCN1D_Encoder(input_size, message_sizes, message_mlp_sizes, output_size, nThreads, activation)
	decoder = Dot_Decoder.Dot_Decoder()
	model = Variational_Autoencoder.VAE(encoder, decoder)
	optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, amsgrad = True)

	print('\n--- Training -------------------------------')
	LOG(report_fn, '\n--- Training -------------------------------')

	for epoch in range(epochs):
		print('\nEpoch ' + str(epoch))
		LOG(report_fn, '\nEpoch ' + str(epoch))

		# Training
		start_time = time.time()
		nTrain = len(data.train_indices)
		start = 0
		avg_loss = 0.0
		nBatch = 0
		while start < nTrain:
			finish = min(start + batch_size, nTrain)
			loss = train_batch(data, data.train_indices[start:finish], model, optimizer)
			avg_loss += loss
			nBatch += 1
			start = finish
		avg_loss /= nTrain
		end_time = time.time()

		# print('Training time = ' + str(end_time - start_time))
		print('Training loss = ' + str(avg_loss))

		# LOG(report_fn, 'Training time = ' + str(end_time - start_time))
		LOG(report_fn, 'Training loss = ' + str(avg_loss))

		# Validation
		start_time = time.time()
		nTest = len(data.test_indices)
		start = 0
		avg_loss = 0.0
		while start < nTest:
			finish = min(start + batch_size, nTest)
			loss = predict_batch(data, data.test_indices[start:finish], model)
			avg_loss += loss
			start = finish
		avg_loss /= nTest	
		end_time = time.time()
		
		# print('Testing time = ' + str(end_time - start_time))
		print('Testing loss = ' + str(avg_loss))

		# LOG(report_fn, 'Testing time = ' + str(end_time - start_time))
		LOG(report_fn, 'Testing loss = ' + str(avg_loss))

	# t-SNE visualization
	if tsne > 0:
		print('t-SNE visualization')
		X = get_representation(data, model)
		X_embedded = TSNE(n_components = 2).fit_transform(X)

		X_train = X_embedded[data.train_indices, :]
		y_train = np.array(data.all_labels[data.train_indices])
		
		X_test = X_embedded[data.test_indices, :]
		y_test = np.array(data.all_labels[data.test_indices])

		visualization(data, X_train, y_train, 0, data_name + " - Training set")
		visualization(data, X_test, y_test, 1, data_name + " - Testing set")

if __name__ == '__main__':
	app.run(main)