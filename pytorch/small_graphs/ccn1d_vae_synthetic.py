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

import BatchGraph
import CCN1D_Encoder
import Dot_Decoder
import Autoencoders
import Synthetic_Dataset

dtype = torch.float
device = torch.device("cpu")

FLAGS = flags.FLAGS

flags.DEFINE_string('data_name', '', 'Data name')
flags.DEFINE_integer('num_samples', 1000, 'Number of samples')
flags.DEFINE_integer('graph_size', 10, 'Graph size')
flags.DEFINE_integer('epochs', 1024, 'Number of epochs')
flags.DEFINE_float('learning_rate', 1e3, 'Learning rate')
flags.DEFINE_integer('input_size', 16, 'Input size')
flags.DEFINE_integer('output_size', 32, 'Output size')
flags.DEFINE_string('message_sizes', '', 'Message sizes')
flags.DEFINE_string('message_mlp_sizes', '', 'Multi-layer perceptron sizes')
flags.DEFINE_integer('nThreads', 14, 'Number of threads')
flags.DEFINE_integer('batch_size', 10, 'Batch size')
flags.DEFINE_string('activation', 'relu', 'Activation')

def CLEAR_LOG(report_fn):
	report = open(report_fn, "w")
	report.close()

def LOG(report_fn, str):
	report = open(report_fn, "a")
	report.write(str + "\n")
	report.close()

def vae_loss(predict, z_mu, z_var, target):
	predict_flat = predict.view(-1)
	target_flat = target.view(-1)
	return F.binary_cross_entropy_with_logits(predict_flat, target_flat, reduction = 'mean')

def train_batch(data, indices, model, optimizer):
	# Graphs concatenation
	graph = BatchGraph.BatchGraph(data.adjacencies[indices], data.vertex_features[indices])

	# Training
	optimizer.zero_grad()
	predict, z_mu, z_var = model(graph)
	target = graph.adj
	loss = vae_loss(predict, z_mu, z_var, target) # F.nll_loss(predict, target)
	loss.backward()
	optimizer.step()

	return loss.item()

def predict_batch(data, indices, model):
	# Graphs concatenation
	graph = BatchGraph.BatchGraph(data.adjacencies[indices], data.vertex_features[indices])

	# Predicting
	with torch.no_grad():
		predict, z_mu, z_var = model(graph)
		target = graph.adj
		loss = vae_loss(predict, z_mu, z_var, target)
	return loss.item()

def main(argv):
	data_name = FLAGS.data_name
	num_samples = FLAGS.num_samples
	graph_size = FLAGS.graph_size
	epochs = FLAGS.epochs
	learning_rate = FLAGS.learning_rate
	input_size = FLAGS.input_size
	output_size = FLAGS.output_size
	message_sizes = [int(element) for element in FLAGS.message_sizes.strip().split(',')]
	message_mlp_sizes = [int(element) for element in FLAGS.message_mlp_sizes.strip().split(',')]
	nThreads = FLAGS.nThreads
	batch_size = FLAGS.batch_size
	activation = FLAGS.activation
	
	report_fn = "reports/synthetic-" + data_name + ".report"

	print("Data name:\n" + data_name)
	print("Report file:\n" + report_fn)

	CLEAR_LOG(report_fn)
	LOG(report_fn, 'Data name:\n' + data_name)

	train_data = Synthetic_Dataset.Synthetic_Dataset(data_name, num_samples, graph_size)
	test_data = Synthetic_Dataset.Synthetic_Dataset(data_name, num_samples, graph_size)

	# Model creation
	input_size = [train_data.vertex_features.shape[2], input_size]

	encoder = CCN1D_Encoder.CCN1D_Encoder(input_size, message_sizes, message_mlp_sizes, output_size, nThreads, activation)
	decoder = Dot_Decoder.Dot_Decoder()
	model = Autoencoders.VAE(encoder, decoder)
	optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, amsgrad = True)

	print('\n--- Training -------------------------------')
	LOG(report_fn, '\n--- Training -------------------------------')

	for epoch in range(epochs):
		print('\nEpoch ' + str(epoch))
		LOG(report_fn, '\nEpoch ' + str(epoch))

		# Training
		start_time = time.time()
		nTrain = train_data.adjacencies.shape[0]
		start = 0
		avg_loss = 0.0
		nBatch = 0
		while start < nTrain:
			finish = min(start + batch_size, nTrain)
			loss = train_batch(train_data, range(start, finish), model, optimizer)
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
		nTest = test_data.adjacencies.shape[0]
		start = 0
		avg_loss = 0.0
		while start < nTest:
			finish = min(start + batch_size, nTest)
			loss = predict_batch(test_data, range(start, finish), model)
			avg_loss += loss
			start = finish
		avg_loss /= nTest	
		end_time = time.time()
		
		# print('Testing time = ' + str(end_time - start_time))
		print('Testing loss = ' + str(avg_loss))

		# LOG(report_fn, 'Testing time = ' + str(end_time - start_time))
		LOG(report_fn, 'Testing loss = ' + str(avg_loss))

if __name__ == '__main__':
	app.run(main)