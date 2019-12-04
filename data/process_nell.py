# Preprocessing NELL dataset

from absl import flags
from absl import logging
from absl import app

import numpy as np
import time

import pickle as pkl

FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', '', 'Dataset directory')
flags.DEFINE_string('rate', '', 'Labelling rate')

def to_edge(input_fn, output_fn):
	obj = pkl.load(open(input_fn, 'rb'), encoding = 'latin1')
	nVertices = len(obj)
	nEdges = 0
	for v in range(nVertices):
		nEdges += len(obj[v])

	output = open(output_fn, "w")
	output.write("Number of edges:\n")
	output.write(str(nEdges) + "\n")
	count = 0
	for v in range(nVertices):
		for i in range(len(obj[v])):
			u = obj[v][i]
			output.write("Index:\n" + str(count) + "\n")
			output.write("From:\n" + str(v) + "\n")
			output.write("To:\n" + str(u) + "\n")
			count += 1
	output.close()
	assert count == nEdges

	print("Done .edge file")

def to_vertex(input_fn, output_fn):
	obj = pkl.load(open(input_fn, 'rb'), encoding = 'latin1')
	nVertices = len(obj)

	output = open(output_fn, "w")
	output.write("Number of vertices:\n" + str(nVertices) + "\n")
	for v in range(nVertices):
		output.write("Reindexed:\n" + str(v) + "\n")
		output.write("Original index:\n" + str(v) + "\n")
	output.close()

	print("Done .vertex file")

def to_feature(x_fn, tx_fn, test_index_fn, output_fn):
	x = pkl.load(open(x_fn, 'rb'), encoding = 'latin1')
	nTrain = x.shape[0]
	nFeatures = x.shape[1]

	tx = pkl.load(open(tx_fn, 'rb'), encoding = 'latin1')
	nTest = tx.shape[0]
	assert nFeatures == tx.shape[1]

	X = x.toarray()
	TX = tx.toarray()

	output = open(output_fn, "w")

	for example in range(nTrain):
		output.write("Index:\n" + str(example) + "\n")
		R = X[example, :]
		index = []
		value = []
		for f in range(nFeatures):
			if R[f] > 0.0:
				index.append(f)
				value.append(R[f])
		output.write("Number of active features:\n" + str(len(index)) + "\n")
		output.write("Indices of active features:\n")
		for i in range(len(index)):
			output.write(str(index[i]) + " ")
		output.write("\n")
		output.write("Values of active features:\n")
		for i in range(len(value)):
			output.write(str(value[i]) + " ")
		output.write("\n")

	test_index = open(test_index_fn, "r")

	for example in range(nTest):
		output.write("Index:\n" + test_index.readline().strip() + "\n")
		R = TX[example, :]
		index = []
		value = []
		for f in range(nFeatures):
			if R[f] > 0.0:
				index.append(f)
				value.append(R[f])
		output.write("Number of active features:\n" + str(len(index)) + "\n")
		output.write("Indices of active features:\n")
		for i in range(len(index)):
			output.write(str(index[i]) + " ")
		output.write("\n")
		output.write("Values of active features:\n")
		for i in range(len(value)):
			output.write(str(value[i]) + " ")
		output.write("\n")

	test_index.close()

	output.close()

	print("Done .feature file")

def to_class(y_fn, ty_fn, test_index_fn, output_fn):
	y = pkl.load(open(y_fn, 'rb'), encoding = 'latin1')
	nClasses = y.shape[1]
	nTrain = y.shape[0]

	ty = pkl.load(open(ty_fn, 'rb'), encoding = 'latin1')
	assert nClasses == ty.shape[1]
	nTest = ty.shape[0]

	output = open(output_fn, "w")

	for example in range(nTrain):
		output.write("Index:\n" + str(example) + "\n")
		R = y[example, :]
		for c in range(nClasses):
			if R[c] > 0:
				output.write("Class:\n" + str(c) + "\n")
				break

	test_index = open(test_index_fn, "r")

	for example in range(nTest):
		output.write("Index:\n" + test_index.readline().strip() + "\n")
		R = ty[example, :]
		for c in range(nClasses):
			if R[c] > 0:
				output.write("Class:\n" + str(c) + "\n")
				break

	test_index.close()

	output.close()

	print("Done .class file")

def to_meta(graph_fn, x_fn, tx_fn, y_fn, ty_fn, output_fn):
	graph = pkl.load(open(graph_fn, 'rb'), encoding = 'latin1')
	nVertices = len(graph)
	nEdges = 0
	for v in range(nVertices):
		nEdges += len(graph[v])

	x = pkl.load(open(x_fn, 'rb'), encoding = 'latin1')
	nFeatures = x.shape[1]
	nTrain = x.shape[0]

	tx = pkl.load(open(tx_fn, 'rb'), encoding = 'latin1')
	assert nFeatures == tx.shape[1]
	nTest = tx.shape[0]

	y = pkl.load(open(y_fn, 'rb'), encoding = 'latin1')
	nClasses = y.shape[1]
	assert nTrain == y.shape[0]

	ty = pkl.load(open(ty_fn, 'rb'), encoding = 'latin1')
	assert nClasses == ty.shape[1]
	assert nTest == ty.shape[0]

	output = open(output_fn, "w")
	output.write("Number of vertices with labels:\n" + str(nTrain + nTest) + "\n")
	output.write("Number of total vertices:\n" + str(nVertices) + "\n")
	output.write("Number of vertices without labels:\n" + str(nVertices - nTrain - nTest) + "\n")
	output.write("Number of edges:\n" + str(nEdges) + "\n")
	output.write("Number of features:\n" + str(nFeatures) + "\n")
	output.write("Number of classes:\n" + str(nClasses) + "\n")
	output.write("Classes:\n")
	for c in range(nClasses):
		output.write(str(c) + "\n")
	output.write("Density:\n" + str((float)(nEdges) / (float)(nVertices)) + "\n")
	output.close()

	print("Done .meta file")

def to_train_val_test(graph_fn, x_fn, tx_fn, test_index_fn, train_fn, val_fn, test_fn):
	graph = pkl.load(open(graph_fn, 'rb'), encoding = 'latin1')
	nVertices = len(graph)

	x = pkl.load(open(x_fn, 'rb'), encoding = 'latin1')
	nTrain = x.shape[0]
	nFeatures = x.shape[1]

	tx = pkl.load(open(tx_fn, 'rb'), encoding = 'latin1')
	nTest = tx.shape[0]
	assert nFeatures == tx.shape[1]

	train = open(train_fn, "w")
	train.write("Number of training examples:\n" + str(nTrain) + "\n")
	train.write("Percentage:\n" + str((float)(nTrain) / (float)(nVertices) * 100.0) + "\n")
	train.write("Indices:\n")
	for example in range(nTrain):
		train.write(str(example) + "\n")
	train.close()

	val = open(val_fn, "w")
	test = open(test_fn, "w")

	val.write("Number of validation/testing examples:\n" + str(nTest) + "\n")
	test.write("Number of validation/testing examples:\n" + str(nTest) + "\n")

	val.write("Percentage:\n" + str((float)(nTest) / (float)(nVertices) * 100.0) + "\n")
	test.write("Percentage:\n" + str((float)(nTest) / (float)(nVertices) * 100.0) + "\n")

	val.write("Indices:\n")
	test.write("Indices:\n")

	test_index = open(test_index_fn, "r")

	for example in range(nTest):
		index = test_index.readline().strip()
		val.write(index + "\n")
		test.write(index + "\n")

	test_index.close()

	val.close()
	test.close()

	print("Done .train, .val, and .test files")

def main(argv):
	data_dir = FLAGS.data_dir
	rate = FLAGS.rate
	data_name = "NELL_" + rate

	to_edge(data_dir + "/ind.nell." + rate + ".graph", data_name + "/" + data_name + ".edge")
	
	to_vertex(data_dir + "/ind.nell." + rate + ".graph", data_name + "/" + data_name + ".vertex")
	
	to_feature(
		data_dir + "/ind.nell." + rate + ".x", 
		data_dir + "/ind.nell." + rate + ".tx",
		data_dir + "/ind.nell." + rate + ".test.index",
		data_name + "/" + data_name + ".feature")
	
	to_class(
		data_dir + "/ind.nell." + rate + ".y", 
		data_dir + "/ind.nell." + rate + ".ty",
		data_dir + "/ind.nell." + rate + ".test.index",
		data_name + "/" + data_name + ".class")
	
	to_meta(
		data_dir + "/ind.nell." + rate + ".graph",
		data_dir + "/ind.nell." + rate + ".x",
		data_dir + "/ind.nell." + rate + ".tx",
		data_dir + "/ind.nell." + rate + ".y",
		data_dir + "/ind.nell." + rate + ".ty",
		data_name + "/" + data_name + ".meta")

	to_train_val_test(
		data_dir + "/ind.nell." + rate + ".graph",
		data_dir + "/ind.nell." + rate + ".x",
		data_dir + "/ind.nell." + rate + ".tx",
		data_dir + "/ind.nell." + rate + ".test.index",
		data_name + "/" + data_name + ".train",
		data_name + "/" + data_name + ".val",
		data_name + "/" + data_name + ".test")

if __name__ == '__main__':
	app.run(main)