# Preprocessing small-graphs datasets

from absl import flags
from absl import logging
from absl import app

import numpy as np
import time

import pickle as pkl

FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', '', 'Dataset directory')
flags.DEFINE_string('data_name', '', 'data_name')
flags.DEFINE_integer('num_folds', 5, 'Number of folds')
flags.DEFINE_integer('num_train_folds', 4, 'Number of folds for training')

def process(input_fn, output_fn, meta_fn):
	input = open(input_fn, "r")
	output = open(output_fn, "w")

	nMolecules = int(input.readline())
	output.write("Total number of molecules:\n" + str(nMolecules) + "\n")

	all_atomic_type = []
	all_label = []
	total_nAtoms = 0
	total_degree = 0
	MIN_nAtoms = int(1e9)
	MAX_nAtoms = 0
	MIN_degree = int(1e9)
	MAX_degree = 0
	MIN_weight = 1e9
	MAX_weight = 0

	for mol in range(nMolecules):
		output.write("Molecule " + str(mol) + "\n")
		
		nAtoms = int(input.readline())
		output.write("Number of atoms in molecule " + str(mol) + ":\n" + str(nAtoms) + "\n")
		MIN_nAtoms = min(MIN_nAtoms, nAtoms)
		MAX_nAtoms = max(MAX_nAtoms, nAtoms)
		total_nAtoms += nAtoms
		
		atomic_type = input.readline().strip()
		output.write("Atomic type for each atom:\n" + atomic_type + "\n")
		
		atomic_type = atomic_type.strip().split(' ')
		for atom in range(nAtoms):
			if not atomic_type[atom] in all_atomic_type:
				all_atomic_type.append(atomic_type[atom])
		
		for atom in range(nAtoms):
			words = input.readline().strip().split(' ')
			degree = int(words[0])
			assert len(words) == 2 * degree + 1
			total_degree += degree
			MIN_degree = min(MIN_degree, degree)
			MAX_degree = max(MAX_degree, degree)
			adj = []
			weight = []
			for i in range(degree):
				adj.append(int(words[2 * i + 1]) - 1) # Atom index must be from 0, not from 1
				weight.append(float(words[2 * i + 2]))
			output.write("Number of neighboring atoms of atom " + str(atom) + ":\n" + str(degree) + "\n")
			output.write("Neighboring atoms of atom " + str(atom) + ":\n")
			for i in range(degree):
				output.write(str(adj[i]) + " ")
			output.write("\n")
			output.write("Weight for each bond:\n")
			for i in range(degree):
				output.write(str(weight[i]) + " ")
				MIN_weight = min(MIN_weight, weight[i])
				MAX_weight = max(MAX_weight, weight[i])
			output.write("\n")
		
		label = input.readline().strip()
		output.write("Label for molecule " + str(mol) + ":\n" + label + "\n")
		if not label in all_label:
			all_label.append(label)

	input.close()
	output.close()

	all_atomic_type.sort()
	all_label.sort()

	assert total_degree % 2 == 0

	meta = open(meta_fn, "w")
	meta.write("Number of molecules:\n" + str(nMolecules) + "\n")
	meta.write("Total number of atoms:\n" + str(total_nAtoms) + "\n")
	meta.write("Total number of bonds:\n" + str(int(total_degree / 2)) + "\n")
	meta.write("Maximum number of atoms in a molecule:\n" + str(MAX_nAtoms) + "\n")
	meta.write("Minimum number of atoms in a molecule:\n" + str(MIN_nAtoms) + "\n")
	meta.write("Maximum degree of an atom:\n" + str(MAX_degree) + "\n")
	meta.write("Minimum degree of an atom:\n" + str(MIN_degree) + "\n")
	meta.write("Maximum weight of a bond:\n" + str(MAX_weight) + "\n")
	meta.write("Mininum weight of a bond:\n" + str(MIN_weight) + "\n")
	meta.write("Average edge ensity:\n" + str(0.5 * total_degree / total_nAtoms) + "\n")
	meta.write("Number of atomic types:\n" + str(len(all_atomic_type)) + "\n")
	meta.write("Atomic types:\n")
	for i in range(len(all_atomic_type)):
		meta.write(all_atomic_type[i] + " ")
	meta.write("\n")
	meta.write("Number of molecular labels:\n" + str(len(all_label)) + "\n")
	meta.write("Molecular labels:\n")
	for i in range(len(all_label)):
		meta.write(all_label[i] + " ")
	meta.write("\n")
	meta.close()

	return nMolecules

def generate_train_test(prefix, nMolecules, num_folds, num_train_folds):
	num_test_folds = num_folds - num_train_folds
	nTrain = int(nMolecules * num_train_folds / num_folds)
	nTest = nMolecules - nTrain

	indices = np.random.permutation(nMolecules)

	start = 0
	for fold in range(num_folds):
		test_fn = prefix + ".test." + str(fold)
		train_fn = prefix + ".train." + str(fold)

		test = open(test_fn, "w")
		test.write("Number of examples:\n" + str(nTest) + "\n")
		test.write("Indices of the examples:\n")
		for i in range(nTest):
			test.write(str(indices[(start + i) % nMolecules]) + "\n")
		test.close()

		train = open(train_fn, "w")
		train.write("Number of examples:\n" + str(nTrain) + "\n")
		train.write("Indices of the examples:\n")
		for i in range(nTrain):
			train.write(str(indices[(start + nTest + i) % nMolecules]) + "\n")
		train.close()

		start = (start + nTest) % nMolecules

def main(argv):
	data_dir = FLAGS.data_dir
	data_name = FLAGS.data_name
	num_folds = FLAGS.num_folds
	num_train_folds = FLAGS.num_train_folds

	assert num_train_folds < num_folds

	input_fn = data_dir + "/" + data_name + ".dat"
	output_fn = data_name + "/" + data_name + ".dat"
	meta_fn = data_name + "/" + data_name + ".meta"

	nMolecules = process(input_fn, output_fn, meta_fn)
	generate_train_test(data_name + "/" + data_name, nMolecules, num_folds, num_train_folds)

	print(data_name)

if __name__ == '__main__':
	app.run(main)