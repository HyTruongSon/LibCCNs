// Framework: GraphFlow
// Author: Machine Learning Group of UChicago
// Main Contributor: Hy Truong Son
// Institution: Department of Computer Science, The University of Chicago
// Copyright 2017 (c) UChicago. All rights reserved.

#include <iostream>
#include <cstring>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <ctime>
#include <sys/time.h>

#include "GraphFlow_gpu/Entity.h"
#include "GraphFlow_gpu/Vector.h"
#include "GraphFlow_gpu/Matrix.h"
#include "GraphFlow_gpu/Tensor3D.h"
#include "GraphFlow_gpu/StackTensor3D.h"
#include "GraphFlow_gpu/Tensor4D.h"
#include "GraphFlow_gpu/RisiContraction_18.h"

using namespace std;

int N;
int nChanels;
const int nContractions = RisiContraction_18::nContractions;

const double epsilon = 1e-8;

const double RANDOM_SEED = 123456789;

// Get the millisecond
void time_ms(long int &ms) {
	struct timeval tp;
	gettimeofday(&tp, NULL);
	ms = tp.tv_sec * 1000 + tp.tv_usec / 1000;
}

// Difference in milliseconds
long int difftime_ms(long int &end, long int &start) {
	return end - start;
}

int get_random(int number) {
	return rand() % number;
}

bool compare_tensor(Tensor3D *A, Tensor3D *B) {
	if (A -> nRows != B -> nRows) {
		return false;
	}
	if (A -> nColumns != B -> nColumns) {
		return false;
	}
	if (A -> nDepth != B -> nDepth) {
		return false;
	}
	if (A -> size != B -> size) {
		return false;
	}
	for (int i = 0; i < A -> size; ++i) {
		if (abs(A -> value[i] != B -> value[i]) > epsilon) {
			return false;
		}
	}
	return true;
}

int main(int argc, char **argv) {
	if (argc < 3) {
		cerr << "Not enough parameters!" << endl;
		return 0;
	} 

	N = atoi(argv[1]);
	nChanels = atoi(argv[2]);

	cout << "-------------------------------------------------------" << endl;
	cout << "N = " << N << endl;
	cout << "nChanels = " << nChanels << endl;

	srand(RANDOM_SEED);

	long int start;
	long int end;

	RisiContraction_18 *ground_truth = new RisiContraction_18(N, nChanels);
	StackTensor3D *stack = new StackTensor3D(N, N, N, nChanels);
	
	// Generate tensors
	Tensor3D **tensors = new Tensor3D* [N];
	for (int i = 0; i < N; ++i) {
		tensors[i] = new Tensor3D(N, N, nChanels);
		for (int chanel = 0; chanel < nChanels; ++chanel) {
			for (int row = 0; row < N; ++row) {
				for (int column = row; column < N; ++column) {
					int index = tensors[i] -> index(row, column, chanel);
					int index_ = tensors[i] -> index(column, row, chanel);

					int random = get_random(10); 
					tensors[i] -> value[index] = random;
					tensors[i] -> value[index_] = random; 
				}
			}
		}
	}	

	// Generate the adjacency matrix
	Matrix *adj = new Matrix(N, N);
	for (int i = 0; i < N; ++i) {
		adj -> value[adj -> index(i, i)] = 1;
		for (int j = i + 1; j < N; ++j) {
			int random = get_random(2);
			adj -> value[adj -> index(i, j)] = random;
			adj -> value[adj -> index(j, i)] = random;
		}
	}

	// Similuation for RisiContraction_18_gpu
	stack -> clear();
	for (int i = 0; i < N; ++i) {
		stack -> add_tensor(tensors[i]);
	}

	ground_truth -> clear();
	for (int i = 0; i < N; ++i) {
		ground_truth -> add_tensor(tensors[i]);
	}
	ground_truth -> set_adjacency(adj);

	// Forward pass
	stack -> forward();

	time_ms(start);
	ground_truth -> forward();
	time_ms(end);

	cout << "CPU forward time: " << difftime_ms(end, start) << endl;

	// Randomized the gradient
	for (int i = 0; i < ground_truth -> size; ++i) {
		ground_truth -> gradient[i] = get_random(100); 
	}

	time_ms(start);
	ground_truth -> backward();
	time_ms(end);

	cout << "CPU backward time: " << difftime_ms(end, start) << endl;

	return 0;
}