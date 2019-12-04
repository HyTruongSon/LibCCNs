#include "common.h"

const int nContractions = 5;

int get_nContractions() {
	return nContractions;
}

void update_receptive_field(vector<int> &A, const vector<int> &B) {
	for (int i = 0; i < B.size(); ++i) {
		const int v = B[i];
		bool found = false;
		for (int j = 0; j < A.size(); ++j) {
			if (v == A[j]) {
				found = true;
				break;
			}
		}
		if (!found) {
			A.push_back(v);
		}
	}
}

static int Binary_Search(const int u, const int v, const int *input_receptive_field_flat, const int *input_start_index_flat) {
	int l = input_start_index_flat[u];
	int r = input_start_index_flat[u + 1] - 1;
	while (l <= r) {
		const int mid = (l + r) / 2;
		if (v == input_receptive_field_flat[mid]) {
			return mid;
		}
		if (v < input_receptive_field_flat[mid]) {
			r = mid - 1;
		} else {
			l = mid + 1;
		}
	}
	return -1;
}

static void contraction_0_forward(
	const int *input_receptive_field_flat, 
	const int *input_start_index_flat, 
	const int *output_receptive_field_flat, 
	const int *output_start_index_flat,
	const float *input_flat,
	float *output_flat,
	const int input_nChannels,
	const int output_nChannels,
	const int nVertices,
	const int channel,
	const int c) {

	for (int vertex = 0; vertex < nVertices; ++vertex) {
		const int d = output_start_index_flat[vertex + 1] - output_start_index_flat[vertex];
		const int start = output_start_index_flat[vertex];

		for (int i = 0; i < d; ++i) {
			const int output_row = start + i;
			const int u = output_receptive_field_flat[output_row];

			for (int j = 0; j < d; ++j) {	
				const int v = output_receptive_field_flat[start + j];

				const int input_row = Binary_Search(u, v, input_receptive_field_flat, input_start_index_flat);
				if (input_row == -1) {
					continue;
				}

				const int input_index = Index(input_row, c, input_nChannels);
				const int output_index = Index(output_row, channel + c, output_nChannels);

				output_flat[output_index] += input_flat[input_index];
			}
		}
	}
}

static void contraction_1_forward(
	const int *input_receptive_field_flat, 
	const int *input_start_index_flat, 
	const int *output_receptive_field_flat, 
	const int *output_start_index_flat,
	const float *input_flat,
	float *output_flat,
	const int input_nChannels,
	const int output_nChannels,
	const int nVertices,
	const int channel,
	const int c) {

	for (int vertex = 0; vertex < nVertices; ++vertex) {
		const int d = output_start_index_flat[vertex + 1] - output_start_index_flat[vertex];
		const int start = output_start_index_flat[vertex];

		for (int i = 0; i < d; ++i) {
			const int output_row = start + i;
			const int u = output_receptive_field_flat[output_row];

			for (int j = 0; j < d; ++j) {	
				const int v = output_receptive_field_flat[start + j];

				const int input_row = Binary_Search(v, u, input_receptive_field_flat, input_start_index_flat);
				if (input_row == -1) {
					continue;
				}
					
				const int input_index = Index(input_row, c, input_nChannels);
				const int output_index = Index(output_row, channel + c, output_nChannels);

				output_flat[output_index] += input_flat[input_index];
			}
		}
	}
}

static void contraction_2_forward(
	const int *input_receptive_field_flat, 
	const int *input_start_index_flat, 
	const int *output_receptive_field_flat, 
	const int *output_start_index_flat,
	const float *input_flat,
	float *output_flat,
	const int input_nChannels,
	const int output_nChannels,
	const int nVertices,
	const int channel,
	const int c) {

	for (int vertex = 0; vertex < nVertices; ++vertex) {
		const int d = output_start_index_flat[vertex + 1] - output_start_index_flat[vertex];
		const int start = output_start_index_flat[vertex];

		for (int i = 0; i < d; ++i) {
			const int output_row = start + i;
			const int u = output_receptive_field_flat[output_row];
			const int v = u;

			const int input_row = Binary_Search(u, v, input_receptive_field_flat, input_start_index_flat);
			if (input_row == -1) {
				continue;
			}

			const int input_index = Index(input_row, c, input_nChannels);
			const int output_index = Index(output_row, channel + c, output_nChannels);

			output_flat[output_index] += input_flat[input_index];
		}
	}
}

static void contraction_3_forward(
	const int *input_receptive_field_flat, 
	const int *input_start_index_flat, 
	const int *output_receptive_field_flat, 
	const int *output_start_index_flat,
	const float *input_flat,
	float *output_flat,
	const int input_nChannels,
	const int output_nChannels,
	const int nVertices,
	const int channel,
	const int c) {

	for (int vertex = 0; vertex < nVertices; ++vertex) {
		const int d = output_start_index_flat[vertex + 1] - output_start_index_flat[vertex];
		const int start = output_start_index_flat[vertex];

		float sum_diagonal = 0.0;
		for (int i = 0; i < d; ++i) {
			const int output_row = start + i;
			const int u = output_receptive_field_flat[output_row];
			const int v = u;

			const int input_row = Binary_Search(u, v, input_receptive_field_flat, input_start_index_flat);
			if (input_row == -1) {
				continue;
			}

			const int input_index = Index(input_row, c, input_nChannels);
			sum_diagonal += input_flat[input_index];
		}

		for (int i = 0; i < d; ++i) {
			const int output_row = start + i;
			const int output_index = Index(output_row, channel + c, output_nChannels);

			output_flat[output_index] += sum_diagonal;
		}
	}
}

static void contraction_4_forward(
	const int *input_receptive_field_flat, 
	const int *input_start_index_flat, 
	const int *output_receptive_field_flat, 
	const int *output_start_index_flat,
	const float *input_flat,
	float *output_flat,
	const int input_nChannels,
	const int output_nChannels,
	const int nVertices,
	const int channel,
	const int c) {

	for (int vertex = 0; vertex < nVertices; ++vertex) {
		const int d = output_start_index_flat[vertex + 1] - output_start_index_flat[vertex];
		const int start = output_start_index_flat[vertex];

		float sum = 0.0;
		for (int i = 0; i < d; ++i) {
			const int output_row = start + i;
			const int u = output_receptive_field_flat[output_row];

			for (int j = 0; j < d; ++j) {	
				const int v = output_receptive_field_flat[start + j];

				const int input_row = Binary_Search(u, v, input_receptive_field_flat, input_start_index_flat);
				if (input_row == -1) {
					continue;
				}

				const int input_index = Index(input_row, c, input_nChannels);
				sum += input_flat[input_index];
			}
		}

		for (int i = 0; i < d; ++i) {
			const int output_row = start + i;
			const int output_index = Index(output_row, channel + c, output_nChannels);

			output_flat[output_index] += sum;
		}
	}
}

void forward_contractions_cpp(
	const int *input_receptive_field_flat, 
	const int *input_start_index_flat, 
	const int *output_receptive_field_flat, 
	const int *output_start_index_flat,
	const float *input_flat,
	float *output_flat,
	const int input_nChannels,
	const int output_nChannels,
	const int nVertices) {

	// Contraction 0
	// cout << "Contraction 0" << endl;

	int channel = 0 * input_nChannels;

	for (int c = 0; c < input_nChannels; ++c) {
		contraction_0_forward(
			input_receptive_field_flat, 
			input_start_index_flat, 
			output_receptive_field_flat, 
			output_start_index_flat,
			input_flat,
			output_flat,
			input_nChannels,
			output_nChannels,
			nVertices,
			channel,
			c
		);
	}

	// Contraction 1
	// cout << "Contraction 1" << endl;

	channel = 1 * input_nChannels;

	for (int c = 0; c < input_nChannels; ++c) {
		contraction_1_forward(
			input_receptive_field_flat, 
			input_start_index_flat, 
			output_receptive_field_flat, 
			output_start_index_flat,
			input_flat,
			output_flat,
			input_nChannels,
			output_nChannels,
			nVertices,
			channel,
			c
		);
	}

	// Contraction 2
	// cout << "Contraction 2" << endl;

	channel = 2 * input_nChannels;

	for (int c = 0; c < input_nChannels; ++c) {
		contraction_2_forward(
			input_receptive_field_flat, 
			input_start_index_flat, 
			output_receptive_field_flat, 
			output_start_index_flat,
			input_flat,
			output_flat,
			input_nChannels,
			output_nChannels,
			nVertices,
			channel,
			c
		);
	}

	// Contraction 3
	// cout << "Contraction 3" << endl;

	channel = 3 * input_nChannels;

	for (int c = 0; c < input_nChannels; ++c) {
		contraction_3_forward(
			input_receptive_field_flat, 
			input_start_index_flat, 
			output_receptive_field_flat, 
			output_start_index_flat,
			input_flat,
			output_flat,
			input_nChannels,
			output_nChannels,
			nVertices,
			channel,
			c
		);
	}

	// Contraction 4
	// cout << "Contraction 4" << endl;

	channel = 4 * input_nChannels;

	for (int c = 0; c < input_nChannels; ++c) {
		contraction_4_forward(
			input_receptive_field_flat, 
			input_start_index_flat, 
			output_receptive_field_flat, 
			output_start_index_flat,
			input_flat,
			output_flat,
			input_nChannels,
			output_nChannels,
			nVertices,
			channel,
			c
		);
	}
}

void forward_contractions_multi_threading_cpp(
	const int *input_receptive_field_flat, 
	const int *input_start_index_flat, 
	const int *output_receptive_field_flat, 
	const int *output_start_index_flat,
	const float *input_flat,
	float *output_flat,
	const int input_nChannels,
	const int output_nChannels,
	const int nVertices,
	const int nThreads) {

	assert(nThreads >= 1);
	std::thread *job = new std::thread [nThreads];

	// Contraction 0
	// cout << "Contraction 0" << endl;

	int channel = 0 * input_nChannels;

	int start = 0;
	while (start < input_nChannels) {
		const int finish = min(start + nThreads, input_nChannels) - 1;

		for (int c = start; c <= finish; ++c) {
			job[c - start] = std::thread(
				contraction_0_forward,
				input_receptive_field_flat, 
				input_start_index_flat, 
				output_receptive_field_flat, 
				output_start_index_flat,
				input_flat,
				output_flat,
				input_nChannels,
				output_nChannels,
				nVertices,
				channel,
				c
			);
		}

		for (int c = start; c <= finish; ++c) {
			job[c - start].join();
		}

		start = finish + 1;
	}

	// Contraction 1
	// cout << "Contraction 1" << endl;

	channel = 1 * input_nChannels;

	start = 0;
	while (start < input_nChannels) {
		const int finish = min(start + nThreads, input_nChannels) - 1;

		for (int c = start; c <= finish; ++c) {
			job[c - start] = std::thread(
				contraction_1_forward,
				input_receptive_field_flat, 
				input_start_index_flat, 
				output_receptive_field_flat, 
				output_start_index_flat,
				input_flat,
				output_flat,
				input_nChannels,
				output_nChannels,
				nVertices,
				channel,
				c
			);
		}

		for (int c = start; c <= finish; ++c) {
			job[c - start].join();
		}

		start = finish + 1;
	}

	// Contraction 2
	// cout << "Contraction 2" << endl;

	channel = 2 * input_nChannels;

	start = 0;
	while (start < input_nChannels) {
		const int finish = min(start + nThreads, input_nChannels) - 1;

		for (int c = start; c <= finish; ++c) {
			job[c - start] = std::thread(
				contraction_2_forward,
				input_receptive_field_flat, 
				input_start_index_flat, 
				output_receptive_field_flat, 
				output_start_index_flat,
				input_flat,
				output_flat,
				input_nChannels,
				output_nChannels,
				nVertices,
				channel,
				c
			);
		}

		for (int c = start; c <= finish; ++c) {
			job[c - start].join();
		}

		start = finish + 1;
	}

	// Contraction 3
	// cout << "Contraction 3" << endl;

	channel = 3 * input_nChannels;

	start = 0;
	while (start < input_nChannels) {
		const int finish = min(start + nThreads, input_nChannels) - 1;

		for (int c = start; c <= finish; ++c) {
			job[c - start] = std::thread(
				contraction_3_forward,
				input_receptive_field_flat, 
				input_start_index_flat, 
				output_receptive_field_flat, 
				output_start_index_flat,
				input_flat,
				output_flat,
				input_nChannels,
				output_nChannels,
				nVertices,
				channel,
				c
			);
		}

		for (int c = start; c <= finish; ++c) {
			job[c - start].join();
		}

		start = finish + 1;
	}

	// Contraction 4
	// cout << "Contraction 4" << endl;

	channel = 4 * input_nChannels;

	start = 0;
	while (start < input_nChannels) {
		const int finish = min(start + nThreads, input_nChannels) - 1;

		for (int c = start; c <= finish; ++c) {
			job[c - start] = std::thread(
				contraction_4_forward,
				input_receptive_field_flat, 
				input_start_index_flat, 
				output_receptive_field_flat, 
				output_start_index_flat,
				input_flat,
				output_flat,
				input_nChannels,
				output_nChannels,
				nVertices,
				channel,
				c
			);
		}

		for (int c = start; c <= finish; ++c) {
			job[c - start].join();
		}

		start = finish + 1;
	}

	delete[] job;
}

static void contraction_0_backward(
	const int *input_receptive_field_flat, 
	const int *input_start_index_flat, 
	const int *output_receptive_field_flat, 
	const int *output_start_index_flat,
	float *input_grad_flat,
	const float *output_grad_flat,
	const int input_nChannels,
	const int output_nChannels,
	const int nVertices,
	const int channel,
	const int c) {

	for (int vertex = 0; vertex < nVertices; ++vertex) {
		const int d = output_start_index_flat[vertex + 1] - output_start_index_flat[vertex];
		const int start = output_start_index_flat[vertex];

		for (int i = 0; i < d; ++i) {
			const int output_row = start + i;
			const int u = output_receptive_field_flat[output_row];

			for (int j = 0; j < d; ++j) {	
				const int v = output_receptive_field_flat[start + j];

				const int input_row = Binary_Search(u, v, input_receptive_field_flat, input_start_index_flat);
				if (input_row == -1) {
					continue;
				}

				const int input_index = Index(input_row, c, input_nChannels);
				const int output_index = Index(output_row, channel + c, output_nChannels);

				input_grad_flat[input_index] += output_grad_flat[output_index];
			}
		}
	}
}

static void contraction_1_backward(
	const int *input_receptive_field_flat, 
	const int *input_start_index_flat, 
	const int *output_receptive_field_flat, 
	const int *output_start_index_flat,
	float *input_grad_flat,
	const float *output_grad_flat,
	const int input_nChannels,
	const int output_nChannels,
	const int nVertices,
	const int channel,
	const int c) {

	for (int vertex = 0; vertex < nVertices; ++vertex) {
		const int d = output_start_index_flat[vertex + 1] - output_start_index_flat[vertex];
		const int start = output_start_index_flat[vertex];

		for (int i = 0; i < d; ++i) {
			const int output_row = start + i;
			const int u = output_receptive_field_flat[output_row];

			for (int j = 0; j < d; ++j) {	
				const int v = output_receptive_field_flat[start + j];

				const int input_row = Binary_Search(v, u, input_receptive_field_flat, input_start_index_flat);
				if (input_row == -1) {
					continue;
				}
					
				const int input_index = Index(input_row, c, input_nChannels);
				const int output_index = Index(output_row, channel + c, output_nChannels);

				input_grad_flat[input_index] += output_grad_flat[output_index];
			}
		}
	}
}

static void contraction_2_backward(
	const int *input_receptive_field_flat, 
	const int *input_start_index_flat, 
	const int *output_receptive_field_flat, 
	const int *output_start_index_flat,
	float *input_grad_flat,
	const float *output_grad_flat,
	const int input_nChannels,
	const int output_nChannels,
	const int nVertices,
	const int channel,
	const int c) {

	for (int vertex = 0; vertex < nVertices; ++vertex) {
		const int d = output_start_index_flat[vertex + 1] - output_start_index_flat[vertex];
		const int start = output_start_index_flat[vertex];

		for (int i = 0; i < d; ++i) {
			const int output_row = start + i;
			const int u = output_receptive_field_flat[output_row];
			const int v = u;

			const int input_row = Binary_Search(u, v, input_receptive_field_flat, input_start_index_flat);
			if (input_row == -1) {
				continue;
			}

			const int input_index = Index(input_row, c, input_nChannels);
			const int output_index = Index(output_row, channel + c, output_nChannels);

			input_grad_flat[input_index] += output_grad_flat[output_index];
		}
	}
}

static void contraction_3_backward(
	const int *input_receptive_field_flat, 
	const int *input_start_index_flat, 
	const int *output_receptive_field_flat, 
	const int *output_start_index_flat,
	float *input_grad_flat,
	const float *output_grad_flat,
	const int input_nChannels,
	const int output_nChannels,
	const int nVertices,
	const int channel,
	const int c) {

	for (int vertex = 0; vertex < nVertices; ++vertex) {
		const int d = output_start_index_flat[vertex + 1] - output_start_index_flat[vertex];
		const int start = output_start_index_flat[vertex];

		float sum_diagonal = 0.0;
		for (int i = 0; i < d; ++i) {
			const int output_row = start + i;
			const int output_index = Index(output_row, channel + c, output_nChannels);

			sum_diagonal += output_grad_flat[output_index];
		}

		for (int i = 0; i < d; ++i) {
			const int output_row = start + i;
			const int u = output_receptive_field_flat[output_row];
			const int v = u;

			const int input_row = Binary_Search(u, v, input_receptive_field_flat, input_start_index_flat);
			if (input_row == -1) {
				continue;
			}

			const int input_index = Index(input_row, c, input_nChannels);
			input_grad_flat[input_index] += sum_diagonal;
		}
	}
}

static void contraction_4_backward(
	const int *input_receptive_field_flat, 
	const int *input_start_index_flat, 
	const int *output_receptive_field_flat, 
	const int *output_start_index_flat,
	float *input_grad_flat,
	const float *output_grad_flat,
	const int input_nChannels,
	const int output_nChannels,
	const int nVertices,
	const int channel,
	const int c) {

	for (int vertex = 0; vertex < nVertices; ++vertex) {
		const int d = output_start_index_flat[vertex + 1] - output_start_index_flat[vertex];
		const int start = output_start_index_flat[vertex];

		float sum = 0.0;
		for (int i = 0; i < d; ++i) {
			const int output_row = start + i;
			const int output_index = Index(output_row, channel + c, output_nChannels);

			sum += output_grad_flat[output_index];
		}

		for (int i = 0; i < d; ++i) {
			const int output_row = start + i;
			const int u = output_receptive_field_flat[output_row];

			for (int j = 0; j < d; ++j) {	
				const int v = output_receptive_field_flat[start + j];

				const int input_row = Binary_Search(u, v, input_receptive_field_flat, input_start_index_flat);
				if (input_row == -1) {
					continue;
				}

				const int input_index = Index(input_row, c, input_nChannels);
				input_grad_flat[input_index] += sum;
			}
		}
	}
}

void backward_contractions_cpp(
	const int *input_receptive_field_flat, 
	const int *input_start_index_flat, 
	const int *output_receptive_field_flat, 
	const int *output_start_index_flat,
	float *input_grad_flat,
	const float *output_grad_flat,
	const int input_nChannels,
	const int output_nChannels,
	const int nVertices) {

	// Contraction 0
	int channel = 0 * input_nChannels;

	for (int c = 0; c < input_nChannels; ++c) {
		contraction_0_backward(
			input_receptive_field_flat, 
			input_start_index_flat, 
			output_receptive_field_flat, 
			output_start_index_flat,
			input_grad_flat,
			output_grad_flat,
			input_nChannels,
			output_nChannels,
			nVertices,
			channel,
			c
		);
	}

	// Contraction 1
	channel = 1 * input_nChannels;

	for (int c = 0; c < input_nChannels; ++c) {
		contraction_1_backward(
			input_receptive_field_flat, 
			input_start_index_flat, 
			output_receptive_field_flat, 
			output_start_index_flat,
			input_grad_flat,
			output_grad_flat,
			input_nChannels,
			output_nChannels,
			nVertices,
			channel,
			c
		);
	}

	// Contraction 2
	channel = 2 * input_nChannels;

	for (int c = 0; c < input_nChannels; ++c) {
		contraction_2_backward(
			input_receptive_field_flat, 
			input_start_index_flat, 
			output_receptive_field_flat, 
			output_start_index_flat,
			input_grad_flat,
			output_grad_flat,
			input_nChannels,
			output_nChannels,
			nVertices,
			channel,
			c
		);
	}

	// Contraction 3
	channel = 3 * input_nChannels;

	for (int c = 0; c < input_nChannels; ++c) {
		contraction_3_backward(
			input_receptive_field_flat, 
			input_start_index_flat, 
			output_receptive_field_flat, 
			output_start_index_flat,
			input_grad_flat,
			output_grad_flat,
			input_nChannels,
			output_nChannels,
			nVertices,
			channel,
			c
		);
	}

	// Contraction 4
	channel = 4 * input_nChannels;

	for (int c = 0; c < input_nChannels; ++c) {
		contraction_4_backward(
			input_receptive_field_flat, 
			input_start_index_flat, 
			output_receptive_field_flat, 
			output_start_index_flat,
			input_grad_flat,
			output_grad_flat,
			input_nChannels,
			output_nChannels,
			nVertices,
			channel,
			c
		);
	}
}

void backward_contractions_multi_threading_cpp(
	const int *input_receptive_field_flat, 
	const int *input_start_index_flat, 
	const int *output_receptive_field_flat, 
	const int *output_start_index_flat,
	float *input_grad_flat,
	const float *output_grad_flat,
	const int input_nChannels,
	const int output_nChannels,
	const int nVertices,
	const int nThreads) {

	assert(nThreads >= 1);
	std::thread *job = new std::thread [nThreads];

	// Contraction 0
	int channel = 0 * input_nChannels;

	int start = 0;
	while (start < input_nChannels) {
		const int finish = min(start + nThreads, input_nChannels) - 1;

		for (int c = start; c <= finish; ++c) {
			job[c - start] = std::thread(
				contraction_0_backward,
				input_receptive_field_flat, 
				input_start_index_flat, 
				output_receptive_field_flat, 
				output_start_index_flat,
				input_grad_flat,
				output_grad_flat,
				input_nChannels,
				output_nChannels,
				nVertices,
				channel,
				c
			);
		}

		for (int c = start; c <= finish; ++c) {
			job[c - start].join();
		}

		start = finish + 1;
	}

	// Contraction 1
	channel = 1 * input_nChannels;

	start = 0;
	while (start < input_nChannels) {
		const int finish = min(start + nThreads, input_nChannels) - 1;

		for (int c = start; c <= finish; ++c) {
			job[c - start] = std::thread(
				contraction_1_backward,
				input_receptive_field_flat, 
				input_start_index_flat, 
				output_receptive_field_flat, 
				output_start_index_flat,
				input_grad_flat,
				output_grad_flat,
				input_nChannels,
				output_nChannels,
				nVertices,
				channel,
				c
			);
		}

		for (int c = start; c <= finish; ++c) {
			job[c - start].join();
		}

		start = finish + 1;
	}

	// Contraction 2
	channel = 2 * input_nChannels;

	start = 0;
	while (start < input_nChannels) {
		const int finish = min(start + nThreads, input_nChannels) - 1;

		for (int c = start; c <= finish; ++c) {
			job[c - start] = std::thread(
				contraction_2_backward,
				input_receptive_field_flat, 
				input_start_index_flat, 
				output_receptive_field_flat, 
				output_start_index_flat,
				input_grad_flat,
				output_grad_flat,
				input_nChannels,
				output_nChannels,
				nVertices,
				channel,
				c
			);
		}

		for (int c = start; c <= finish; ++c) {
			job[c - start].join();
		}

		start = finish + 1;
	}

	// Contraction 3
	channel = 3 * input_nChannels;

	start = 0;
	while (start < input_nChannels) {
		const int finish = min(start + nThreads, input_nChannels) - 1;

		for (int c = start; c <= finish; ++c) {
			job[c - start] = std::thread(
				contraction_3_backward,
				input_receptive_field_flat, 
				input_start_index_flat, 
				output_receptive_field_flat, 
				output_start_index_flat,
				input_grad_flat,
				output_grad_flat,
				input_nChannels,
				output_nChannels,
				nVertices,
				channel,
				c
			);
		}

		for (int c = start; c <= finish; ++c) {
			job[c - start].join();
		}

		start = finish + 1;
	}

	// Contraction 4
	channel = 4 * input_nChannels;

	start = 0;
	while (start < input_nChannels) {
		const int finish = min(start + nThreads, input_nChannels) - 1;

		for (int c = start; c <= finish; ++c) {
			job[c - start] = std::thread(
				contraction_4_backward,
				input_receptive_field_flat, 
				input_start_index_flat, 
				output_receptive_field_flat, 
				output_start_index_flat,
				input_grad_flat,
				output_grad_flat,
				input_nChannels,
				output_nChannels,
				nVertices,
				channel,
				c
			);
		}

		for (int c = start; c <= finish; ++c) {
			job[c - start].join();
		}

		start = finish + 1;
	}

	delete[] job;
}

static void forward_shrinking_job(
	const int *start_index_flat,
	const float *input_flat,
	float *output_flat,
	const int nVertices,
	const int nChannels,
	const int c
) {
	for (int vertex = 0; vertex < nVertices; ++vertex) {
		float sum = 0.0;
		for (int row = start_index_flat[vertex]; row < start_index_flat[vertex + 1]; ++row) {
			sum += input_flat[Index(row, c, nChannels)];
		}
		output_flat[Index(vertex, c, nChannels)] += sum;
	}
}

void forward_shrinking_cpp(
	const int *start_index_flat,
	const float *input_flat,
	float *output_flat,
	const int nVertices,
	const int nChannels) {

	for (int c = 0; c < nChannels; ++c) {
		forward_shrinking_job(
			start_index_flat,
			input_flat,
			output_flat,
			nVertices,
			nChannels,
			c
		);
	}
}

void forward_shrinking_multi_threading_cpp(
	const int *start_index_flat,
	const float *input_flat,
	float *output_flat,
	const int nVertices,
	const int nChannels,
	const int nThreads) {

	assert(nThreads >= 1);
	std::thread *job = new std::thread [nThreads];

	int start = 0;
	while (start < nChannels) {
		const int finish = min(start + nThreads, nChannels) - 1;

		for (int c = start; c <= finish; ++c) {
			job[c - start] = std::thread(
				forward_shrinking_job,
				start_index_flat,
				input_flat,
				output_flat,
				nVertices,
				nChannels,
				c
			);
		}

		for (int c = start; c <= finish; ++c) {
			job[c - start].join();
		}

		start = finish + 1;
	}

	delete[] job;
}

static void backward_shrinking_job(
	const int *start_index_flat,
	float *input_grad_flat,
	const float *output_grad_flat,
	const int nVertices,
	const int nChannels,
	const int c
) {
	for (int vertex = 0; vertex < nVertices; ++vertex) {
		float grad = output_grad_flat[Index(vertex, c, nChannels)];
		for (int row = start_index_flat[vertex]; row < start_index_flat[vertex + 1]; ++row) {
			input_grad_flat[Index(row, c, nChannels)] += grad;
		}		
	}
}

void backward_shrinking_cpp(
	const int *start_index_flat,
	float *input_grad_flat,
	const float *output_grad_flat,
	const int nVertices,
	const int nChannels) {

	for (int c = 0; c < nChannels; ++c) {
		backward_shrinking_job(
			start_index_flat,
			input_grad_flat,
			output_grad_flat,
			nVertices,
			nChannels,
			c
		);
	}
}

void backward_shrinking_multi_threading_cpp(
	const int *start_index_flat,
	float *input_grad_flat,
	const float *output_grad_flat,
	const int nVertices,
	const int nChannels,
	const int nThreads) {

	assert(nThreads >= 1);
	std::thread *job = new std::thread [nThreads];

	int start = 0;
	while (start < nChannels) {
		const int finish = min(start + nThreads, nChannels) - 1;

		for (int c = start; c <= finish; ++c) {
			job[c - start] = std::thread(
				backward_shrinking_job,
				start_index_flat,
				input_grad_flat,
				output_grad_flat,
				nVertices,
				nChannels,
				c
			);
		}

		for (int c = start; c <= finish; ++c) {
			job[c - start].join();
		}

		start = finish + 1;
	}

	delete[] job;
}

static void forward_normalizing_job(
	const int *start_index_flat,
	const float *input_flat,
	float *output_flat,
	const int nVertices,
	const int nChannels,
	const int c
) {
	for (int vertex = 0; vertex < nVertices; ++vertex) {
		if (start_index_flat[vertex] == start_index_flat[vertex + 1]) {
			continue;
		}
		float d = start_index_flat[vertex + 1] - start_index_flat[vertex];
		for (int row = start_index_flat[vertex]; row < start_index_flat[vertex + 1]; ++row) {
			output_flat[Index(row, c, nChannels)] += input_flat[Index(row, c, nChannels)] / d;
		}
	}
}

void forward_normalizing_cpp(
	const int *start_index_flat,
	const float *input_flat,
	float *output_flat,
	const int nVertices,
	const int nChannels) {

	for (int c = 0; c < nChannels; ++c) {
		forward_normalizing_job(
			start_index_flat,
			input_flat,
			output_flat,
			nVertices,
			nChannels,
			c
		);
	}
}

static void backward_normalizing_job(
	const int *start_index_flat,
	float *input_grad_flat,
	const float *output_grad_flat,
	const int nVertices,
	const int nChannels,
	const int c
) {
	for (int vertex = 0; vertex < nVertices; ++vertex) {
		if (start_index_flat[vertex] == start_index_flat[vertex + 1]) {
			continue;
		}
		float d = start_index_flat[vertex + 1] - start_index_flat[vertex];
		for (int row = start_index_flat[vertex]; row < start_index_flat[vertex + 1]; ++row) {
			input_grad_flat[Index(row, c, nChannels)] += output_grad_flat[Index(row, c, nChannels)] / d;
		}
	}
}

void backward_normalizing_cpp(
	const int *start_index_flat,
	float *input_grad_flat,
	const float *output_grad_flat,
	const int nVertices,
	const int nChannels) {

	for (int c = 0; c < nChannels; ++c) {
		backward_normalizing_job(
			start_index_flat,
			input_grad_flat,
			output_grad_flat,
			nVertices,
			nChannels,
			c
		);
	}
}