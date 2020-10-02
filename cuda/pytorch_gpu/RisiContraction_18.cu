#include <iostream>
#include <cstring>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <thread>
#include <assert.h>

#include <torch/torch.h>

using namespace std;

// +----------------------------+
// | Additional Functionalities |
// +----------------------------+

// Number of contractions
const int NCONTRACTIONS = 18;

// Ceiling
int rounded_division(int number1, int number2) {
	if (number1 % number2 == 0) {
		return number1 / number2;
	}
	return number1 / number2 + 1;
}

// +-------------------------------------------+
// | Atomic Addition Operation For Double Type |
// +-------------------------------------------+ 

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
	static __inline__ __device__ double atomicAdd(double *address, double val) {
		unsigned long long int* address_as_ull = (unsigned long long int*) address;
		unsigned long long int old = *address_as_ull, assumed;
		if (val == 0.0) {
			return __longlong_as_double(old);
		}
		do {
			assumed = old;
			old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val +__longlong_as_double(assumed)));
		} while (assumed != old);
		return __longlong_as_double(old);
	}
#endif

// +-------------------------------------+
// | Kernel Function For The Forward Job |
// +-------------------------------------+

__global__ void RisiContraction_18_forward_job(double *tensor, double *adj, double *value, int N, int nChannels) {
	__shared__ int nContractions;
	__shared__ int A;
	__shared__ int B;
	__shared__ int C;
	__shared__ int Y;

	nContractions = NCONTRACTIONS;

	int global_threadId = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (global_threadId < N * N * nChannels * nContractions) {	
		C = nChannels;
		B = N * C;
		A = N * B;

		Y = nChannels * nContractions;
		
		int f = (global_threadId % Y) % nChannels;
		int Case = (global_threadId % Y) / nChannels + 1;
		int y = (global_threadId / Y) % N;
		int x = (global_threadId / Y) / N;

		int a, b, c, d, e;
		double adj_value;

		double sum = 0.0;

		// +-----------+
		// | 1 + 1 + 1 |
		// +-----------+

		// Case 1 (1/50): Fix a, b. Contract c, d, e.
		if (Case == 1) {
			a = x;
			b = y;

			for (d = 0; d < N; ++d) {
				for (e = 0; e < N; ++e) {
					adj_value = adj[d * N + e];
					if (adj_value > 0) {
						for (c = 0; c < N; ++c) {
							sum += tensor[a * A + b * B + c * C + f] * adj_value;
						}
					}
				}
			}
		}
				
		// Case 2 (3/50): Fix a, d. Contract b, c, e.
		if (Case == 2) {		
			a = x;
			d = y;

			for (e = 0; e < N; ++e) {
				adj_value = adj[d * N + e];
				if (adj_value > 0) {
					for (b = 0; b < N; ++b) {
						for (c = 0; c < N; ++c) {
							sum += tensor[a * A + b * B + c * C + f] * adj_value;
						}
					}
				}
			}	
		}
		
		// Case 3 (5/50): Fix b, c. Contract a, d, e.
		if (Case == 3) {		
			b = x;
			c = y;

			for (d = 0; d < N; ++d) {
				for (e = 0; e < N; ++e) {
					adj_value = adj[d * N + e];
					if (adj_value > 0) {
						for (a = 0; a < N; ++a) {
							sum += tensor[a * A + b * B + c * C + f] * adj_value;
						}
					}
				}
			}	
		}

		// Case 4 (6/50): Fix b, d. Contract a, c, e.
		if (Case == 4) {
			b = x;
			d = y;

			for (e = 0; e < N; ++e) {
				adj_value = adj[d * N + e];
				if (adj_value > 0) {
					for (a = 0; a < N; ++a) {
						for (c = 0; c < N; ++c) {
							sum += tensor[a * A + b * B + c * C + f] * adj_value;
						}
					}
				}
			}
		}

		// Case 5 (10/50): Fix d, e. Contract a, b, c.
		if (Case == 5) {		
			d = x;
			e = y;

			adj_value = adj[d * N + e];
			if (adj_value > 0) {
				for (a = 0; a < N; ++a) {
					for (b = 0; b < N; ++b) {
						for (c = 0; c < N; ++c) {
							sum += tensor[a * A + b * B + c * C + f] * adj_value;
						}
					}
				}
			}
		}

		// +-------+
		// | 1 + 2 |
		// +-------+

		// Case 6 (11/50): (a, b). Contract (c, d). Singleton (e).
		if (Case == 6) {
			a = x;
			b = y;

			for (d = 0; d < N; ++d) {
				for (e = 0; e < N; ++e) {
					adj_value = adj[d * N + e];
					c = d;
					sum += tensor[a * A + b * B + c * C + f] * adj_value;
				}
			}
		}

		// Case 7 (13/50): (a, b). Contract (d, e). Singleton (c).
		if (Case == 7) {
			a = x;
			b = y;

			for (d = 0; d < N; ++d) {
				e = d;
				adj_value = adj[d * N + e];
				if (adj_value > 0) {
					for (c = 0; c < N; ++c) {
						sum += tensor[a * A + b * B + c * C + f] * adj_value;
					}
				}
			}
		}

		// Case 8 (17/50): (a, d). Contract (b, c). Singleton (e).
		if (Case == 8) {
			a = x;
			d = y;

			for (e = 0; e < N; ++e) {
				adj_value = adj[d * N + e];
				if (adj_value > 0) {
					for (b = 0; b < N; ++b) {
						c = b;
						sum += tensor[a * A + b * B + c * C + f] * adj_value;
					}
				}
			}
		}

		// Case 9 (18/50): (a, d). Contract (b, e). Singleton (c).
		if (Case == 9) {
			a = x;
			d = y;

			for (e = 0; e < N; ++e) {
				adj_value = adj[d * N + e];
				if (adj_value > 0) {
					b = e;
					for (c = 0; c < N; ++c) {
						sum += tensor[a * A + b * B + c * C + f] * adj_value;
					}
				}
			}
		}

		// Case 10 (23/50): (b, c). Contract (a, d). Singleton (e).
		if (Case == 10) {
			b = x;
			c = y;

			for (d = 0; d < N; ++d) {
				for (e = 0; e < N; ++e) {
					adj_value = adj[d * N + e];
					if (adj_value > 0) {
						a = d;
						sum += tensor[a * A + b * B + c * C + f] * adj_value;
					}
				}
			}
		}

		// Case 11 (26/50): (b, d). Contract (a, c). Singleton (e).
		if (Case == 11) {
			b = x;
			d = y;

			for (e = 0; e < N; ++e) {
				adj_value = adj[d * N + e];
				if (adj_value > 0) {
					for (a = 0; a < N; ++a) {
						c = a;
						sum += tensor[a * A + b * B + c * C + f] * adj_value;
					}
				}
			}
		}

		// Case 12 (27/50): (b, d). Contract (a, e). Singleton (c).
		if (Case == 12) {
			b = x;
			d = y;

			for (e = 0; e < N; ++e) {
				adj_value = adj[d * N + e];
				if (adj_value > 0) {
					a = e;
					for (int c = 0; c < N; ++c) {
						sum += tensor[a * A + b * B + c * C + f] * adj_value;
					}
				}
			}
		}

		// Case 13 (28/50): (b, d). Contract (c, e). Singleton (a).
		if (Case == 13) {
			b = x;
			d = y;

			for (e = 0; e < N; ++e) {
				adj_value = adj[d * N + e];
				if (adj_value > 0) {
					c = e;
					for (int a = 0; a < N; ++a) {
						sum += tensor[a * A + b * B + c * C + f] * adj_value;
					}
				}
			}
		}

		// Case 14 (38/50): (d, e). Contract (a, b). Singleton (c).
		if (Case == 14) {
			d = x;
			e = y;

			adj_value = adj[d * N + e];
			if (adj_value > 0) {
				for (int a = 0; a < N; ++a) {
					b = a;
					for (int c = 0; c < N; ++c) {
						sum += tensor[a * A + b * B + c * C + f] * adj_value;
					}
				}
			}
		}

		// Case 15 (40/50): (d, e). Contract (b, c). Singleton (a).
		if (Case == 15) {
			d = x;
			e = y;

			adj_value = adj[d * N + e];
			if (adj_value > 0) {
				for (int b = 0; b < N; ++b) {
					c = b;
					for (int a = 0; a < N; ++a) {
						sum += tensor[a * A + b * B + c * C + f] * adj_value;
					}
				}
			}
		}

		// +---+
		// | 3 |
		// +---+

		// Case 16 (43/50): (a, d). Contract (b, c, e).
		if (Case == 16) {
			a = x;
			d = y;

			for (int e = 0; e < N; ++e) {
				adj_value = adj[d * N + e];
				if (adj_value > 0) {
					b = e;
					c = e;
					sum += tensor[a * A + b * B + c * C + f] * adj_value;
				}
			}
		}	

		// Case 17 (46/50): (b, d). Contract (a, c, e).
		if (Case == 17) {
			b = x;
			d = y;

			for (int e = 0; e < N; ++e) {
				adj_value = adj[d * N + e];
				if (adj_value > 0) {
					a = e;
					c = e;
					sum += tensor[a * A + b * B + c * C + f] * adj_value;
				}
			}
		}

		// Case 18 (50/50): (d, e). Contract (a, b, c).
		if (Case == 18) {
			d = x;
			e = y;

			adj_value = adj[d * N + e];
			if (adj_value > 0) {
				for (int a = 0; a < N; ++a) {
					b = a;
					c = a;
					sum += tensor[a * A + b * B + c * C + f] * adj_value;
				}
			}
		}

		value[global_threadId] = sum;
	}
}

// +--------------------------------------+
// | Kernel Function For The Backward Job |
// +--------------------------------------+

__global__ void RisiContraction_18_backward_job(double *tensor_gradient, double *adj, double *gradient, int N, int nChannels) {
	
	__shared__ int nContractions;
	__shared__ int X;
	__shared__ int Y;

	nContractions = NCONTRACTIONS;

	int global_threadId = blockIdx.x * blockDim.x + threadIdx.x;

	if (global_threadId < N * N * N * nChannels) {
		X = N * nChannels * nContractions;
		Y = nChannels * nContractions;

		int f = global_threadId % nChannels;
		int c = (global_threadId / nChannels) % N;
		int b = ((global_threadId / nChannels) / N) % N;
		int a = ((global_threadId / nChannels) / N) / N;

		double sum = 0.0;

		int ind;
		double adj_value;

		for (int d = 0; d < N; ++d) {
			for (int e = 0; e < N; ++e) {
				adj_value = adj[d * N + e];

				if (adj_value > 0) {
					// +-----------+
					// | 1 + 1 + 1 |
					// +-----------+

					// Case 1 (1/50): Fix a, b. Contract c, d, e.
					ind = a * X + b * Y + 0 * nChannels + f;
					sum += gradient[ind] * adj_value;

					// Case 2 (3/50): Fix a, d. Contract b, c, e.
					ind = a * X + d * Y + 1 * nChannels + f;
					sum += gradient[ind] * adj_value;

					// Case 3 (5/50): Fix b, c. Contract a, d, e.
					ind = b * X + c * Y + 2 * nChannels + f;
					sum += gradient[ind] * adj_value;

					// Case 4 (6/50): Fix b, d. Contract a, c, e.
					ind = b * X + d * Y + 3 * nChannels + f;
					sum += gradient[ind] * adj_value;

					// Case 5 (10/50): Fix d, e. Contract a, b, c.
					ind = d * X + e * Y + 4 * nChannels + f;
					sum += gradient[ind] * adj_value;

					// +-------+
					// | 1 + 2 |
					// +-------+

					// Case 6 (11/50): (a, b). Contract (c, d). Singleton (e).
					if (c == d) {
						ind = a * X + b * Y + 5 * nChannels + f;
						sum += gradient[ind] * adj_value;
					}

					// Case 7 (13/50): (a, b). Contract (d, e). Singleton (c).
					if (d == e) {
						ind = a * X + b * Y + 6 * nChannels + f;
						sum += gradient[ind] * adj_value;
					}

					// Case 8 (17/50): (a, d). Contract (b, c). Singleton (e).
					if (b == c) {
						ind = a * X + d * Y + 7 * nChannels + f;
						sum += gradient[ind] * adj_value;
					}

					// Case 9 (18/50): (a, d). Contract (b, e). Singleton (c).
					if (b == e) {
						ind = a * X + d * Y + 8 * nChannels + f;
						sum += gradient[ind] * adj_value;
					}

					// Case 10 (23/50): (b, c). Contract (a, d). Singleton (e).
					if (a == d) {
						ind = b * X + c * Y + 9 * nChannels + f;
						sum += gradient[ind] * adj_value;
					}

					// Case 11 (26/50): (b, d). Contract (a, c). Singleton (e).
					if (a == c) {
						ind = b * X + d * Y + 10 * nChannels + f;
						sum += gradient[ind] * adj_value;
					}

					// Case 12 (27/50): (b, d). Contract (a, e). Singleton (c).
					if (a == e) {
						ind = b * X + d * Y + 11 * nChannels + f;
						sum += gradient[ind] * adj_value;
					}

					// Case 13 (28/50): (b, d). Contract (c, e). Singleton (a).
					if (c == e) {
						ind = b * X + d * Y + 12 * nChannels + f;
						sum += gradient[ind] * adj_value;
					}

					// Case 14 (38/50): (d, e). Contract (a, b). Singleton (c).
					if (a == b) {
						ind = d * X + e * Y + 13 * nChannels + f;
						sum += gradient[ind] * adj_value;
					}

					// Case 15 (40/50): (d, e). Contract (b, c). Singleton (a).
					if (b == c) {
						ind = d * X + e * Y + 14 * nChannels + f;
						sum += gradient[ind] * adj_value;
					}

					// +---+
					// | 3 |
					// +---+

					// Case 16 (43/50): (a, d). Contract (b, c, e).
					if ((b == c) && (c == e))  {
						ind = a * X + d * Y + 15 * nChannels + f;
						sum += gradient[ind] * adj_value;
					}

					// Case 17 (46/50): (b, d). Contract (a, c, e).
					if ((a == c) && (c == e))  {
						ind = b * X + d * Y + 16 * nChannels + f;
						sum += gradient[ind] * adj_value;
					}

					// Case 18 (50/50): (d, e). Contract (a, b, c).
					if ((a == b) && (b == c))  {
						ind = d * X + e * Y + 17 * nChannels + f;
						sum += gradient[ind] * adj_value;
					}
				}
			}
		}
		
		tensor_gradient[global_threadId] += sum;
	}
}

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void RisiContraction_18_forward(
	const torch::Tensor &tensor,
	const torch::Tensor &adj,
	torch::Tensor &value,
	const int nThreads = 1024
) {
	CHECK_INPUT(tensor);
	CHECK_INPUT(adj);
	CHECK_INPUT(value);

	assert(tensor.dim() == 4);
	assert(adj.dim() == 2);
	assert(value.dim() == 3);

	const int N = tensor.size(0);
	const int nChannels = tensor.size(3);

	assert(tensor.size(1) == N);
	assert(tensor.size(2) == N);
	assert(adj.size(0) == N);
	assert(adj.size(1) == N);
	assert(value.size(0) == N);
	assert(value.size(1) == N);
	assert(value.size(2) == nChannels * NCONTRACTIONS);

	const int size = value.numel();
	dim3 dimGrid(rounded_division(size, nThreads));
	dim3 dimBlock(nThreads);

	// Kernel launch
	RisiContraction_18_forward_job <<< dimGrid, dimBlock >>> (
		tensor.data<double>(), 
		adj.data<double>(), 
		value.data<double>(), 
		N, 
		nChannels);
}

void RisiContraction_18_backward(
	torch::Tensor &tensor_gradient,
	const torch::Tensor &adj,
	const torch::Tensor &value_gradient,
	const int nThreads = 1024
) {
	CHECK_INPUT(tensor_gradient);
	CHECK_INPUT(adj);
	CHECK_INPUT(value_gradient);

	assert(tensor_gradient.dim() == 4);
	assert(adj.dim() == 2);
	assert(value_gradient.dim() == 3);

	const int N = tensor_gradient.size(0);
	const int nChannels = tensor_gradient.size(3);

	assert(tensor_gradient.size(1) == N);
	assert(tensor_gradient.size(2) == N);
	assert(adj.size(0) == N);
	assert(adj.size(1) == N);
	assert(value_gradient.size(0) == N);
	assert(value_gradient.size(1) == N);
	assert(value_gradient.size(2) == nChannels * NCONTRACTIONS);

	const int size = tensor_gradient.numel();
	dim3 dimGrid(rounded_division(size, nThreads));
	dim3 dimBlock(nThreads);

	// Kernel launch
	RisiContraction_18_backward_job <<< dimGrid, dimBlock >>> (
		tensor_gradient.data<double>(), 
		adj.data<double>(), 
		value_gradient.data<double>(), 
		N, 
		nChannels);
}

std::vector<at::Tensor> test_api(const std::vector<at::Tensor> &tensors) {
	const int N = tensors.size();
	std::vector<at::Tensor> result;
	for (int i = 0; i < N; ++i) {
		result.push_back(torch::zeros({}));
	}
	return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("test_api", &test_api, "Test API");
	m.def("RisiContraction_18_forward", &RisiContraction_18_forward, "Forward functionality");
	m.def("RisiContraction_18_backward", &RisiContraction_18_backward, "Backward functionality");
}
