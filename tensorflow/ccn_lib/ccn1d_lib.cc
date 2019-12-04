#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <list>
#include <iterator>
#include <assert.h>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "../../cpp/ccn1d_cpu.h"

using namespace tensorflow;

void ZERO_OUT(Tensor *tensor) {
	float *tensor_flat = tensor -> flat<float>().data();
	for (int i = 0; i < tensor -> NumElements(); ++i) {
		tensor_flat[i] = 0.0;
	}
}

class GetNumContractionsOp : public OpKernel {
public:
	explicit GetNumContractionsOp(OpKernelConstruction *context) : OpKernel(context) {
	}

	void Compute(OpKernelContext *context) override {
		Tensor *output_tensor = NULL;
		OP_REQUIRES_OK(context, context -> allocate_output(0, TensorShape(), &output_tensor));
		auto output_flat = output_tensor -> flat<int64>();
		output_flat(0) = get_nContractions();
	}

private:
};

REGISTER_OP("GetNumContractions")
.Output("output: int64")
;

REGISTER_KERNEL_BUILDER(Name("GetNumContractions").Device(DEVICE_CPU), GetNumContractionsOp);


class PrecomputeNeighborsOp : public OpKernel {
public:
	explicit PrecomputeNeighborsOp(OpKernelConstruction *context) : OpKernel(context) {
		OP_REQUIRES_OK(context, context -> GetAttr("nVertices", &nVertices));
		OP_REQUIRES_OK(context, context -> GetAttr("nLayers", &nLayers));
		OP_REQUIRES_OK(context, context -> GetAttr("nTensors", &nTensors));

		assert(nLayers > 0);
		assert(nTensors == 2 * (nLayers + 1));
	}

	void Compute(OpKernelContext *context) override {
		const Tensor& edges = context -> input(0);
		assert(edges.dims() == 2);
		assert(edges.dim_size(1) == 2);
		const int nEdges = edges.dim_size(0);
		assert(2 * nEdges == edges.NumElements());
		const int *edges_flat = edges.flat<int>().data();

		vector<int> *adj = new vector<int> [nVertices];
		vector<int> *old_adj = new vector<int> [nVertices];

		for (int u = 0; u < nVertices; ++u) {
			adj[u].clear();
			adj[u].push_back(u);
		}

		for (int i = 0; i < nEdges; ++i) {
			const int u = edges_flat[2 * i];
			const int v = edges_flat[2 * i + 1];

			assert(u < nVertices);
			assert(v < nVertices);
			
			assert(u >= 0);
			assert(v >= 0);

			if (u != v) {
				adj[u].push_back(v);
			}
		}

		for (int u = 0; u < nVertices; ++u) {
			sort(adj[u].begin(), adj[u].end());
		}

		for (int layer = 0; layer <= nLayers; ++layer) {
			if (layer == 0) {
				Tensor *receptive_field = NULL;
				OP_REQUIRES_OK(context, context -> allocate_output(2 * layer, TensorShape({nVertices, 1}), &receptive_field));

				Tensor *start_index = NULL;
				OP_REQUIRES_OK(context, context -> allocate_output(2 * layer + 1, TensorShape({nVertices + 1, 1}), &start_index));

				int *receptive_field_flat = receptive_field -> flat<int>().data();
				int *start_index_flat = start_index -> flat<int>().data();

				for (int u = 0; u < nVertices; ++u) {
					receptive_field_flat[u] = u;
					start_index_flat[u] = u;
				}

				start_index_flat[nVertices] = receptive_field -> dim_size(0);
				continue;
			}

			// Extending the receptive fields
			int N = 0;
			if (layer == 1) {
				for (int u = 0; u < nVertices; ++u) {
					N += adj[u].size();
				}
			} else {
				for (int u = 0; u < nVertices; ++u) {
					old_adj[u].clear();
					for (int i = 0; i < adj[u].size(); ++i) {
						old_adj[u].push_back(adj[u][i]);
					}
				}

				for (int u = 0; u < nVertices; ++u) {
					for (int i = 0; i < old_adj[u].size(); ++i) {
						const int v = old_adj[u][i];
						update_receptive_field(adj[u], old_adj[v]);
					}
					N += adj[u].size();
					sort(adj[u].begin(), adj[u].end());
				}
			}

			Tensor *receptive_field = NULL;
			OP_REQUIRES_OK(context, context -> allocate_output(2 * layer, TensorShape({N, 1}), &receptive_field));

			Tensor *start_index = NULL;
			OP_REQUIRES_OK(context, context -> allocate_output(2 * layer + 1, TensorShape({nVertices + 1, 1}), &start_index));

			int *receptive_field_flat = receptive_field -> flat<int>().data();
			int *start_index_flat = start_index -> flat<int>().data();

			int count = 0;
			for (int u = 0; u < nVertices; ++u) {
				start_index_flat[u] = count;
				for (int i = 0; i < adj[u].size(); ++i) {
					receptive_field_flat[count] = adj[u][i];
					++count;
				}
			}

			assert(count == N);
			assert(count == receptive_field -> dim_size(0));

			start_index_flat[nVertices] = receptive_field -> dim_size(0);
		}

		delete[] adj;
		delete[] old_adj;
	}

private:
	int nVertices;
	int nLayers;
	int nTensors;
};

REGISTER_OP("PrecomputeNeighbors")
.Attr("nVertices: int")
.Attr("nLayers: int")
.Attr("nTensors: int")
.Input("edges: int32")
.Output("output: nTensors * int32")
;

REGISTER_KERNEL_BUILDER(Name("PrecomputeNeighbors").Device(DEVICE_CPU), PrecomputeNeighborsOp);


class DynamicPrecomputeNeighborsOp : public OpKernel {
public:
	explicit DynamicPrecomputeNeighborsOp(OpKernelConstruction *context) : OpKernel(context) {
		OP_REQUIRES_OK(context, context -> GetAttr("nLayers", &nLayers));
		OP_REQUIRES_OK(context, context -> GetAttr("nTensors", &nTensors));

		assert(nLayers > 0);
		assert(nTensors == 2 * (nLayers + 1));
	}

	void Compute(OpKernelContext *context) override {
		const Tensor& edges = context -> input(0);
		assert(edges.dims() == 2);
		assert(edges.dim_size(1) == 2);
		const int nEdges = edges.dim_size(0);
		assert(2 * nEdges == edges.NumElements());
		const int *edges_flat = edges.flat<int>().data();

		const Tensor& nVertices_tensor = context -> input(1);
		const int *nVertices_flat = nVertices_tensor.flat<int>().data();
		const int nVertices = nVertices_flat[0];

		vector<int> *adj = new vector<int> [nVertices];
		vector<int> *old_adj = new vector<int> [nVertices];

		for (int u = 0; u < nVertices; ++u) {
			adj[u].clear();
			adj[u].push_back(u);
		}

		for (int i = 0; i < nEdges; ++i) {
			const int u = edges_flat[2 * i];
			const int v = edges_flat[2 * i + 1];

			assert(u < nVertices);
			assert(v < nVertices);
			
			assert(u >= 0);
			assert(v >= 0);

			if (u != v) {
				adj[u].push_back(v);
			}
		}

		for (int u = 0; u < nVertices; ++u) {
			sort(adj[u].begin(), adj[u].end());
		}

		for (int layer = 0; layer <= nLayers; ++layer) {
			if (layer == 0) {
				Tensor *receptive_field = NULL;
				OP_REQUIRES_OK(context, context -> allocate_output(2 * layer, TensorShape({nVertices, 1}), &receptive_field));

				Tensor *start_index = NULL;
				OP_REQUIRES_OK(context, context -> allocate_output(2 * layer + 1, TensorShape({nVertices + 1, 1}), &start_index));

				int *receptive_field_flat = receptive_field -> flat<int>().data();
				int *start_index_flat = start_index -> flat<int>().data();

				for (int u = 0; u < nVertices; ++u) {
					receptive_field_flat[u] = u;
					start_index_flat[u] = u;
				}

				start_index_flat[nVertices] = receptive_field -> dim_size(0);
				continue;
			}

			// Extending the receptive fields
			int N = 0;
			if (layer == 1) {
				for (int u = 0; u < nVertices; ++u) {
					N += adj[u].size();
				}
			} else {
				for (int u = 0; u < nVertices; ++u) {
					old_adj[u].clear();
					for (int i = 0; i < adj[u].size(); ++i) {
						old_adj[u].push_back(adj[u][i]);
					}
				}

				for (int u = 0; u < nVertices; ++u) {
					for (int i = 0; i < old_adj[u].size(); ++i) {
						const int v = old_adj[u][i];
						update_receptive_field(adj[u], old_adj[v]);
					}
					N += adj[u].size();
					sort(adj[u].begin(), adj[u].end());
				}
			}

			Tensor *receptive_field = NULL;
			OP_REQUIRES_OK(context, context -> allocate_output(2 * layer, TensorShape({N, 1}), &receptive_field));

			Tensor *start_index = NULL;
			OP_REQUIRES_OK(context, context -> allocate_output(2 * layer + 1, TensorShape({nVertices + 1, 1}), &start_index));

			int *receptive_field_flat = receptive_field -> flat<int>().data();
			int *start_index_flat = start_index -> flat<int>().data();

			int count = 0;
			for (int u = 0; u < nVertices; ++u) {
				start_index_flat[u] = count;
				for (int i = 0; i < adj[u].size(); ++i) {
					receptive_field_flat[count] = adj[u][i];
					++count;
				}
			}

			assert(count == N);
			assert(count == receptive_field -> dim_size(0));

			start_index_flat[nVertices] = receptive_field -> dim_size(0);
		}

		delete[] adj;
		delete[] old_adj;
	}

private:
	int nLayers;
	int nTensors;
};

REGISTER_OP("DynamicPrecomputeNeighbors")
.Attr("nLayers: int")
.Attr("nTensors: int")
.Input("edges: int32")
.Input("vertices: int32")
.Output("output: nTensors * int32")
;

REGISTER_KERNEL_BUILDER(Name("DynamicPrecomputeNeighbors").Device(DEVICE_CPU), DynamicPrecomputeNeighborsOp);


class ForwardContractionsOp : public OpKernel {
public:
	explicit ForwardContractionsOp(OpKernelConstruction *context) : OpKernel(context) {
	}

	void Compute(OpKernelContext *context) override {
		const Tensor& input_receptive_field = context -> input(0);
		const Tensor& input_start_index = context -> input(1);
		const Tensor& output_receptive_field = context -> input(2);
		const Tensor& output_start_index = context -> input(3);
		const Tensor& input = context -> input(4);

		assert(input_receptive_field.dims() == 2);
		assert(input_receptive_field.dim_size(0) == input.dim_size(0));

		assert(input_start_index.dims() == 2);
		const int nVertices = input_start_index.dim_size(0) - 1;

		assert(output_receptive_field.dims() == 2);

		assert(output_start_index.dims() == 2);
		assert(nVertices == output_start_index.dim_size(0) - 1);

		assert(input.dims() == 2);
		const int input_nChannels = input.dim_size(1);
		const int output_nChannels = input_nChannels * nContractions;

		Tensor *output = NULL;
		OP_REQUIRES_OK(context, context -> allocate_output(0, TensorShape({output_receptive_field.dim_size(0), output_nChannels}), &output));
		ZERO_OUT(output);

		const int *input_receptive_field_flat = input_receptive_field.flat<int>().data();
		const int *input_start_index_flat = input_start_index.flat<int>().data();
		const int *output_receptive_field_flat = output_receptive_field.flat<int>().data();
		const int *output_start_index_flat = output_start_index.flat<int>().data();
		const float *input_flat = input.flat<float>().data();
		float *output_flat = output -> flat<float>().data();

		forward_contractions_cpp(
			input_receptive_field_flat, 
			input_start_index_flat, 
			output_receptive_field_flat, 
			output_start_index_flat,
			input_flat,
			output_flat,
			input_nChannels,
			output_nChannels,
			nVertices
		);
	}

private:
	
};

REGISTER_OP("ForwardContractions")
.Input("input_receptive_field: int32")
.Input("input_start_index: int32")
.Input("output_receptive_field: int32")
.Input("output_start_index: int32")
.Input("input: float32")
.Output("output: float32")
;

REGISTER_KERNEL_BUILDER(Name("ForwardContractions").Device(DEVICE_CPU), ForwardContractionsOp);


class ForwardContractionsMultiThreadingOp : public OpKernel {
public:
	explicit ForwardContractionsMultiThreadingOp(OpKernelConstruction *context) : OpKernel(context) {
		OP_REQUIRES_OK(context, context -> GetAttr("nThreads", &nThreads));

		assert(nThreads >= 1);
	}

	void Compute(OpKernelContext *context) override {
		const Tensor& input_receptive_field = context -> input(0);
		const Tensor& input_start_index = context -> input(1);
		const Tensor& output_receptive_field = context -> input(2);
		const Tensor& output_start_index = context -> input(3);
		const Tensor& input = context -> input(4);

		assert(input_receptive_field.dims() == 2);
		assert(input_receptive_field.dim_size(0) == input.dim_size(0));

		assert(input_start_index.dims() == 2);
		const int nVertices = input_start_index.dim_size(0) - 1;

		assert(output_receptive_field.dims() == 2);

		assert(output_start_index.dims() == 2);
		assert(nVertices == output_start_index.dim_size(0) - 1);

		assert(input.dims() == 2);
		const int input_nChannels = input.dim_size(1);
		const int output_nChannels = input_nChannels * nContractions;

		Tensor *output = NULL;
		OP_REQUIRES_OK(context, context -> allocate_output(0, TensorShape({output_receptive_field.dim_size(0), output_nChannels}), &output));
		ZERO_OUT(output);

		const int *input_receptive_field_flat = input_receptive_field.flat<int>().data();
		const int *input_start_index_flat = input_start_index.flat<int>().data();
		const int *output_receptive_field_flat = output_receptive_field.flat<int>().data();
		const int *output_start_index_flat = output_start_index.flat<int>().data();
		const float *input_flat = input.flat<float>().data();
		float *output_flat = output -> flat<float>().data();

		forward_contractions_multi_threading_cpp(
			input_receptive_field_flat, 
			input_start_index_flat, 
			output_receptive_field_flat, 
			output_start_index_flat,
			input_flat,
			output_flat,
			input_nChannels,
			output_nChannels,
			nVertices,
			nThreads
		);
	}

private:
	int nThreads;
};

REGISTER_OP("ForwardContractionsMultiThreading")
.Attr("nThreads: int")
.Input("input_receptive_field: int32")
.Input("input_start_index: int32")
.Input("output_receptive_field: int32")
.Input("output_start_index: int32")
.Input("input: float32")
.Output("output: float32")
;

REGISTER_KERNEL_BUILDER(Name("ForwardContractionsMultiThreading").Device(DEVICE_CPU), ForwardContractionsMultiThreadingOp);


class BackwardContractionsOp : public OpKernel {
public:
	explicit BackwardContractionsOp(OpKernelConstruction *context) : OpKernel(context) {
	}

	void Compute(OpKernelContext *context) override {
		const Tensor& input_receptive_field = context -> input(0);
		const Tensor& input_start_index = context -> input(1);
		const Tensor& output_receptive_field = context -> input(2);
		const Tensor& output_start_index = context -> input(3);
		const Tensor& output_grad = context -> input(4);

		assert(input_receptive_field.dims() == 2);

		assert(input_start_index.dims() == 2);
		const int nVertices = input_start_index.dim_size(0) - 1;

		assert(output_receptive_field.dims() == 2);
		assert(output_receptive_field.dim_size(0) == output_grad.dim_size(0));

		assert(output_start_index.dims() == 2);
		assert(nVertices == output_start_index.dim_size(0) - 1);

		assert(output_grad.dims() == 2);
		const int output_nChannels = output_grad.dim_size(1);
		assert(output_nChannels % nContractions == 0);
		const int input_nChannels = output_nChannels / nContractions;

		Tensor *input_grad = NULL;
		OP_REQUIRES_OK(context, context -> allocate_output(0, TensorShape({input_receptive_field.dim_size(0), input_nChannels}), &input_grad));
		ZERO_OUT(input_grad);

		const int *input_receptive_field_flat = input_receptive_field.flat<int>().data();
		const int *input_start_index_flat = input_start_index.flat<int>().data();
		const int *output_receptive_field_flat = output_receptive_field.flat<int>().data();
		const int *output_start_index_flat = output_start_index.flat<int>().data();
		float *input_grad_flat = input_grad -> flat<float>().data();
		const float *output_grad_flat = output_grad.flat<float>().data();

		backward_contractions_cpp(
			input_receptive_field_flat, 
			input_start_index_flat, 
			output_receptive_field_flat, 
			output_start_index_flat,
			input_grad_flat,
			output_grad_flat,
			input_nChannels,
			output_nChannels,
			nVertices
		);
	}

private:
	
};

REGISTER_OP("BackwardContractions")
.Input("input_receptive_field: int32")
.Input("input_start_index: int32")
.Input("output_receptive_field: int32")
.Input("output_start_index: int32")
.Input("output_grad: float32")
.Output("input_grad: float32")
;

REGISTER_KERNEL_BUILDER(Name("BackwardContractions").Device(DEVICE_CPU), BackwardContractionsOp);


class BackwardContractionsMultiThreadingOp : public OpKernel {
public:
	explicit BackwardContractionsMultiThreadingOp(OpKernelConstruction *context) : OpKernel(context) {
		OP_REQUIRES_OK(context, context -> GetAttr("nThreads", &nThreads));

		assert(nThreads >= 1);
	}

	void Compute(OpKernelContext *context) override {
		const Tensor& input_receptive_field = context -> input(0);
		const Tensor& input_start_index = context -> input(1);
		const Tensor& output_receptive_field = context -> input(2);
		const Tensor& output_start_index = context -> input(3);
		const Tensor& output_grad = context -> input(4);

		assert(input_receptive_field.dims() == 2);

		assert(input_start_index.dims() == 2);
		const int nVertices = input_start_index.dim_size(0) - 1;

		assert(output_receptive_field.dims() == 2);
		assert(output_receptive_field.dim_size(0) == output_grad.dim_size(0));

		assert(output_start_index.dims() == 2);
		assert(nVertices == output_start_index.dim_size(0) - 1);

		assert(output_grad.dims() == 2);
		const int output_nChannels = output_grad.dim_size(1);

		assert(output_nChannels % nContractions == 0);
		const int input_nChannels = output_nChannels / nContractions;

		Tensor *input_grad = NULL;
		OP_REQUIRES_OK(context, context -> allocate_output(0, TensorShape({input_receptive_field.dim_size(0), input_nChannels}), &input_grad));
		ZERO_OUT(input_grad);

		const int *input_receptive_field_flat = input_receptive_field.flat<int>().data();
		const int *input_start_index_flat = input_start_index.flat<int>().data();
		const int *output_receptive_field_flat = output_receptive_field.flat<int>().data();
		const int *output_start_index_flat = output_start_index.flat<int>().data();
		float *input_grad_flat = input_grad -> flat<float>().data();
		const float *output_grad_flat = output_grad.flat<float>().data();

		backward_contractions_multi_threading_cpp(
			input_receptive_field_flat, 
			input_start_index_flat, 
			output_receptive_field_flat, 
			output_start_index_flat,
			input_grad_flat,
			output_grad_flat,
			input_nChannels,
			output_nChannels,
			nVertices,
			nThreads
		);
	}

private:
	int nThreads;
};

REGISTER_OP("BackwardContractionsMultiThreading")
.Attr("nThreads: int")
.Input("input_receptive_field: int32")
.Input("input_start_index: int32")
.Input("output_receptive_field: int32")
.Input("output_start_index: int32")
.Input("output_grad: float32")
.Output("input_grad: float32")
;

REGISTER_KERNEL_BUILDER(Name("BackwardContractionsMultiThreading").Device(DEVICE_CPU), BackwardContractionsMultiThreadingOp);


class ForwardShrinkingOp : public OpKernel {
public:
	explicit ForwardShrinkingOp(OpKernelConstruction *context) : OpKernel(context) {
	}

	void Compute(OpKernelContext *context) override {
		const Tensor& start_index = context -> input(0);
		const Tensor& input = context -> input(1);

		assert(start_index.dims() == 2);
		const int nVertices = start_index.dim_size(0) - 1;

		assert(input.dims() == 2);
		const int nChannels = input.dim_size(1);

		Tensor *output = NULL;
		OP_REQUIRES_OK(context, context -> allocate_output(0, TensorShape({nVertices, nChannels}), &output));
		ZERO_OUT(output);

		const int *start_index_flat = start_index.flat<int>().data();
		const float *input_flat = input.flat<float>().data();
		float *output_flat = output -> flat<float>().data();

		forward_shrinking_cpp(
			start_index_flat,
			input_flat,
			output_flat,
			nVertices,
			nChannels
		);
	}

private:
	
};

REGISTER_OP("ForwardShrinking")
.Input("start_index: int32")
.Input("input: float32")
.Output("output: float32")
;

REGISTER_KERNEL_BUILDER(Name("ForwardShrinking").Device(DEVICE_CPU), ForwardShrinkingOp);


class ForwardShrinkingMultiThreadingOp : public OpKernel {
public:
	explicit ForwardShrinkingMultiThreadingOp(OpKernelConstruction *context) : OpKernel(context) {
		OP_REQUIRES_OK(context, context -> GetAttr("nThreads", &nThreads));

		assert(nThreads >= 1);
	}

	void Compute(OpKernelContext *context) override {
		const Tensor& start_index = context -> input(0);
		const Tensor& input = context -> input(1);

		assert(start_index.dims() == 2);
		const int nVertices = start_index.dim_size(0) - 1;

		assert(input.dims() == 2);
		const int nChannels = input.dim_size(1);

		Tensor *output = NULL;
		OP_REQUIRES_OK(context, context -> allocate_output(0, TensorShape({nVertices, nChannels}), &output));
		ZERO_OUT(output);

		const int *start_index_flat = start_index.flat<int>().data();
		const float *input_flat = input.flat<float>().data();
		float *output_flat = output -> flat<float>().data();

		forward_shrinking_multi_threading_cpp(
			start_index_flat,
			input_flat,
			output_flat,
			nVertices,
			nChannels,
			nThreads
		);
	}

private:
	int nThreads;
};

REGISTER_OP("ForwardShrinkingMultiThreading")
.Attr("nThreads: int")
.Input("start_index: int32")
.Input("input: float32")
.Output("output: float32")
;

REGISTER_KERNEL_BUILDER(Name("ForwardShrinkingMultiThreading").Device(DEVICE_CPU), ForwardShrinkingMultiThreadingOp);


class BackwardShrinkingOp : public OpKernel {
public:
	explicit BackwardShrinkingOp(OpKernelConstruction *context) : OpKernel(context) {
	}

	void Compute(OpKernelContext *context) override {
		const Tensor& start_index = context -> input(0);
		const Tensor& output_grad = context -> input(1);

		assert(start_index.dims() == 2);
		const int nVertices = start_index.dim_size(0) - 1;

		assert(output_grad.dims() == 2);
		assert(output_grad.dim_size(0) == nVertices);
		const int nChannels = output_grad.dim_size(1);

		const int *start_index_flat = start_index.flat<int>().data();
		const float *output_grad_flat = output_grad.flat<float>().data();
		
		Tensor *input_grad = NULL;
		OP_REQUIRES_OK(context, context -> allocate_output(0, TensorShape({start_index_flat[nVertices], nChannels}), &input_grad));
		ZERO_OUT(input_grad);

		float *input_grad_flat = input_grad -> flat<float>().data();

		backward_shrinking_cpp(
			start_index_flat,
			input_grad_flat,
			output_grad_flat,
			nVertices,
			nChannels
		);
	}

private:
	
};

REGISTER_OP("BackwardShrinking")
.Input("start_index: int32")
.Input("output_grad: float32")
.Output("input_grad: float32")
;

REGISTER_KERNEL_BUILDER(Name("BackwardShrinking").Device(DEVICE_CPU), BackwardShrinkingOp);


class BackwardShrinkingMultiThreadingOp : public OpKernel {
public:
	explicit BackwardShrinkingMultiThreadingOp(OpKernelConstruction *context) : OpKernel(context) {
		OP_REQUIRES_OK(context, context -> GetAttr("nThreads", &nThreads));

		assert(nThreads >= 1);
	}

	void Compute(OpKernelContext *context) override {
		const Tensor& start_index = context -> input(0);
		const Tensor& output_grad = context -> input(1);

		assert(start_index.dims() == 2);
		const int nVertices = start_index.dim_size(0) - 1;

		assert(output_grad.dims() == 2);
		assert(output_grad.dim_size(0) == nVertices);
		const int nChannels = output_grad.dim_size(1);

		const int *start_index_flat = start_index.flat<int>().data();
		const float *output_grad_flat = output_grad.flat<float>().data();
		
		Tensor *input_grad = NULL;
		OP_REQUIRES_OK(context, context -> allocate_output(0, TensorShape({start_index_flat[nVertices], nChannels}), &input_grad));
		ZERO_OUT(input_grad);

		float *input_grad_flat = input_grad -> flat<float>().data();

		backward_shrinking_multi_threading_cpp(
			start_index_flat,
			input_grad_flat,
			output_grad_flat,
			nVertices,
			nChannels,
			nThreads
		);
	}

private:
	int nThreads;
};

REGISTER_OP("BackwardShrinkingMultiThreading")
.Attr("nThreads: int")
.Input("start_index: int32")
.Input("output_grad: float32")
.Output("input_grad: float32")
;

REGISTER_KERNEL_BUILDER(Name("BackwardShrinkingMultiThreading").Device(DEVICE_CPU), BackwardShrinkingMultiThreadingOp);


class ForwardNormalizingOp : public OpKernel {
public:
	explicit ForwardNormalizingOp(OpKernelConstruction *context) : OpKernel(context) {
	}

	void Compute(OpKernelContext *context) override {
		const Tensor& start_index = context -> input(0);
		const Tensor& input = context -> input(1);

		assert(start_index.dims() == 2);
		const int nVertices = start_index.dim_size(0) - 1;

		assert(input.dims() == 2);
		const int nChannels = input.dim_size(1);

		Tensor *output = NULL;
		OP_REQUIRES_OK(context, context -> allocate_output(0, TensorShape({input.dim_size(0), nChannels}), &output));
		ZERO_OUT(output);

		const int *start_index_flat = start_index.flat<int>().data();
		const float *input_flat = input.flat<float>().data();
		float *output_flat = output -> flat<float>().data();

		forward_normalizing_cpp(
			start_index_flat,
			input_flat,
			output_flat,
			nVertices,
			nChannels
		);
	}

private:
	
};

REGISTER_OP("ForwardNormalizing")
.Input("start_index: int32")
.Input("input: float32")
.Output("output: float32")
;

REGISTER_KERNEL_BUILDER(Name("ForwardNormalizing").Device(DEVICE_CPU), ForwardNormalizingOp);


class BackwardNormalizingOp : public OpKernel {
public:
	explicit BackwardNormalizingOp(OpKernelConstruction *context) : OpKernel(context) {
	}

	void Compute(OpKernelContext *context) override {
		const Tensor& start_index = context -> input(0);
		const Tensor& output_grad = context -> input(1);

		assert(start_index.dims() == 2);
		const int nVertices = start_index.dim_size(0) - 1;

		assert(output_grad.dims() == 2);
		const int nChannels = output_grad.dim_size(1);
		
		Tensor *input_grad = NULL;
		OP_REQUIRES_OK(context, context -> allocate_output(0, TensorShape({output_grad.dim_size(0), nChannels}), &input_grad));
		ZERO_OUT(input_grad);

		const int *start_index_flat = start_index.flat<int>().data();
		const float *output_grad_flat = output_grad.flat<float>().data();
		float *input_grad_flat = input_grad -> flat<float>().data();

		backward_normalizing_cpp(
			start_index_flat,
			input_grad_flat,
			output_grad_flat,
			nVertices,
			nChannels
		);
	}

private:
	
};

REGISTER_OP("BackwardNormalizing")
.Input("start_index: int32")
.Input("output_grad: float32")
.Output("input_grad: float32")
;

REGISTER_KERNEL_BUILDER(Name("BackwardNormalizing").Device(DEVICE_CPU), BackwardNormalizingOp);


class TestOp : public OpKernel {
public:
	explicit TestOp(OpKernelConstruction *context) : OpKernel(context) {
	}

	void Compute(OpKernelContext *context) override {
	}

private:
};

REGISTER_OP("Test")
.Input("in: float")
.Output("out: float")
;

REGISTER_KERNEL_BUILDER(Name("Test").Device(DEVICE_CPU), TestOp);