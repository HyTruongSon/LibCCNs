#include "../../cpp/ccn1d_cpu.h"
#include <torch/torch.h>

vector< pair<at::Tensor, at::Tensor> > precompute_neighbors(const at::Tensor &edges, const int &nVertices, const int &nLayers) {
	assert(edges.dim() == 2);
	assert(nLayers > 0);

	const int nEdges = edges.size(0);
	assert(2 == edges.size(1));

	// int *edges_flat = reinterpret_cast<int*>(edges.data<int>());
	float *edges_flat = reinterpret_cast<float*>(edges.data<float>());

	vector< pair<at::Tensor, at::Tensor> > result;
	result.clear();

	vector<int> *adj = new vector<int> [nVertices];
	vector<int> *old_adj = new vector<int> [nVertices];

	for (int u = 0; u < nVertices; ++u) {
		adj[u].clear();
		adj[u].push_back(u);
	}

	for (int i = 0; i < nEdges; ++i) {
		const int u = (int)(edges_flat[2 * i]);
		const int v = (int)(edges_flat[2 * i + 1]);

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
			// at::Tensor receptive_field = torch::zeros(torch::CPU(at::kInt), {nVertices});
			// at::Tensor start_index = torch::zeros(torch::CPU(at::kInt), {nVertices + 1});

			at::Tensor receptive_field = torch::zeros({nVertices}, torch::CPU(at::kInt));
			at::Tensor start_index = torch::zeros({nVertices + 1}, torch::CPU(at::kInt));

			int *receptive_field_flat = reinterpret_cast<int*>(receptive_field.data<int>());
			int *start_index_flat = reinterpret_cast<int*>(start_index.data<int>());

			for (int u = 0; u < nVertices; ++u) {
				receptive_field_flat[u] = u;
				start_index_flat[u] = u;
			}

			start_index_flat[nVertices] = receptive_field.size(0);
			result.push_back(make_pair(receptive_field, start_index));
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

		// at::Tensor receptive_field = torch::zeros(torch::CPU(at::kInt), {N});
		// at::Tensor start_index = torch::zeros(torch::CPU(at::kInt), {nVertices + 1});

		at::Tensor receptive_field = torch::zeros({N}, torch::CPU(at::kInt));
		at::Tensor start_index = torch::zeros({nVertices + 1}, torch::CPU(at::kInt));

		int *receptive_field_flat = reinterpret_cast<int*>(receptive_field.data<int>());
		int *start_index_flat = reinterpret_cast<int*>(start_index.data<int>());

		int count = 0;
		for (int u = 0; u < nVertices; ++u) {
			start_index_flat[u] = count;
			for (int i = 0; i < adj[u].size(); ++i) {
				receptive_field_flat[count] = adj[u][i];
				++count;
			}
		}

		assert(count == N);
		assert(count == receptive_field.size(0));

		start_index_flat[nVertices] = int(receptive_field.size(0));
		result.push_back(make_pair(receptive_field, start_index));
	}

	delete[] adj;
	delete[] old_adj;

	return result;
} 

int forward_contractions(
	const at::Tensor &input_receptive_field, 
	const at::Tensor &input_start_index, 
	const at::Tensor &output_receptive_field, 
	const at::Tensor &output_start_index,
	const at::Tensor &input,
	at::Tensor &output) {

	assert(input_receptive_field.dim() == 1);
	assert(input_receptive_field.size(0) == input.size(0));

	assert(input_start_index.dim() == 1);
	const int nVertices = input_start_index.size(0) - 1;

	assert(output_receptive_field.dim() == 1);
	assert(output_receptive_field.size(0) == output.size(0));

	assert(output_start_index.dim() == 1);
	assert(output_start_index.size(0) == nVertices + 1);

	assert(input.dim() == 2);
	assert(output.dim() == 2);

	const int input_nChannels = input.size(1);
	const int output_nChannels = output.size(1);
	assert(input_nChannels * nContractions == output_nChannels);

	int *input_receptive_field_flat = reinterpret_cast<int*>(input_receptive_field.data<int>());
	int *input_start_index_flat = reinterpret_cast<int*>(input_start_index.data<int>());
	int *output_receptive_field_flat = reinterpret_cast<int*>(output_receptive_field.data<int>());
	int *output_start_index_flat = reinterpret_cast<int*>(output_start_index.data<int>());

	float *input_flat = reinterpret_cast<float*>(input.data<float>());
	float *output_flat = reinterpret_cast<float*>(output.data<float>());

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

	return 0;
}

int forward_contractions_multi_threading(
	const at::Tensor &input_receptive_field, 
	const at::Tensor &input_start_index, 
	const at::Tensor &output_receptive_field, 
	const at::Tensor &output_start_index,
	const at::Tensor &input,
	at::Tensor &output,
	const int nThreads) {

	assert(input_receptive_field.dim() == 1);
	assert(input_receptive_field.size(0) == input.size(0));

	assert(input_start_index.dim() == 1);
	const int nVertices = input_start_index.size(0) - 1;

	assert(output_receptive_field.dim() == 1);
	assert(output_receptive_field.size(0) == output.size(0));

	assert(output_start_index.dim() == 1);
	assert(output_start_index.size(0) == nVertices + 1);

	assert(input.dim() == 2);
	assert(output.dim() == 2);

	const int input_nChannels = input.size(1);
	const int output_nChannels = output.size(1);
	assert(input_nChannels * nContractions == output_nChannels);

	int *input_receptive_field_flat = reinterpret_cast<int*>(input_receptive_field.data<int>());
	int *input_start_index_flat = reinterpret_cast<int*>(input_start_index.data<int>());
	int *output_receptive_field_flat = reinterpret_cast<int*>(output_receptive_field.data<int>());
	int *output_start_index_flat = reinterpret_cast<int*>(output_start_index.data<int>());

	float *input_flat = reinterpret_cast<float*>(input.data<float>());
	float *output_flat = reinterpret_cast<float*>(output.data<float>());

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

	return 0;
}

int backward_contractions(
	const at::Tensor &input_receptive_field, 
	const at::Tensor &input_start_index, 
	const at::Tensor &output_receptive_field, 
	const at::Tensor &output_start_index,
	at::Tensor &input_grad,
	const at::Tensor &output_grad) {

	assert(input_receptive_field.dim() == 1);
	assert(input_receptive_field.size(0) == input_grad.size(0));

	assert(input_start_index.dim() == 1);
	const int nVertices = input_start_index.size(0) - 1;

	assert(output_receptive_field.dim() == 1);
	assert(output_receptive_field.size(0) == output_grad.size(0));

	assert(output_start_index.dim() == 1);
	assert(output_start_index.size(0) == nVertices + 1);

	assert(input_grad.dim() == 2);
	assert(output_grad.dim() == 2);

	const int input_nChannels = input_grad.size(1);
	const int output_nChannels = output_grad.size(1);
	assert(input_nChannels * nContractions == output_nChannels);

	int *input_receptive_field_flat = reinterpret_cast<int*>(input_receptive_field.data<int>());
	int *input_start_index_flat = reinterpret_cast<int*>(input_start_index.data<int>());
	int *output_receptive_field_flat = reinterpret_cast<int*>(output_receptive_field.data<int>());
	int *output_start_index_flat = reinterpret_cast<int*>(output_start_index.data<int>());

	float *input_grad_flat = reinterpret_cast<float*>(input_grad.data<float>());
	float *output_grad_flat = reinterpret_cast<float*>(output_grad.data<float>());

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

	return 0;
}

int backward_contractions_multi_threading(
	const at::Tensor &input_receptive_field, 
	const at::Tensor &input_start_index, 
	const at::Tensor &output_receptive_field, 
	const at::Tensor &output_start_index,
	at::Tensor &input_grad,
	const at::Tensor &output_grad,
	const int nThreads) {

	assert(input_receptive_field.dim() == 1);
	assert(input_receptive_field.size(0) == input_grad.size(0));

	assert(input_start_index.dim() == 1);
	const int nVertices = input_start_index.size(0) - 1;

	assert(output_receptive_field.dim() == 1);
	assert(output_receptive_field.size(0) == output_grad.size(0));

	assert(output_start_index.dim() == 1);
	assert(output_start_index.size(0) == nVertices + 1);

	assert(input_grad.dim() == 2);
	assert(output_grad.dim() == 2);

	const int input_nChannels = input_grad.size(1);
	const int output_nChannels = output_grad.size(1);
	assert(input_nChannels * nContractions == output_nChannels);

	int *input_receptive_field_flat = reinterpret_cast<int*>(input_receptive_field.data<int>());
	int *input_start_index_flat = reinterpret_cast<int*>(input_start_index.data<int>());
	int *output_receptive_field_flat = reinterpret_cast<int*>(output_receptive_field.data<int>());
	int *output_start_index_flat = reinterpret_cast<int*>(output_start_index.data<int>());

	float *input_grad_flat = reinterpret_cast<float*>(input_grad.data<float>());
	float *output_grad_flat = reinterpret_cast<float*>(output_grad.data<float>());

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

	return 0;
}

int forward_shrinking(
	// const at::Tensor &receptive_field, 
	const at::Tensor &start_index,
	const at::Tensor &input,
	at::Tensor &output) {

	// assert(receptive_field.dim() == 1);
	// assert(receptive_field.size(0) == input.size(0));

	assert(start_index.dim() == 1);
	const int nVertices = start_index.size(0) - 1;
	
	assert(input.dim() == 2);
	assert(input.size(1) == output.size(1));

	assert(output.dim() == 2);
	assert(output.size(0) == nVertices);

	// int *receptive_field_flat = reinterpret_cast<int*>(receptive_field.data<int>());
	int *start_index_flat = reinterpret_cast<int*>(start_index.data<int>());

	float *input_flat = reinterpret_cast<float*>(input.data<float>());
	float *output_flat = reinterpret_cast<float*>(output.data<float>());

	const int nChannels = input.size(1);

	forward_shrinking_cpp(
		start_index_flat,
		input_flat,
		output_flat,
		nVertices,
		nChannels
	);

	return 0;
}

int forward_shrinking_multi_threading(
	// const at::Tensor &receptive_field, 
	const at::Tensor &start_index,
	const at::Tensor &input,
	at::Tensor &output,
	const int nThreads) {

	// assert(receptive_field.dim() == 1);
	// assert(receptive_field.size(0) == input.size(0));

	assert(start_index.dim() == 1);
	const int nVertices = start_index.size(0) - 1;
	
	assert(input.dim() == 2);
	assert(input.size(1) == output.size(1));

	assert(output.dim() == 2);
	assert(output.size(0) == nVertices);

	// int *receptive_field_flat = reinterpret_cast<int*>(receptive_field.data<int>());
	int *start_index_flat = reinterpret_cast<int*>(start_index.data<int>());

	float *input_flat = reinterpret_cast<float*>(input.data<float>());
	float *output_flat = reinterpret_cast<float*>(output.data<float>());

	const int nChannels = input.size(1);

	forward_shrinking_multi_threading_cpp(
		start_index_flat,
		input_flat,
		output_flat,
		nVertices,
		nChannels,
		nThreads
	);

	return 0;
}

int backward_shrinking(
	// const at::Tensor &receptive_field, 
	const at::Tensor &start_index,
	const at::Tensor &input_grad,
	at::Tensor &output_grad) {

	// assert(receptive_field.dim() == 1);
	// assert(receptive_field.size(0) == input_grad.size(0));

	assert(start_index.dim() == 1);
	const int nVertices = start_index.size(0) - 1;
	
	assert(input_grad.dim() == 2);
	assert(input_grad.size(1) == output_grad.size(1));

	assert(output_grad.dim() == 2);
	assert(output_grad.size(0) == nVertices);

	// int *receptive_field_flat = reinterpret_cast<int*>(receptive_field.data<int>());
	int *start_index_flat = reinterpret_cast<int*>(start_index.data<int>());

	float *input_grad_flat = reinterpret_cast<float*>(input_grad.data<float>());
	float *output_grad_flat = reinterpret_cast<float*>(output_grad.data<float>());

	const int nChannels = input_grad.size(1);

	backward_shrinking_cpp(
		start_index_flat,
		input_grad_flat,
		output_grad_flat,
		nVertices,
		nChannels
	);

	return 0;
}

int backward_shrinking_multi_threading(
	// const at::Tensor &receptive_field, 
	const at::Tensor &start_index,
	const at::Tensor &input_grad,
	at::Tensor &output_grad,
	const int nThreads) {

	// assert(receptive_field.dim() == 1);
	// assert(receptive_field.size(0) == input_grad.size(0));

	assert(start_index.dim() == 1);
	const int nVertices = start_index.size(0) - 1;
	
	assert(input_grad.dim() == 2);
	assert(input_grad.size(1) == output_grad.size(1));

	assert(output_grad.dim() == 2);
	assert(output_grad.size(0) == nVertices);

	// int *receptive_field_flat = reinterpret_cast<int*>(receptive_field.data<int>());
	int *start_index_flat = reinterpret_cast<int*>(start_index.data<int>());

	float *input_grad_flat = reinterpret_cast<float*>(input_grad.data<float>());
	float *output_grad_flat = reinterpret_cast<float*>(output_grad.data<float>());

	const int nChannels = input_grad.size(1);

	backward_shrinking_multi_threading_cpp(
		start_index_flat,
		input_grad_flat,
		output_grad_flat,
		nVertices,
		nChannels,
		nThreads
	);

	return 0;
}

int forward_normalizing(
	// const at::Tensor &receptive_field, 
	const at::Tensor &start_index,
	const at::Tensor &input,
	at::Tensor &output) {

	// assert(receptive_field.dim() == 1);
	// assert(receptive_field.size(0) == input.size(0));

	assert(start_index.dim() == 1);
	const int nVertices = start_index.size(0) - 1;
	
	assert(input.dim() == 2);
	assert(input.size(0) == output.size(0));
	assert(input.size(1) == output.size(1));

	assert(output.dim() == 2);

	// int *receptive_field_flat = reinterpret_cast<int*>(receptive_field.data<int>());
	int *start_index_flat = reinterpret_cast<int*>(start_index.data<int>());

	float *input_flat = reinterpret_cast<float*>(input.data<float>());
	float *output_flat = reinterpret_cast<float*>(output.data<float>());

	const int nChannels = input.size(1);

	forward_normalizing_cpp(
		start_index_flat,
		input_flat,
		output_flat,
		nVertices,
		nChannels
	);

	return 0;
}

int backward_normalizing(
	// const at::Tensor &receptive_field, 
	const at::Tensor &start_index,
	const at::Tensor &input_grad,
	at::Tensor &output_grad) {

	// assert(receptive_field.dim() == 1);
	// assert(receptive_field.size(0) == input.size(0));

	assert(start_index.dim() == 1);
	const int nVertices = start_index.size(0) - 1;
	
	assert(input_grad.dim() == 2);
	assert(input_grad.size(0) == output_grad.size(0));
	assert(input_grad.size(1) == output_grad.size(1));

	assert(output_grad.dim() == 2);

	// int *receptive_field_flat = reinterpret_cast<int*>(receptive_field.data<int>());
	int *start_index_flat = reinterpret_cast<int*>(start_index.data<int>());

	float *input_grad_flat = reinterpret_cast<float*>(input_grad.data<float>());
	float *output_grad_flat = reinterpret_cast<float*>(output_grad.data<float>());

	const int nChannels = input_grad.size(1);

	backward_normalizing_cpp(
		start_index_flat,
		input_grad_flat,
		output_grad_flat,
		nVertices,
		nChannels
	);

	return 0;
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
	m.def("precompute_neighbors", &precompute_neighbors, "Precomputing neighborhood");

	m.def("forward_contractions", &forward_contractions, "Forward contractions");
	m.def("backward_contractions", &backward_contractions, "Backward contractions");
	
	m.def("forward_contractions_multi_threading", &forward_contractions_multi_threading, "Forward contractions (multi-threading)");
	m.def("backward_contractions_multi_threading", &backward_contractions_multi_threading, "Backward contractions (multi-threading)");
	
	m.def("forward_shrinking", &forward_shrinking, "Forward shrinking");
	m.def("backward_shrinking", &backward_shrinking, "Backward shrinking");
	
	m.def("forward_shrinking_multi_threading", &forward_shrinking_multi_threading, "Forward shrinking (multi-threading)");
	m.def("backward_shrinking_multi_threading", &backward_shrinking_multi_threading, "Backward shrinking (multi-threading)");

	m.def("forward_normalizing", &forward_normalizing, "Forward normalizing");
	m.def("backward_normalizing", &backward_normalizing, "Backward normalizing");

	m.def("get_nContractions", &get_nContractions, "Return the number of contractions");
	m.def("test_api", &test_api, "Test API");
}