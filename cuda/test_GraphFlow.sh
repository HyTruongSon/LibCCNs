# Number of vertices
num_vertices=10

# Number of channels
num_channels=20

# CPU version compilation
rm -f test_GraphFlow_cpu
g++ test_GraphFlow.cpp -o test_GraphFlow_cpu

# CPU version executation
./test_GraphFlow_cpu $num_vertices $num_channels

# GPU (CUDA) version compilation
rm -f test_GraphFlow_cuda
nvcc test_GraphFlow.cu -o test_GraphFlow_cuda

# GPU (CUDA) version executation
./test_GraphFlow_cuda $num_vertices $num_channels

# Cleaning
rm -f test_GraphFlow_cpu test_GraphFlow_cuda