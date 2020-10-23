# Compile the library
cd ../ccn_lib/
sh compile.sh
cd ../small_graphs

# Data name
data_name=tree
#data_name=star
#data_name=er

# Number of samples
num_samples=100

# Graph size
graph_size=10

# Number of epochs
epochs=1024

# Learning rate
learning_rate=1e-2

# Architecture
input_size=16
output_size=32
message_sizes=16,16
message_mlp_sizes=16

# Activation
activation=sigmoid
# activation=relu

# Multi-threading
nThreads=10

# Batch size
batch_size=4

# Training program
python3 ccn1d_dgvae_synthetic.py --data_name=$data_name --num_samples=$num_samples --graph_size=$graph_size --epochs=$epochs --learning_rate=$learning_rate --input_size=$input_size --output_size=$output_size --message_sizes=$message_sizes --message_mlp_sizes=$message_mlp_sizes --nThreads=$nThreads --activation=$activation