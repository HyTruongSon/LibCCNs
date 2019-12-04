# Compile the library
cd ../ccn_lib/
sh compile.sh
cd ../large_graph/

# Number of epochs
epochs=1024

# Learning rate
learning_rate=1e-2

# Architecture
input_size=16
message_sizes=16
message_mlp_sizes=16

# Activation
activation=sigmoid
# activation=relu

# Multi-threading
nThreads=10

# Dataset's directory
data_dir=../../data

# Dataset: Citation graph
# data_name=citeseer
# data_name=cora
# data_name=WebKB
# data_name=Pubmed-Diabetes

# Dataset: Knowledge graph (NELL)
data_name=NELL_0.1
# data_name=NELL_0.01
# data_name=NELL_0.001

# t-SNE visualization
tsne=1

# Training program
python3 ccn1d_training_program.py --data_dir=$data_dir --data_name=$data_name --epochs=$epochs --learning_rate=$learning_rate --input_size=$input_size --message_sizes=$message_sizes --message_mlp_sizes=$message_mlp_sizes --nThreads=$nThreads --activation=$activation --tsne=$tsne