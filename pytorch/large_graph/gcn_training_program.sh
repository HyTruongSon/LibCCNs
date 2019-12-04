# Number of epochs
epochs=1024

# Learning rate
learning_rate=1e-3

# Using sparse representation or not
# sparse=dense
sparse=sparse

# Architecture
input_size=64
message_sizes=16,16
message_mlp_sizes=16

# Dataset's directory
data_dir=../../data

# Dataset
data_name=citeseer
# data_name=cora

# Training program
python3 gcn_training_program.py --data_dir=$data_dir --data_name=$data_name --epochs=$epochs --learning_rate=$learning_rate --sparse=$sparse --input_size=$input_size --message_sizes=$message_sizes --message_mlp_sizes=$message_mlp_sizes