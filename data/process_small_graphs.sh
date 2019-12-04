# Dataset's directory
data_dir=small_graphs_datasets

# Number of folds
num_folds=5

# Number of folds for training
num_train_folds=4

# Data name
data_name=MUTAG

# Training program
python3 process_small_graphs.py --data_dir=$data_dir --data_name=$data_name --num_folds=$num_folds --num_train_folds=$num_train_folds

# Data name
data_name=DD

# Training program
python3 process_small_graphs.py --data_dir=$data_dir --data_name=$data_name --num_folds=$num_folds --num_train_folds=$num_train_folds

# Data name
data_name=ENZYMES

# Training program
python3 process_small_graphs.py --data_dir=$data_dir --data_name=$data_name --num_folds=$num_folds --num_train_folds=$num_train_folds

# Data name
data_name=NCI1

# Training program
python3 process_small_graphs.py --data_dir=$data_dir --data_name=$data_name --num_folds=$num_folds --num_train_folds=$num_train_folds

# Data name
data_name=NCI109

# Training program
python3 process_small_graphs.py --data_dir=$data_dir --data_name=$data_name --num_folds=$num_folds --num_train_folds=$num_train_folds

# Data name
data_name=PTC

# Training program
python3 process_small_graphs.py --data_dir=$data_dir --data_name=$data_name --num_folds=$num_folds --num_train_folds=$num_train_folds