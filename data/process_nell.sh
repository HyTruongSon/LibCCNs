# Dataset's directory
data_dir=NELL/nell_data

# Labelling rate
rate=0.1

# Training program
python3 process_nell.py --data_dir=$data_dir --rate=$rate

# Labelling rate
rate=0.01

# Training program
python3 process_nell.py --data_dir=$data_dir --rate=$rate

# Labelling rate
rate=0.001

# Training program
python3 process_nell.py --data_dir=$data_dir --rate=$rate