# Decoder as a simple dot product
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

torch.manual_seed(123456789)

dtype_float = torch.float
dtype_int = torch.int
device = torch.device("cpu")

class Dot_Decoder(nn.Module):
	def __init__(self):
		super(Dot_Decoder, self).__init__()

	def forward(self, X):
		# Prediction without Sigmoid
		predict = torch.matmul(X, torch.transpose(X, 0, 1))

		# Prediction with Sigmoid
		# predict = torch.sigmoid(torch.matmul(X, torch.transpose(X, 0, 1)))
		
		return predict