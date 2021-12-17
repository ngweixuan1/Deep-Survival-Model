import os
import pickle
import pandas as pd
import random
random.seed(1)
from lifelines.utils import concordance_index
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import math
import random
import numpy as np
import pandas as pd
from scipy import sparse
from torch.utils.data import TensorDataset, Dataset
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

'''
Input: Train, test, validation split for labels and features
Output: Training and validation curves, validation and test scores
'''

#GRU: 32 hidden units bidirectional,, 1 layer of GRU
#Linear Layer: Output layer 2 output units


# load dataset
PATH_OUTPUT = "../../models/"
train_labels = pd.read_csv("../../data/train_labels.csv")
valid_labels = pd.read_csv("../../data/valid_labels.csv")
test_labels = pd.read_csv("../../data/test_labels.csv")

train_seqs = pd.read_csv("../../data/train_seqs.csv")
valid_seqs = pd.read_csv("../../data/valid_seqs.csv")
test_seqs = pd.read_csv("../../data/test_seqs.csv")

num_features = train_seqs.shape[1]-3

# Hyperparameters here
USE_CUDA = True 
BATCH_SIZE = 64
learning_rate = 0.001
dropout = 0.3
NUM_EPOCHS = 15
num_layers = 1
device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA == True else "cpu")
torch.manual_seed(0)


def plot_learning_curves(train_losses, valid_losses, train_c_indexes, valid_c_indexes):
	# plot training and validation loss curves
	plt.plot(list(range(len(train_losses))), train_losses, color='blue', label = "Training Loss")
	plt.plot(list(range(len(valid_losses))), valid_losses, color='orange', label = "Validation Loss")
	plt.title("Loss Curve")
	plt.ylabel('Loss')
	plt.xlabel('epoch')
	plt.legend(loc = 'upper right')
	plt.ion()
	plt.show(block=True)

	plt.clf()
	# plot training and validation concrordance index curves
	plt.plot(list(range(len(train_c_indexes))), train_c_indexes, color='blue', label = "Training C Index")
	plt.plot(list(range(len(valid_c_indexes))), valid_c_indexes, color='orange', label = "Validation C Index")
	plt.title("Concordance Index Curve")
	plt.ylabel('C Index')
	plt.xlabel('epoch')
	plt.legend(loc = 'upper right')
	plt.ion()
	plt.show(block=True)

	plt.clf()		


class featureslabel(Dataset):
	# Approach similar to HW5 part2, heavily modified code
	def __init__(self, seqs, labels, num_periods = 1024):
		
		self.labels = labels[['num_days',	'mortality']].values
		seqs_id = list(seqs['subject_id'].unique())
		a,b = seqs.shape

		total_seq = torch.zeros(len(seqs_id),num_periods,b-3)
		for i, id in enumerate(seqs_id):
				# Drop irrelevant columns
				temp_seq = seqs[ seqs['subject_id'] == id].drop(['Unnamed: 0', 'subject_id','period','record_date', 'first_admittime',	'end_time_icu', 'SUBJECT_ID',	'DOD',	'num_days1',	'num_days',	'mortality'] ,axis=1)
				# For each id, place them into the correct tensor
				total_seq[i, :temp_seq.shape[0], :temp_seq.shape[1]] = torch.tensor(temp_seq.values)
		self.seqs = total_seq

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, index):
		return self.seqs[index], self.labels[index]

# Compute concordence index
def compute_c_index(output, target):
	with torch.no_grad():
		a,b =  output[:, 0].cpu(), output[:, 1].cpu()
		c_index = concordance_index(target[:,0].cpu(), a, b)
		return c_index

def train(model, device, data_loader, criterion, optimizer, epoch):
	# Approach similar to HW5 part2, heavily modified code
	losses = []
	c_index = []
	model.train()
	for i, (input, target) in enumerate(data_loader):
		if isinstance(input, tuple):
			input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
		else:
			input = input.to(device)
		target = target.to(device)
		# Optimize
		optimizer.zero_grad()
		output = model(input)
		# Loss
		loss = criterion(output, target)
		loss.backward()
		#Append Concordence index and loss
		optimizer.step()
		losses.append(loss.item())
		
		c_index.append(compute_c_index(output, target).item())

	# returns loss and concordence index for training
	return sum(losses)/len(losses),sum(c_index)/len(c_index)


def evaluate(model, device, data_loader, criterion):
	# Approach similar to HW5 part2, heavily modified code
	losses = []
	c_index = []
	results = []
	model.eval()
	with torch.no_grad():
		for i, (input, target) in enumerate(data_loader):
			if isinstance(input, tuple):
				input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
			else:
				input = input.to(device)
			target = target.to(device)
			output = model(input)
			# Loss
			loss = criterion(output, target)
			# Append losses and c index
			losses.append(loss.item())
			c_index.append(compute_c_index(output, target).item())
			y_true = target.detach().to('cpu').numpy().tolist()
			y_pred = output.detach().to('cpu').max(1)[1].numpy().tolist()
			results.extend(list(zip(y_true, y_pred)))


	return sum(losses)/len(losses), sum(c_index)/len(c_index), results

def custom_loss(output, target):
		# References: Martinsson E (2016). WTTE-RNN: Weibull time to event recurrent neural network. Master’s thesis, University of Gothenburg, Sweden.	
		a,b = output[:, 0], output[:, 1]
		y, u = target[:, 0], target[:, 1]
		hazard_r1 = torch.pow((y+1e-30)/a,b)
		hazard_r2 = torch.pow((y+1)/a, b)
		loss = -torch.mean(torch.log(torch.exp(hazard_r2-hazard_r1) -1.0)*u-hazard_r2)
		return loss

# final model
class GRUnetwork(nn.Module):
	def __init__(self, dim_input, init_alpha = 1.0, dropout =0.1, num_layers=1):
		# initialize linear layers, sequence based layers etc		
		super(GRUnetwork, self).__init__()
		self.dim_input = dim_input
		self.rnn1 = nn.GRU(input_size=dim_input, hidden_size=32, batch_first=True, num_layers = num_layers, bidirectional=True, dropout = dropout)
		self.linear2 = nn.Linear(in_features = 64, out_features =16)
		self.linear3 = nn.Linear(in_features = 16, out_features =2)
		self.relu = nn.ReLU()
		self.init_alpha = init_alpha
		self.batchnorm1 = torch.nn.BatchNorm1d(num_features = 125)
		self.batchnorm2 = torch.nn.BatchNorm1d(num_features = 64)
		self.batchnorm3 = torch.nn.BatchNorm1d(num_features = 16)
	
	# activation function to get alpha and beta  
	def activation_function(self, output):
		a = self.init_alpha * torch.exp(output[:, 0])
		b = torch.nn.functional.softplus(output[:, 1])
		a = torch.reshape(a, (a.shape[0], 1))
		b = torch.reshape(b, (b.shape[0], 1))

		final_output = torch.cat((a, b), axis=1)
		return final_output
						
	def forward(self, input_tuple):
		# forward pass
		seqs = input_tuple

		lengths = torch.tensor(seqs.shape[1]).repeat(seqs.shape[0])
		seqs = pack_padded_sequence(self.batchnorm1(seqs), lengths.cpu(), batch_first=True)
		seqs, shape_out = self.rnn1(seqs)
		seqs, len_out = pad_packed_sequence(seqs, batch_first=True)
		seqs = self.linear2(self.batchnorm2(seqs[:, 0]))
		seqs = self.linear3(self.relu(seqs))
		# Activation function included in the forward function for loss calculation later
		seqs = self.activation_function(seqs)
		
		return seqs

# Create object with features and labels
num_periods = train_seqs.period.max()+1 # num_periods for padding
train_dataset = featureslabel(train_seqs, train_labels, num_periods = num_periods)
valid_dataset = featureslabel(valid_seqs, valid_labels, num_periods =num_periods )
test_dataset = featureslabel(test_seqs, test_labels, num_periods =num_periods )

# Load into dataloader
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# alpha intialized to mean in order to factilitate faster training
init_alpha = np.mean(train_dataset.labels[:,0])
print("Training started")
model = GRUnetwork(num_features, init_alpha = init_alpha, dropout = dropout, num_layers=num_layers)
criterion = custom_loss
optimizer = optim.Adam(model.parameters(), lr= learning_rate)
model.to(device)

best_c_indexes = 0.0
train_losses, train_mses, train_maes, train_c_indexes = [], [], [], []
valid_losses, valid_mses, valid_maes, valid_c_indexes = [], [], [], []

# Track loss and metrices
for epoch in range(NUM_EPOCHS):
	train_loss, train_c_index = train(model, device, train_loader, criterion, optimizer, epoch)
	valid_loss, valid_c_index, valid_results = evaluate(model, device, valid_loader, criterion)
	train_losses.append(train_loss)
	valid_losses.append(valid_loss)
	train_c_indexes.append(train_c_index)
	valid_c_indexes.append(valid_c_index)
	if valid_c_index > best_c_indexes :
		best_c_indexes = valid_c_index
		best_model = model
# Test
print("Training completed")
plot_learning_curves(train_losses, valid_losses, train_c_indexes, valid_c_indexes)
test_loss, test_c_index, test_results = evaluate(best_model, device, test_loader, criterion)
print( "Test Concordance Index: ", test_c_index)

