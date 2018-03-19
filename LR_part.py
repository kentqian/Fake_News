import torch
import torchvision.models as models
import torchvision
from torch.autograd import Variable
import numpy as np
import  matplotlib.pyplot as plt
from scipy.misc import imread,imresize,imshow
import torch.nn as nn

from data_gain import getSets

dtype_float = torch.FloatTensor
dtype_long = torch.LongTensor

fake = open("clean_fake.txt", 'r')
real = open("clean_real.txt", 'r')

def random_permutation(seed, train_x):
	np.random.seed(seed)
	train_idx = np.random.permutation(range(train_x.shape[0]))
	return train_idx

def learning_part(training_matrix, label_matrix, train_idx, model):
	x = Variable(torch.from_numpy(training_matrix[train_idx]), requires_grad=False).type(dtype_float)
	y_classes = Variable(torch.from_numpy(label_matrix[train_idx]), requires_grad=False).type(dtype_long)

	loss_fn = torch.nn.CrossEntropyLoss()

	learning_rate = 1e-3
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 0.001)

	for t in range(1000):
		y_pred = model(x)
		loss = loss_fn(y_pred, y_classes)
		model.zero_grad()  # Zero out the previous gradient computation
		loss.backward()    # Compute the gradient
		optimizer.step()

		if t % 1000 == 0 :
			print "Current loss for one minibatch: " + str(loss.data.numpy()[0])

def get_accuracy(model, vali_matrix, vali_label, test_matrix, test_label, training_matrix, training_label):
	for i in range(0,1):
		train_idx = random_permutation(i, training_matrix)
		batches = np.array_split(train_idx, 1)

		for j in range(0,len(batches)):
			print "Epoch No. %s with minibatch No. %s. " %(i + 1, j + 1)

			learning_part(training_matrix, training_label, batches[j], model)

			train_x = Variable(torch.from_numpy(training_matrix), requires_grad=False).type(dtype_float)
			y_pred = model(train_x).data.numpy()
			print "Minibatch Accuracy(training_set): %s. " %(np.mean(np.argmax(y_pred, 1) == training_label))

			vali_x = Variable(torch.from_numpy(vali_matrix), requires_grad=False).type(dtype_float)
			y_pred = model(vali_x).data.numpy()
			print "Minibatch Accuracy(validation_set): %s. " %(np.mean(np.argmax(y_pred, 1) == vali_label))

			test_x = Variable(torch.from_numpy(test_matrix), requires_grad=False).type(dtype_float)
			y_pred = model(test_x).data.numpy()
			print "Minibatch Accuracy(test_set): %s. " %(np.mean(np.argmax(y_pred, 1) == test_label))


class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        out = self.linear(x)
        return out

vali_matrix, vali_label, test_matrix, test_label, training_matrix, training_label, total_words = getSets(fake, real)

if __name__ == '__main__':

	print "==============Data Ready==============="

	model = LogisticRegression(training_matrix.shape[1],2)

	get_accuracy(model, vali_matrix, vali_label, test_matrix, test_label, training_matrix, training_label)

	weight = model.linear.weight.data.numpy()
	for i in range(1,-1,-1):
		a = np.argsort(weight)[i,-10:][::-1]
		print "======weight top10========"
		print weight[i,a]
		print list(np.array(total_words)[a])

		b = np.argsort(weight)[i,:10]
		print "======weight low10========"
		print weight[i,b]
		print list(np.array(total_words)[b])

