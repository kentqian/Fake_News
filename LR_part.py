import torch
import torchvision.models as models
import torchvision
from torch.autograd import Variable
import numpy as np
import  matplotlib.pyplot as plt
from scipy.misc import imread,imresize,imshow
import torch.nn as nn
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from data_gain import getSets

dtype_float = torch.FloatTensor
dtype_long = torch.LongTensor

fake = open("clean_fake.txt", 'r')
real = open("clean_real.txt", 'r')

def random_permutation(seed, train_x):
	np.random.seed(seed)
	train_idx = np.random.permutation(range(train_x.shape[0]))
	return train_idx

def learning_part(training_matrix, training_label,vali_matrix, vali_label, test_matrix, test_label, train_idx, model):
	x = Variable(torch.from_numpy(training_matrix[train_idx]), requires_grad=False).type(dtype_float)
	y_classes = Variable(torch.from_numpy(training_label[train_idx]), requires_grad=False).type(dtype_long)

	train_x = Variable(torch.from_numpy(training_matrix), requires_grad=False).type(dtype_float)
	vali_x = Variable(torch.from_numpy(vali_matrix), requires_grad=False).type(dtype_float)
	test_x = Variable(torch.from_numpy(test_matrix), requires_grad=False).type(dtype_float)

	loss_fn = torch.nn.CrossEntropyLoss()

	train_acc_list = []
	vali_acc_list = []
	test_acc_list = []

	learning_rate = 1e-3
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 0.001)

	for t in range(10000):
		y_pred = model(x)
		loss = loss_fn(y_pred, y_classes)
		model.zero_grad()  # Zero out the previous gradient computation
		loss.backward()    # Compute the gradient
		optimizer.step()

		if t % 500 == 0 :
			print "Current loss for one minibatch: " + str(loss.data.numpy()[0])
			y_pred = model(train_x).data.numpy()
			train_acc_list.append(np.mean(np.argmax(y_pred, 1) == training_label))
			print "Minibatch Accuracy(training_set): %s. " %(np.mean(np.argmax(y_pred, 1) == training_label))

			y_pred = model(vali_x).data.numpy()
			vali_acc_list.append(np.mean(np.argmax(y_pred, 1) == vali_label))
			print "Minibatch Accuracy(validation_set): %s. " %(np.mean(np.argmax(y_pred, 1) == vali_label))

			y_pred = model(test_x).data.numpy()
			test_acc_list.append(np.mean(np.argmax(y_pred, 1) == test_label))
			print "Minibatch Accuracy(test_set): %s. " %(np.mean(np.argmax(y_pred, 1) == test_label))

	return train_acc_list, vali_acc_list, test_acc_list

def get_accuracy(model, vali_matrix, vali_label, test_matrix, test_label, training_matrix, training_label):
	size = np.arange(0,20,1)
	for i in range(0,1):
		train_idx = random_permutation(i, training_matrix)
		batches = np.array_split(train_idx, 1)

		for j in range(0,len(batches)):
			print "Epoch No. %s with minibatch No. %s. " %(i + 1, j + 1)
			train_acc_list, vali_acc_list, test_acc_list = learning_part(training_matrix, training_label,vali_matrix, vali_label, test_matrix, test_label, batches[j], model)
	
	line1,line2,line3 = plt.plot(size,train_acc_list,size,vali_acc_list,size,test_acc_list)
	plt.legend((line1,line2,line3),("training_set","validation_set","test_set"))
	plt.show()


def learning_part_tP(training_matrix, training_label,vali_matrix, vali_label, model, learning_rate, L2):
	x = Variable(torch.from_numpy(training_matrix), requires_grad=False).type(dtype_float)
	y_classes = Variable(torch.from_numpy(training_label), requires_grad=False).type(dtype_long)

	vali_x = Variable(torch.from_numpy(vali_matrix), requires_grad=False).type(dtype_float)

	loss_fn = torch.nn.CrossEntropyLoss()

	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = L2)

	for t in range(10000):
		y_pred = model(x)
		loss = loss_fn(y_pred, y_classes)
		model.zero_grad()  # Zero out the previous gradient computation
		loss.backward()    # Compute the gradient
		optimizer.step()

	y_pred = model(vali_x).data.numpy()
	acc = np.mean(np.argmax(y_pred, 1) == vali_label)
	print "Minibatch Accuracy(validation_set): %s. " %(acc)
	return acc

def LR_tune_perform(training_matrix, training_label,vali_matrix, vali_label, model):
	accuracy_list = []
	learning_fig_list = []

	for learning_rate in [1e-1, 1e-2, 1e-3]:
		for L2 in [0.1, 0.05, 0.01, 0.001]:
			print learning_rate, L2
			one_accuracy = learning_part_tP(training_matrix, training_label,vali_matrix, vali_label, model, learning_rate, L2)
			accuracy_list.append(one_accuracy)
			learning_fig_list.append("{},{}".format(learning_rate,L2))

	index_max = np.argmax(accuracy_list)
	print "Final Pick: " + learning_fig_list[index_max]

def part_4():
	vali_matrix, vali_label, test_matrix, test_label, training_matrix, training_label, total_words = getSets(fake, real)
	print "==============Data Ready==============="

	model = LogisticRegression(training_matrix.shape[1],2)

	get_accuracy(model, vali_matrix, vali_label, test_matrix, test_label, training_matrix, training_label)

	return model, total_words

def part_6(model, total_words, flag):
	weight = model.linear.weight.data.numpy()
	if flag == 'a':
		a = np.argsort(weight)[1,-10:][::-1]
		print "======weight top10========"
		print weight[1,a]
		print list(np.array(total_words)[a])

		b = np.argsort(weight)[1,:10]
		print "======weight low10========"
		print weight[1,b]
		print list(np.array(total_words)[b])
	elif flag == 'b':
		print "======weight top10========"
		a = np.argsort(weight)[1][::-1]
		a = [element for element in list(np.array(total_words)[a]) if element not in ENGLISH_STOP_WORDS]
		print a[:10]
		print "======weight low10========"
		b = np.argsort(weight)[1]
		b = [element for element in list(np.array(total_words)[b]) if element not in ENGLISH_STOP_WORDS]
		print b[:10]


class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        out = self.linear(x)
        return out


