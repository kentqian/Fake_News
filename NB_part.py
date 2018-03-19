import operator
import random
import math
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

random.seed(10)

def NB_preprocess(word_set, m, p_hat):	
	train_dic = {}

	# Count the existence of each word
	for headtitle in word_set:
		for word in headtitle:
			train_dic[word] = train_dic.get(word,0) + 1

	# Create a dictionary for each word with P(word_i|class)
	for word in train_dic:
		train_dic[word] = (train_dic[word] + (m*p_hat)) / float(len(word_set) + m)

	return train_dic


def NB_prediction(train_dic, laplacian, headtitle, prior):
	P_features = 0

    # Get P(headline|class)
	for word in headtitle:
		P_specific_word = train_dic.get(word,None)
		if (not P_specific_word):
			P_features += math.log(laplacian)
		else:
			P_features += math.log(P_specific_word)

	for word in train_dic:
		if word not in headtitle:
			P_features += math.log(1 - train_dic[word])
    
    # Get the prediction
	return math.exp(P_features + math.log(prior))

def NB_accuracy(real_train_dic, fake_train_dic, laplacian_real, laplacian_fake, P_real, P_fake, data_set_real, data_set_fake):
	real_accurate_count = 0
	fake_accurate_count = 0
	accurate_count = 0

	 # Determine by the greater values between P_real_predict and P_fake_predict
	for headtitle in data_set_real:
		P_real_predict = NB_prediction(real_train_dic, laplacian_real, headtitle, P_real)
		P_fake_predict = NB_prediction(fake_train_dic, laplacian_fake, headtitle, P_fake)
		if P_real_predict > P_fake_predict:
			real_accurate_count += 1
			accurate_count += 1

	for headtitle in data_set_fake:
		P_real_predict = NB_prediction(real_train_dic, laplacian_real, headtitle, P_real)
		P_fake_predict = NB_prediction(fake_train_dic, laplacian_fake, headtitle, P_fake)
		if P_real_predict < P_fake_predict:
			fake_accurate_count += 1
			accurate_count += 1

	print "Real Accuracy: {:10.2f}%".format(real_accurate_count*100.0/len(data_set_real))
	print "Fake Accuracy: {:10.2f}%".format(fake_accurate_count*100.0/len(data_set_fake))
	print "Total Accuracy: {:10.2f}%".format(accurate_count*100.0/(len(data_set_fake) + len(data_set_real)))

	return accurate_count*100.0/(len(data_set_fake) + len(data_set_real))

def part_2(real_train, real_vali, real_test, fake_train, fake_vali, fake_test, m, p_hat):
	P_real = len(real_train) / float(len(real_train)+len(fake_train))
	P_fake = len(fake_train) / float(len(real_train)+len(fake_train))

	laplacian_real = m*p_hat / float(len(real_train)+m)
	laplacian_fake = m*p_hat / float(len(fake_train)+m)

	real_train_dic = NB_preprocess(real_train, m, p_hat)
	fake_train_dic = NB_preprocess(fake_train, m, p_hat)


	print "============training set==============="
	NB_accuracy(real_train_dic, fake_train_dic, laplacian_real, laplacian_fake, P_real, P_fake, real_train, fake_train)
	print "===========validation set=============="
	vali_acc = NB_accuracy(real_train_dic, fake_train_dic, laplacian_real, laplacian_fake, P_real, P_fake, real_vali, fake_vali)
	print "==============test set================="
	NB_accuracy(real_train_dic, fake_train_dic, laplacian_real, laplacian_fake, P_real, P_fake, real_test, fake_test)

	return vali_acc

def P_word_given_class_times_piror(word_set, piror, m, p_hat, flag):	
	set_dic = {}

	for headtitle in word_set:
		for word in headtitle:
			set_dic[word] = set_dic.get(word,0) + 1

	if flag:
		for word in set_dic:
			set_dic[word] = math.exp(math.log((set_dic[word] + (m*p_hat)) / float(len(word_set) + m)) + math.log(piror))
	else:
		for word in set_dic:
			set_dic[word] = math.exp(math.log(1 - ((set_dic[word] + (m*p_hat)) / float(len(word_set) + m))) + math.log(piror))

	return set_dic

def create_P_word_dic(real_numerator_dic, fake_numerator_dic, laplacian_fake, laplacian_real, P_fake, P_real):
	total_words = list(set(real_numerator_dic.keys() + fake_numerator_dic.keys()))

	P_word_dic = {}
	for word in total_words:
		if real_numerator_dic.get(word,None) and fake_numerator_dic.get(word, None):
			P_word_dic[word] = real_numerator_dic[word] + fake_numerator_dic[word]
		elif real_numerator_dic.get(word,None) and not fake_numerator_dic.get(word, None):
			P_word_dic[word] = real_numerator_dic[word] + math.exp(math.log(laplacian_fake) + math.log(P_fake))
		elif not real_numerator_dic.get(word,None) and fake_numerator_dic.get(word, None):
			P_word_dic[word] = fake_numerator_dic[word] + math.exp(math.log(laplacian_real) + math.log(P_real))

	return P_word_dic
	
def prior_and_laplacian(real_total, fake_total, m, p_hat):
	P_real = real_total / float(real_total+fake_total)
	P_fake = fake_total / float(real_total+fake_total)

	laplacian_real = m*p_hat / float(real_total+m)
	laplacian_fake = m*p_hat / float(fake_total+m)

	return P_real, P_fake, laplacian_real, laplacian_fake

def P_c_given_dic(numerator_dic, ab_numerator_dic, P_word_dic):
	P_c_given_word_dic = {}
	P_c_given_not_word_dic = {}

	for word in numerator_dic:
		P_c_given_word_dic[word] = numerator_dic[word] / P_word_dic[word]

	for word in numerator_dic:
		P_c_given_not_word_dic[word] = ab_numerator_dic[word] / (1 - P_word_dic[word])

	return P_c_given_word_dic, P_c_given_not_word_dic

def part_3(real_total_news, fake_total_news, real_total, fake_total, m, p_hat, flag):

	P_real, P_fake, laplacian_real, laplacian_fake = prior_and_laplacian(real_total, fake_total, m, p_hat)

	real_numerator_dic = P_word_given_class_times_piror(real_total_news, P_real, m, p_hat, True)
	fake_numerator_dic = P_word_given_class_times_piror(fake_total_news, P_fake, m, p_hat, True)

	ab_real_numerator_dic = P_word_given_class_times_piror(real_total_news, P_real, m, p_hat, False)
	ab_fake_numerator_dic = P_word_given_class_times_piror(fake_total_news, P_fake, m, p_hat, False)	

	P_word_dic =  create_P_word_dic(real_numerator_dic, fake_numerator_dic, laplacian_fake, laplacian_real, P_fake, P_real)

	P_real_given_word_dic, P_real_given_not_word_dic = P_c_given_dic(real_numerator_dic, ab_real_numerator_dic, P_word_dic)
	P_fake_given_word_dic, P_fake_given_not_word_dic = P_c_given_dic(fake_numerator_dic, ab_fake_numerator_dic, P_word_dic)

	a = sorted(P_real_given_word_dic.iteritems(), key=lambda (k,v): (v,k), reverse=True)
	b = sorted(P_fake_given_word_dic.iteritems(), key=lambda (k,v): (v,k), reverse=True)
	c = sorted(P_real_given_not_word_dic.iteritems(), key=lambda (k,v): (v,k), reverse=True)
	d = sorted(P_fake_given_not_word_dic.iteritems(), key=lambda (k,v): (v,k), reverse=True)

	if flag == 'a':
		print "=====P(real|word)====="
		print "10 words whose presence most strongly predicts that the news is real"
		for i in range(0,10):
			print "Word: {}, Prob: {}%.".format(a[i][0],a[i][1]*100)
		print "\n"
		print "=====P(fake|word)====="
		print "10 words whose presence most strongly predicts that the news is fake"
		for i in range(0,10):
			print "Word: {}, Prob: {}%.".format(b[i][0],b[i][1]*100)
		print "\n"
		print "===P(real|not word)==="
		print "10 words whose absence most strongly predicts that the news is real"
		for i in range(0,10):
			print "Word: {}, Prob: {}%.".format(c[i][0],c[i][1]*100)
		print "\n"
		print "===P(fake|not word)==="
		print "10 words whose absence most strongly predicts that the news is fake"
		for i in range(0,10):
			print "Word: {}, Prob: {}%.".format(d[i][0],d[i][1]*100)
	elif flag == 'b':
		a = [element for element in a if element[0] not in ENGLISH_STOP_WORDS]
		print "=====P(real|non-stopword)====="
		for i in range(0,10):
			print "Word: {}, Prob: {}%.".format(a[i][0],a[i][1]*100)
		print "\n"
		b = [element for element in b if element[0] not in ENGLISH_STOP_WORDS]
		print "=====P(fake|non-stopword)====="
		for i in range(0,10):
			print "Word: {}, Prob: {}%.".format(b[i][0],b[i][1]*100)


	
