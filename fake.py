import random
from NB_part import part_2, part_3
from LR_part import part_4, part_6
from DT_part import part_7

def part_1(input):
	with open(input) as f:
		content = f.readlines()
	dic = {}
	news_total_list = []
	total_headlines = len(content)

	for headtitle in content:
		one_news = list(set(headtitle.strip().split()))
		news_total_list.append(one_news)
		for word in one_news:
			dic[word] = dic.get(word,0) + 1

	sorted_list = sorted(dic.iteritems(), key=lambda (k,v): (v,k), reverse=True)

	# for i in range(0,15):
	# 	print 'Words: {}, Frequency:{:10.2f}%.\n'.format(sorted_list[i][0], int(sorted_list[i][1])*100.0/total_headlines)
	
	# for i in ['hillary','says','clinton']:
	# 	if i in dic.keys():
	# 		print 'Words: {}, Frequency:{:10.2f}%.\n'.format(i, int(dic[i])*100.0/total_headlines)
	# 	else:
	# 		print 'Words: {}, Frequency:{:10.2f}%.\n'.format(i, 0)

	random.shuffle(news_total_list)

	training_set = news_total_list[0:total_headlines*70/100]
	validation_set = news_total_list[total_headlines*70/100:total_headlines*70/100 + total_headlines*15/100]
	test_set = news_total_list[total_headlines*70/100 + total_headlines*15/100:total_headlines]
	
	return training_set, validation_set, test_set, news_total_list, total_headlines

def tune_performance_part2(real_train, real_vali, real_test, fake_train, fake_vali, fake_test):
	accuracy_list = []
	mp_list = []

	m = 1
	for m in [10,50,100,500,1000]:
		for p_hat in [0.1,0.05,0.01,0.005,0.001]:
			print m, p_hat
			one_accuracy = part_2(real_train, real_vali, real_test, fake_train, fake_vali, fake_test, m, p_hat)
			accuracy_list.append(one_accuracy)
			mp_list.append("{},{}".format(m,p_hat))

	index_max = np.argmax(accuracy_list)
	print "Final Pick: " + mp_list[index_max]


# print '========real_news========='
# real_train, real_vali, real_test, real_total_news, real_total = part_1("clean_real.txt")
# print '========fake_news========='
# fake_train, fake_vali, fake_test, fake_total_news, fake_total = part_1("clean_fake.txt")

# tune_performance_part2(real_train, real_vali, real_test, fake_train, fake_vali, fake_test)

# part_2(real_train, real_vali, real_test, fake_train, fake_vali, fake_test, 1000, 0.001)

# part_3(real_total_news, fake_total_news, real_total, fake_total, 1000, 0.001, 'a')

# part_3(real_total_news, fake_total_news, real_total, fake_total, 1000, 0.001, 'b')

model, total_words = part_4()
part_6(model, total_words,'a')
# part_6(model, total_words,'b')

# part_7()



