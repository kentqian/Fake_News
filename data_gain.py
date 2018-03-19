import numpy as np

fake = open("clean_fake.txt", 'r')
real = open("clean_real.txt", 'r')

def countWords(headline):
    wordCount = {}
    for w in headline:
        if w in wordCount.keys():
            wordCount[w] += 1
        else:
            wordCount[w] = 1
    return wordCount

def countWord(headline):
    wordCount = {}
    for w in headline:
        wordCount[w] = 1
    return wordCount

def createData(file):

    keyWords = {}
    # index of the headline
    i = 0
    for line in file:
        eachLine = line.split()
        #wordCount = countWords(eachLine)
        wordCount = countWord(eachLine)

        # for each word in new headline
        # if it appeared before, add 1 to keyWord dic
        # else set num of seen head lines 0s and add 1
        # to current index of head lines
        for word in wordCount:
            if word in keyWords.keys():
                keyWords[word].append(wordCount[word])
            else:
                newData = [0 for _ in range(i)]
                keyWords[word] = newData
                keyWords[word].append(wordCount[word])
        for w in keyWords.keys():
            if len(keyWords[w]) != i + 1:
                keyWords[w].append(0)
        i += 1
    return keyWords

def dataNaturalJoint(data1, data2):
    result = {}
    for k in data1.keys():
        if k in data2.keys():
            result[k] = data1[k] + data2[k]
        else:
            result[k] = data1[k] + [0 for _ in range(len(data2[data2.keys()[0]]))]
    for k in data2.keys():
        if k not in data1.keys():
            result[k] = [0 for _ in range(len(data1[data1.keys()[0]]))] + data2[k]
    return result

def dataLeftJoint(data1, data2):
    result = {}
    for k in data1.keys():
        if k in data2.keys():
            result[k] = data1[k] + data2[k]
    return result

def getSets(fileFake, fileClean):
    f = createData(fake)
    c = createData(real)
    data = dataNaturalJoint(f,c)


    datas = map(lambda x: data[x], data.keys())
    datas = np.vstack(tuple(datas))
    datas = datas.T

    # validation set 15% of the datas 490 and where first 245 are fake
    validationSet1 = datas[0:245]
    validationSet2 = datas[3021:3266]
    validationSet = np.vstack((validationSet1, validationSet2))

    validataionTarget = np.zeros((490,))
    validataionTarget[245:] = 1
    testTarget = np.zeros((490,))
    testTarget[245:] = 1

    # test set, first half are (245) are fake news head lines
    testSet1 = datas[245: 490]
    testSet2 = datas[2776: 3021]
    testSet = np.vstack((testSet1, testSet2))

    # training set
    trainSet1 = datas[490:1298]
    trainSet2 = datas[1298: 2776]
    trainSet = np.vstack((trainSet1, trainSet2))

    trainTarget = np.zeros((2776-490,))
    trainTarget[1298-490:] = 1
    return validationSet, validataionTarget, testSet, testTarget, trainSet, trainTarget, data.keys()

# testSet, testTarget, trainSet, trainTarget = getSets(fake, real)[2:]

# print trainSet
