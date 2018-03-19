import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus
import matplotlib.pyplot as plt


def part7(depth, trainSet, trainLable, testSet, testLable, features, picName):
    # train the decision tree
    clf = DecisionTreeClassifier(criterion="entropy" ,max_depth=depth)
    clf.fit(trainSet, trainLable)

    # draw the tree with hegiht 3
    dot_data = StringIO()
    tree.export_graphviz(clf, feature_names=features, out_file=dot_data, max_depth=2)

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png(picName)

    # test the decision tree with test set and give accuracy
    pred = clf.predict(testSet)
    result = map(lambda x, y: np.abs(x-y), pred, testLable)
    return 1-sum(result)/490.0


if __name__ == "__main__":
    testSet, testTarget, trainSet, trainTarget, features = getSets(fake, real)[2:]
    depths = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    result = []
    for i in depths:
        accuracy = part7(i, trainSet, trainTarget, testSet, testTarget, features, 'tree' + str(i) + '.png')
        result.append(accuracy)
    plt.xlabel('max depths')
    plt.ylabel('accuracy of test set')
    plt.title('DT with entropy')
    plt.plot(depths, result)
