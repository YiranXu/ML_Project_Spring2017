
import pandas as pd
import csv
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from itertools import cycle


data = pd.read_csv('/Users/WeisenZhao/Documents/Weisen Zhao/Machine Learning/Project/Data/track_rate.csv'
                   , header = 0, dtype={'userID':'str','itemID':'str','label':'str'})


data.head()


data.shape



data = data[data['rating'] != 'rating']



trainningNum = math.floor(27168056 * 0.8)
data.head()



data['rating'] = data['rating'].astype('int')
trainning = data[:trainningNum]
testing = data[trainningNum:]


ave = data.groupby(data['itemID']).mean()
ave.columns = ['average rating']


ave.head()


dataWithAverage = data.join(ave, on = 'itemID')


dataWithAverage.head()


testingWithAverage = testing.join(ave, on = 'itemID')
testingWithAverage.head()



mean_squared_error(testingWithAverage['rating'], testingWithAverage['average rating'])



for index, row in testingWithAverage.iterrows():
    score = row['rating']
    if score >= 80:
        testingWithAverage.set_value(index, 'rating', 4)
    if score < 20:
        testingWithAverage.set_value(index, 'rating', 0)
    if score >= 20 and score < 40:
        testingWithAverage.set_value(index, 'rating', 1)
    if score >= 40 and score < 60:
        testingWithAverage.set_value(index, 'rating', 2)
    if score >= 60 and score < 80:
        testingWithAverage.set_value(index, 'rating', 3) 


testingWithAverage.head()
for index, row in testingWithAverage.iterrows():
    score = row['average rating']
    if score >= 80:
        testingWithAverage.set_value(index, 'average rating', 4)
    if score < 20:
        testingWithAverage.set_value(index, 'average rating', 0)
    if score >= 20 and score < 40:
        testingWithAverage.set_value(index, 'average rating', 1)
    if score >= 40 and score < 60:
        testingWithAverage.set_value(index, 'average rating', 2)
    if score >= 60 and score < 80:
        testingWithAverage.set_value(index, 'average rating', 3)

testingWithAverage.head()

fpr = dict()
tpr = dict()
roc_auc = dict()
y_test = testingWithAverage['rating']
y_test = label_binarize(y_test, classes=[0, 1, 2, 3, 4])
y_score = testingWithAverage['average rating']
y_score = label_binarize(y_score, classes=[0, 1, 2, 3, 4])
n_classes = y_score.shape[1]
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'r', 'g'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()