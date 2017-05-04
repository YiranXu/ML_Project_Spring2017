import sys
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import *
from sklearn.preprocessing import label_binarize
from itertools import cycle
from sklearn.cross_validation import train_test_split
import warnings
import pickle


data = pd.read_csv('/scratch/yt1209/MLproject_data/xgboost/sample1.csv')


#fitting
X = data.ix[:,3:]
y = data.ix[:,2]


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8,random_state=42)

print "Data Splitted"
params = {'n_estimators': 10, 'max_depth': 15, 'learning_rate': 0.1}


xgboost = xgb.XGBRegressor(**params)
xgboost.fit(X_train, y_train)

filename = 'xgboost_model1.sav'
pickle.dump(xgboost, open(filename, 'wb'))


print 'Data Fitted'
y_pred = xgboost.predict(X_test)

print 'Prediction'

#evaluation
def root_mean_square_error(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_square_error(y_test, y_pred)

print mae
print rmse
