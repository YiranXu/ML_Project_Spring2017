import pandas as pd
import numpy as np
import datetime
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import *
from sklearn.preprocessing import label_binarize
from itertools import cycle
from sklearn.cross_validation import train_test_split

data = pd.read_csv("fourmillion_withItemID_unwrapped.csv")

#fitting
X = data.ix[:,3:]
y = data.ix[:,2]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8,random_state=42)

params = {'n_estimators': 100, 'max_depth': 15, 'min_samples_split': 4,
          'learning_rate': 0.001, 'loss': 'ls'}


timestart = datetime.datetime.now()
clf = GradientBoostingRegressor(**params)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

#evaluation
def root_mean_square_error(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_square_error(y_test, y_pred)

print("MAE: %.4f" % mae)
print("MSE: %.4f" % mse)
print("RMSE: %.4f" % rmse)

timesend = datetime.datetime.now()
timedelta = round((timeend - timestart).total_seconds()//3600, 2) 
print("Time taken to execute this model is: " + str(timedelta) + " hours.\n")





timestart = datetime.datetime.now()

#grid search
param_grid_gbr = {'learning_rate':[0.001,0.005,0.01], 'n_estimators':[200, 500], 'max_depth':[5,10,15,20,25], 'min_samples_split':[6,10,15,20,30]}
gbr_grid_search = GridSearchCV(GradientBoostingRegressor(), param_grid_gbr, cv = 5, scoring = 'neg_mean_squared_error') 
gbr_grid_search.fit(X_train, y_train)


best = grid_search.best_estimator_
best.fit(X_train, y_train)
y_pred = best.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_square_error(y_test, y_pred)

print("MAE: %.4f" % mae)
print("MSE: %.4f" % mse)
print("RMSE: %.4f" % rmse)


timesend = datetime.datetime.now()
timedelta = round((timeend - timestart).total_seconds()//3600, 2) 
print("Time taken to execute this model is: " + str(timedelta) + " hours.\n")

