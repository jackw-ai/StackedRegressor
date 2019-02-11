# Import libraries

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as Lin_Reg
from sklearn import metrics
from sklearn.metrics import mean_squared_error
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import scipy as sp

from collections import Counter
import time
import math

# libraries for boosting
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


""" Timer Function """

def elapsed_time(since):
    ''' for keeping track of training and running time '''
    def minutes(s):
        ''' formats to minutes '''
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    now = time.time()
    s = now - since
    return 'time elapsed: %s (%s to %s)' % (minutes(s), minutes(since), minutes(now))


""" Computing RMSE """

def scoreRMSE(predictor, X, true_y, plot = True):
    ''' function to compute RMSE '''
    
    predictions = predictor.predict(X)
    print("\nModel Report")
    print ("RMSE Score : %f" % np.sqrt(mean_squared_error(predictions, true_y)))
    accuracy = predictor.score(X, true_y) * 100
    print("Accuracy : %.4g" % accuracy, '%')         

    if plot:
        # Print f-score:    
        feat_imp = pd.Series(predictor.get_booster().get_fscore()).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='f-score', figsize = (20, 10))
        plt.ylabel('f-score')


""" Generating Output """

def generate_predictions(model, test, filename = "submissions.csv"):
    ''' generates predictions to csv file '''
    
    # remove first column (ID) to make predictions
    X_test = test.iloc[:, 1:]
    X_test.head()
    
    # make prediction
    predictions = model.predict(X_test)
    
    # format predictions with Id column
    submission = pd.DataFrame(data=predictions, columns=['Predicted'])
    submission.insert(0, "Id", range(1, 1 + X_test.shape[0]))
    submission['Id'] = submission['Id'].astype(str)
    
    # Save predictions to .csv file
    submission.to_csv(filename, index = False)
    print("saved predictions to ", filename)
    return submission
