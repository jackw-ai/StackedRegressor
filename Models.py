# All the models in the first layer of the stacked regressor

from utils import *

from preprocess import *

# libraries for XGboost
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV

# libraries for random forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler

# libraries for NN
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten

# libraries for kNN
from sklearn.neighbors import KNeighborsRegressor

# libraries for Adaboost
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

# libraries for SVM
from sklearn import svm

# Load data sets
X_train, y_train, test = load_data()

""" Linear Regression """

# Linear regression
linReg = Lin_Reg()
linReg.fit(X_train, y_train)

scoreRMSE(linReg, X_train, y_train, plot = False)


""" XGBoost """

data_dmatrix = xgb.DMatrix(data = X_train, label = y_train)

# helper for plotting trees
def plot_xgb(xg_reg):
    xgb.plot_importance(xg_reg, max_num_features = 50)
    plt.show()
    
    xgb.plot_tree(xg_reg)
    plt.show()

# parameters
params = {
    'objective":"reg:linear',
    'min_child_weight': 0,
    'colsample_bytree': 0.6,
    'subsample': 0.9,
    'learning_rate': 0.01,
    'n_estimators': 525,
    'max_depth': 4,
    'gamma': 0,
    'alpha': 5,
    'random_state': 102022
    }

# gradient boost regression model
xg_reg = xgb.XGBRegressor(**params)

start = time.time()
xg_reg.fit(X_train,y_train)
print(elapsed_time(start))

# divide dataset into 5 folds
cv_folds = 5

# early stop after 50 rounds of no improvement
early_stopping_rounds = 50

cv_results = xgb.cv(dtrain = data_dmatrix,
                    params = params,
                    nfold = cv_folds,
                    num_boost_round = params['n_estimators'],
                    early_stopping_rounds = early_stopping_rounds,
                    metrics = "rmse",
                    as_pandas = True,
                    verbose_eval = True)

plot_loss(cv_results)

# fit optimal number of trees
xg_reg.set_params(n_estimators=cv_results.shape[0])

xg_reg.fit(X_train,y_train)

scoreRMSE(xg_reg, X_train, y_train)

""" Random Forest """

sc = StandardScaler()
rf_train = sc.fit_transform(X_train)

rfparams = {
    'n_estimators': 250,
    'max_depth': 8,
    'random_state': 40,
    'verbose': False
}

rfr = RandomForestRegressor(**rfparams)

rfr.fit(X_train, y_train)

scoreRMSE(rfr, X_train, y_train, plot = False)

scores = cross_validate(rfr, X_train, y_train, cv = 5, scoring = 'neg_mean_squared_error', verbose = True)

plot_loss(scores, model_type = 'rfr')


""" Adaboost """

ada = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 8),
                        n_estimators = 200,
                        learning_rate = 0.5,
                        loss = 'linear',
                        random_state = np.random.RandomState(1)
                        )

ada.fit(X_train, y_train)

scoreRMSE(ada, X_train, y_train, plot = False)

""" SVM """

svr = svm.SVR(cache_size = 500, kernel= 'linear')

svr.fit(X_train, y_train)

scoreRMSE(svr, X_train, y_train, plot = False)

""" Neural Net """
def build_NN():
    NN_model = Sequential()
    
    # Input Layer
    NN_model.add(Dense(128, kernel_initializer='normal',input_dim = X_train.shape[1], activation='relu'))

    # Hidden Layers
    NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
    NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))

    # Output Layer
    NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

    # compile NN
    NN_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    
    # get modell summary
    NN_model.summary()

    return NN_model

def fit_NN(X_train, y_train):
    NN_model = build_NN()
    history = NN_model.fit(X_train, y_train, epochs= 150, batch_size = 32, validation_split = 0.2)

generate_sub(NN_model)


""" KNN """
knn = KNeighborsRegressor(n_neighbors = 2)

knn.fit(X_train, y_train)

scoreRMSE(knn, X_train, y_train, plot = False)

knn2 = KNeighborsRegressor(n_neighbors = 3)
knn2.fit(X_train, y_train)

scoreRMSE(knn2, X_train, y_train, plot = False)



