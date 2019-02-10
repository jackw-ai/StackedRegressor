from StackedRegressor import *
from models import *

param2 = {"objective":"reg:linear",
          "min_child_weight": 10, 
          'colsample_bytree': 0.8,
          'subsample': 0.8,
          'learning_rate': 0.1,
          'n_estimators': 50,
          'max_depth': 8, 
          'gamma': 0,       
          'alpha': 0.5,
          'seed': 102022
         }

xg2 = xgb.XGBRegressor(**param2)

stackedreg = StackedRegressor(xg2, models = [knn, knn2, ada, rfr, xg_reg, NN_model], model_names = ["KNN", "KNN2", "ADA", "RFR", "XGB", "NN"], trainx = X_train, trainy = y_train)

cv_results = stackedreg.cv(params = paramS, 
                    nfold = cv_folds,
                    early_stopping_rounds = early_stopping_rounds,
                    metrics = "rmse", 
                    as_pandas = True,
                    verbose_eval = True)

print(cv_results["test-rmse-mean"].tail(1))
cv_results.loc[cv_results['test-rmse-mean'].idxmin()]

res = stackedreg.fit()

generate_predictions(stackedreg)




