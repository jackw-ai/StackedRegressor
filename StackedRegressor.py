from utils import *

class StackedRegressor():
  ''' Stacked Regression model that uses a set of models as first layer before passing output as second layer featurs to final model '''
  
  def __init__(self, final_model, models = [], model_names = [], trainx = pd.DataFrame(), trainy = pd.DataFrame()):
    ''' initializes second layer final_model as well as first layer models '''
    
    self.models = models
    self.final_model = final_model
    
    # optional names
    self.model_names = model_names

    # optional to initialize training set here
    if not trainx.empty and not trainy.empty:
      self.trainx = trainx
      self.trainy = trainy
     
  
  def add_model(self, model, model_name = None):
    ''' adds models to stack'''
    
    self.models.append(model)
    
    if model_name:
      self.model_names.append(model_name)
    
  def get_model(self, i):
    ''' retrieves models'''
    
    return self.models[i]
   
  def fit(self, trainx = False, trainy = False):
    ''' fits model and returns second level features'''
    
    # in case we didn't pass in training set yet
    if trainx and trainy:
      self.trainx = trainx
      self.trainy = trainy
    
    self._fit_bottom_layer()
    
    self._get_second_level_features(self.trainx)
    
    # pass in second level features
    self.final_model.fit(self.first_preds, self.trainy)
    
    # print total loss
    scoreRMSE(self.final_model, self.first_preds, self.trainy, plot= False)
    
    return self.first_preds
  
  def _fit_bottom_layer(self):
    ''' fits first layer of models '''
    
    print("Training first layer models...")

    for model in self.models:
      model.fit(self.trainx, self.trainy)
      
  def _get_second_level_features(self, trainx):
    ''' extract second layer features '''
    
    # build the second level features using first level models
    res = np.concatenate([np.vstack(model.predict(trainx)) for model in self.models], axis = 1)
    
    # optional: set column names to be model names
    if len(self.model_names) == len(self.models):
      self.first_preds = pd.DataFrame(data = res, columns = self.model_names)
    else:
      self.first_preds = pd.DataFrame(data = res)

    print("Second layer features computed...")
    
  def predict(self, test):
    ''' get predictions from stacked model'''
    self._get_second_level_features(test)
    
    return self.final_model.predict(self.first_preds)
  
  def cv(self, params, nfold = 5, early_stopping_rounds = 50, 
         metrics = "rmse", as_pandas = True, verbose_eval = True, plot = True):
    ''' k-folds cross-validation for xgboost'''
    
    data_dmatrix = xgb.DMatrix(self.trainx, label = self.trainy)
    
    try:
      cv_results = xgb.cv(dtrain = data_dmatrix, params = params, nfold = nfold, num_boost_round = params['n_estimators'], early_stopping_rounds = early_stopping_rounds, 
         metrics = metrics, as_pandas = as_pandas, verbose_eval = verbose_eval)
      
      if plot:
        plt.plot(cv_results["test-rmse-mean"], color='red', label = "test-rmse")
        plt.plot(cv_results["train-rmse-mean"], color = 'navy', label = "train-rmse")
        plt.title("RMSE")
        plt.legend()
        plt.show()
      
      return cv_results
    except:
      # self.model.cross_validate() 
      raise NotImplementedError
