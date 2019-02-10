from utils import *

import seaborn as sns

from sklearn.feature_selection import mutual_info_regression

""" Preprocessing """

TRAIN_PATH = "./data/train.csv"
TEST_PATH = "./data/test.csv"

def load_data(plot = False):
    '''
        loads training and test data
        Optional: plot histogram of training data
    '''
    
    print("loading data...")
    
    # load train and test sets
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    print("splitting training data...")
    
    # split training set into x and y
    # removes ID column
    X_train = train.iloc[:, 1:-1]
    y_train = train.iloc[:, -1]

    print("Training data shape, ", X_train.shape)

    # check null values
    if X_train.isnull().values.any():
        print("NOTE: Contains null values, do something?")
    
    if plot:
        #  plot histogram of target
        plt.hist(train['Target'], bins = 100)
        plt.title("Distribution of Dependent Variable")
        plt.show()

    print("Loaded successfully")
    
    return X_train, y_train, test

""" Visualization """

def plot_heatmap(train):
    ''' plot heatmap of features '''
    
    sns.set(style = "white")
    
    
    # compute correlation matrix
    corr = train.iloc[:, 1:].corr()
    
    # matplotlib figure
    f, ax = plt.subplots(figsize = (20, 9))
    
    # custom diverging colormap
    cmap = sns.diverging_palette(200, 20, as_cmap = True)
    
    # heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, cmap = cmap, vmax = .3, center = 0,
                square = True, cbar_kws = {"shrink": .5})
                
    plt.show()

def plot_loss(history, model_type = 'xgb'):
    # plot loss graph
    
    if model_type = 'xgb':
        plt.plot(history["test-rmse-mean"], color = 'red', label = "test-rmse")
        plt.plot(history["train-rmse-mean"], color = 'navy', label = "train-rmse")
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('n_rounds')
        plt.legend(loc = 'upper right')
        plt.show()

    elif model_type = 'nn':
        
        plt.plot(history.history['loss'], color = 'red', label = 'train')
        plt.plot(history.history['val_loss'], color = 'navy', label = 'validation')
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(loc = 'upper left')
        plt.show()

    elif model_type == 'rfr':
        plt.plot(history['test_score'], color = 'red', label = "test-rmse")
        plt.plot(history['train_score'], color = 'navy', label = "train-rmse")
        plt.title('model loss')
        plt.ylabel('error')
        plt.xlabel('cv')
        plt.legend(loc = 'upper right')
        plt.show()




""" Feature Selection """

def filter_features(X_train, y_train, threshold = 0):
    '''
        filters out irrelevant data points with correlation
        less than threshold value.
        
        Mutual information between two random variables is a non-negative
        value, which measures the dependency between the variables. It is
        equal to zero if and only if two random variables are independent,
        and higher values mean higher dependency.
    '''
    
    mir = mutual_info_regression(X_train, y_train)
    
    mirfeature_scores = sorted([(mir[i], i) for i in range(251)])
    
    filterd_features = [feature[1] for feature in mirfeature_scores if feature[0] > threshold]
    
    filtered_trainingset = X_train.iloc[:, filterd_features]
    
    return filtered_trainingset
