!pip install scikit-learn

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot
from pandas.plotting import scatter_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import KFold

def load_dataset():
    return pd.read_csv('iris.csv')
    
def print_plot_univariate(dataset):
    dataset.hist(figsize = (8, 10))
    pyplot.show()

    
def print_plot_multivariate(dataset):
    scatter_matrix(dataset, figsize = (8, 10))
    pyplot.show()
    
def summarize_dataset(dataset):
    print(f"Dataset dimension:\\n{dataset.shape}\\n\\n\")
    print(f"First 10 rows of dataset:\\n{dataset.head(10)}\\n\\n\")
    print(f"Statistical summary:\\n{dataset.describe()}\\n\\n\")
    print(f"Class Distribution:\\n{dataset.groupby('class').size()}\\n\") 
      
def test_summarize_dataset(dataset):
    X = dataset.drop('class', axis = 1)
    y = dataset['class']
    X_train, X_test, Y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
          
    model_dt = DecisionTreeClassifier()
    model_dt_pred = cross_val_score(model_dt, X_train, Y_train, cv=KFold(n_splits=2), scoring='accuracy')
    print(f"\nDecisionTree: {round(model_dt_pred.mean(), 6)}  ({round(model_dt_pred.std(), 6)})") 
          
    model_gnb = GaussianNB()
    model_gnb_pred = cross_val_score(model_gnb, X_train, Y_train, cv=KFold(n_splits=2), scoring='accuracy')
    print(f"GaussianNB: {round(model_gnb_pred.mean(), 6)}  ({round(model_gnb_pred.std(), 6)})")
    
    model_kn = KNeighborsClassifier()
    model_kn_pred = cross_val_score(model_kn, X_train, Y_train, cv=KFold(n_splits=2), scoring='accuracy')
    print(f"KNeighbors: {round(model_kn_pred.mean(), 6)}  ({round(model_kn_pred.std(), 6)})")
    
    model_lgr = LogisticRegression(solver='liblinear', multi_class='ovr')
    model_lgr_pred = cross_val_score(model_lgr, X_train, Y_train, cv=KFold(n_splits=2), scoring='accuracy')
    print(f"LogisticRegression: {round(model_lgr_pred.mean(), 6)}  ({round(model_lgr_pred.std(), 6)})")
          
    model_lda = LinearDiscriminantAnalysis()
    model_lda_pred = cross_val_score(model_lda, X_train, Y_train, cv=KFold(n_splits=2), scoring='accuracy')
    print(f"LinearDiscriminant: {round(model_lda_pred.mean(), 6)}  ({round(model_lda_pred.std(), 6)})")
          
    model_svc = SVC(gamma='auto')
    model_svc_pred = cross_val_score(model_svc, X_train, Y_train, cv=KFold(n_splits=2), scoring='accuracy')
    print(f"SVM: {round(model_svc_pred.mean(), 6)}  ({round(model_svc_pred.std(), 6)})")
 


dataset = load_dataset()
summarize_dataset(dataset)
print_plot_univariate(dataset)
print_plot_multivariate(dataset)
test_summarize_dataset(dataset)
