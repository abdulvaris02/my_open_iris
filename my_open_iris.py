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

def load_dataset():
    return pd.read_csv('iris.csv')
    
def summarize_dataset(dataset):
    print("Dataset dimension:\n", dataset.shape, "\n")
    print("First 10 rows of dataset:\n", dataset.head(20), "\n")
    print("Statistical summary:\n", dataset.describe(), "\n")
    print("Class Distribution:\n", dataset.groupby('class').size(), "\n")    
    
    
def print_plot_univariate(dataset):
    dataset.hist()
    pyplot.show()
    print("\n\n\n")
        
    
def print_plot_multivariate(dataset):
    scatter_matrix(dataset)
    pyplot.show()
    
def my_print_and_test_models(dataset):
    array = dataset.values
    X = array[:,0:4]
    y = array[:,4]
    X_train, X_test, Y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
    
    model_dt = DecisionTreeClassifier()
    model_dt.fit(X_train, Y_train)
    model_dt_pred = model_dt.predict(X_test)
    print("\n")
    print(f"mine -> DecisionTree:  {accuracy_score(y_test, model_dt_pred).mean()}  {accuracy_score(y_test, model_dt_pred).std()}")
    
    dectree_name = '__DecisionTree__'
    dectree_model = DecisionTreeClassifier()
    dectree_results = cross_val_score(dectree_model, X_train, Y_train, cv=2, scoring='accuracy')
    print('%s: %f (%f)' % (dectree_name, dectree_results.mean(), dectree_results.std()))



    model_gnb = GaussianNB()
    model_gnb.fit(X_train, Y_train)
    model_gnb_pred = model_gnb.predict(X_test)
    print("GaussianNB: ", accuracy_score(y_test, model_gnb_pred))

    model_kn = KNeighborsClassifier()
    model_kn.fit(X_train, Y_train)
    model_kn_pred = model_kn.predict(X_test)
    print("KNeighbors: ", accuracy_score(y_test, model_kn_pred))

    model_lgr = LogisticRegression(solver='liblinear', multi_class='ovr')
    model_lgr.fit(X_train, Y_train)
    model_lgr_pred = model_lgr.predict(X_test)
    print("LogisticRegression: ", accuracy_score(y_test, model_lgr_pred))

    model_lda = LinearDiscriminantAnalysis()
    model_lda.fit(X_train, Y_train)
    model_lda_pred = model_lda.predict(X_test)
    print("LinearDiscriminant: ", accuracy_score(y_test, model_lda_pred))

    model_svc = SVC(gamma='auto')
    model_svc.fit(X_train, Y_train)
    model_svc_pred = model_svc.predict(X_test)
    print("SVM: ", accuracy_score(y_test, model_svc_pred))
    
dataset = load_dataset()
    summarize_dataset(dataset)
    print_plot_univariate(dataset)
    print_plot_multivariate(dataset)
    my_print_and_test_models(dataset)
