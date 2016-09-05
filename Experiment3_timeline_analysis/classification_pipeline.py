import numpy as np
import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt
#%matplotlib inline

import sklearn as sk
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

class classification_pipeline:
    def __init__(self):


    def do_classification(dataset):
        # creating the models
        # clf = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        models = [(LogisticRegression(), 'Logistic Regression'),
                  (GaussianNB(), 'Naive Bayes'),
                  (svm.LinearSVC(C=1.0), 'Linear SVM'),
                  (svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                           decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
                           max_iter=-1, probability=False, random_state=None, shrinking=True,
                           tol=0.001, verbose=True), 'SVM with RBF kernel'),
                  (RandomForestClassifier(n_estimators=100), 'Random Forest'),
                  (KNeighborsClassifier(n_neighbors=8), 'KNN'),
                  (DecisionTreeClassifier(max_depth=5), 'Decision Tree'),
                  (AdaBoostClassifier(), 'AdaBoost'),
                  (LinearDiscriminantAnalysis(), 'Linear Discriminant Analysis'),
                  (QuadraticDiscriminantAnalysis(), 'Quadratic Discriminant Analysis')]
        # applying the models
        n_folds = 10
        k_fold = cross_validation.KFold(n=len(dataset), n_folds=n_folds, shuffle=False, random_state=None)
        accuracy = {}
        for train_index, test_index in k_fold:
            X_train = dataset[train_index, :-1]
            y_train = dataset[train_index, -1]
            X_test = dataset[test_index, :-1]
            y_test = dataset[test_index, -1]

            for clf, name in models:
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                accuracy_percent = 100 * sk.metrics.accuracy_score(y_test, y_pred) / n_folds
                if name not in accuracy:
                    accuracy[name] = accuracy_percent
                else:
                    accuracy[name] += accuracy_percent
        print('\n')
        for key, value in accuracy.items():
            print(key, ':', round(value, 2))


    def plot_TSNE(data, labels):
        tsne_model = TSNE(n_components=2, random_state=0)
        np.set_printoptions(suppress=True)
        points = tsne_model.fit_transform(data)
        df = pd.DataFrame(data=np.column_stack([points, labels]), columns=["x", "y", "class"])
        sns.lmplot("x", "y", data=df, hue='class', fit_reg=False, palette=sns.color_palette('colorblind'))
        # sns.plt.plot(figsize=(20, 20))
        sns.plt.show()


        pca = PCA(n_components=2)
        pca.fit(data)
        # print(pca.explained_variance_ratio_)
        data = pca.transform(data)
        df = pd.DataFrame(data=np.column_stack([data, labels]), columns=["x", "y", "class"])
        sns.lmplot("x", "y", data=df, hue='class', fit_reg=False, palette=sns.color_palette('colorblind'))
        sns.plt.show()
