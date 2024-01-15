import random

import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.svm import SVC
import scipy.io
import time
from statistics import mean

from model.OneClassGen import OneClassGenerativeModel


def func(X):
    pass

def load_titanic():
    arr = np.genfromtxt("data/titanic.csv", delimiter=",", usemask=True)
    inliers = arr[arr[:, -1] == -1, :]
    outliers = arr[arr[:, -1] == 1, :]
    X_inliers = inliers[:, :-1]
    X_outliers = outliers[:, :-1]
    return X_inliers, X_outliers, 'Titanic'

def load_magic():
    arr = np.genfromtxt("data/magic.csv", delimiter=",", usemask=True)
    inliers = arr[arr[:, -1] == -1, :]
    outliers = arr[arr[:, -1] == 1, :]
    X_inliers = inliers[:, :-1]
    X_outliers = outliers[:, :-1]
    return X_inliers, X_outliers, 'Magic'


def load_mulcross():
    arr = np.genfromtxt("data/mulcross.csv", delimiter=",", usemask=True)
    inliers = arr[arr[:, -1] == 1, :]
    outliers = arr[arr[:, -1] == -1, :]
    X_inliers = inliers[:, :-1]
    X_outliers = outliers[:, :-1]
    return X_inliers, X_outliers, 'Mulcross'


def load_thyroid():
    return load_mat('data/thyroid.mat', 'Thyroid')


def load_shuttle():
    return load_mat('data/shuttle.mat', 'Shuttle')


def load_vowels():
    return load_mat('data/vowels.mat', 'Vowels')

def load_pendigits():
    return load_mat('data/pendigits.mat', 'Pendigits')

def load_cardio():
    return load_mat('data/cardio.mat', 'Cardio')


def load_mat(path, ds_name):
    mat = scipy.io.loadmat(path)
    X = mat['X']
    y = mat['y']
    X_y = np.c_[X, y]
    inliers = X_y[X_y[:, -1] == 0, :]
    outliers = X_y[X_y[:, -1] == 1, :]
    X_inliers = inliers[:, :-1]
    X_outliers = outliers[:, :-1]
    return X_inliers, X_outliers, ds_name


def process_dataset(X_inliers, X_outliers, ds_name=None, k=5, limit_examples=None, model=None, **kwargs):
    np.random.shuffle(X_inliers)
    if limit_examples:
        X_inliers = X_inliers[:limit_examples, :]
    chunks = np.array_split(X_inliers, k)

    roc = []
    p_r = []
    for i, value in enumerate(chunks):
        print(f'Iteration: {i}')
        X_train = np.concatenate([x for idx, x in enumerate(chunks) if idx != i])
        clf = OneClassGenerativeModel(measure_time=i == k - 1, **kwargs) if model is None else model(**kwargs)
        clf.fit(X_train)
        np.random.shuffle(X_outliers)
        out = X_outliers[:chunks[i].shape[0], :]
        X_test = np.concatenate((chunks[i], out))
        Y_test = np.concatenate((np.ones(chunks[i].shape[0]), np.negative(np.ones(out.shape[0]))))
        roc.append(roc_auc_score(Y_test, clf.predict(X_test)))
        p_r.append(average_precision_score(Y_test, clf.predict(X_test)))
    if ds_name:
        print(f'Processed dataset {ds_name}')
    print(kwargs)
    print('Area under ROC: ', mean(roc))
    print('Area under P-R: ', mean(p_r))


def main(name):
    from sklearn.linear_model import SGDOneClassSVM
    X_inliers, X_outliers, ds_name = load_thyroid()
    k=5

    process_dataset(X_inliers, X_outliers, ds_name, k, model=LocalOutlierFactor, novelty=True)

    
if __name__ == '__main__':
    main('PyCharm')
