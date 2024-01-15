import random

import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
import time


class OneClassGenerativeModel:

    def __init__(self, classifier=None, n=1, h=1, linspace_extend=0.1, measure_time=False, **classifier_kwargs):
        if classifier:
            if callable(classifier):
                self.clf = classifier(**classifier_kwargs)
            else:
                self.clf = classifier
        else:
            self.clf = RandomForestClassifier(random_state=42)
        self.n = n
        self.h = h
        self.linspace_extend = linspace_extend
        self.measure_time = measure_time

    def fit(self, X, y=None):
        if self.measure_time:
            start = time.time()

        X_inliers = X
        Y_inliers = np.ones(X.shape[0])
        X_outliers = self.get_outliers(X)
        Y_outliers = np.negative(np.ones(X_outliers.shape[0]))
        X_train = np.concatenate([X_inliers, X_outliers])
        Y_train = np.concatenate([Y_inliers, Y_outliers])

        self.clf.fit(X_train, Y_train)

        if self.measure_time:
            end = time.time()
            print(f'fit() method execution time: {round(end - start, 4)}s')
        return self

    def get_outliers(self, X):
        if self.measure_time:
            start = time.time()
        outliers = self.gen_outliers(X)
        if self.measure_time:
            end = time.time()
            print(f'Outlier generation time: {round(end - start, 4)}s for data of shape {X.shape}')
        return outliers

    def gen_outliers(self, X):
        arr = None
        for i in range(X.shape[1]):
            data = X[:, i]
            data = np.sort(data)
            _min = data[0]
            _max = data[data.shape[0] - 1]
            _tmp = (_max - _min) * self.linspace_extend
            _min -= _tmp
            _max += _tmp
            space = np.linspace(_min, _max, int((data.shape[0] + 1) * self.h))
            intervals = [(space[i - 1], x) for i, x in enumerate(space) if i != 0]
            buckets = []
            curr_idx = 0
            max_counter = 0
            for i, interval in enumerate(intervals):
                counter = 0
                while curr_idx < len(data) and data[curr_idx] < interval[1]:
                    counter += 1
                    curr_idx += 1
                buckets.append({'interval': interval, 'counter': counter})
                if counter > max_counter:
                    max_counter = counter

            for bucket in buckets:
                bucket['counter'] = max_counter - bucket['counter'] + 1

            intervals_to_sample = random.choices([x['interval'] for x in buckets],
                                                 [x['counter'] for x in buckets],
                                                 k=int(len(data) * self.n))

            gen = [random.uniform(interval[0], interval[1]) for interval in intervals_to_sample]
            arr = np.c_[arr, gen] if arr is not None else np.array(gen)
        return arr

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)