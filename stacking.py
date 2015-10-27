# -*- coding: utf-8 -*-

"""Stacking learning method library"""

import numpy as np
from sklearn.cross_validation import Kfold

class Classifier(object):
    """Classifier wrapper"""
    def __init__(self, predict_function):
        self.predict = predict_function


class Stacking(object):
    """Base class for stacking method of learning"""

    def __init__(self, base_fitter, meta_fitter, 
                 split=lambda I: list(KFold(n=I.size, n_folds=2, shuffle=True)),
                 decision_rule=lambda estimations: max(set(estimations), key=estimations.count)):
        self.split = split
        self.base_fitter = base_fitter
        self.meta_fitter = meta_fitter
        self.decision_rule = decision_rule


    def fit(self, X, y):
        """Build compositions of classifiers.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
         
        y : array-like, shape = [n_samples]
        
        Returns
        -------
        self : object
            Returns self.
        """
        self.classifiers = []
        I = np.arange(y.size)
        partitions = self.split(I)

        for partition in partitions:
            base_subsample, meta_subsample = partition

            base_classifier = self.base_fitter(X[base_subsample], y[base_subsample])

            meta_features = base_classifier.predict(X[meta_subsample]).reshape(meta_subsample.size, -1)

            X_meta = np.hstack((X[meta_subsample], meta_features))

            meta_classifier = self.meta_fitter(X_meta, y[meta_subsample])

            self.classifiers.append(
                namedtuple('classifier', ['base', 'meta'])(base_classifier, meta_classifier)
            )

        return self


    def predict(self, X):      
        """Predict class for X.

        The predicted class of an input sample is computed as the majority
        prediction of the meta-classifiers.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes.
        """
        y_predicted = []

        estimations_matrix = np.array([], dtype=int).reshape(X.shape[0], 0)

        for classifier in self.classifiers:
            meta_features = classifier.base.predict(X)
            meta_estimations = classifier.meta.predict(np.column_stack((X, meta_features)))

            estimations_matrix = np.column_stack((estimations_matrix, meta_estimations))

        y_predicted = [self.decision_rule(list(estimations)) for estimations in estimations_matrix]

        return y_predicted


class MultiStacking(object):    
    """Base class for stacking method of learning"""

    def __init__(self, fitters, 
                 split=lambda I: [[I[0::3], I[1::3], I[2::3]], [I[1::3], I[2::3], I[0::3]]],
                 decision_rule=lambda estimations: max(set(estimations), key=estimations.count)):
        self.split = split
        self.fitters = fitters
        self.decision_rule = decision_rule  


    def fit(self, X, y):
        """Build compositions of classifiers.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
         
        y : array-like, shape = [n_samples]
        
        Returns
        -------
        self : object
            Returns self.
        """

        self.classifiers = []
        I = np.arange(y.size)
        partitions = self.split(I)

        for partition in partitions:

            classifiers = []
            meta_features = []
            base_subsample = partition[0]
            base_fitter = self.fitters[0]          
            base_classifier = base_fitter(X[base_subsample], y[base_subsample])
            classifiers.append(base_classifier)
            meta_features.append([base_classifier.predict(X[subsample])
                                                        for subsample in partition[1:]]) 

            for (i, fitter) in enumerate(self.fitters[1:], 1):
                subsample = partition[i]
                X_meta = np.hstack(
                    (X[subsample], meta_features[i-1][0].reshape(subsample.size, -1))
                )
                meta_classifier = fitter(X_meta, y[subsample])
                classifiers.append(meta_classifier)

                for (j, feature) in enumerate(meta_features[i - 1][1:], 1):
                    subsample = partition[i + j]
                    X_meta = np.hstack(
                        (X[subsample], meta_features[i-1][j].reshape(subsample.size, -1))
                    )
                    meta_features.append([meta_classifier.predict(X_meta)])             

            self.classifiers.append(classifiers)

        return self


    def predict(self, X):
        """Predict class for X.

        The predicted class of an input sample is computed as the majority
        prediction of the meta-classifiers.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes.
        """

        y_predicted = []

        estimations_matrix = np.array([], dtype=int).reshape(X.shape[0], 0)

        for classifiers in self.classifiers:
            meta_features = np.array([], dtype = int).reshape(X.shape[0], 0)
            for classifier in classifiers:
                X_new = np.column_stack((X, meta_features))
                meta_features = classifier.predict(X_new)
            estimations_matrix = np.column_stack((estimations_matrix, meta_features))

        y_predicted = [self.decision_rule(list(estimations)) for estimations in estimations_matrix]

        return y_predicted
