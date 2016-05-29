# -*- coding: utf-8 -*-


"""Stacking learning method library"""


import numpy as np
from sklearn.cross_validation import KFold
from sklearn.base import ClassifierMixin, BaseEstimator


class Stacking(BaseEstimator, ClassifierMixin):
    """Base class for stacking method of learning"""

    def __init__(self, base_estimators, meta_fitter, n_folds=3, extend_meta=False):
        """Initialize Stacking

        Input parameters:
            base_estimators --- list of tuples (fit(X, y), predict(clf, X)) -- base estimators
            meta_fitter --- meta classifier
            split --- split strategy
        """
        self.base_estimators = base_estimators
        self.meta_fitter = meta_fitter
        self.n_folds = n_folds
        self.extend_meta = extend_meta

    def fit(self, X, y):
        """Build compositions of classifiers.

        Input parameters:
            X : array-like or sparse matrix of shape = [n_samples, n_features]         
            y : array-like, shape = [n_samples]

        Output parameters:
            self : object
        """
        self.base_predictors = []
        X_meta, y_meta = [], []
        y = np.array(y)

        for base_subsample, meta_subsample in KFold(X.shape[0], self.n_folds, True):
            meta_features = [X[meta_subsample]] if self.extend_meta else []
            for fit, predict in self.base_estimators:
                base_clf = fit(X[base_subsample], y[base_subsample])

                meta_features.append(
                    predict(base_clf, X[meta_subsample]).reshape(meta_subsample.size, -1)
                )
                
            X_meta.append(np.hstack(meta_features))
            y_meta.extend(y[meta_subsample])

        X_meta = np.vstack(X_meta)
        self.meta_classifier = self.meta_fitter(X_meta, y_meta)

        self.base_classifiers = [(fit(X, y), predict)
                                    for (fit, predict) in self.base_estimators]
        return self


    def predict(self, X):      
        """Predict class for X.

        The predicted class of an input sample is computed as the majority
        prediction of the meta-classifiers.

        Input parameters:
            X : array-like or sparse matrix of shape = [n_samples, n_features]

        Output:
            y : array of shape = [n_samples] -- predicted classes
        """
        estimations_meta = [X] if self.extend_meta else []

        for base_clf, predict in self.base_classifiers:
            estimations_meta.append(predict(base_clf, X).reshape(X.shape[0], -1))

        return self.meta_classifier.predict(np.hstack(estimations_meta))
