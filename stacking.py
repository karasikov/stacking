# -*- coding: utf-8 -*-


"""Stacking learning method library"""


import numpy as np
from sklearn.model_selection import KFold
from sklearn.base import ClassifierMixin, BaseEstimator
import scipy


class Stacking(BaseEstimator, ClassifierMixin):
    """Base class for stacking method of learning"""

    def __init__(self, base_estimators, meta_fitter, get_folds=None, n_folds=3, extend_meta=False):
        """Initialize Stacking

        Input parameters:
            base_estimators --- list of tuples (fit(X, y), predict(clf, X)) -- base estimators
            meta_fitter --- fit(X, y)
            split --- split strategy
        """
        self.base_estimators = base_estimators
        self.meta_fitter = meta_fitter
        self.get_folds = get_folds if get_folds else lambda y, n_folds: KFold(n_folds, True).split(y)
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
        X = scipy.sparse.csr_matrix(X)
        if isinstance(y, list):
            y = np.array(y)

        X_meta, meta_indices = [], []
        for _, meta_subsample in self.get_folds(y, self.n_folds):
            mask_meta = np.zeros(len(y), dtype=bool)
            mask_meta[meta_subsample] = True

            meta_subsample = np.where(mask_meta)[0]

            meta_indices.append(meta_subsample)

            X_meta.append(scipy.sparse.hstack([
                scipy.sparse.csc_matrix(predict(fit(X[~mask_meta], y[~mask_meta]),
                                                X[meta_subsample]).reshape(mask_meta.sum(), -1))
                for fit, predict in self.base_estimators
            ]).tocsr())

        meta_indices = np.hstack(meta_indices)
        X_meta = scipy.sparse.vstack(X_meta)[np.argsort(meta_indices)]

        self.base_classifiers = [(fit(X, y), predict) for (fit, predict) in self.base_estimators]

        X = csr_vstack(X.tocsc().T, X_meta.tocsc().T).T.tocsr() if self.extend_meta else X_meta
        self.meta_classifier = self.meta_fitter(X, y)
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
        if not hasattr(self, 'meta_classifier'):
            raise Exception("Fit meta classifier first")

        X = scipy.sparse.csr_matrix(X)
        X_meta = scipy.sparse.hstack([
            scipy.sparse.csc_matrix(predict(base_clf, X).reshape(X.shape[0], -1))
            for base_clf, predict in self.base_classifiers
        ]).tocsr()
        X = csr_vstack(X.tocsc().T, X_meta.tocsc().T).T.tocsr() if self.extend_meta else X_meta
        return self.meta_classifier.predict(X)


def csr_vstack(a, b):
    """Takes 2 matrices and appends the second one to the bottom of the first one

    Works for big matrices using int64 indices
    """
    assert(type(a) is scipy.sparse.csr_matrix)
    assert(type(b) is scipy.sparse.csr_matrix)
    assert(a.shape[1] == b.shape[1])
    a.data = np.hstack((a.data, b.data))
    a.indices = np.hstack((a.indices, b.indices))
    a.indptr = np.hstack((a.indptr, (b.indptr.astype(np.int64) + a.nnz)[1:]))
    a._shape = (a.shape[0] + b.shape[0], b.shape[1])
    return a

