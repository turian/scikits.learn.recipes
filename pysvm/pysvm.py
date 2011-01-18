"""
pysvm module for training an SVM in Python.
"""

from common.sparsedicttocsrmatrix import SparseDictToCSRMatrix

import numpy
from scikits.learn import svm

def train(examples, Y):
    """
    examples is a list of dict of {"feature name": value}.
    Y is a list of target values.

    TODO: Don't assume a sparse matrix?
    TODO: Don't assume svm.sparse.NuSVC()
    """
    csrmatrix = SparseDictToCSRMatrix()
    X = csrmatrix.train(examples)
    Y = numpy.array(Y)
    assert X.shape[0] == len(Y)

    clf = svm.sparse.NuSVC()
    clf.fit(X, Y)

    from scikits.learn.cross_val import LeaveOneOut
    loo = LeaveOneOut(len(Y))
    print loo
    for train, test in loo:
        trainidx = [idx for idx in range(len(train)) if train[idx]]
        testidx = [idx for idx in range(len(test)) if test[idx]]
        X_train, X_test, y_train, y_test = X[trainidx], X[testidx], Y[trainidx], Y[testidx]
        print "train", X_train.shape, y_train.shape
        print "test", X_test.shape, y_test.shape

        clf = svm.sparse.NuSVC()
        clf.fit(X_train, y_train)
        print y_test
        print clf.predict(X_test)
        print clf.score(X_test, y_test)
        print

