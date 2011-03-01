"""
pysvm module for training an SVM in Python.
"""

import numpy
import math
import random
import sys
from scikits.learn import svm

import common.str
from common.stats import stats

def train(X, Y):
    """
    Pick hyperparams through a random grid-search, and return the best classifier.
    examples is a list of dict of {"feature name": value}.
    X should have been converted to a csrmatrix, maybe using SparseDictToCSRMatrix().
    Y is a list of target values.

    TODO: Don't assume a sparse matrix?
    TODO: Make list of hyperparams more generic, not hard-coded
    """
    Y = numpy.array(Y)
    assert X.shape[0] == len(Y)
    assert Y.ndim == 1

    ALL_ALPHA = [0.32 ** i for i in range(-3, 14)]
#    print ALL_ALPHA
#    ALL_N_ITER = [2 ** i for i in range(10)]
    ALL_N_ITER = [2 ** i for i in range(7)]
#    print ALL_N_ITER

    bestnll = 1e100
    bestalpha = None
    bestn_iter = None

    # Random-order grid search
    hyperparams = []
    for alpha in ALL_ALPHA:
        for n_iter in ALL_N_ITER:
            hyperparams.append((alpha, n_iter))
    random.shuffle(hyperparams)
    for i, (alpha, n_iter) in enumerate(hyperparams):
        nll = evaluate(X, Y, alpha=alpha, n_iter=n_iter)
        if nll < bestnll:
            bestnll = nll
            bestalpha = alpha
            bestn_iter = n_iter
            print >> sys.stderr, "new best nlltotal %f (alpha=%f, n_iter=%d)" % (bestnll, bestalpha, bestn_iter)
#        if (i+1)%25 == 0:
#            print >> sys.stderr, "Done with %s of hyperparams..." % (common.str.percent(i+1, len(hyperparams)))
#            print >> sys.stderr, stats()
    # Don't want hyperparameters at the extremum
    if not(bestalpha != ALL_ALPHA[0] and bestalpha != ALL_ALPHA[-1]):
        print >> sys.stderr, "WARNING: Hyperparameter alpha=%s is at the extremum" % bestalpha
    if not((bestn_iter != ALL_N_ITER[0] or ALL_N_ITER[0]==1) and bestn_iter != ALL_N_ITER[-1]):
        print >> sys.stderr, "WARNING: Hyperparameter n_iter=%s is at the extremum" % bestn_iter

    print >> sys.stderr, "BEST NLL %f (alpha=%f, n_iter=%d)" % (bestnll, bestalpha, bestn_iter)
        
##    clf = svm.sparse.NuSVC()
#    clf = svm.sparse.NuSVR()
#    clf.fit(X, Y)
    return fit_classifier(X, Y, bestalpha, bestn_iter)

def fit_classifier(X, Y, alpha, n_iter):
    """
    Train a classifier on X and Y with the given hyperparameters, and return it.
    TODO: Hyperparameters should be a kwarg and passed to the classifier constructor.
    TODO: Don't assume svm.sparse.NuSVC() / logistic regression
    """
    # SVM
#    clf = svm.sparse.NuSVC(probability=True)

    # Logistic Regression
    from scikits.learn import linear_model
    clf = linear_model.sparse.SGDClassifier(loss='log', shuffle=True, alpha=alpha, n_iter=n_iter)
    clf.fit(X, Y)
    return clf

def evaluate(X, Y, alpha, n_iter):
    """
    Evaluate X and Y using leave-one-out crossvalidation, and return the nll.
    TODO: Hyperparameters should be a kwarg and passed to the classifier constructor.
    """
    from scikits.learn.cross_val import LeaveOneOut
    loo = LeaveOneOut(len(Y))
#    print loo

    nlltotal = 0.
    for train, test in loo:
        trainidx = [idx for idx in range(len(train)) if train[idx]]
        testidx = [idx for idx in range(len(test)) if test[idx]]
        X_train, X_test, y_train, y_test = X[trainidx], X[testidx], Y[trainidx], Y[testidx]
#        print "train", X_train.shape, y_train.shape
#        print "test", X_test.shape, y_test.shape

        if len(frozenset(y_train)) == 1:
            # Skip training on this LOO set if there is only one y-value in the training set
            continue

        clf = fit_classifier(X_train, y_train, alpha=alpha, n_iter=n_iter)

#        print "target", y_test
##        print "predict", clf.predict(X_test)
#        print "predict", clf.predict_proba(X_test)
##        print "df", clf.decision_function(X_test)
##        print "score", clf.score(X_test, y_test)

        y_test_predict = clf.predict_proba(X_test)
        assert y_test.shape == (1,)
        assert y_test_predict.shape == (1,)
        if y_test[0] == 1:
            probtarget = y_test_predict[0]
        else:
            assert y_test[0] == 0
            probtarget = 1-y_test_predict[0]
#        print "probtarget", probtarget
        nll = -math.log(probtarget)
#        print "nll", nll
#        print

        nlltotal += nll
    nlltotal /= len(Y)
#    print "nlltotal %f (alpha=%f, n_iter=%d)" % (nlltotal, alpha, n_iter)
    return nlltotal
