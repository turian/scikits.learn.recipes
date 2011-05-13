"""
pysvm module for training an SVM in Python.
"""

import numpy
import math
import random
import sys
from scikits.learn import svm

from scikits.learn.metrics import f1_score

from scikits.learn.cross_val import StratifiedKFold
from scikits.learn.grid_search import GridSearchCV

import common.str
from common.stats import stats

try:
    from itertools import product
except:
    def product(*args, **kwds):
        pools = map(tuple, args) * kwds.get('repeat', 1)
        result = [[]]
        for pool in pools:
            result = [x + [y] for x in result for y in pool]
        for prod in result:
            yield tuple(prod)

def train(X, Y):
    """
    Pick hyperparams through a random grid-search, and return the best classifier.
    examples is a list of dict of {"feature name": value}.
    X should have been converted to a csrmatrix, maybe using SparseDictToCSRMatrix().
    Y is a list of target values.

    Yield classifiers that are increasingly better.

    TODO: Don't assume a sparse matrix?
    TODO: Make list of hyperparams more generic, not hard-coded
    """
    Y = numpy.array(Y)
    assert X.shape[0] == len(Y)
    assert Y.ndim == 1

    tuned_parameters = [{'alpha': [0.32 ** i for i in range(-3, 11)],
                         'n_iter': [2 ** i for i in range(11)],
                         'penalty': ["l1", "l2"],
                         'loss': ['log', 'hinge', 'modified_huber'],
                         'shuffle': [True],
#                         'fit_intercept': [True],
                         'fit_intercept': [False],
                         'verbose': [0],
                        },
                        {'alpha': [0.32 ** i for i in range(-3, 11)],
                         'n_iter': [2 ** i for i in range(11)],
#                         'rho': [0.0, 0.1, 0.35, 0.6, 0.85, 1.0],
                         'rho': [0.1, 0.35, 0.6, 0.85],
                         'penalty': ["elasticnet"],
                         'loss': ['log', 'hinge', 'modified_huber'],
                         'shuffle': [True],
#                         'fit_intercept': [True],
                         'fit_intercept': [False],
                         'verbose': [0],
                        }]

#    # Sparse l1 model
#    tuned_parameters = [{'alpha': [0.32 ** i for i in range(-3, 4)],
#                         'n_iter': [2 ** i for i in range(8)],
#                         'penalty': ["l1"],
#                         'loss': ['log', 'hinge', 'modified_huber'],
#                         'shuffle': [True],
#                         'fit_intercept': [True],
#                         'verbose': [0],
#                        }]

#    tuned_parameters = [{'alpha': [0.32 ** i for i in range(2, 10)],
#                         'n_iter': [8],
#                         'penalty': ["elasticnet"],
#                         'loss': ['hinge'],
#                         'shuffle': [True],
#                         'fit_intercept': [True],
#                         'verbose': [0],
#                        }]

#    clf = linear_model.sparse.SGDClassifier(loss='log', shuffle=True, fit_intercept=True, **hyperparams)

#    from scikits.learn import linear_model
#    clf = GridSearchCV(linear_model.sparse.SGDClassifier(shuffle=True, fit_intercept=True), tuned_parameters, n_jobs=2, score_func=f1_score, verbose=True)
#    clf.fit(X, Y, cv=StratifiedKFold(Y, 10))
##    y_true, y_pred = y[test], clf.predict(X[test])
#    return


#    bestnll = 1e100
    bestscore = 0.
    besthyperparams= None

    all_hyperparams =  []
    for p in tuned_parameters:
        # Always sort the keys of a dictionary, for reproducibility
        items = sorted(p.items())
        keys, values = zip(*items)
        for v in product(*values):
            params = dict(zip(keys, v))
            all_hyperparams.append(params)
    random.shuffle(all_hyperparams)


    for i, hyperparams in enumerate(all_hyperparams):
        score = evaluate(X, Y, hyperparams)
        if score > bestscore:
            bestscore = score
            besthyperparams = hyperparams
            clf = fit_classifier(X, Y, besthyperparams)
            print >> sys.stderr, "new best f1 %f (%s) %d non-zero weights" % (bestscore, besthyperparams, len(clf.sparse_coef_.indices))
            yield clf
        if (i+1)%25 == 0:
            print >> sys.stderr, "Done with %s of hyperparams..." % (common.str.percent(i+1, len(all_hyperparams)))
            print >> sys.stderr, stats()
#    # Don't want hyperparameters at the extremum
#    if not(bestalpha != ALL_ALPHA[0] and bestalpha != ALL_ALPHA[-1]):
#        print >> sys.stderr, "WARNING: Hyperparameter alpha=%s is at the extremum" % bestalpha
#    if not((bestn_iter != ALL_N_ITER[0] or ALL_N_ITER[0]==1) and bestn_iter != ALL_N_ITER[-1]):
#        print >> sys.stderr, "WARNING: Hyperparameter n_iter=%s is at the extremum" % bestn_iter

    print >> sys.stderr, "BEST F1 %f (%s)" % (bestscore, besthyperparams)
        
##    clf = svm.sparse.NuSVC()
#    clf = svm.sparse.NuSVR()
#    clf.fit(X, Y)
#    return fit_classifier(X, Y, besthyperparams)

def fit_classifier(X, Y, hyperparams):
    """
    Train a classifier on X and Y with the given hyperparameters, and return it.
    TODO: Hyperparameters should be a kwarg and passed to the classifier constructor.
    TODO: Don't assume svm.sparse.NuSVC() / logistic regression
    """
    # SVM
#    clf = svm.sparse.NuSVC(probability=True)

    # Logistic Regression
    from scikits.learn import linear_model
#    clf = linear_model.sparse.SGDClassifier(loss='log', shuffle=True, alpha=alpha, n_iter=n_iter)
    clf = linear_model.sparse.SGDClassifier(**hyperparams)
    clf.fit(X, Y)
    return clf

def evaluate(X, Y, hyperparams):
    """
    Evaluate X and Y using leave-one-out or K-fold crossvalidation, and return the nll.
    TODO: Hyperparameters should be a kwarg and passed to the classifier constructor.
    """

#    from scikits.learn.cross_val import LeaveOneOut
#    loo = LeaveOneOut(len(Y))
    from scikits.learn.cross_val import KFold
    K = 5
#    print >> sys.stderr, "Using 10-fold cross-validation"
    loo = KFold(len(Y), K)
#    print loo

    all_y_test = []
    all_y_test_predict = []

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

        clf = fit_classifier(X_train, y_train, hyperparams)

#        print "target", y_test
##        print "predict", clf.predict(X_test)
#        print "predict", clf.predict_proba(X_test)
##        print "df", clf.decision_function(X_test)
##        print "score", clf.score(X_test, y_test)

#        y_test_predict = clf.predict_proba(X_test)
        y_test_predict = clf.predict(X_test)
#        print y_test_predict

        all_y_test.append(y_test)
        all_y_test_predict.append(y_test_predict)

##        print clf.best_estimator
#        print precision_score(y_test, y_test_predict)
#        print recall_score(y_test, y_test_predict)
#        print classification_report(y_test, y_test_predict)
#
#
#        assert y_test.shape == (1,)
#        assert y_test_predict.shape == (1,)
#        if y_test_predict[0] >= 1.:
##            print >> sys.stderr, "WHA? y_test_predict[0] %f >= 1. !!!" % y_test_predict[0]
#            y_test_predict[0] = 1-1e-9
#        elif y_test_predict[0] <= 0.:
##            print >> sys.stderr, "WHA? y_test_predict[0] %f <= 0. !!!" % y_test_predict[0]
#            y_test_predict[0] = 1e-9
#
#        if y_test[0] == 1:
#            probtarget = y_test_predict[0]
#        else:
#            assert y_test[0] == 0
#            probtarget = 1-y_test_predict[0]
##        print "probtarget", probtarget
##        print y_test[0], y_test_predict[0], repr(probtarget)
#        nll = -math.log(probtarget)
##        print "nll", nll
##        print
#
#        nlltotal += nll
#    nlltotal /= len(Y)
##    print "nlltotal %f (alpha=%f, n_iter=%d)" % (nlltotal, alpha, n_iter)
#    return nlltotal

    y_test = numpy.hstack(all_y_test)
    y_test_predict = numpy.hstack(all_y_test_predict)
    assert y_test.ndim == 1
    assert y_test_predict.ndim == 1
    assert Y.shape == y_test.shape
    assert y_test.shape == y_test_predict.shape
#    import plot
#    print "precision_recall_fscore_support", scikits.learn.metrics.precision_recall_fscore_support(y_test, y_test_predict)
    f1 = f1_score(y_test, y_test_predict)
#    print "\tf1 = %0.3f when evaluating with %s" % (f1, hyperparams)
#    sys.stdout.flush()
#    precision, recall, thresholds = scikits.learn.metrics.precision_recall_curve(y_test, y_test_predict)
#    plot.plot_precision_recall(precision, recall)
#    print "confusion_matrix", scikits.learn.metrics.confusion_matrix(y_test, y_test_predict)
#    print "roc_curve", scikits.learn.metrics.roc_curve(y_test, y_test_predict)
#    fpr, tpr, thresholds = scikits.learn.metrics.roc_curve(y_test, y_test_predict)
#    print "auc", scikits.learn.metrics.auc(fpr, tpr)
#    plot.plot_roc(fpr, tpr)
    return f1
