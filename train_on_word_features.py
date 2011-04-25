#!/usr/bin/python

import sys
from common.stats import stats
from pystringfeatures import stringfeatures
from common.sparsedicttocsrmatrix import SparseDictToCSRMatrix
import scikitslearnrecipes.train

lines = open("/home/joseph/Downloads/valid-keyphrases.txt").readlines()
Y = [int(l[0]) for l in lines]
features = [stringfeatures(l[2:]) for l in lines]

print >> sys.stderr, "Generated features"
print >> sys.stderr, stats()

cnt = int(len(Y) * .9)
features_alltrain = features[:cnt]
features_test = features[cnt:]
Y_alltrain = Y[:cnt]
Y_test = Y[cnt:]

print >> sys.stderr, "Split into %d (train+dev), %d test" % (len(features_alltrain), len(features_test))
print >> sys.stderr, stats()

fmap = SparseDictToCSRMatrix()
fmap.train(features_alltrain)

print >> sys.stderr, "Created feature map"
print >> sys.stderr, stats()

X_alltrain = fmap(features_alltrain)
X_test = fmap(features_test)

print >> sys.stderr, "Applied feature map"
print >> sys.stderr, stats()

scikitslearnrecipes.train.train(X_alltrain, Y_alltrain)
#clf = scikitslearnrecipes.train.fit_classifier(X_alltrain, Y_alltrain, 0.01, 10)
#assert clf.sparse_coef_.shape[0] == 1
#assert clf.sparse_coef_.ndim == 2
#for f in clf.sparse_coef_.indices:
#    print clf.sparse_coef_[0, f], fmap.idmap.key(f)

#    f = l[2:]
#    Y = [l[0] for
#    print l[0], stringfeatures(f)


