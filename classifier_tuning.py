__author__ = "Ehsaneddin Asgari"
__copyright__ = "Copyright 2017, HH-HZI Project"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Ehsaneddin Asgari"
__email__ = "asgari@berkeley.edu"

import codecs
import itertools
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import _pickle as pickle

class ClassifierTuning(object):
    def __init__(self, X, Y, clf_0, parameters):
        self.cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=1)
        self.clf = GridSearchCV(estimator=clf_0, param_grid=parameters, cv=self.cv, n_jobs=30, scoring='f1')
        self.X = X
        self.Y = Y

    def find_best(self, filename):
        self.clf.fit(self.X, self.Y)
        with open( filename + '.pickle', 'wb') as f:
            pickle.dump(self.clf, f)
        return self.clf
