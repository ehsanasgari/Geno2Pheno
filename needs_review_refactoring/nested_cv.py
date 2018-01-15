__author__ = "Ehsaneddin Asgari"
__copyright__ = "Copyright 2017, HH-HZI Project"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Ehsaneddin Asgari"
__email__ = "asgari@berkeley.edu"

import codecs
import itertools
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
import _pickle as pickle

class NestedCrossVal(object):
    def __init__(self, X, Y, inner_folds=10, outer_folds=10, random_state=1):
        self.inner_cv = KFold(n_splits=inner_folds, shuffle=True, random_state=random_state)
        self.outer_cv = KFold(n_splits=outer_folds, shuffle=True, random_state=random_state)
        self.X = X
        self.Y = Y

    def tune_and_evaluate(self, estimator, parameters, optimized_for='neg-f1'):
        '''
        :param estimator: the classifier
        :param parameters: the search space for the classifier
        :param optimized_for: options: neg-f1, f1, precision, recall, f1_micro, f1_macro, roc_auc
        :return:
        '''
        if optimized_for=='neg-f1':
            optimized_for='f1'
            optimization_label=(1-np.array(self.Y))
        else:
            optimization_label=self.Y
        # inner cross_validation
        self.greed_search = GridSearchCV(estimator=estimator, param_grid=parameters, cv=self.inner_cv, scoring=optimized_for)
        self.greed_search.fit(self.X, optimization_label)
        # Nested CV with parameter optimization
        self.nested_score = cross_val_score(self.greed_search, X=self.X, y=optimization_label, cv=self.outer_cv,)
