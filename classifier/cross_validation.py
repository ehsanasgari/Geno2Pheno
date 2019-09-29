__author__ = "Ehsaneddin Asgari"
__license__ = "Apache 2"
__version__ = "1.0.0"
__maintainer__ = "Ehsaneddin Asgari"
__email__ = "asgari@berkeley.edu"
__project__ = "GENO2PHENO of SEQ2GENO2PHENO"
__website__ = ""

import sys

sys.path.append('../')
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_val_score
from utility.file_utility import FileUtility
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics.classification import precision_recall_fscore_support
from sklearn.metrics.scorer import make_scorer
from sklearn.metrics import f1_score
import numpy as np

__author__ = "Ehsaneddin Asgari"
__license__ = "Apache 2"
__version__ = "1.0.0"
__maintainer__ = "Ehsaneddin Asgari"
__email__ = "asgari@berkeley.edu"
__project__ = "GENO2PHENO of SEQ2GENO2PHENO"
__website__ = ""

import sys

sys.path.append('../')
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_val_score, KFold, \
    cross_validate

from utility.file_utility import FileUtility
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics.classification import precision_recall_fscore_support
from sklearn.metrics.scorer import make_scorer
from sklearn.metrics import f1_score
import numpy as np
import tqdm
import itertools


class CrossValidator(object):
    '''
     The Abstract Cross-Validator
    '''

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.scoring = {  # 'auc_score_macro': make_scorer(self.roc_auc_macro),
            # 'auc_score_micro': make_scorer(self.roc_auc_micro),
            'accuracy': 'accuracy',
            'scores_p_1': 'precision',
            'scores_r_1': 'recall',
            'scores_f1_1': 'f1',
            'scores_f1_0': make_scorer(self.f1_0),
            'scores_p_0': make_scorer(self.precision_0),
            'scores_r_0': make_scorer(self.recall_0),
            'precision_micro': 'precision_micro',
            'precision_macro': 'precision_macro', 'recall_macro': 'recall_macro',
            'recall_micro': 'recall_micro', 'f1_macro': 'f1_macro', 'f1_micro': 'f1_micro'}

    def roc_auc_macro(self, y_true, y_score):
        return roc_auc_score(y_true, y_score, average="macro")

    def roc_auc_micro(self, y_true, y_score):
        return roc_auc_score(y_true, y_score, average="micro")

    def precision_0(self, y_true, y_pred, labels=None, average='binary', sample_weight=None):
        '''
        :param y_true:
        :param y_pred:
        :param labels:
        :param average:
        :param sample_weight:
        :return: calculate prec for neg class
        '''
        p, _, _, _ = precision_recall_fscore_support(y_true, y_pred,
                                                     beta=1,
                                                     labels=labels,
                                                     pos_label=0,
                                                     average=average,
                                                     warn_for=('f-score',),
                                                     sample_weight=sample_weight)
        return p

    def recall_0(self, y_true, y_pred, labels=None, average='binary', sample_weight=None):
        '''
        :param y_true:
        :param y_pred:
        :param labels:
        :param average:
        :param sample_weight:
        :return: calculate recall for neg class
        '''
        _, r, _, _ = precision_recall_fscore_support(y_true, y_pred,
                                                     beta=1,
                                                     labels=labels,
                                                     pos_label=0,
                                                     average=average,
                                                     warn_for=('f-score',),
                                                     sample_weight=sample_weight)
        return r

    def f1_0(self, y_true, y_pred, labels=None, average='binary', sample_weight=None):
        '''
        :param y_true:
        :param y_pred:
        :param labels:
        :param average:
        :param sample_weight:
        :return: calculate f1 for neg class
        '''
        _, _, f, _ = precision_recall_fscore_support(y_true, y_pred,
                                                     beta=1,
                                                     labels=labels,
                                                     pos_label=0,
                                                     average=average,
                                                     warn_for=('f-score',),
                                                     sample_weight=sample_weight)
        return f


class PredefinedFoldCrossVal(CrossValidator):
    '''
        Predefined folds
    '''

    def __init__(self, X, Y, isolate_list, fold_file, test_file):
        '''
        :param X:
        :param Y:
        :param folds:
        :param random_state:
        '''
        CrossValidator.__init__(self, X, Y)

        map_to_idx = {isolate: idx for idx, isolate in enumerate(isolate_list)}

        test_idx = [map_to_idx[test] for test in FileUtility.load_list(test_file)[0].split() if test in map_to_idx]

        train_idx = [map_to_idx[train] for train in
                     list(itertools.chain(*[l.split() for l in FileUtility.load_list(fold_file)]))]

        self.X_test = X[test_idx, :]
        self.Y_test = [Y[idy] for idy in test_idx]

        train_idx = list(set(map_to_idx.values()) - set(test_idx))

        X = X[train_idx, :]
        Y = [Y[idy] for idy in train_idx]
        isolate_list = [isolate_list[idx] for idx in train_idx]

        self.train_isolate_list = isolate_list
        map_to_idx = {isolate: idx for idx, isolate in enumerate(isolate_list)}

        splits = [[map_to_idx[item] for item in fold_list.split() if item in map_to_idx] for fold_list in
                  FileUtility.load_list(fold_file)]

        new_splits = []
        for i in range(len(splits)):
            train = [j for i in splits[:i] + splits[i + 1:] for j in i]
            test = splits[i]
            new_splits.append([train, test])

        self.cv = new_splits
        self.X = X
        self.Y = Y

    def tune_and_evaluate(self, estimator, parameters, cv_inner=5, score='f1_macro', n_jobs=-1, file_name='results',
                          NUM_TRIALS=3):
        '''
        :param estimator:
        :param parameters:p
        :param score:
        :param n_jobs:
        :param file_name: directory/tuning/classifier/features/
        :return:
        '''
        self.nested_scores = []
        cv_dicts = []
        test_predictions_in_trials = []
        best_params_in_trials = []

        # Loop for each trial
        for i in tqdm.tqdm(range(NUM_TRIALS)):
            # Choose cross-validation techniques for the inner and outer loops,
            # independently of the dataset.
            # E.g "GroupKFold", "LeaveOneOut", "LeaveOneGroupOut", etc.
            inner_cv = StratifiedKFold(n_splits=cv_inner, shuffle=True, random_state=i)

            # parameter search and scoring
            self.greed_search = GridSearchCV(estimator=estimator, param_grid=parameters, cv=inner_cv,
                                             scoring=self.scoring,
                                             refit=score, error_score=0, n_jobs=n_jobs, verbose=0)

            # Nested CV with parameter optimization
            nested_score = cross_val_score(self.greed_search, X=self.X, y=self.Y, cv=self.cv, n_jobs=1, scoring=score)
            self.nested_scores.append(nested_score)

            # Nested CV with parameter optimization
            cv_dict_pred = cross_val_predict(self.greed_search, X=self.X, y=self.Y, cv=self.cv, n_jobs=1)
            cv_dicts.append(cv_dict_pred)

        # get the cv results
        cv_predictions_pred = []
        cv_predictions_trues = []

        # Non_nested parameter search and scoring
        self.greed_search = GridSearchCV(estimator=estimator, param_grid=parameters, cv=self.cv, scoring=self.scoring,
                                         refit=score, error_score=0, n_jobs=n_jobs, verbose=0)

        self.greed_search.fit(X=self.X, y=self.Y)

        isolates = []
        for train, test in self.cv:
            self.greed_search.best_estimator_.fit(self.X[train, :], [self.Y[idx] for idx in train])
            preds = self.greed_search.best_estimator_.predict(self.X[test, :])
            trues = [self.Y[idx] for idx in test]
            [cv_predictions_pred.append(pred) for pred in preds]
            [cv_predictions_trues.append(tr) for tr in trues]
            for i in test:
                isolates.append(i)

        label_set = list(set(self.Y))
        label_set.sort()

        isolates = [self.train_isolate_list[iso] for iso in isolates]
        conf = confusion_matrix(cv_predictions_trues, cv_predictions_pred, labels=label_set)

        Y_test_pred = self.greed_search.best_estimator_.predict(self.X_test)

        # save in file
        FileUtility.save_obj(file_name,
                             [self.nested_scores, cv_dicts, label_set, conf, label_set, self.greed_search.best_score_,
                              self.greed_search.best_estimator_,
                              self.greed_search.cv_results_, self.greed_search.best_params_,
                              (cv_predictions_pred, cv_predictions_trues, isolates), (Y_test_pred, self.Y_test)])

class NestedCrossVal(CrossValidator):
    '''
    Nested cross-validation
    '''

    def __init__(self, X, Y, inner_folds=10, outer_folds=10, random_state=1, opt_f1_class=0):
        '''
        :param X:
        :param Y:
        :param inner_folds:
        :param outer_folds:
        :param random_state:
        :param opt_f1_class:
        '''
        CrossValidator.__init__(self, X, Y, opt_f1_class=opt_f1_class)
        self.inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=random_state)
        self.outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=random_state)

    def tune_and_evaluate(self, estimator, parameters, score='f1_macro', file_name='results'):
        '''
        :param estimator:
        :param parameters:
        :param score:
        :param file_name: directory/tuning/classifier/features/
        :return:
        '''
        # inner cross_validation
        self.greed_search = GridSearchCV(estimator=estimator, param_grid=parameters, cv=self.inner_cv,
                                         scoring=self.scoring, refit=score, error_score=0)
        # Nested CV with parameter optimization
        self.nested_score = cross_val_score(self.greed_search, X=self.X, y=self.Y, cv=self.outer_cv)

        # saving
        FileUtility.save_obj([self.greed_search, self.nested_score], file_name)


class PredefinedFoldCrossVal(CrossValidator):
    '''
        Predefined folds
    '''

    def __init__(self, X, Y, isolate_list, fold_file, test_file):
        '''
        :param X:
        :param Y:
        :param folds:
        :param random_state:
        '''
        CrossValidator.__init__(self, X, Y)



        map_to_idx = {isolate: idx for idx, isolate in enumerate(isolate_list)}


        test_idx = [map_to_idx[test] for test in FileUtility.load_list(test_file)[0].split() if test in map_to_idx]


        self.X_test=X[test_idx,:]
        self.Y_test=[Y[idy] for idy in test_idx]

        train_idx=list(set(map_to_idx.values())-set(test_idx))

        X=X[train_idx,:]
        Y=[Y[idy] for idy in train_idx]

        isolate_list=[isolate_list[idx] for idx in train_idx]
        self.train_isolate_list=isolate_list
        map_to_idx = {isolate: idx for idx, isolate in enumerate(isolate_list)}
        splits = [[map_to_idx[item] for item in fold_list.split() if item in map_to_idx] for fold_list in
                  FileUtility.load_list(fold_file)]

        new_splits = []
        for i in range(len(splits)):
            train = [j for i in splits[:i] + splits[i + 1:] for j in i]
            test = splits[i]
            new_splits.append([train, test])

        self.cv = new_splits
        self.X = X
        self.Y = Y

    def tune_and_evaluate(self, estimator, parameters, score='f1_macro', n_jobs=-1, file_name='results', NUM_TRIALS=10):
        '''
        :param estimator:
        :param parameters:p
        :param score:
        :param n_jobs:
        :param file_name: directory/tuning/classifier/features/
        :return:
        '''

        # Loop for each trial
        nested_scores = [0]*NUM_TRIALS

        for i in range(NUM_TRIALS):
            # Choose cross-validation techniques for the inner and outer loops,
            # independently of the dataset.
            # E.g "GroupKFold", "LeaveOneOut", "LeaveOneGroupOut", etc.
            inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=i)

            # Non_nested parameter search and scoring
            clf = GridSearchCV(estimator=svm, param_grid=p_grid, cv=inner_cv,
                               iid=False)
            # greed_search
            self.greed_search = GridSearchCV(estimator=estimator, param_grid=parameters, cv=inner_cv,
                                             scoring=self.scoring,
                                             refit=score, error_score=0, n_jobs=n_jobs, verbose=0)

            # Nested CV with parameter optimization
            nested_score = cross_val_score(self.greed_search, X=X_iris, y=y_iris, cv=self.cv)
            nested_scores[i] = nested_score.mean()


        label_set = list(set(self.Y))
        label_set.sort()

        # fitting
        self.greed_search.fit(X=self.X, y=self.Y)

        # get the cv results
        cv_predictions_pred=[]
        cv_predictions_trues=[]

        isolates=[]
        for train, test in self.cv:
            self.greed_search.best_estimator_.fit(self.X[train,:],[self.Y[idx] for idx in train])
            preds=self.greed_search.best_estimator_.predict(self.X[test,:])
            trues=[self.Y[idx] for idx in test]
            [cv_predictions_pred.append(pred) for pred in preds]
            [cv_predictions_trues.append(tr) for tr in trues]
            for i in test:
                isolates.append(i)
        isolates=[self.train_isolate_list[iso] for  iso in isolates]

        Y_test_pred=self.greed_search.best_estimator_.predict(self.X_test)
        f1_test=f1_score(self.Y_test,Y_test_pred)

        conf = confusion_matrix(cv_predictions_trues, cv_predictions_pred, labels=label_set)
        # save in file
        FileUtility.save_obj(file_name,
                             [label_set, conf, label_set, self.greed_search.best_score_, self.greed_search.best_estimator_,
                              self.greed_search.cv_results_, self.greed_search.best_params_,  (cv_predictions_pred,cv_predictions_trues,isolates), (Y_test_pred, self.Y_test) ])
