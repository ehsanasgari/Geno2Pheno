import sys

sys.path.append('../')
from sklearn.svm import LinearSVC, SVC
from classifier.cross_validation import KFoldCrossVal
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

class SVM:
    def __init__(self, X, Y, clf_model='LSVM'):
        if clf_model == 'LSVM':
            self.model = LinearSVC(C=1.0, multi_class='ovr')
            self.type = 'linear'
        else:
            self.model = SVC(C=1.0, kernel='rbf')
            self.type = 'rbf'
        self.X = X
        self.Y = Y

    def tune_and_eval(self, results_file,
                      params=[{'C': [1000, 500, 200, 100, 50, 20, 10, 5, 2, 1, 0.2, 0.5, 0.01, 0.02, 0.05, 0.001]}]):
        CV = KFoldCrossVal(self.X, self.Y, folds=10)
        CV.tune_and_evaluate(self.model, parameters=params, score='f1_macro', file_name=results_file + '_SVM',
                             n_jobs=10)


class RFClassifier:
    def __init__(self, X, Y):
        self.model = RandomForestClassifier(bootstrap=True, criterion='gini',
                                            min_samples_split=2, max_features='auto', min_samples_leaf=1,
                                            n_estimators=1000)
        self.X = X
        self.Y = Y

    def tune_and_eval(self, results_file, params=None):
        if params is None:
            params = [{"n_estimators": [100, 200, 500, 1000],
                       "criterion": ["entropy"],  # "gini",
                       'max_features': ['sqrt'],  # 'auto',
                       'min_samples_split': [5],  # 2,5,10
                       'min_samples_leaf': [1]}]
        self.CV = KFoldCrossVal(self.X, self.Y, folds=10)
        self.CV.tune_and_evaluate(self.model, parameters=params, score='f1_macro', file_name=results_file + '_RF',
                                  n_jobs=15)

class KNN:
    def __init__(self, X, Y):
        self.model = KNeighborsClassifier(n_neighbors=3)
        self.X = X
        self.Y = Y

    def tune_and_eval(self, results_file, params=None):
        if params is None:
            params = [{"n_neighbors": [1,2,3,4,5,6,7,8,9,10,15,20]}]
        self.CV = KFoldCrossVal(self.X, self.Y, folds=10)
        self.CV.tune_and_evaluate(self.model, parameters=params, score='f1_macro', file_name=results_file + '_KNN',
                                  n_jobs=15)


