from sklearn.svm import LinearSVC, SVC
from classifier.cross_validation import KFoldCrossVal
from utility.file_utility import FileUtility

class SVM:
    def __init__(self, X, Y, clf_model='LSVM'):
        if clf_model=='LSVM':
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
        CV.tune_and_evaluate(self.model, parameters=params, score='f1_macro', file_name=results_file + '_' + self.type, n_jobs=10)
