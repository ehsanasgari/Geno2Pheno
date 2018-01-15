__author__ = "Ehsaneddin Asgari"
__copyright__ = "Copyright 2017, HH-HZI Project"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Ehsaneddin Asgari"
__email__ = "asgari@berkeley.edu"

from sklearn.cross_validation import LeaveOneOut
from sklearn import svm
import numpy as np
import math
import operator
import codecs
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt

class SVMClassifier(object):
    def __init__(self,X,Y,feature_names, c=1,kernel='linear'):
        self.X=X
        self.Y=Y
        self.feature_names=feature_names
        self.clf_svm=svm.SVC(kernel=kernel, C=c)

    def evaluateOLO(self):
        results=[('gtruth','prediction')]
        self.loo = LeaveOneOut(self.X.shape[0])
        for train_index, test_index in self.loo:
            Y_train=[self.Y[idx] for idx in train_index]
            Y_test=[self.Y[idx] for idx in test_index]
            y_pred=self.clf_random_forest.fit(self.X[train_index,:], Y_train).predict(self.X[test_index,:])
            results.append((Y_test[0],y_pred[0]))
        return results
    
    def kFoldCV(self,folds=10):
        cv = ShuffleSplit(n_splits=folds, test_size=0.1, random_state=1)
        scores = cross_val_score(self.clf_svm, self.X, self.Y, cv=cv, scoring='precision', n_jobs=-1)
        precision=(scores.mean(), scores.std())
        scores = cross_val_score(self.clf_svm, self.X, self.Y, cv=cv, scoring='recall', n_jobs=-1)
        recall=(scores.mean(), scores.std())
        scores = cross_val_score(self.clf_svm, self.X, self.Y, cv=cv, scoring='f1',n_jobs=-1)
        f1=(scores.mean(), scores.std())
        return dict([('PRE',precision),('REC',recall),('F1',f1)])


    def get_important_features(self,file_name, N):
        self.clf_svm.fit(self.X, self.Y)
        f = codecs.open(file_name,'w')
        f.write('\t'.join(['feature', 'score', 'std', '#I-out-of-'+str(np.sum(self.Y)), '#O-out-of-'+str(len(self.Y)-np.sum(self.Y))])+'\n')
        for w, score in scores:
            feature_array=self.X[:,self.feature_names.index(w)]
            pos=[feature_array[idx] for idx, x in enumerate(self.Y) if x==1]
            neg=[feature_array[idx] for idx, x in enumerate(self.Y) if x==0]
            f.write('\t'.join([str(w), str(score[0]), str(score[1]), str(np.sum(pos)), str(np.sum(neg))])+'\n')
        f.close()

    def plot_coefficients(classifier, feature_names, top_features=20):
        coef = classifier.coef_.ravel()
        top_positive_coefficients = np.argsort(coef)[-top_features:]
        top_negative_coefficients = np.argsort(coef)[:top_features]
        top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
        # create plot
        plt.figure(figsize=(15, 5))
        colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
        plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
        feature_names = np.array(feature_names)
        plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
        plt.show()
        cv = CountVectorizer()
        cv.fit(data)
        print (len(cv.vocabulary_))
        print (cv.get_feature_names())

        #X_train = cv.transform(data)
        #svm = LinearSVC()
        #svm.fit(X_train, target)
        #plot_coefficients(svm, cv.get_feature_names())
        #std = np.std([tree.feature_importances_ for tree in self.clf_random_forest.estimators_],axis=0)

        