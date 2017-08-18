__author__ = "Ehsaneddin Asgari"
__copyright__ = "Copyright 2017, HH-HZI Project"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Ehsaneddin Asgari"
__email__ = "asgari@berkeley.edu"

from sklearn.cross_validation import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import math
import operator
import codecs
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import confusion_matrix, f1_score

class RandomForest(object):
    def __init__(self,X,Y,feature_names):
        self.X=X
        self.Y=Y
        self.feature_names=feature_names
        self.clf_random_forest=RandomForestClassifier(bootstrap=True, criterion='gini',
            min_samples_split= 2, max_features='auto', min_samples_leaf=1, n_estimators=1000, n_jobs=-1)

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
        scores = cross_val_score(self.clf_random_forest, self.X, self.Y, cv=cv, scoring='precision')
        precision=(scores.mean(), scores.std())
        scores = cross_val_score(self.clf_random_forest, self.X, self.Y, cv=cv, scoring='recall')
        recall=(scores.mean(), scores.std())
        scores = cross_val_score(self.clf_random_forest, self.X, self.Y, cv=cv, scoring='f1')
        f1=(scores.mean(), scores.std())
        return dict([('PRE',precision),('REC',recall),('F1',f1)])




    def get_important_features(self,file_name, N):
        self.clf_random_forest.fit(self.X, self.Y)
        std = np.std([tree.feature_importances_ for tree in self.clf_random_forest.estimators_],axis=0)

        scores = {self.feature_names[i]: (s,std[i]) for i, s in enumerate(list(self.clf_random_forest.feature_importances_)) if not math.isnan(s) }
        scores = sorted(scores.items(), key=operator.itemgetter([1][0]),reverse=True)[0:N]
        f = codecs.open(file_name,'w')
        f.write('\t'.join(['feature', 'score', 'std', '#I-out-of-'+str(np.sum(self.Y)), '#O-out-of-'+str(len(self.Y)-np.sum(self.Y))])+'\n')
        for w, score in scores:
            feature_array=self.X[:,self.feature_names.index(w)]
            pos=[feature_array[idx] for idx, x in enumerate(self.Y) if x==1]
            neg=[feature_array[idx] for idx, x in enumerate(self.Y) if x==0]
            f.write('\t'.join([str(w), str(score[0]), str(score[1]), str(np.sum(pos)), str(np.sum(neg))])+'\n')
        f.close()