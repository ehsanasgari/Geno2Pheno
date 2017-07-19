__author__ = "Ehsaneddin Asgari"
__copyright__ = "Copyright 2017, HH-HZI Project"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Ehsaneddin Asgari"
__email__ = "asgari@berkeley.edu"

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier

class RandomForest(object):
    def __init__(self,X,Y):
        self.X=X
        self.Y=Y
        self.clf_random_forest=RandomForestClassifier(bootstrap=True, criterion='gini',
            max_depth=None, max_features='auto', min_samples_leaf=1, n_estimators=500, n_jobs=30)

    def evaluate_through_cv(self,folds=10, test_portion=0.1, multiclass=True):
        cv = ShuffleSplit(n_splits=folds, test_size=test_portion, random_state=1)
        if not multiclass:
            scores = cross_val_score(self.clf_random_forest, X, Y, cv=cv, scoring='precision')
            self.precision=(scores.mean(), scores.std())
            scores = cross_val_score(self.clf_random_forest, X, Y, cv=cv, scoring='recall')
            self.recall=(scores.mean(), scores.std())
            scores = cross_val_score(self.clf_random_forest, X, Y, cv=cv, scoring='f1')
            self.f1=(scores.mean(), scores.std())
        if multiclass:
            #scores = cross_val_score(self.clf_random_forest, X, Y, cv=cv, scoring='precision_macro')
            #self.precision_macro=(scores.mean(), scores.std())
            #scores = cross_val_score(self.clf_random_forest, X, Y, cv=cv, scoring='recall_macro')
            #self.recall_macro=(scores.mean(), scores.std())
            #scores = cross_val_score(self.clf_random_forest, X, Y, cv=cv, scoring='precision_weighted')
            #self.precision_weighted=(scores.mean(), scores.std())
            #scores = cross_val_score(self.clf_random_forest, X, Y, cv=cv, scoring='recall_weighted')
            #self.recall_weighted=(scores.mean(), scores.std())
            scores = cross_val_score(self.clf_random_forest, X, Y, cv=cv, scoring='f1_weighted')
            self.f1_weighted=(scores.mean(), scores.std())
            scores = cross_val_score(self.clf_random_forest, X, Y, cv=cv, scoring='f1_macro')
            self.f1_macro=(scores.mean(), scores.std())

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        y_pred=self.clf_random_forest.fit(X_train, y_train).predict(X_test)
        self.showing_labels=list(set(Y))
        self.showing_labels.sort()
        self.confusion=confusion_matrix(y_test, y_pred,labels=self.showing_labels)


    def get_important_features(self, N, labels):
        self.clf_random_forest.fit(self.X, self.Y)
        importances = self.clf_random_forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in self.clf_random_forest.estimators_],axis=0)
        indices = np.argsort(importances)[::-1]
        # Print the feature ranking
        results=[]
        print("Feature ranking:")
        for f in range(self.X.shape[1]):
            if f<N:
                print("%d. feature %s (%f, %f)" % (f + 1, labels[indices[f]], importances[indices[f]],std[indices[f]]))
            results.append((f + 1, labels[indices[f]], importances[indices[f]],std[indices[f]]))
        return results
