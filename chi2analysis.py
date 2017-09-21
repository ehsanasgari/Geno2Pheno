__author__ = "Ehsaneddin Asgari"
__copyright__ = "Copyright 2017, HH-HZI Project"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Ehsaneddin Asgari"
__email__ = "asgari@berkeley.edu ehsaneddin.asgari@helmholtz-hzi.de"

import math
import codecs
import numpy as np
from sklearn.feature_selection import SelectKBest, SelectFdr
from sklearn.feature_selection import chi2
import operator


class Chi2Anlaysis(object):
    # X^2 is statistically significant at the p-value level
    def __init__(self, X, Y, feature_names):
        '''
        :param X:
        :param Y:
        :param feature_names:
        '''
        self.X = X
        self.Y = Y
        self.feature_names = feature_names

    def extract_features_kbest(self, file_name, K='all'):
        '''
            Extraction of top-k features
        '''
        selector = SelectKBest(chi2, k=K)
        selector.fit_transform(self.X, self.Y)

        scores = {self.feature_names[i]: (s, selector.pvalues_[i]) for i, s in enumerate(list(selector.scores_)) if
                  not math.isnan(s)}
        scores = sorted(scores.items(), key=operator.itemgetter([1][0]), reverse=True)
        f = codecs.open(file_name, 'w')
        f.write('\t'.join(['feature', 'score', 'p-value', '#I-out-of-' + str(np.sum(self.Y)),
                           '#O-out-of-' + str(len(self.Y) - np.sum(self.Y))]) + '\n')
        for w, score in scores:
            feature_array = self.X[:, self.feature_names.index(w)]
            pos = [feature_array[idx] for idx, x in enumerate(self.Y) if x == 1]
            neg = [feature_array[idx] for idx, x in enumerate(self.Y) if x == 0]
            f.write('\t'.join([str(w), str(score[0]), str(score[1]), str(np.sum(pos)), str(np.sum(neg))]) + '\n')
        f.close()

    def extract_features_fdr(self, file_name, N, alpha=5e-2):
        '''
            Feature extraction with fdr-correction
        '''
        # https://brainder.org/2011/09/05/fdr-corrected-fdr-adjusted-p-values/
        # Filter: Select the p-values for an estimated false discovery rate
        # This uses the Benjamini-Hochberg procedure. alpha is an upper bound on the expected false discovery rate.
        selector = SelectFdr(chi2, alpha=alpha)
        selector.fit_transform(self.X, self.Y)
        scores = {self.feature_names[i]: (s, selector.pvalues_[i]) for i, s in enumerate(list(selector.scores_)) if
                  not math.isnan(s)}
        scores = sorted(scores.items(), key=operator.itemgetter([1][0]), reverse=True)[0:N]
        f = codecs.open(file_name, 'w')
        f.write('\t'.join(['feature', 'score', 'p-value', '#I-out-of-' + str(np.sum(self.Y)),
                           '#O-out-of-' + str(len(self.Y) - np.sum(self.Y))]) + '\n')
        self.X = self.X.toarray()
        for w, score in scores:
            feature_array = self.X[:, self.feature_names.index(w)]
            pos = [feature_array[idx] for idx, x in enumerate(self.Y) if x == 1]
            neg = [feature_array[idx] for idx, x in enumerate(self.Y) if x == 0]
            f.write('\t'.join([str(w), str(score[0]), str(score[1]), str(np.sum(pos)), str(np.sum(neg))]) + '\n')
        f.close()
        return scores
