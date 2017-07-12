# -*- coding: utf8 -*-
'''
Created 2014-2016

@author: Ehsaneddin Asgari
@email: asgari@berkeley.edu
'''
from sklearn.manifold import TSNE
import codecs
from numpy import *
import numpy as np

from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
from sklearn.cluster import KMeans
from nltk import FreqDist


class VisualizeMatrix(object):
    '''
    classdocs
    '''

    def __init__(self, mat, labels, texts, path, rows_to_be_vis=[], start_unmarked=0):
        '''
        Constructor
        '''
        self.start_unmarked=start_unmarked
        self.ndim = mat
        self.rows_to_be_vis=rows_to_be_vis
        self.names = [texts[idx] for idx in self.rows_to_be_vis]
        self.classes = np.zeros((self.ndim.shape[0], 1))
        self.classes[:, 0] = labels
        self.path = path
        self._updateTSNEres()
        self._generateHTML()

    @staticmethod
    def convert_matrix(B, make_binary=False):
        '''
        to be fixed
        :param B:
        :param make_binary:
        :return:
        '''
        class_dict = dict()
        if make_binary:
            B /= max(np.max(B), 1)
            B = np.round(B)
        B.astype(int)
        c = np.zeros(B.shape[0])
        frequent_class = []
        for row in range(0, B.shape[0]):
            idx = ''.join([str(int(x)) for x in list(B[row, :])])
            frequent_class.append(idx)
            if idx in class_dict:
                c[row] = class_dict[idx]
            else:
                class_dict[idx] = len(class_dict) + 1
                c[row] = class_dict[idx]
        return B, c, frequent_class

    def _updateTSNEres(self):
        model = TSNE(n_components=2)
        np.set_printoptions(suppress=False)
        mat=self.ndim[self.rows_to_be_vis,:]
        z= zeros((len(self.rows_to_be_vis),1))
        z[self.start_unmarked:-1]=1
        mat=np.append(mat, z, axis=1)
        print (mat.shape)
        self.tsne_res = model.fit_transform(mat)
        np.savetxt(self.path+'.txt',np.hstack((self.tsne_res, self.classes[self.rows_to_be_vis])))
        np.savetxt(self.path+'_include.txt',self.rows_to_be_vis)
        # self.tsne_res = self.tsne_res[self.rows_to_be_vis,:]


        #kmeans = KMeans(n_clusters=20, random_state=0).fit(self.original_mat)
        #self.classes[:, 0] = (kmeans.labels_ + 1)
        H, xedges, yedges = np.histogram2d(self.tsne_res[:, 0], self.tsne_res[:, 1], bins=15)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        plt.contour(H, extent=extent)
        plt.scatter(self.tsne_res[:, 0], self.tsne_res[:, 1])
        plt.savefig(self.path + '.png', dpi=1200)
        self.tsne_res = np.hstack((self.tsne_res, self.classes[self.rows_to_be_vis]))

    def _generateHTML(self):
        f = codecs.open(self.path + ".html", 'w', 'utf-8')
        lines = [l.strip() for l in codecs.open('html_template', 'r', 'utf-8').readlines()]
        for i, line in enumerate(lines):
            f.write(line)
            if i == 137:
                for row_num, row in enumerate(list(self.tsne_res)):
                    print(row)
                    if not row_num == 0:
                        f.write('\r')
                    f.write('\t'.join([str(row[0]), str(row[1]), str(int(row[2]))]))
            elif i == 140:
                [f.write(row + '\r') for row in self.names]
            else:
                f.write('\r')
        f.close()


if __name__ == '__main__':
    V2 = VisualizeMatrix(np.array([[1, 0, 0], [1, 0, 1], [1, 1, 1], [0, 0, 0]]), ['1', '2', '3', '4'], ['1', '2', '3', '4'], 'index')
    V2._generateHTML()
