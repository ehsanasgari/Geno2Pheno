__author__ = "Ehsaneddin Asgari"
__copyright__ = "Copyright 2017, HH-HZI Project"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Ehsaneddin Asgari"
__email__ = "asgari@berkeley.edu"

from data_access_utility import ABRAccessUtility
from random_forest import RandomForest
from file_utility import  FileUtility
import itertools
import numpy as np 

def normal_features():
    for feat in ['genePA','geneexp','SNPs']:
        ABRAccess=ABRAccessUtility('/mounts/data/proj/asgari/github_data/data/pseudomonas/data_v2/extracted_features/',[feat])
        for drug in ABRAccess.drugs:
            print(feat, drug)
            X,Y,features=ABRAccess.getXYPredictionMat(drug, mapping={'0':0,'0.0':0,'1':1,'1.0':1})
            RF=RandomForest(X,Y,features)
            results=RF.evaluateOLO()
            FileUtility.save_obj(results,'/mounts/data/proj/asgari/github_repos/abr_prediction/results/classification/random_forest/param1/'+'_'.join([feat,drug]))
            RF.get_important_features('/mounts/data/proj/asgari/github_repos/abr_prediction/results/classification/random_forest/param1/'+'_'.join([feat,drug])+'_features.txt',100)

def sequence_features():
    for feat in ['seq_6_gram']:
        ABRAccess=ABRAccessUtility('/mounts/data/proj/asgari/github_data/data/pseudomonas/data_v2/extracted_features/',[feat])
        for drug in ABRAccess.drugs:
            print(feat, drug)
            X,Y,features=ABRAccess.getXYPredictionMat_scaffolds(drug, mapping={'0':0,'0.0':0,'1':1,'1.0':1})
            print (X.shape)
            print (np.sum(Y))
            #RF=RandomForest(X,Y,features)
            #results=RF.kFoldCV()
            #FileUtility.save_obj(results,'/mounts/data/proj/asgari/github_repos/abr_prediction/results/classification/random_forest/param1/'+'_'.join([feat,drug]))
            #RF.get_important_features('/mounts/data/proj/asgari/github_repos/abr_prediction/results/classification/random_forest/param1/'+'_'.join([feat,drug])+'_features.txt',100)

sequence_features()

