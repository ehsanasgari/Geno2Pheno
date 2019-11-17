import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

import warnings
from sklearn.exceptions import DataConversionWarning, FitFailedWarning, UndefinedMetricWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=FitFailedWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)

import sys
sys.path.append('../')
from data_access.data_access import GenotypePhenotypeAccess
import numpy as np
from utility.file_utility import FileUtility
from classifier.classical_classifiers import SVM

# ariel ['Ciprofloxacin_S-vs-R']
# triton ['Tobramycin_S-vs-R']
for drug in  ['Ceftazidim_S-vs-R', 'Meropenem_S-vs-R']:
    gp_access = GenotypePhenotypeAccess(f"/net/sgi/metagenomics/projects/pseudo_genomics/results/geno2pheno_package/Pseudogenomics_nestedSVM/", mapping={'1':1,'1.0':1,'0':0,'0.0':0})
    for r in np.arange(0.9,0,-0.1):    
        feature_type = 'gexpgpa'+str(round(r,1))
        X,Y,features, iso=gp_access.get_xy_prediction_mats([feature_type],drug,mapping={'0': 0, '0.0': 0, '1': 1, '1.0': 1})
        print(X.shape, len(Y), len(features), len(iso))
        svm = SVM(X,Y)
        svm.tune_and_eval_predefined(f"/net/sgi/metagenomics/projects/pseudo_genomics/results/geno2pheno_package/Pseudogenomics_nestedSVM/classifications/S_vs_R/{drug}/{feature_type}_CV_tree_SVM", iso, f"/net/sgi/metagenomics/projects/pseudo_genomics/results/geno2pheno_package/Pseudogenomics_nestedSVM/classifications/S_vs_R/{drug}/cv/tree/{drug}_S_vs_R_folds.txt", f"/net/sgi/metagenomics/projects/pseudo_genomics/results/geno2pheno_package/Pseudogenomics_nestedSVM/classifications/S_vs_R/{drug}/cv/tree/{drug}_S_vs_R_test.txt",  njobs=50, feature_names=features, optimized_for='f1_macro',params=[{'C': [1000, 500, 200, 100, 50, 20, 10, 5, 2, 1, 0.2, 0.5, 0.01, 0.02, 0.05, 0.001]}])