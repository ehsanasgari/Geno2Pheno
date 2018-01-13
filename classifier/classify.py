import sys
sys.path.append('../')
from classifier.random_forest import RFClassifier
from classifier.svm import SVM
from utility.file_utility import FileUtility

#path='../../datasets/processed_data/body-site/cpe/'
#for x in FileUtility.recursive_glob(path, 'npe*.npz'):
#    if x.split('/')[-1].split('.')[0] not in ['npe_1000_2000','npe_10000_-1','npe_5000_-1','npe_2000_500']:
#        X=FileUtility.load_sparse_csr(x)
#        Y=FileUtility.load_list('../../datasets/processed_data/body-site/data_config/labels_phen.txt')
#        MRF = RFClassifier(X, Y)
#        MRF.tune_and_eval('../../datasets/results/body-sites/K/RF_'+x.split('/')[-1].split('.')[0])


path='../../datasets/processed_data/crohn/npe/'
for x in FileUtility.recursive_glob(path, 'npe*.npz'):
    if x.split('/')[-1].split('.')[0] in ['npe_5000_100','npe_10000_1000']:
        X=FileUtility.load_sparse_csr(x)
        Y=FileUtility.load_list('../../datasets/processed_data/crohn/data_config/labels_disease_complete1359.txt')
        MRF = RFClassifier(X, Y)
        MRF.tune_and_eval('../../datasets/results/crohn/features/RF_'+x.split('/')[-1].split('.')[0])


path='../../datasets/processed_data/body-site/cpe/'
for x in FileUtility.recursive_glob(path, 'npe*.npz'):
    if x.split('/')[-1].split('.')[0]  in ['npe_5000_100','npe_5000_1000']:
        X=FileUtility.load_sparse_csr(x)
        Y=FileUtility.load_list('../../datasets/processed_data/body-site/data_config/labels_phen.txt')
        MRF = RFClassifier(X, Y)
        MRF.tune_and_eval('../../datasets/results/body-sites/K/RF_'+x.split('/')[-1].split('.')[0])



import sys
sys.path.append('../')
from classifier.random_forest import RFClassifier
from classifier.svm import SVM
from utility.file_utility import FileUtility

#path='../../datasets/processed_data/body-site/cpe/'
#for x in FileUtility.recursive_glob(path, 'npe*.npz'):
#    if x.split('/')[-1].split('.')[0] not in ['npe_1000_2000','npe_10000_-1','npe_5000_-1','npe_2000_500']:
#        X=FileUtility.load_sparse_csr(x)
#        Y=FileUtility.load_list('../../datasets/processed_data/body-site/data_config/labels_phen.txt')
#        MRF = RFClassifier(X, Y)
#        MRF.tune_and_eval('../../datasets/results/body-sites/K/RF_'+x.split('/')[-1].split('.')
