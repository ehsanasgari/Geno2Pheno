
import codecs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from dimreduction.visualize_matrix import VisualizeMatrix
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import numpy as np
from sklearn.ensemble import RandomForestClassifier


snp_dir='/mounts/data/proj/asgari/github_data/data/pseudomonas/data/'
label_files='MIC/v2/mic_bin_with_intermediate.txt'
gene_expression_file='gene_expression/rpg_log_transformed.txt'

label_mapping={str(l.strip().split('\t')[0]):[int(float(str(x)))  for x in l.strip().split('\t')[1::]] if len(l.strip().split('\t')[1::])==5 else [0,0,0,0,0] for l in codecs.open(snp_dir+label_files,'r','utf-8').readlines()[1::]}
classes= ['Ciprofloxacin','Tobramycin','Colistin','Ceftazidim','Meropenem']

# question 'MHH2417', does have only one entry
gene_expression=[l.strip() for l in codecs.open(snp_dir+gene_expression_file,'r','utf-8').readlines()]
gene_expression_mapping={str(entry.split('\t')[0]):[float(str(x)) for x in entry.split('\t')[1::]] for entry in gene_expression[1::]}
training_instances=set(label_mapping.keys()).intersection(gene_expression_mapping.keys())

Y=[label_mapping[x][0] for x in training_instances]
X=np.array([gene_expression_mapping[x] for x in training_instances])

model = RandomForestClassifier(n_estimators=100, criterion='gini')

model = model.fit(X, Y)
