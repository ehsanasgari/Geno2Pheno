
import codecs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import numpy as np
from sklearn.ensemble import RandomForestClassifier


data_dir='/mounts/data/proj/asgari/github_data/data/pseudomonas/data/'
label_files='MIC/v2/mic_bin_with_intermediate.txt'
gene_expression_file='gene_expression/rpg_log_transformed.txt'

label_mapping={str(l.strip().split('\t')[0]):[int(float(str(x)))  for x in l.strip().split('\t')[1::]] if len(l.strip().split('\t')[1::])==5 else [0,0,0,0,0] for l in codecs.open(data_dir+label_files,'r','utf-8').readlines()[1::]}
classes= ['Ciprofloxacin','Tobramycin','Colistin','Ceftazidim','Meropenem']

# question 'MHH2417', does have only one entry
gene_expression=[l.strip() for l in codecs.open(data_dir+gene_expression_file,'r','utf-8').readlines()]
gene_expression_mapping={str(entry.split('\t')[0]):[float(str(x)) for x in entry.split('\t')[1::]] for entry in gene_expression[1::]}
training_instances=set(label_mapping.keys()).intersection(gene_expression_mapping.keys())

Y=[str(label_mapping[x][0]) for x in training_instances]
X=np.array([gene_expression_mapping[x] for x in training_instances])
print (X.shape)
print (len(Y))
f=open('results_SVM.txt','w')
clf = svm.SVC(kernel='rbf', C=1)
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=1)
scores = cross_val_score(clf, X, Y, cv=cv, scoring='f1_macro')
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
f.write("F1: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std())+'\n')
y_pred=clf.fit(X, Y).predict(X)
showing_labels=list(set(Y))
showing_labels.sort()
print(confusion_matrix(Y, y_pred,labels=showing_labels))
confusion=confusion_matrix(Y, y_pred,labels=showing_labels)
f.write('\n\nConfusion matix\n')
f.write(' '.join([str(x) for x in showing_labels])+'\n')
for row in confusion:
    f.write(' '.join([str(elem) for elem in row])+'\n')
f.close()
