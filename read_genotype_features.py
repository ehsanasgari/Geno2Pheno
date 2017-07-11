
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
import itertools




data_dir='/mounts/data/proj/asgari/github_data/data/pseudomonas/data/'
label_files='MIC/v2/mic_bin_with_intermediate.txt'
gene_expression_file='gene_expression/rpg_log_transformed.txt'
snps_files='snp/v2/all_SNPs_final_bin.txt'

# data reading
label_mapping={str(l.strip().split('\t')[0]):[int(float(str(x)))  for x in l.strip().split('\t')[1::]] if len(l.strip().split('\t')[1::])==5 else [0,0,0,0,0] for l in codecs.open(data_dir+label_files,'r','utf-8').readlines()[1::]}

#gene_expression=[l.strip() for l in codecs.open(data_dir+gene_expression_file,'r','utf-8').readlines()]
#gene_expression_mapping={str(entry.split('\t')[0]):[float(str(x)) for x in entry.split('\t')[1::]] for entry in gene_expression[1::]}

snps=[l.strip().replace(' ','') for l in codecs.open(data_dir+snps_files,'r','utf-8').readlines()]
mapping={'0.0':-1,'0':-1,'1.0':1,'1':1,'':0}
snps_mapping={str(entry.split('\t')[0]):[mapping[str(x)] for x in entry.split('\t')[1::]] for entry in snps[1::]}

#training_instances=set(label_mapping.keys()).intersection(gene_expression_mapping.keys())
#X=np.array([gene_expression_mapping[x] for x in training_instances])
training_instances=set(label_mapping.keys()).intersection(snps_mapping.keys())
X=np.array([snps_mapping[x] for x in training_instances])
# question 'MHH2417', does have only one entry

classes_idx=[0,1,2,3,4]
classes= ['Ciprofloxacin','Tobramycin','Colistin','Ceftazidim','Meropenem']
classifier_method=['svm','rf']

f=[]
for x in classes:
    f.append(open('results_snps_'+x+'.txt','w'))

clf_random_forest=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=2,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)

clf_svm = svm.SVC(kernel='rbf', C=1)

classifier_method={'rf':clf_random_forest, 'svm':clf_svm}
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=1)

for classifier,class_idx in list(itertools.product(classifier_method, classes_idx)):

    Y=[label_mapping[x][class_idx] for x in training_instances]

    scores = cross_val_score(classifier_method[classifier], X, Y, cv=cv, scoring='precision')
    print("precision "+classifier+" : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
    f[class_idx].write("precision "+classifier+": %0.2f (+/- %0.2f)" % (scores.mean(), scores.std())+'\n')

    scores = cross_val_score(classifier_method[classifier], X, Y, cv=cv, scoring='recall')
    print("recall "+classifier+" : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
    f[class_idx].write("recall "+classifier+": %0.2f (+/- %0.2f)" % (scores.mean(), scores.std())+'\n')

    scores = cross_val_score(classifier_method[classifier], X, Y, cv=cv, scoring='f1')
    print("f1 "+classifier+" : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
    f[class_idx].write("f1 "+classifier+": %0.2f (+/- %0.2f)" % (scores.mean(), scores.std())+'\n')

    scores = cross_val_score(classifier_method[classifier], X, Y, cv=cv, scoring='roc_auc')
    print("roc_auc "+classifier+" : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
    f[class_idx].write("roc_auc "+classifier+": %0.2f (+/- %0.2f)" % (scores.mean(), scores.std())+'\n')


    y_pred=classifier_method[classifier].fit(X, Y).predict(X)
    showing_labels=list(set(Y))
    showing_labels.sort()
    print(confusion_matrix(Y, y_pred,labels=showing_labels))
    confusion=confusion_matrix(Y, y_pred,labels=showing_labels)
    f[class_idx].write('\nConfusion matrix\n')
    f[class_idx].write(' '.join([str(x) for x in showing_labels])+'\n')
    for row in confusion:
        f[class_idx].write(' '.join([str(elem) for elem in row])+'\n')

for class_idx in classes_idx:
    f[class_idx].close()
