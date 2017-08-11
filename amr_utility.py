__author__ = "Ehsaneddin Asgari"
__copyright__ = "Copyright 2017, HH-HZI Project"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Ehsaneddin Asgari"
__email__ = "asgari@berkeley.edu"

import codecs
import numpy as np
import random
from sklearn import metrics
import matplotlib.pyplot as plt
from nltk import FreqDist
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

class ABRUtility(object):
    '''
    This class is written to read/load/save
    data for AMR prediction of Pseudomonas Aeruginosa
    '''
    def __init__(self, data_dir, labels_addr, genexp_data_addr, snps_data_addr, gen_pres_abs_data_addr):
        print ('Data access created..')
        self.data_dir=data_dir
        self.labels_addr=labels_addr
        self.genexp_data_addr=genexp_data_addr
        self.snps_data_addr=snps_data_addr
        self.gen_pres_abs_data_addr=gen_pres_abs_data_addr

    def produce_feature_matrix(self, feature_file_address):
        '''
        :param feature_file_address:
        :return:
        '''
        rows=[l.strip() for l in codecs.open(feature_file_address,'r','utf-8').readlines()]
        mapping_isolate2feature={str(entry.split('\t')[0]):[float(str(x)) for x in entry.split('\t')[1::]] for entry in rows[1::]}
        features=rows[0].rstrip().split('\t')
        return mapping_isolate2feature, features

    def produce_label_str_vec(self, label_file_address):
        '''
            This function produces instances like: ZG02420619 ['1.0', '0.0', '0.0', '', '1.0']
        '''
        rows=[l.replace('\n','') for l in codecs.open(label_file_address,'r','utf-8').readlines()]
        isolate2label_vec_mapping={str(entry.split('\t')[0]):[str(x) for idx,x in enumerate(entry.split('\t')[1::])] for entry in rows[1::]}
        labels=rows[0].rstrip().split('\t')[1::]

        # init
        drug2labeledisolates_mapping=dict()
        for label in labels:
            drug2labeledisolates_mapping[label]=[]

        # only consider non-empty values
        for isolate,resist_vec in isolate2label_vec_mapping.items():
            for idx, val in enumerate(resist_vec):
                if val in ['0','0.0','1','1.0']:
                    drug2labeledisolates_mapping[labels[idx]].append((isolate,int(float(val))))
        # generate dict of labels for each class
        for label in labels:
            drug2labeledisolates_mapping[label]=dict(drug2labeledisolates_mapping[label])
        return isolate2label_vec_mapping, labels, drug2labeledisolates_mapping


    def common_isolates(self, list_of_list_of_isolates):
        '''
        :param list_of_list_of_isolates:
        :return:
        '''
        common_islt=set(list_of_list_of_isolates[0])
        for next_list in list_of_list_of_isolates[1::]:
            common_islt=common_islt.intersection(next_list)
        common_islt=list(common_islt)
        common_islt.sort()
        return common_islt

    def make_matrix(self, mapping, isolate_lsit):
        '''
        :param mapping:
        :param isolate_lsit:
        :return:
        '''
        return np.array([mapping[x] for x in isolate_lsit])

    def makeXY_prediction(self, X, label_dict, common_islt):
        '''
        :param X:
        :param label_dict:
        :param common_islt:
        :return:
        '''
        rows=[]
        labels=[]
        for isolate, label in label_dict.items():
            if isolate in common_islt:
                rows.append(common_islt.index(isolate))
                labels.append(label)
        return X[rows,:],labels

    def make_balanced_dataset(self, X, Y, coeff=1):
        '''
        :param X:
        :param Y:
        :param coeff:
        :return:
        '''
        all_idx=list(range(len(Y)))
        select_idx=[idx for idx,v in enumerate(Y) if v ==1]
        large_list=list(set(all_idx) - set(select_idx))
        random.shuffle(large_list)
        select_rand=large_list[0:coeff*len(select_idx)]
        rows=select_idx+select_rand
        return X[rows,:], [Y[x] for x in rows]

    def load_all_features(self):
        '''
        :param data_dir:
        :param labels_addr:
        :param genexp_data_addr:
        :param snps_data_addr:
        :param gen_pres_abs_data_addr:
        :return:
        '''

        # feature reading
        genexp_isolate2feature_mapping, genexp_features=self.produce_feature_matrix(self.data_dir+self.genexp_data_addr)
        snps_isolate2feature_mapping, snps_features=self.produce_feature_matrix(self.data_dir+self.snps_data_addr)
        genpa_isolate2feature_mapping, genpa_features=self.produce_feature_matrix(self.data_dir+self.gen_pres_abs_data_addr)

        # find isolates with all features
        self.common_islt=self.common_isolates([genexp_isolate2feature_mapping.keys(),snps_isolate2feature_mapping.keys(),genpa_isolate2feature_mapping.keys()])
        X_gene_exp=self.make_matrix(genexp_isolate2feature_mapping, self.common_islt)
        X_snp=self.make_matrix(snps_isolate2feature_mapping, self.common_islt)
        X_gene_pna=self.make_matrix(genpa_isolate2feature_mapping, self.common_islt)

        # concatinate the features
        self.X=np.concatenate((X_gene_exp,X_snp,X_gene_pna), axis=1)

        # label of features
        self.feature_labels=['genexp_'+x for x in genexp_features]+['snp_'+x for x in snps_features]+ ['gene_pa_'+x for x in genpa_features]

        # label reading
        self.isolate2label_vec_mapping, self.drug_names, self.drug2labeledisolates_mapping=self.produce_label_str_vec(self.data_dir+self.labels_addr)

    def get_multilabeltoword(self):
        mapping={'':'I','0':'D','0.0':'D','1':'R','1.0':'R'}
        return {k:''.join([mapping[x] for x in list(v)]) for k,v in self.isolate2label_vec_mapping.items()}

    def drug_similarity(self, fileaddress):
        multilabels=list(DA.get_multilabeltoword().values())
        N=len(self.drug_names)
        MI=np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                MI[i,j]=metrics.mutual_info_score([label[i] for label in list(multilabels)],[label[j] for label in list(multilabels)])

        fig, ax = plt.subplots()
        ax.set_xticklabels(['']+self.drug_names,rotation=45)
        ax.set_yticklabels(['']+self.drug_names)
        plt.imshow(MI, cmap='binary' )
        plt.savefig(fileaddress)
        return MI

    def resistance_frequency_analysis(self):
        multilabels=list(self.get_multilabeltoword().values())
        return FreqDist(multilabels)

    def return_most_k_frequent_classes(self, n):
        most_common_labels=[x for x,y in self.resistance_frequency_analysis().most_common(n)]
        joint_labels=self.get_multilabeltoword()
        selected_isolates=[x for x in self.common_islt if joint_labels[x] in most_common_labels]
        X,Y=self.makeXY_prediction(self.X, self.get_multilabeltoword(), selected_isolates)
        return X, Y, selected_isolates

    def return_ova_labeling(self):
        joint_labels=self.get_multilabeltoword()
        all_labels=list(set(joint_labels.values()))
        X,Y=self.makeXY_prediction(self.X, self.get_multilabeltoword(), self.common_islt)
        labeling_schemes=dict()
        for label in all_labels:
            labeling_schemes[label]=[1 if y==label else 0 for y in Y]
        return X, labeling_schemes

    def classifier_tuning(self, name, classifier, parameters):
        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=1)
        params=dict()
        for drug in self.drug_names:
            print ('parameter tuning for drug ',drug)
            Xpart, Y=self.makeXY_prediction(self.X, self.drug2labeledisolates_mapping[drug], self.common_islt)
            CT=ClassifierTuning(Xpart, Y, classifier, parameters)
            CT.find_best(name+'_'+drug)
            params[drug]=CT.clf
        return params

    def classifier_testing_RF(self, parameters_file_prefix):
        results=dict()
        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=1)
        for drug in self.drug_names:
            print (drug)
            results[drug]=[]
            Xpart, Y=self.makeXY_prediction(self.X, self.drug2labeledisolates_mapping[drug], self.common_islt)
            RF = pickle.load(open('tuned_params/'+parameters_file_prefix+"_"+drug+".pickle", "rb"))
            parameters=RF.best_params_
            clf_random_forest=RandomForestClassifier(bootstrap=True, criterion='gini',
            max_depth=None, max_features=parameters['max_features'], min_samples_split=parameters['min_samples_split'] , min_samples_leaf=1, n_estimators=parameters['n_estimators'], n_jobs=30)
            scores = cross_val_score(clf_random_forest, Xpart, Y, cv=cv, scoring='precision')
            print("precision: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
            results[drug].append(('RF','precision',scores.mean(), scores.std()))

            scores = cross_val_score(clf_random_forest,Xpart, Y, cv=cv, scoring='recall')
            print("recall: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
            results[drug].append(('RF','recall',scores.mean(), scores.std()))

            scores = cross_val_score(clf_random_forest, Xpart, Y, cv=cv, scoring='f1')
            print("f1: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
            results[drug].append(('RF','f1',scores.mean(), scores.std()))

            #scores = cross_val_score(clf_random_forest, Xpart, Y, cv=cv, scoring='roc_auc')
            #print("roc_auc: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
            #results[drug].append(('RF','roc_auc',scores.mean(), scores.std()))
    def extract_joint_relevant_features(self):
        X,labeling_scheme=self.return_ova_labeling()
        feature_names = self.feature_labels
        selector = SelectKBest(chi2,k='all')


        frequents=DA.resistance_frequency_analysis().most_common(100)

        for rank, (l,freq) in enumerate(frequents):
            print (l)
            L=labeling_scheme[l]
            selector.fit_transform(X, L )
            scores = {feature_names[i]: x for i, x in enumerate(list(selector.scores_))}
            scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[0:100]
            f = codecs.open('extracted_features/joint_drug/'+str(rank+1)+'_'+l+'_'+str(freq)+'_chi2.txt','w')
            f.write('\t'.join(['feature', 'score', 'avg I', 'avg O'])+'\n')
            for w, score in scores:
                feature_array=X[:,feature_names.index(w)]
                pos=[feature_array[idx] for idx, x in enumerate(L) if x==1]
                neg=[feature_array[idx] for idx, x in enumerate(L) if x==0]
                f.write('\t'.join([str(w), str(score), str(round(np.average(pos),2))+'(+/-)'+str(round(np.std(pos),2)), str(round(np.average(neg),2))+'(+/-)'+str(round(np.std(neg),2))])+'\n')
            f.close()
    def extract_drug_specific_features(self):
        feature_names = self.feature_labels
        selector = SelectKBest(chi2,k='all')

        for drug in self.drug_names:
            print (drug)
            Xpart, Y=self.makeXY_prediction(self.X, self.drug2labeledisolates_mapping[drug], self.common_islt)
            selector.fit_transform(Xpart, Y )
            scores = {feature_names[i]: x for i, x in enumerate(list(selector.scores_))}
            scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[0:100]
            f = codecs.open('extracted_features/separate_drug/'+drug+'_chi2.txt','w')
            f.write('\t'.join(['feature', 'score', 'avg I', 'avg O'])+'\n')
            for w, score in scores:
                feature_array=Xpart[:,feature_names.index(w)]
                pos=[feature_array[idx] for idx, x in enumerate(Y) if x==1]
                neg=[feature_array[idx] for idx, x in enumerate(Y) if x==0]
                f.write('\t'.join([str(w), str(score), str(round(np.average(pos),2))+'(+/-)'+str(round(np.std(pos),2)), str(round(np.average(neg),2))+'(+/-)'+str(round(np.std(neg),2))])+'\n')
            f.close()


data_dir='/mounts/data/proj/asgari/github_data/data/pseudomonas/data/'
labels_addr='MIC/v2/mic_bin_without_intermediate.txt'
genexp_data_addr='new/gene_expression/rpg_log_transformed_426.txt'
snps_data_addr='snp/v2/non-syn_SNPs_bin.txt'
gen_pres_abs_data_addr='new/annot.txt'

DA=ABRUtility(data_dir, labels_addr, genexp_data_addr, snps_data_addr, gen_pres_abs_data_addr)
DA.load_all_features()
