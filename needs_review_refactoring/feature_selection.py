__author__ = "Ehsaneddin Asgari"
__copyright__ = "Copyright 2017, HH-HZI Project"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Ehsaneddin Asgari"
__email__ = "asgari@berkeley.edu ehsaneddin.asgari@helmholtz-hzi.de"

from data_access_utility import ABRAccessUtility
import numpy as np
from scipy import sparse
from chi2analysis import Chi2Anlaysis
import operator
import codecs
from file_utility import FileUtility

def single_drug_pattern(file_name, list_of_feature=['genePA','geneexp','SNPs'], mapping={'0':0,'0.0':0,'1':1,'1.0':1}):
    ABRAccess=ABRAccessUtility('/mounts/data/proj/asgari/github_data/data/pseudomonas/data_v2/extracted_features/',list_of_feature)
    drug2labeledisolates_mapping=ABRAccess.get_labels(mapping=mapping)
    global_selected=set()
    for drug in ABRAccess.drugs:
        iso2label=drug2labeledisolates_mapping[drug]
        scores=dict()
        for key in list(ABRAccess.X.keys()):
            print (drug,' ',key)
            iso2label=dict(iso2label)
            idx_removed=[idx for idx, iso in enumerate(ABRAccess.isolates[key]) if iso not in iso2label]
            refined_isos=[iso for iso in ABRAccess.isolates[key] if iso in iso2label]
            Y=[iso2label[iso] for iso in refined_isos]
            X=sparse.csr_matrix(np.delete(ABRAccess.X[key].toarray(),idx_removed, 0))
            CH2=Chi2Anlaysis(X,Y,ABRAccess.feature_names[key])
            sc=CH2.extract_features_fdr('results/separate_drugs_separate_features/details/'+file_name+'_'+drug+'_'+key+'.txt',100)
            scores[key]=sc
        score_aggregated=[]
        for feature, score_list in scores.items():
            [score_aggregated.append(('##'.join([feature,feat_detail]),value)) for feat_detail,value in score_list]
        score_aggregated=dict(score_aggregated)
        score_aggregated = sorted(score_aggregated.items(), key=operator.itemgetter([1][0]),reverse=True)
        f = codecs.open('results/separate_drugs_separate_features/aggregation/'+file_name+'_all_'+drug+'.txt','w')
        f.write('\t'.join(['feature', 'score', 'p-value'])+'\n')
        selected=set()
        for w, score in score_aggregated:
            f.write('\t'.join([str(w), str(score[0]), str(score[1])])+'\n')
            if score[0]>10:
                selected.add(str(w))
        f.close()
        if len(selected)>0 and len(global_selected)==0:
            global_selected=selected
        elif len(global_selected)>0:
            global_selected.intersection(selected)
    FileUtility.save_list('results/separate_drugs_separate_features/'+file_name+'_alldrugs.txt',list(global_selected))
            

def multiple_drug_feature_selection():
    ABRAccess=ABRAccessUtility('/mounts/data/proj/asgari/github_data/data/pseudomonas/data_v2/extracted_features/',['genePA','geneexp','SNPs','k4-9mers'])
    joint_labels=ABRACCESS.get_multilabel_labels()
    label_freq=FreqDist(list(joint_labels.values()))
    all_labels=list(set(joint_labels.values()))
    labeling_schemes=dict()

    for label in all_labels:
        scores=dict()
        print (label,' ', label_freq[label],'------------')
        for key in list(ABRAccess.X.keys()):
            print (label,' ',key)
            idx_removed=[idx for idx, iso in enumerate(ABRAccess.isolates[key]) if iso not in joint_labels]
            refined_isos=[iso for iso in ABRAccess.isolates[key] if iso in joint_labels]
            Y=[joint_labels[iso] for iso in refined_isos]
            Y=[1 if y==label else 0 for y in Y]
            X=sparse.csr_matrix(np.delete(ABRAccess.X[key].toarray(),idx_removed, 0))
            CH2=Chi2Anlaysis(X,Y,ABRAccess.feature_names[key])
            sc=CH2.extract_features_fdr('results/multiple_drugs_separate_features/details/'+label+'_'+key+'.txt',100)
            scores[key]=sc
        score_aggregated=[]
        for feature, score_list in scores.items():
            [score_aggregated.append(('##'.join([feature,feat_detail]),value)) for feat_detail,value in score_list]
        score_aggregated=dict(score_aggregated)
        score_aggregated = sorted(score_aggregated.items(), key=operator.itemgetter([1][0]),reverse=True)
        f = codecs.open('results/multiple_drugs_separate_features/'+label+'_all_frq'+str(label_freq[label])+'.txt','w')
        f.write('\t'.join(['feature', 'score', 'p-value'])+'\n')
        selected=set()
        for w, score in score_aggregated:
            f.write('\t'.join([str(w), str(score[0]), str(score[1])])+'\n')
            if score[0]>10:
                selected.add(str(w))
        f.close()

def multiple_drug_feature_selection_specific(pos,neg):
    ABRAccess=ABRAccessUtility('/mounts/data/proj/asgari/github_data/data/pseudomonas/data_v2/extracted_features/',['genePA','geneexp','SNPs'])
    joint_labels=ABRAccess.get_multilabel_labels()
    joint_labels={k:v for k,v in joint_labels.items() if v in [pos,neg]}
    
    scores=dict()
    for key in list(ABRAccess.X.keys()):
        print (pos,' ',key)
        idx_removed=[idx for idx, iso in enumerate(ABRAccess.isolates[key]) if iso not in joint_labels]
        refined_isos=[iso for iso in ABRAccess.isolates[key] if iso in joint_labels]
        Y=[joint_labels[iso] for iso in refined_isos]
        Y=[1 if y==pos else 0 for y in Y]
        X=sparse.csr_matrix(np.delete(ABRAccess.X[key].toarray(),idx_removed, 0))
        CH2=Chi2Anlaysis(X,Y,ABRAccess.feature_names[key])
        sc=CH2.extract_features_fdr('/mounts/data/proj/asgari/github_repos/abr_prediction/results/multiple_drugs_separate_features/chi2/details/specific_'+pos+'_vs_'+neg+'_'+key+'.txt',100)
        scores[key]=sc
    score_aggregated=[]
    for feature, score_list in scores.items():
        [score_aggregated.append(('##'.join([feature,feat_detail]),value)) for feat_detail,value in score_list]
    score_aggregated=dict(score_aggregated)
    score_aggregated = sorted(score_aggregated.items(), key=operator.itemgetter([1][0]),reverse=True)
    f = codecs.open('/mounts/data/proj/asgari/github_repos/abr_prediction/results/multiple_drugs_separate_features/chi2/specific_'+pos+'_vs_'+neg+'.txt','w')
    f.write('\t'.join(['feature', 'score', 'p-value'])+'\n')
    selected=set()
    for w, score in score_aggregated:
        f.write('\t'.join([str(w), str(score[0]), str(score[1])])+'\n')
        if score[0]>10:
            selected.add(str(w))
    f.close()
        
        
def all_single_drugs():
    settings={'R-vs-V':{'0':0,'0.0':0,'1':1,'1.0':1},'R-vs-I':{'':0,'1':1,'1.0':1},'I-vs-V':{'':1,'0':0,'0.0':0},'R-vs-all':{'':0,'0':0,'0.0':0,'1':1,'1.0':1},'I-vs-all':{'':1,'0':0,'0.0':0,'1':0,'1.0':0},'V-vs-all':{'':0,'0':1,'0.0':1,'1':0,'1.0':0}}
    for label, mapping in settings.items():
        print(label+'=========================================')
        single_drug_pattern(label, list_of_feature=['genePA','geneexp','SNPs','k4-9mers'], mapping=mapping)

multiple_drug_feature_selection_specific('RRRRR','RRVRR')        