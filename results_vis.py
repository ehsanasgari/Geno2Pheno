__author__ = "Ehsaneddin Asgari"
__copyright__ = "Copyright 2017, HH-HZI Project"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Ehsaneddin Asgari"
__email__ = "asgari@berkeley.edu / ehsaneddin.asgari@helmholtz-hzi.de"

from file_utility import  FileUtility
from data_access_utility import ABRAccessUtility
import numpy as np
import codecs
from sklearn.preprocessing import normalize
import numpy as np
from file_utility import  FileUtility


def create_table(dict_results,average_dict):
    
    text = """\\begin{table}
    \centering
    \\resizebox{\columnwidth}{!}
    {\\begin{tabular}{|l|c|c|c|c|c|c|c||c|c|c|c|c|}           \hline 
    \multirow{2}{*}{Drug} & \multirow{2}{*}{$\# $R} & \multirow{2}{*}{$\# \\neg$R} & \multicolumn{5}{c||}{Random Forest} & \multicolumn{5}{c|}{SVM - Linear} \\\   \cline{4-13}
     & &  &{Prec.} & {Rec.} & {F1} & {TNR} & {ACC} & {Prec.} & {Rec.} & {F1} & {TNR} & {ACC}\\\   \hline """
    for k,res_dict in dict_results.items():
        text+=" {"+k+"}  & "+ str(round(res_dict['R'],2))+" & "+ str(round(res_dict['NR'],2))+""" & """+ str(round(res_dict['Prec'],2)) +""" & """+ str(round(res_dict['TPR/Rec'],2)) +""" & """+ str(round(res_dict['F'],2)) +""" & """+ str(round(res_dict['TNR'],2)) +""" & """+ str(round(res_dict['ACC'],2)) +""" & & & & & \\\ \hline """     

    text+=" \hline {Average}  & "+ average_dict['R']+" & "+ average_dict['NR']+""" & """+ average_dict['Prec'] +""" & """+ average_dict['TPR/Rec'] +""" & """+ average_dict['F'] +""" & """+ average_dict['TNR'] +""" & """+ average_dict['ACC'] +""" & & & & & \\\ \hline """     
    text+=""" \end{tabular}}
     \end{table}"""
    return text


def create_result_table(direct='results/random_forest/', out_file='table.tex')
    ABRAccess=ABRAccessUtility('/mounts/data/proj/asgari/github_data/data/pseudomonas/data_v2/extracted_features/',['genePA','geneexp','SNPs'])

    results=dict()
    for drug in ABRAccess.drugs:
        results[drug]=dict()
        res=FileUtility.load_obj(direct+drug+'.pickle')[1::]
        TP=res.count((1,1))
        TN=res.count((0,0))
        FP=res.count((0,1))
        FN=res.count((1,0))
        TPR_recal=TP/(TP+FN)
        PRE=TP/(TP+FP)
        TNR=TN/(TN+FP)
        ACC=(TP+TN)/(TN+FP+FN+TP)
        F=2/(1/PRE+1/TPR_recal)
        R=TP+FN
        NR=TN+FP
        results[drug]['R']=(R)
        results[drug]['NR']=(NR)
        results[drug]['Prec']=(round(PRE,2))
        results[drug]['TPR/Rec']=(round(TPR_recal,2))
        results[drug]['TNR']=(round(TNR,2))
        results[drug]['ACC']=(round(ACC,2))
        results[drug]['F']=(round(F,2))

    average_dict=dict()
    average_dict['R']=str(round(np.mean([results[drug]['R'] for drug in ABRAccess.drugs]),1))
    average_dict['NR']=str(round(np.mean([results[drug]['NR'] for drug in ABRAccess.drugs]),2))
    average_dict['Prec']=str(round(np.mean([results[drug]['Prec'] for drug in ABRAccess.drugs]),2))
    average_dict['TPR/Rec']=str(round(np.mean([results[drug]['TPR/Rec'] for drug in ABRAccess.drugs]),2))
    average_dict['TNR']=str(round(np.mean([results[drug]['TNR'] for drug in ABRAccess.drugs]),2))
    average_dict['ACC']=str(round(np.mean([results[drug]['ACC'] for drug in ABRAccess.drugs]),2))
    average_dict['F']=str(round(np.mean([results[drug]['F'] for drug in ABRAccess.drugs]),2))

    f=open(out_file,'w')
    f.write(create_table(results,average_dict))
    f.close()
    
def create_separate_drug_visualization():
    directory='/mounts/data/proj/asgari/github_repos/abr_prediction/results/separate_drugs_separate_features/chi2/details/'
    possiblities=[('R','I'),('R','V'),('I','V')]
    color_dict={'I':'#FFF176', 'IR':'#FFCC80', 'R':'#E57373', 'V':'#76FF03','IV':'#C6FF00','RV':'#E0E0E0'}
    ABRAccess=ABRAccessUtility('/mounts/data/proj/asgari/github_data/data/pseudomonas/data_v2/extracted_features/',['genePA','SNPs','geneexp'])

    thrsh=10
    for drug in ABRAccess.drugs:
        print(drug)
        features_label=dict()
        features_idx=dict()
        X,Y,features_matrix=ABRAccess.getXYPredictionMat(drug, mapping={'0':0,'0.0':0,'1':1,'1.0':1,'':0})
        features_label=dict()
        for x,y in possiblities:
            lines=[l.strip().split() for l in codecs.open(directory+x+'-vs-'+y+'_'+drug+'_SNPs.txt','r', 'utf-8').readlines()][1::]    
            for l in lines:
                l[0]='SNPs##'+l[0]
                if float(l[1])>=thrsh: 
                    if float(l[3])>float(l[4]):
                        if l[0] not in features_label:
                            features_label[l[0]]=set(x)
                        else:
                            features_label[l[0]].add(x)
                    else:
                        if l[0] not in features_label:
                            features_label[l[0]]=set([y])
                        else:
                            features_label[l[0]].add(y)
            lines=[l.strip().split() for l in codecs.open(directory+x+'-vs-'+y+'_'+drug+'_genePA.txt','r', 'utf-8').readlines()][1::]    
            for l in lines:
                if float(l[1])>=thrsh: 
                    l[0]='genePA##'+l[0]
                    if float(l[3])>float(l[4]):
                        if l[0] not in features_label:
                            features_label[l[0]]=set(x)
                        else:
                            features_label[l[0]].add(x)
                    else:
                        if l[0] not in features_label:
                            features_label[l[0]]=set(y)
                        else:
                            features_label[l[0]].add(y)
            lines=[l.strip().split() for l in codecs.open(directory+x+'-vs-'+y+'_'+drug+'_geneexp.txt','r', 'utf-8').readlines()][1::]    
            for l in lines:
                if float(l[1])>=thrsh: 
                    l[0]='geneexp##'+l[0]
                    if float(l[3])>float(l[4]):
                        if l[0] not in features_label:
                            features_label[l[0]]=set(x)
                        else:
                            features_label[l[0]].add(x)
                    else:
                        if l[0] not in features_label:
                            features_label[l[0]]=set(y)
                        else:
                            features_label[l[0]].add(y)
            important_features=list(features_label.keys())
            important_features.sort()
            new_dict={x:features_matrix.index(x) for x in important_features}
            features_idx.update(new_dict)
        features_label={k:''.join(sorted(list(v))) for k,v in features_label.items()}
        features_list=list(features_label.keys())
        features_list.sort()
        idx=[features_idx[x] for x in features_list]
        features_names_list=[x.replace('_','-').replace(',','#')  for x in features_list]
        mat=normalize(X[:,idx].T.toarray()+1e-100, norm='l1')
        out = stats.entropy(mat.T[:,:,None], mat.T[:,None,:])
        HC=HierarchicalClutering(out, features_names_list)
        out_nwk=HC.nwk
        out_color_dict={features_names_list[idx]:color_dict[features_label[feat]] for idx,feat in enumerate(features_list)}
        FileUtility.save_obj([out_nwk,out_color_dict,color_dict],'/mounts/data/proj/asgari/github_repos/abr_prediction/results/hierclustering_chi2/pickles/separate_chi2_'+str(thrsh)+'_'+drug)

def creat_visualization_for_profile():
    directory='/mounts/data/proj/asgari/github_repos/abr_prediction/results/multiple_drugs_separate_features/chi2/details/'
    color_dict={'R':'#E57373', 'U':'white','RU':'#E0E0E0'}
    ABRAccess=ABRAccessUtility('/mounts/data/proj/asgari/github_data/data/pseudomonas/data_v2/extracted_features/',['genePA','SNPs','geneexp'])

    thrsh=10
    pttrn='RRRRR'

    features_label=dict()
    features_idx=dict()
    X,Y,features_matrix=ABRAccess.getXYPredictionMat(drug, mapping={'0':0,'0.0':0,'1':1,'1.0':1,'':0})
    features_label=dict()
    x='R'
    y='U'
    lines=[l.strip().split() for l in codecs.open(directory+pttrn+'_SNPs.txt','r', 'utf-8').readlines()][1::]    
    for l in lines:
        l[0]='SNPs##'+l[0]
        if float(l[1])>=thrsh: 
            if float(l[3])>float(l[4]):
                if l[0] not in features_label:
                    features_label[l[0]]=set(x)
                else:
                    features_label[l[0]].add(x)
            else:
                if l[0] not in features_label:
                    features_label[l[0]]=set([y])
                else:
                    features_label[l[0]].add(y)
    lines=[l.strip().split() for l in codecs.open(directory+pttrn+'_genePA.txt','r', 'utf-8').readlines()][1::]    
    for l in lines:
        if float(l[1])>=thrsh: 
            l[0]='genePA##'+l[0]
            if float(l[3])>float(l[4]):
                if l[0] not in features_label:
                    features_label[l[0]]=set(x)
                else:
                    features_label[l[0]].add(x)
            else:
                if l[0] not in features_label:
                    features_label[l[0]]=set(y)
                else:
                    features_label[l[0]].add(y)
    lines=[l.strip().split() for l in codecs.open(directory+pttrn+'_geneexp.txt','r', 'utf-8').readlines()][1::]    
    for l in lines:
        if float(l[1])>=thrsh: 
            l[0]='geneexp##'+l[0]
            if float(l[3])>float(l[4]):
                if l[0] not in features_label:
                    features_label[l[0]]=set(x)
                else:
                    features_label[l[0]].add(x)
            else:
                if l[0] not in features_label:
                    features_label[l[0]]=set(y)
                else:
                    features_label[l[0]].add(y)
    important_features=list(features_label.keys())
    important_features.sort()
    new_dict={x:features_matrix.index(x) for x in important_features}
    features_idx.update(new_dict)
    features_label={k:''.join(sorted(list(v))) for k,v in features_label.items()}
    features_list=list(features_label.keys())
    features_list.sort()
    idx=[features_idx[x] for x in features_list]
    features_names_list=[x.replace('_','-').replace(',','#')  for x in features_list]
    mat=normalize(X[:,idx].T.toarray()+1e-100, norm='l1')
    out = stats.entropy(mat.T[:,:,None], mat.T[:,None,:])
    HC=HierarchicalClutering(out, features_names_list)
    out_nwk=HC.nwk
    out_color_dict={features_names_list[idx]:color_dict[features_label[feat]] for idx,feat in enumerate(features_list)}
    FileUtility.save_obj([out_nwk,out_color_dict,color_dict],'/mounts/data/proj/asgari/github_repos/abr_prediction/results/hierclustering_chi2/multi_chi2_'+pttrn+'_'+str(thrsh))
