

import sys
sys.path.append('../')
from utility.genotype_file_utility import GenotypeReader
from utility.file_utility import FileUtility
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import tqdm
from decimal import Decimal

def getIDXs(isolates,df):
    isolates_all=df['Isolates'].values.tolist()
    return [isolates_all.index(x) for x in isolates]

def getIvalue(drug,mode,delta=0.005):
    scaler = MinMaxScaler()
    df=pd.read_table('/mounts/data/proj/asgari/dissertation/git_repos/Geno2Pheno/data_config/before_march_2018/Final_MICs_16.06.16.txt')
    isolates_all=df['Isolates'].values.tolist()
    res=df[['Isolates','CIP MIC','TOB MIC','COL MIC','CAZ MIC','MEM MIC']]
    R_S=df[['Unnamed: 2','Unnamed: 4','Unnamed: 6','Unnamed: 8','Unnamed: 10']].as_matrix().tolist()
    resistances_before=np.array([[float(str(x).replace('<=','').replace('≤','').replace('<=','').replace('≥','').replace('>=','')) for x in row] for row in res[['CIP MIC','TOB MIC','COL MIC','CAZ MIC','MEM MIC']].as_matrix()])
    isolates=[x[0] for idx, x in enumerate(list(df[['Isolates']].values))]
    drugs=['CIP','TOB','COL','CAZ','MEM']
    drugs_name=['Ciprofloxacin','Tobramycin','Colistin','Ceftazidim','Meropenem']
    idx=drugs.index(drug)
    resistances_after=resistances_before.copy()
    column=np.zeros((resistances_before.shape[0],1))
    column[:,0]=resistances_before[:,idx]
    resistances_after[:,idx]=scaler.fit_transform(column)[:,0]

    ###
    I_isolates=df[df['Unnamed: '+str(idx*2+2)]=='I']['Isolates'].values.tolist()
    I_idxs=getIDXs(I_isolates,df)
    mean_I=np.mean(resistances_after[I_idxs,idx])
    mean_I_before=np.mean(resistances_before[I_idxs,idx])

    #plt.hist(resistances_after[:,idx], bins='auto')
    #plt.title("Histogram with 'auto' bins")
    #plt.show()
    ###

    dfmiss=pd.read_table('/mounts/data/proj/asgari/final_proj/Geno2Pheno/notebooks/miscl_all.txt')
    dfmiss=dfmiss[dfmiss['mode']==mode]
    dfmiss=dfmiss[dfmiss['drug']==drugs_name[idx]]

    miss_classified=list(dfmiss['sample'].values)
    miss_idxs=getIDXs(miss_classified,df)
    miss_idxs_close = np.where(np.abs(resistances_after[miss_idxs,idx]-mean_I)<delta)[0].tolist()
    miss_idxs_close = [miss_idxs[i] for i in miss_idxs_close]
    miss_idxs_not_close = list(set(miss_idxs) - set(miss_idxs_close))

    all_idxs=list(range(len(isolates_all)))
    all_idxs=list(set(all_idxs) - set(I_idxs))
    corr_idxs=list(set(all_idxs) - set(miss_idxs))

    corr_idxs_close = np.where(np.abs(resistances_after[corr_idxs,idx]-mean_I)<delta)[0].tolist()
    corr_idxs_close = [corr_idxs[i] for i in corr_idxs_close]
    corr_idxs_not_close = list(set(corr_idxs) - set(corr_idxs_close))

    contig=[[len(corr_idxs_not_close),len(corr_idxs_close)],[len(miss_idxs_not_close),len(miss_idxs_close)]]
    contig=np.array(contig)
    c, p, dof, expected = chi2_contingency(contig)
    return p, contig, mean_I, scaler, mean_I_before


def find_the_best_thr(drug,mode):
    p=1
    t=0
    final_table=[]
    for i in np.arange(0.001,0.999,0.001):
        try:
            p_c,tab, mean_I, scaler, mean_I_before=getIvalue(drug,mode,delta=i)
            if p_c<p:
                p=p_c
                t=scaler.inverse_transform([[i]])
                final_table=tab
        except:
            a=1

    return (p,t,final_table,mean_I_before)



for mode in ['expr','gpa','snps', 'gpa_expr' ,'expr_snps' ,'gpa_snps','gpa_expr_snps' ]:
    results=dict()
    for drug in ['CAZ','CIP','MEM','TOB']:
        p,t,table,mean_I_before=find_the_best_thr(drug,mode)
        print (mode,'\t',drug,'\t',"{:.2E}".format(Decimal(p)),'\t',np.round(mean_I_before,4),'\t',np.round(t[0][0],3) )

