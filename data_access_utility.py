__author__ = "Ehsaneddin Asgari"
__copyright__ = "Copyright 2017, HH-HZI Project"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Ehsaneddin Asgari"
__email__ = "asgari@berkeley.edu"

from file_utility import  FileUtility
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer
import codecs
import numpy as np
import itertools

class ABRAccessUtility(object):
    '''
    This class is written to read/load/save
    data for AMR prediction of Pseudomonas Aeruginosa
    '''
    def __init__(self, dir, prefix_list):
        print ('Data access created..')
        self.load_data(dir, prefix_list)
        self.load_labels()

    def getXYPredictionMat_multidrug(self):
        # get common isolates
        list_of_list_of_isolates=list(self.isolates.values())
        mapping=self.get_multilabel_labels()
        list_of_list_of_isolates.append(list(mapping.keys()))
        final_isolates=ABRAccessUtility.common_isolates(list_of_list_of_isolates)
        final_isolates.sort()

        # feature types
        features=list(self.X.keys())

        tf=TfidfTransformer(norm=None, use_idf=True, smooth_idf=True)

        feature_names=[]
        feature_matrices=[]
        for feature in features:
            if not feature=='geneexp':
                tf.fit(self.X[feature])
                temp=tf.transform(self.X[feature])
            else:
                temp=self.X[feature]
            idx=[self.isolates[feature].index(isol) for isol in final_isolates]
            temp=temp[idx,:]
            feature_matrices.append(temp.toarray())
            feature_names+=['##'.join([feature, x]) for x in self.feature_names[feature]]

        X=np.concatenate(tuple(feature_matrices), axis=1)
        X=sparse.csr_matrix(X)
        Y=[mapping[isol] for isol in final_isolates]
        return X, Y, feature_names
    
    def getXYPredictionMat(self, drug, mapping={'0':0,'0.0':0,'1':1,'1.0':1}):
        # get common isolates
        list_of_list_of_isolates=list(self.isolates.values())
        mapping=dict(self.get_labels(mapping)[drug])
        list_of_list_of_isolates.append(list(mapping.keys()))
        final_isolates=ABRAccessUtility.common_isolates(list_of_list_of_isolates)
        final_isolates.sort()

        # feature types
        features=list(self.X.keys())

        tf=TfidfTransformer(norm=None, use_idf=True, smooth_idf=True)

        feature_names=[]
        feature_matrices=[]
        for feature in features:
            if not feature=='geneexp':
                tf.fit(self.X[feature])
                temp=tf.transform(self.X[feature])
            else:
                temp=self.X[feature]
            idx=[self.isolates[feature].index(isol) for isol in final_isolates]
            temp=temp[idx,:]
            feature_matrices.append(temp.toarray())
            feature_names+=['##'.join([feature, x]) for x in self.feature_names[feature]]

        X=np.concatenate(tuple(feature_matrices), axis=1)
        X=sparse.csr_matrix(X)
        Y=[mapping[isol] for isol in final_isolates]
        return X, Y, feature_names

    def getXYPredictionMat_scaffolds(self, drug, mapping={'0':0,'0.0':0,'1':1,'1.0':1}):
        # get common isolates
        list_of_list_of_isolates=list(self.isolates.values())
        mapping=dict(self.get_labels(mapping)[drug])
        list_of_list_of_isolates.append(list(mapping.keys()))
        final_isolates=ABRAccessUtility.common_isolates(list_of_list_of_isolates)
        final_isolates.sort()

        # feature types
        features=list(self.X.keys())

        tf=TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True)

        feature_names=[]
        feature_matrices=[]
        for feature in features:
            temp=self.X[feature]
            idx=list(itertools.chain(*[[ids for ids, x in enumerate(self.isolates[feature]) if x==isol]  for isol in final_isolates]))
            temp=temp[idx,:]
            feature_matrices.append(temp.toarray())
            feature_names+=['##'.join([feature, x]) for x in self.feature_names[feature]]

        X=np.concatenate(tuple(feature_matrices), axis=1)
        X=sparse.csr_matrix(X)
        Y=[mapping[self.isolates[features[0]][ids]] for ids in idx]
        return X, Y, feature_names

    def get_labels(self,mapping={'0':0,'0.0':0,'1':1,'1.0':1}):
        '''
        Load labels
        :param mapping:
        :return:
        '''
        new_drug2labeledisolates_mapping=dict()
        for drug in self.drugs:
            new_drug2labeledisolates_mapping[drug]=[]

        # only consider non-empty values
        for isolate,resist_vec in self.isolate2label_vec_mapping.items():
            for idx, val in enumerate(resist_vec):
                if val in mapping:
                    new_drug2labeledisolates_mapping[self.drugs[idx]].append((isolate,mapping[val]))
        return new_drug2labeledisolates_mapping

    def get_multilabel_labels(self):
        '''
            drug resistance profile label
        '''
        mapping={'':'I','0':'V','0.0':'V','1':'R','1.0':'R'}
        return {k:''.join([mapping[x] for x in list(v)]) for k,v in self.isolate2label_vec_mapping.items()}



    def load_data(self, dir, prefix_list):
        '''
        Load list of features
        :param dir:
        :param prefix_list:
        :return:
        '''
        self.X=dict()
        self.feature_names=dict()
        self.isolates=dict()
        for save_pref in prefix_list:
            self.X[save_pref]=FileUtility.load_sparse_csr('_'.join([dir+save_pref,'feature','vect.npz']))
            self.feature_names[save_pref]=FileUtility.load_list('_'.join([dir+save_pref,'feature','list.txt']))
            self.isolates[save_pref]=FileUtility.load_list('_'.join([dir+save_pref,'isolates','list.txt']))

    def load_labels(self):
        '''
            This function load labels ZG02420619 ['1.0', '0.0', '0.0', '', '1.0']
        '''
        label_file_address='/mounts/data/proj/asgari/github_data/data/pseudomonas/data_v2/mic_bin_without_intermediate.txt'
        rows=[l.replace('\n','') for l in codecs.open(label_file_address,'r','utf-8').readlines()]
        self.isolate2label_vec_mapping={str(entry.split('\t')[0]):[str(x) for idx,x in enumerate(entry.split('\t')[1::])] for entry in rows[1::]}
        self.drugs=rows[0].rstrip().split('\t')[1::]

        # init
        self.drug2labeledisolates_mapping=dict()
        for drug in self.drugs:
            self.drug2labeledisolates_mapping[drug]=[]

        mapping={'0':0,'0.0':0,'1':1,'1.0':1}
        # only consider non-empty values
        for isolate,resist_vec in self.isolate2label_vec_mapping.items():
            for idx, val in enumerate(resist_vec):
                if val in mapping:
                    self.drug2labeledisolates_mapping[self.drugs[idx]].append((isolate,mapping[val]))
        # generate dict of labels for each class
        for drug in self.drugs:
            self.drug2labeledisolates_mapping[drug]=dict(self.drug2labeledisolates_mapping[drug])

    @staticmethod
    def common_isolates(list_of_list_of_isolates):
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
    
    @staticmethod
    def read_data_from_scratch():
        '''
        Used only once to create the data for later loading
        :return:
        '''
        data_dir='/mounts/data/proj/asgari/github_data/data/pseudomonas/data_v2/'
        genexp_data_addr='rpg_log_transformed_426.txt'
        snps_data_addr='all_SNPs_final_bin_uq.txt'
        gen_pres_abs_data_addr='annot.txt'
        FileUtility._read_signle_file_continuous_feature(data_dir+genexp_data_addr,save_pref=data_dir+'geneexp')
        FileUtility._read_signle_file_binary_feature(data_dir+snps_data_addr,save_pref=data_dir+'SNPs')
        FileUtility._read_signle_file_binary_feature(data_dir+gen_pres_abs_data_addr,save_pref=data_dir+'genePA')

    def read_assembled_sequences(self):
        '''
        Used only once to create the data for later loading
        :return:
        '''
        data_dir='/mounts/data/proj/asgari/github_data/data/pseudomonas/data_v2/sequences_assembled/'
        save_dir='/mounts/data/proj/asgari/github_data/data/pseudomonas/data_v2/extracted_features/'
        fasta_files=FileUtility.recursive_glob(data_dir,'*.fasta')
        FileUtility._read_fasta_kmer_features(fasta_files,save_pref=save_dir+'k4-9mers',analyzer='char', ngram=(4,9), idf=True, aggregate=True)
        
    def create_labeled_scaffolds(self,min_length=10000):
        '''
        To create sequence scaffold for prediction
        '''
        joint_labels=self.get_multilabel_labels()
        data_dir='/mounts/data/proj/asgari/github_data/data/pseudomonas/data_v2/sequences_assembled/'
        save_dir='/mounts/data/proj/asgari/github_data/data/pseudomonas/data_v2/extracted_features/'
        fasta_files=FileUtility.recursive_glob(data_dir,'*.fasta')
        FileUtility._create_fasta_sequences(fasta_files, save_dir+'labeled_sequences_min_'+str(min_length)+'.fasta', joint_labels, min_length=min_length)

