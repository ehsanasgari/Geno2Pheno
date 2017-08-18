__author__ = "Ehsaneddin Asgari"
__copyright__ = "Copyright 2017, HH-HZI Project"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Ehsaneddin Asgari"
__email__ = "asgari@berkeley.edu"

import codecs
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
import fnmatch
import os
from Bio import SeqIO
from featurizer import TextFeature
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import generic_dna
import _pickle as pickle

class FileUtility(object):
    def __init__(self):
        print ('File utility object created..')


    @staticmethod
    def _read_fasta_kmer_features(fasta_files, save_pref='_', analyzer='char', ngram=(4,8), idf=True, aggregate=True):
        corpus=[]
        isolates=[]
        for fasta_file in fasta_files:
            iso=fasta_file.split('/')[-1].split('_')[0]
            line=''
            isolates.append(iso)
            if aggregate:
                for cur_record in SeqIO.parse(fasta_file, "fasta") :
                    line+=str(cur_record.seq).upper().replace(' ','')
                corpus.append(line)
            else:
                for cur_record in SeqIO.parse(fasta_file, "fasta") :
                    temp=str(cur_record.seq).upper().replace(' ','')
                    if len(temp)>len(line):
                        line=temp
                corpus.append(line)

        TF=TextFeature(corpus, analyzer=analyzer, ngram=ngram, idf=idf, norm='l2')

        FileUtility.save_sparse_csr('_'.join([save_pref,'feature','vect.npz']),TF.tf_vec)
        FileUtility.save_list('_'.join([save_pref,'feature','list.txt']), TF.feature_names)
        FileUtility.save_list('_'.join([save_pref,'isolates','list.txt']),isolates)

    @staticmethod
    def _create_fasta_sequences(fasta_files, filename, mapping, min_length=100000):
        records=[]
        for fasta_file in fasta_files:
            iso=fasta_file.split('/')[-1].split('_')[0]
            if iso in mapping:
                selected_sequence=''
                isolate='###'.join([iso,mapping[iso]])

                for cur_record in SeqIO.parse(fasta_file, "fasta") :
                    seq_select=str(cur_record.seq).upper()
                    if len(seq_select)>min_length:
                        records.append(SeqRecord(Seq(seq_select, generic_dna), id=isolate, description=''))
        print (len(records), ' records added for training')
        SeqIO.write(records, filename, "fasta")
        

    @staticmethod
    def _read_signle_file_binary_feature(path, save_pref='_' ):
        '''
        :param path:
        :return:
        '''
        rows=[l.strip() for l in codecs.open(path,'r','utf-8').readlines()]
        features=[feat.replace(' ','') for feat in rows[0].rstrip().split('\t')]
        isolates=[row.split()[0] for row in rows[1::]]

        # corpus creation
        corpus=[]
        for entry in rows[1::]:
            corpus.append(' '.join([features[idx] for idx,x in enumerate(entry.split('\t')[1::]) if FileUtility._float_or_zero(x)>0]))

        tfm = TfidfVectorizer(use_idf=False, analyzer='word', norm=None, tokenizer=str.split, stop_words=[], lowercase=False)
        tf_vec = tfm.fit_transform(corpus)
        feature_names = tfm.get_feature_names()

        FileUtility.save_sparse_csr('_'.join([save_pref,'feature','vect.npz']),tf_vec)
        FileUtility.save_list('_'.join([save_pref,'feature','list.txt']),feature_names)
        FileUtility.save_list('_'.join([save_pref,'isolates','list.txt']),isolates)

    @staticmethod
    def _read_signle_file_continuous_feature(path, save_pref='_' ):
        '''
        :param path:
        :return:
        '''
        rows=[l.strip() for l in codecs.open(path,'r','utf-8').readlines()]
        tf_vec=sparse.csr_matrix([[FileUtility._float_or_zero(x)  for x in entry.split('\t')[1::]] for entry in rows[1::]])
        isolates=[row.split()[0] for row in rows[1::]]
        feature_names=[feat.replace(' ','') for feat in rows[0].rstrip().split('\t')]

        FileUtility.save_sparse_csr('_'.join([save_pref,'feature','vect.npz']),tf_vec)
        FileUtility.save_list('_'.join([save_pref,'feature','list.txt']),feature_names)
        FileUtility.save_list('_'.join([save_pref,'isolates','list.txt']),isolates)

    @staticmethod
    def save_obj(value, filename):
        with open( filename + '.pickle', 'wb') as f:
            pickle.dump(value, f)

    @staticmethod
    def load_obj(filename):
        return pickle.load(open(filename,"rb"))
                    
    @staticmethod
    def save_list(filename, list_names):
        f=codecs.open(filename, 'w','utf-8')
        for x in list_names:
            f.write(x+'\n')
        f.close()

    @staticmethod
    def load_list(filename):
        return [line.strip() for line in codecs.open(filename, 'r','utf-8').readlines()]

    @staticmethod
    def save_sparse_csr(filename,array):
        np.savez(filename,data = array.data ,indices=array.indices,
                 indptr =array.indptr, shape=array.shape )
    @staticmethod
    def load_sparse_csr(filename):
        loader = np.load(filename)
        return sparse.csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                             shape = loader['shape'])

    @staticmethod
    def _float_or_zero(value):
        try:
            return float(value)
        except:
            return 0.0

    @staticmethod
    def recursive_glob(treeroot, pattern):
        '''
        :param treeroot: the path to the directory
        :param pattern:  the pattern of files
        :return:
        '''
        results = []
        for base, dirs, files in os.walk(treeroot):
            good_files = fnmatch.filter(files, pattern)
            results.extend(os.path.join(base, f) for f in good_files)
        return results