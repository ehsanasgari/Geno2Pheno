__author__ = "Ehsaneddin Asgari"
__copyright__ = "Copyright 2017, HH-HZI Project"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Ehsaneddin Asgari"
__email__ = "asgari@berkeley.edu ehsaneddin.asgari@helmholtz-hzi.de"

from Bio import SeqIO
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
from file_utility import FileUtility

corpus=[]
isolates=[]
for cur_record in SeqIO.parse("/mounts/data/proj/asgari/github_data/data/pseudomonas/data_v2/extracted_features/labeled_sequences_min_10K.fasta", "fasta") :
    line=str(cur_record.seq).upper()
    isolates.append(str(cur_record.id).split('###')[0])
    corpus.append(line)

TF=TextFeature(corpus, analyzer='char', ngram=(6,6), idf=True, norm='l1')

save_pref='/mounts/data/proj/asgari/github_data/data/pseudomonas/data_v2/extracted_features/seq_6_gram'

FileUtility.save_sparse_csr('_'.join([save_pref,'feature','vect.npz']),TF.tf_vec)
FileUtility.save_list('_'.join([save_pref,'feature','list.txt']), TF.feature_names)
FileUtility.save_list('_'.join([save_pref,'isolates','list.txt']),isolates)