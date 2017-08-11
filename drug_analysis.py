from sklearn.manifold import TSNE
import codecs
from numpy import *
import numpy as np

from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
from sklearn.cluster import KMeans
from nltk import FreqDist


data_dir='/mounts/data/proj/asgari/github_data/data/pseudomonas/data/'
label_files='MIC/v2/mic_bin_with_intermediate.txt'
gene_expression_file='gene_expression/rpg_log_transformed.txt'
snps_files='snp/v2/all_SNPs_final_bin.txt'

# data reading
label_mapping={str(l.strip().split('\t')[0]):[int(float(str(x)))  for x in l.strip().split('\t')[1::]] if len(l.strip().split('\t')[1::])==5 else [0,0,0,0,0] for l in codecs.open(data_dir+label_files,'r','utf-8').readlines()[1::]}

gene_expression=[l.strip() for l in codecs.open(data_dir+gene_expression_file,'r','utf-8').readlines()]
gene_expression_mapping={str(entry.split('\t')[0]):[float(str(x)) for x in entry.split('\t')[1::]] for entry in gene_expression[1::]}

snps=[l.strip().replace(' ','') for l in codecs.open(data_dir+snps_files,'r','utf-8').readlines()]
mapping={'0.0':-1,'0':-1,'1.0':1,'1':1,'':0}
snps_mapping={str(entry.split('\t')[0]):[mapping[str(x)] for x in entry.split('\t')[1::]] for entry in snps[1::]}

training_instances=list(set(label_mapping.keys()).intersection(gene_expression_mapping.keys()).intersection(snps_mapping.keys()))

X1=np.array([gene_expression_mapping[x] for x in training_instances])

X2=np.array([snps_mapping[x] for x in training_instances])

X=np.concatenate((X1,X2), axis=1)

D=np.array([label_mapping[x] for x in training_instances])

kmeans = KMeans(n_clusters=20, random_state=0).fit(D)
classes = np.zeros((D.shape[0], 1))
classes[:,0] = (kmeans.labels_ + 1)

model = TSNE(n_components=2)
np.set_printoptions(suppress=False)

tsne_res = model.fit_transform(X)
np.savetxt('.txt',np.hstack((self.tsne_res, self.classes[self.rows_to_be_vis])))

