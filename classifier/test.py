import sys
sys.path.append('../')
from make_representations.representation_maker import Metagenomic16SRepresentation
import time
import sys
from utility.file_utility import FileUtility

#files=FileUtility.load_list('../../datasets/crohn/files_with_label.txt')
fasta_files, mapping = FileUtility.read_fasta_directory('/mounts/data/proj/asgari/dissertation/datasets/deepbio/microbiome/crohn/','fastq')#,only_files=[f.split('/')[-1] for f in files])



sampling_dict={6:[100,1000,5000,10000,-1]}
for k in range(6,7):
    for s in sampling_dict[k]:
        print(k)
        RS=Metagenomic16SRepresentation(fasta_files, mapping, s, 5)
        start = time.time()
        a=RS.generate_kmers_all(k, save=False)
        end = time.time()
        print(s,(end - start))