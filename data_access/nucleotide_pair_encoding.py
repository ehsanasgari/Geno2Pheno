__author__ = "Ehsaneddin Asgari"
__license__ = "Apache 2"
__version__ = "1.0.0"
__maintainer__ = "Ehsaneddin Asgari"
__email__ = "asgari@berkeley.edu"
__project__ = "LLP - DiTaxa"
__website__ = "https://llp.berkeley.edu/ditaxa/"


import sys

sys.path.append('../')
import random
from utility.file_utility import FileUtility
from Bio import SeqIO
from multiprocessing import Pool
import tqdm
from make_representations.npe_efficient import train_npe
import sentencepiece as spm
import timeit


class NPETrain:
    '''
        Training the NPE segmentation
    '''

    def __init__(self, file_directory, file_extenstion, onlyfiles=[]):
        '''
        :param file_directory: the samples directory
        :param file_extenstion: the file extension fastq or fasta
        :param onlyfiles: filter a list of files
        :param backend: which backend to use
        '''
        print('Segmentation training')
        self.file_directory = file_directory
        self.file_extenstion = file_extenstion
        self.fasta_files, self.filename_mapping = FileUtility.read_fasta_directory(self.file_directory,
                                                                                   self.file_extenstion,
                                                                                   only_files=onlyfiles)
        print(str(len(self.fasta_files)), 'fasta files found in', self.file_directory)

    def generate(self, vocab_size, output_dir, num_p=2, backend='Sentencepiece'):
        '''
        :param vocab_size: the size of final vocabulary
        :param sample_size: how many reads from each file
        :param output_dir: where to write the results
        :param num_p: number of cores
        :return:
        '''
        start = timeit.default_timer()
        fasta_files = [x for x in self.fasta_files]
        isolates=['CF592_Iso2', 'CF609_Iso3', 'CH2500', 'CH2502', 'CH2522', 'CH2543', 'CH2560', 'CH2582', 'CH2591', 'CH2598', 'CH2608', 'CH2623', 'CH2639', 'CH2657', 'CH2658', 'CH2660', 'CH2665', 'CH2672', 'CH2674', 'CH2675', 'CH2677', 'CH2678', 'CH2682', 'CH2685', 'CH2687', 'CH2698', 'CH2699', 'CH2705', 'CH2706', 'CH2707', 'CH2713', 'CH2724', 'CH2734', 'CH2735', 'CH2747', 'CH2748', 'CH2764', 'CH2824', 'CH2860', 'CH2875', 'CH3173', 'CH3177', 'CH3290', 'CH3325', 'CH3462', 'CH3466', 'CH3484', 'CH3570', 'CH3613', 'CH3648', 'CH3797', 'CH3882', 'CH4035', 'CH4083', 'CH4411', 'CH4433', 'CH4438', 'CH4443', 'CH4489', 'CH4528', 'CH4548', 'CH4560', 'CH4584', 'CH4634', 'CH4681', 'CH4703', 'CH4704', 'CH4745', 'CH4755', 'CH4757', 'CH4766', 'CH4780', 'CH4785', 'CH4840', 'CH4860', 'CH4862', 'CH4877', 'CH4878', 'CH4916', 'CH4990', 'CH4992', 'CH5022', 'CH5052', 'CH5066', 'CH5159', 'CH5174', 'CH5182', 'CH5193', 'CH5206', 'CH5267', 'CH5291', 'CH5334', 'CH5353', 'CH5356', 'CH5363', 'CH5387', 'CH5432', 'CH5462', 'CH5464', 'CH5478', 'CH5528', 'CH5531', 'CH5548', 'CH5550', 'CH5551', 'CH5591', 'CH5596', 'CH5597', 'CH5621', 'CH5638', 'CH5666', 'CH5688', 'CH5695', 'ESP002', 'ESP004', 'ESP006', 'ESP012', 'ESP013', 'ESP023', 'ESP025', 'ESP027', 'ESP039', 'ESP040', 'ESP043', 'ESP047', 'ESP050', 'ESP053', 'ESP055', 'ESP059', 'ESP060', 'ESP061', 'ESP063', 'ESP064', 'ESP066', 'ESP067', 'ESP068', 'ESP069', 'ESP070', 'ESP071', 'ESP072', 'ESP073', 'ESP074', 'ESP075', 'ESP076', 'ESP077', 'ESP078', 'ESP079', 'ESP081', 'ESP082', 'ESP083', 'ESP084', 'ESP088', 'ESP091', 'F1659', 'F1688', 'F1689', 'F1697', 'F1706', 'F1712', 'F1724', 'F1727', 'F1741', 'F1745', 'F1746', 'F1747', 'F1748', 'F1752', 'F1754', 'F1757', 'F1758', 'F1759', 'F1760', 'F1763', 'F1764', 'F1775', 'F1787', 'F1789', 'F1795', 'F1796', 'F1798', 'F1801', 'F1810', 'F1812', 'F1821', 'F1823', 'F1853', 'F1862', 'F1864', 'F1869', 'F1880', 'F1883', 'F1894', 'F1957', 'F1968', 'F1979', 'F1983', 'F1997', 'F2000', 'F2003', 'F2005', 'F2006', 'F2010', 'F2017', 'F2020', 'F2029', 'F2030', 'F2034', 'F2035', 'F2040', 'F2044', 'F2045', 'F2054', 'F2055', 'F2056', 'F2059', 'F2064', 'F2065', 'F2073', 'F2075', 'F2081', 'F2093', 'F2097', 'F2098', 'F2119', 'F2128', 'F2137', 'F2148', 'F2165', 'F2166', 'F2167', 'F2172', 'F2176', 'F2199', 'F2208', 'F2230', 'F2234', 'F2235', 'F2236', 'F2240', 'M70563004', 'M70564993', 'M70565254', 'M70565897', 'M70604538', 'M70608424', 'M70635118', 'M70638412', 'M70639645', 'M70640096', 'M70647319', 'M70649365', 'M70652746', 'M70663247', 'MHH0985', 'MHH14929', 'MHH15083', 'MHH15103', 'MHH15151', 'MHH15204', 'MHH1525', 'MHH15275', 'MHH15817', 'MHH16050', 'MHH16208', 'MHH16371', 'MHH16379', 'MHH16427', 'MHH16459', 'MHH16513', 'MHH16530', 'MHH16610', 'MHH16798', 'MHH16951', 'MHH17233', 'MHH17247', 'MHH17441', 'MHH17501', 'MHH17546', 'MHH17767', 'MHH17783', 'MS2', 'PSAE1438', 'PSAE1439', 'PSAE1641', 'PSAE1645', 'PSAE1742', 'PSAE1745', 'PSAE1872', 'PSAE1903', 'PSAE1912', 'PSAE1934', 'PSAE1975', 'PSAE2125', 'PSAE2126', 'PSAE2127', 'PSAE2139', 'PSAE2325', 'Vb20477', 'Vb3320', 'ZG02420619', 'ZG02488718', 'ZG02512057', 'ZG205864', 'ZG301975', 'ZG302370', 'ZG322541', 'ZG5003493', 'ZG5021922', 'ZG5048010', 'ZG5051896', 'ZG5089456', 'ZG8006959', 'ZG8038581181', 'ZG8510487', 'ZG8525123']
        isolates=[x+'.fa' for x in isolates]
        fasta_files=[x for x in fasta_files if x.split('/')[-1] in isolates]
        print (fasta_files)
        corpus = []
        pool = Pool(processes=num_p)
        for ky, v in tqdm.tqdm(pool.imap_unordered(self._get_corpus, fasta_files, chunksize=num_p),
                               total=len(fasta_files)):
            corpus = corpus + v
        pool.close()
        print('\t✔ Corpus size for training NPE is ', len(corpus))
        if backend == 'Sentencepiece':
            FileUtility.save_list('/net/sgi/metagenomics/projects/pseudo_genomics/results/geno2pheno_package/Pseudogenomics/temp/tmp_txt', corpus)
            spm.SentencePieceTrainer.Train(
                '--input=/net/sgi/metagenomics/projects/pseudo_genomics/results/geno2pheno_package/Pseudogenomics/temp/tmp_txt --model_prefix=' + output_dir + '_500K --add_dummy_prefix=0 --max_sentencepiece_length=512 --model_type=bpe --mining_sentence_size=5000000 --input_sentence_size=10000000 --vocab_size=500000')
            #FileUtility.save_list('/net/sgi/metagenomics/projects/pseudo_genomics/results/geno2pheno_package/Pseudogenomics/temp/tmp_txt', corpus[0:10])
        elif backend == 'normalbpe':
            train_npe(corpus, output_dir, vocab_size, output_dir + '_freq')
        print('\t✔ The segmentation training took ', timeit.default_timer() - start, ' ms.')

    def _get_corpus(self, file_name_sample,sample_size=-1, unqiue_reads=True):
        '''
        :param file_name_sample:
        :return:
        '''

        file_name = file_name_sample
        corpus = []
        if file_name[-1] == 'q':
            for cur_record in SeqIO.parse(file_name, "fastq"):
                corpus.append(str(cur_record.seq).lower())
        else:
            for cur_record in SeqIO.parse(file_name, "fasta"):
                corpus.append(str(cur_record.seq).lower())
        if unqiue_reads:
            corpus = [x.replace(' ', '') for x in corpus]
            corpus = list(set(corpus))
        if sample_size == -1:
            return file_name, corpus
        else:
            random.seed(0)
            return file_name, random.sample(corpus, min(sample_size, len(corpus)))


if __name__ == '__main__':
    GNPE = NPETrain('/net/sgi/metagenomics/projects/pseudo_genomics/forRuben/genomes/',
                                          'fa')
    GNPE.generate(100000,
                  '/net/sgi/metagenomics/projects/pseudo_genomics/results/geno2pheno_package/Pseudogenomics/temp/pseudonpe',
                  backend='Sentencepiece')
