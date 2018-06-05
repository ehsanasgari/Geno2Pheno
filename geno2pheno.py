#! /usr/bin/python

# -*- coding: utf-8 -*-
__author__ = "Ehsaneddin Asgari"
__license__ = "Apache 2"
__version__ = "1.0.0"
__maintainer__ = "Ehsaneddin Asgari"
__email__ = "asgari@berkeley.edu"
__project__ = "GENO2PHENO of SEQ2GENO2PHENO"
__website__ = ""

import argparse
import os
import os.path
import sys

from data_access.intermediate_representation_utility import IntermediateRepCreate
from utility.file_utility import FileUtility


class Geno2Pheno:
    def __init__(self,outdir, labels):
        '''
            Geno2Pheno commandline use
        '''
        print('Geno2Pheno of Seq2Geno2Pheno 1.0.0')
        self.outdir=outdir
        self.label_file=labels
        self.labeling=dict([tuple(a.split()) for a in FileUtility.load_list(self.label_file)])

    @staticmethod
    def representation_creation(input_dir, output_dir, filetype):
        IC = IntermediateRepCreate(output_dir+'/intermediate_representations/')
        files = FileUtility.recursive_glob(input_dir, '*' + filetype)
        for file in files:
            IC.create_table(file, file.split('/')[-1].replace('.' + filetype, ''), 'binary')

    @staticmethod
    def classification(output_dir,classifier='SVM'):




def checkArgs(args):
    '''
        This function checks the input arguments and returns the errors (if exist) otherwise reads the parameters
    '''
    # keep all errors
    err = "";
    # Using the argument parser in case of -h or wrong usage the correct argument usage
    # will be prompted
    parser = argparse.ArgumentParser()

    # primary #################################################################################################
    parser.add_argument('--outdir', action='store', dest='output_dir', default=False, type=str,
                        help='output directory')

    parser.add_argument('--labels', action='store', dest='labels', default=False, type=str,
                        help='label file')


    # top level ######################################################################################################
    parser.add_argument('--rep', action='store_true', help='Create representation from genotype tables')
    ## normalizations need to be added

    # representation input #################################################################################################
    parser.add_argument('--indir', action='store', dest='input_dir', default=False, type=str,
                        help='input directory', required='--rep' in sys.argv)

    # general to bootstrap  and rep ##################################################################################
    parser.add_argument('--filetype', action='store', dest='filetype', type=str, default='mat',
                        help='the suffix of input files', required='--indir' in sys.argv)


    parsedArgs = parser.parse_args()

    try:
        os.stat(parsedArgs.output_dir)
    except:

        os.mkdir(parsedArgs.output_dir)

    if (not os.access(parsedArgs.labels, os.F_OK)):
        err = err + "\nError: Permission denied or could not find the labels!"
        return err

    G2P=Geno2Pheno(parsedArgs.output_dir, parsedArgs.labels)





    if parsedArgs.rep:
        '''
            bootstrapping functionality
        '''
        print('Bootstrapping requested..\n')
        if (not os.access(parsedArgs.input_dir, os.F_OK)):
            err = err + "\nError: Permission denied or could not find the directory!"
            return err
        else:
            if len(FileUtility.recursive_glob(parsedArgs.input_dir, '*' + parsedArgs.filetype)) == 0:
                err = err + "\nThe filetype " + parsedArgs.filetype + " could not find the directory!"
                return err
            Geno2Pheno.representation_creation(parsedArgs.input_dir, parsedArgs.output_dir, parsedArgs.filetype)
        return False

    else:
        err = err + "\nError: You need to specify a valid command!"
        print('others')

    return False


if __name__ == '__main__':
    err = checkArgs(sys.argv)
    if err:
        print(err)
        exit()
