__author__ = "Ehsaneddin Asgari"
__copyright__ = "Copyright 2017, HH-HZI Project / LLP"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Ehsaneddin Asgari"
__email__ = "asgari@berkeley.edu"

from scipy import stats
from sklearn.preprocessing import normalize

def get_kl_rows(A):
    '''
    :param A: matrix A
    :return: Efficient implementation to calculate kl-divergence between rows in A
    '''
    norm_A=normalize(A+1e-100, norm='l1')
    return stats.entropy(norm_A.T[:,:,None], norm_A.T[:,None,:])
