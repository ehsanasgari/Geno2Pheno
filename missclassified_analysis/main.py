

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

