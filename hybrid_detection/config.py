
"""
Set EXP_ROOT_DIR here
This is where all training data, models and predictions
from './full_ai_exp/' and './stem_ai_exp/' experiments will be exported.
"""

#EXP_ROOT_DIR = None  # e.g. /path/to/ismir26_hybrid/experiments

EXP_ROOT_DIR = '/Users/frigaud/workspace/data/ismir26'  # TODO RM
#EXP_ROOT_DIR = '/data/nfs/analysis/frigaud/ismir26'

assert EXP_ROOT_DIR is not None, "Please set EXP_ROOT_DIR environment variable"
