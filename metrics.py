import os
from glob import glob
import re
import argparse
import pandas as pd
import numpy as np

import scipy.stats as stats
import utils
import binary_metric as bm
import matplotlib.pyplot as plt
import seaborn as sns

import logging


def compute_metrics_on_directories_raw(dir_gt, dir_pred):
    '''
    - Dice
    - Hausdorff distance
    - Average surface distance
    - Predicted volume
    - Volume error w.r.t. ground truth
    :param dir_gt: Directory of the ground truth segmentation maps.
    :param dir_pred: Directory of the predicted segmentation maps.
    :return: Pandas dataframe with all measures in a row for each prediction and each structure
    """
    
    


def main(path_pred, path_gt, eval_dir):
    logging.info(path_gt)
    logging.info(path_pred)
    logging.info(eval_dir)
    
    if os.path.isdir(path_gr) and os.path.isdir(path_pred):
        
        df = compute_metrics_on_directories_raw(path_gt, path_pred)
        
        print_stats(df, eval_dir)
        print_latex_tables(df, eval_dir)
        boxplot_metrics(df, eval_dir)

        logging.info('------------Average Dice Figures----------')
        logging.info('Dice 1: %f' % np.mean(df.loc[df['struc'] == 'LV']['dice']))
        logging.info('Dice 2: %f' % np.mean(df.loc[df['struc'] == 'RV']['dice']))
        logging.info('Dice 3: %f' % np.mean(df.loc[df['struc'] == 'Myo']['dice']))
        logging.info('Mean dice: %f' % np.mean(np.mean(df['dice'])))
        logging.info('------------------------------------------')
    
    else:
        raise ValueError(
            "The paths given needs to be two directories or two files.")
