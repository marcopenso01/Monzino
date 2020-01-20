import os
import glob
import numpy as np
import logging
import cv2

import argparse
import metrics
import time
from importlib.machinery import SourceFileLoader
import tensorflow as tf
from skimage import transform
import matplotlib.pyplot as plt

import configuration as config
import model as model
import utils
import acdc_data
import image_utils

if __name__ == '__main__':
  log_root = config.log_root
  model_path = os.path.join(log_root, config.experiment_name)
  logging.info(model_path)

  logging.warning('EVALUATING ON TEST SET')
  input_path = config.test_data_root
  img_path = os.path.join(input_path, 'img')
  output_path = os.path.join(model_path, 'predictions')

  path_pred = os.path.join(output_path, 'prediction')
  utils.makefolder(path_pred)

  init_iteration = score_data(img_path,
                              output_path,
                              model_path,
                              config=config,
                              do_postprocessing=True,
                              gt_exists= config.test_gt)
  if config.test_gt:
    path_gt = os.path.join(input_path, 'mask')
    path_eval = os.path.join(output_path, 'eval')
    utils.makefolder(path_eval)
    metrics.main(path_gt, path_pred, path_eval)
