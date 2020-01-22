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

def score_data(input_folder, output_folder, model_path, config, do_postprocessing=False, gt_exists=True):
    nx, ny = config.image_size[:2]
    batch_size = 1
    num_channels = config.nlabels

    image_tensor_shape = [batch_size] + list(config.image_size) + [1]
    images_pl = tf.placeholder(tf.float32, shape=image_tensor_shape, name='images')

    mask_pl, softmax_pl = model.predict(images_pl, config)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)
        checkpoint_path = utils.get_latest_model_checkpoint_path(model_path, 'model_best_dice.ckpt')
        saver.restore(sess, checkpoint_path)
        init_iteration = int(checkpoint_path.split('/')[-1].split('-')[-1])
        total_time = 0
        total_volumes = 0

        for folder in sorted(os.listdir(input_folder)):
            folder_path = os.path.join(input_folder, folder) #ciclo su cartelle paz
            if os.path.isdir(folder_path):
                for phase in os.listdir(folder_path):
                    phase_path = os.path.join(folder_path, phase) #ciclo su phase ES ED
                    img_path = os.path.join(phase_path, 'img')
                    
                    if gt_exists:
                        mask_path = os.path.join(phase_path, 'mask')
                        mask_arr = []

                    start_time = time.time()
                    predictions = []
                    img_arr = []

                    for file in sorted(glob.glob(img_path)):  #elenco delle img
                        img_addr = os.path.join(img_path, file)
                        img = cv2.imread(img_addr, 0)
                        if config.standardize:
                            img = image_utils.standardize_image(img)
                        if config.normalize:
                            img = image_utils.normalize_image(img)
                        img = cv2.resize(img, (nx, ny), interpolation = cv2.INTER_AREA)
                        
                        img = np.float32(img)
                        x = image_utils.reshape_2Dimage_to_tensor(img)
                        
                        # GET PREDICTION
                        feed_dict = {
                        images_pl: x,
                        }  

                        mask_out, logits_out = sess.run([mask_pl, softmax_pl], feed_dict=feed_dict)
                        prediction_cropped = np.squeeze(logits_out[0,...])

                        slice_predictions = np.zeros((nx,ny,num_channels))
                        slice_predictions = prediction_cropped
                        prediction = cv2.resize(slice_predictions, (nx, ny, num_channels), interpolation = cv2.INTER_LINEAR)
                        prediction = np.uint8(np.argmax(prediction, axis=-1))
                        predictions.append(prediction)
                        img_arr.append(np.squeeze(x))

                        if gt_exists:
                            for filem in sorted(glob.glob(mask_path)):
                                mask_addr = os.path.join(mask_path, filem)
                                mask = cv2.imread(mask_addr, 0)
                                mask = cv2.resize(mask, (nx, ny), interpolation=cv2.INTER_NEAREST)
                                mask = np.asarray(mask, dtype=np.uint8)
                                y = image_utils.reshape_2Dimage_to_tensor(mask)
                                mask_arr.append(np.squeeze(y))

                    prediction_arr = np.transpose(np.asarray(predictions, dtype=np.uint8), (1,2,0))
                    img_arrs = np.transpose(np.asarray(img_arr, dtype=np.float32), (1,2,0))                   
                    if gt_exists:
                        mask_arrs = np.transpose(np.asarray(mask_arr, dtype=np.uint8), (1,2,0))
                    

                    if do_postprocessing:
                        prediction_arr = image_utils.keep_largest_connected_components(prediction_arr)

                    elapsed_time = time.time() - start_time
                    total_time += elapsed_time
                    total_volumes += 1 
                    logging.info('Evaluation of volume took %f secs.' % elapsed_time)

                    # Save prediced mask
                    out_file_name = os.path.join(output_folder, 'prediction', folder, phase)
                    utils.makefolder(out_file_name)
                    for zz in range(prediction_arr.shape[2]):
                        slice_img = np.squeeze(prediction_arr[:,:,zz])
                        cv2.imwrite(os.path.join(out_file_name, 'img' + zz + '.png'))

                    # Save images
                    image_file_name = os.path.join(output_folder, 'image', folder, phase)
                    utils.makefolder(image_file_name)
                    for zz in range(img_arr.shape[2]):
                        slice_img = np.squeeze(img_arr[:,:,zz])
                        cv2.imwrite(os.path.join(image_file_name, 'img' + zz + '.png'))
                    
                    # Save mask
                    if gt_exists:
                        mask_file_name = os.path.join(output_folder, 'mask', folder, phase)
                        utils.makefolder(mask_file_name)
                        for zz in range(mask_arr.shape[2]):
                            slice_img = np.squeeze(mask_arr[:,:,zz])
                            cv2.imwrite(os.path.join(mask_file_name, 'img', zz + '.png'))

        logging.info('Average time per volume: %f' % (total_time/total_volumes))
   
    return init_iteration
    
 
if __name__ == '__main__':
    log_root = config.log_root
    model_path = os.path.join(log_root, config.experiment_name)
    logging.info(model_path)

    logging.warning('EVALUATING ON TEST SET')
    input_path = config.test_data_root
    output_path = os.path.join(model_path, 'predictions')

    path_pred = os.path.join(output_path, 'prediction')
    utils.makefolder(path_pred)
    path_image = os.path.join(output_path, 'image')
    utils.makefolder(path_image)

    init_iteration = score_data(input_path,
                                output_path,
                                model_path,
                                config=config,
                                do_postprocessing=True,
                                gt_exists= config.gt_exists)
    if config.gt_exists:
        path_gt = os.path.join(input_path, 'mask')
        path_eval = os.path.join(output_path, 'eval')
        utils.makefolder(path_eval)
        metrics.main(path_gt, path_pred, path_eval)
