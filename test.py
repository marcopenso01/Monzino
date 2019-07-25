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
import read_data
import image_utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def score_data(input_folder, output_folder, model_path, config, do_postprocessing=False, gt_exists=True):

    nx, ny = config.image_size[:2]
    batch_size = 1
    num_channels = config.nlabels

    image_tensor_shape = [batch_size] + list(config.image_size) + [1]
    images_pl = tf.placeholder(tf.float32, shape=image_tensor_shape, name='images')

    # According to the experiment config, pick a model and predict the output
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

        for folder in os.listdir(input_folder):

            folder_path = os.path.join(input_folder, folder)

            if os.path.isdir(folder_path):

                if evaluate_test_set or evaluate_all:
                    train_test = 'test'  # always test
                else:
                    train_test = 'test' if (int(folder[-3:]) % 5 == 0) else 'train'


                if train_test == 'test':

                    infos = {}
                    for line in open(os.path.join(folder_path, 'Info.cfg')):
                        label, value = line.split(':')
                        infos[label] = value.rstrip('\n').lstrip(' ')

                    patient_id = folder.lstrip('patient')
                    ED_frame = int(infos['ED'])
                    ES_frame = int(infos['ES'])

                    for file in glob.glob(os.path.join(folder_path, 'patient???_frame??.nii.gz')):

                        logging.info(' ----- Doing image: -------------------------')
                        logging.info('Doing: %s' % file)
                        logging.info(' --------------------------------------------')

                        file_base = file.split('.nii.gz')[0]

                        frame = int(file_base.split('frame')[-1])
                        img_dat = utils.load_nii(file)
                        img = img_dat[0].copy()
                        #img = cv2.normalize(img, dst=None, alpha=config.min, beta=config.max, norm_type=cv2.NORM_MINMAX)
                        #img = image_utils.normalize_image(img)
                        print('img')
                        print(img.shape)
                        print(img.dtype)

                        if gt_exists:
                            file_mask = file_base + '_gt.nii.gz'
                            mask_dat = utils.load_nii(file_mask)
                            mask = mask_dat[0]

                        start_time = time.time()

                        if config.data_mode == '2D':

                            pixel_size = (img_dat[2].structarr['pixdim'][1], img_dat[2].structarr['pixdim'][2])
                            scale_vector = (pixel_size[0] / config.target_resolution[0],
                                            pixel_size[1] / config.target_resolution[1])
                            print('pixel_size', pixel_size)
                            print('scale_vector', scale_vector)

                            predictions = []
                            mask_arr = []
                            img_arr = []

                            for zz in range(img.shape[2]):

                                slice_img = np.squeeze(img[:,:,zz])
                                slice_rescaled = transform.rescale(slice_img,
                                                                   scale_vector,
                                                                   order=1,
                                                                   preserve_range=True,
                                                                   multichannel=False,
                                                                   anti_aliasing=True,
                                                                   mode='constant')
                                print('slice_img', slice_img.shape)
                                print('slice_rescaled', slice_rescaled.shape)
                                
                                slice_mask = np.squeeze(mask[:, :, zz])
                                mask_rescaled = transform.rescale(slice_mask,
                                                                  scale_vector,
                                                                  order=0,
                                                                  preserve_range=True,
                                                                  multichannel=False,
                                                                  anti_aliasing=True,
                                                                  mode='constant')

                                slice_cropped = acdc_data.crop_or_pad_slice_to_size(slice_rescaled, nx, ny)
                                print('slice_cropped', slice_cropped.shape)
                                mask_cropped = acdc_data.crop_or_pad_slice_to_size(mask_rescaled, nx, ny)
                                
                                slice_cropped = np.float32(slice_cropped)
                                mask_cropped = np.asarray(mask_cropped, dtype=np.uint8)
                                
                                x = image_utils.reshape_2Dimage_to_tensor(slice_cropped)
                                y = image_utils.reshape_2Dimage_to_tensor(mask_cropped)
                                
                                # GET PREDICTION
                                feed_dict = {
                                images_pl: x,
                                }
                                
                                mask_out, logits_out = sess.run([mask_pl, softmax_pl], feed_dict=feed_dict)

                            
                                prediction_cropped = np.squeeze(logits_out[0,...])

                                # ASSEMBLE BACK THE SLICES
                                slice_predictions = np.zeros((nx,ny,num_channels))
                                slice_predictions = prediction_cropped
                                # RESCALING ON THE LOGITS
                                if gt_exists:
                                    prediction = transform.resize(slice_predictions,
                                                                  (nx, ny, num_channels),
                                                                  order=1,
                                                                  preserve_range=True,
                                                                  anti_aliasing=True,
                                                                  mode='constant')
                                

                                # prediction = transform.resize(slice_predictions,
                                #                               (mask.shape[0], mask.shape[1], num_channels),
                                #                               order=1,
                                #                               preserve_range=True,
                                #                               mode='constant')

                                prediction = np.uint8(np.argmax(prediction, axis=-1))
                                predictions.append(prediction)
                                mask_arr.append(np.squeeze(y))
                                img_arr.append(np.squeeze(x))
                                
                            prediction_arr = np.transpose(np.asarray(predictions, dtype=np.uint8), (1,2,0))
                            mask_arrs = np.transpose(np.asarray(mask_arr, dtype=np.uint8), (1,2,0))
                            img_arrs = np.transpose(np.asarray(img_arr, dtype=np.float32), (1,2,0))                   

                            
                        # This is the same for 2D and 3D again
                        if do_postprocessing:
                            prediction_arr = image_utils.keep_largest_connected_components(prediction_arr)

                        elapsed_time = time.time() - start_time
                        total_time += elapsed_time
                        total_volumes += 1

                        logging.info('Evaluation of volume took %f secs.' % elapsed_time)

                        if frame == ED_frame:
                            frame_suffix = '_ED'
                        elif frame == ES_frame:
                            frame_suffix = '_ES'
                        else:
                            raise ValueError('Frame doesnt correspond to ED or ES. frame = %d, ED = %d, ES = %d' %
                                             (frame, ED_frame, ES_frame))

                        # Save prediced mask
                        out_file_name = os.path.join(output_folder, 'prediction',
                                                     'patient' + patient_id + frame_suffix + '.nii.gz')
                        if gt_exists:
                            out_affine = mask_dat[1]
                            out_header = mask_dat[2]
                        else:
                            out_affine = img_dat[1]
                            out_header = img_dat[2]

                        logging.info('saving to: %s' % out_file_name)
                        utils.save_nii(out_file_name, prediction_arr, out_affine, out_header)

                        # Save image data to the same folder for convenience
                        image_file_name = os.path.join(output_folder, 'image',
                                                'patient' + patient_id + frame_suffix + '.nii.gz')
                        logging.info('saving to: %s' % image_file_name)
                        utils.save_nii(image_file_name, img_dat[0], out_affine, out_header)

                        if gt_exists:

                            # Save GT image
                            gt_file_name = os.path.join(output_folder, 'ground_truth', 'patient' + patient_id + frame_suffix + '.nii.gz')
                            logging.info('saving to: %s' % gt_file_name)
                            utils.save_nii(gt_file_name, mask_arrs, out_affine, out_header)

                            # Save difference mask between predictions and ground truth
                            difference_mask = np.where(np.abs(prediction_arr-mask_arrs) > 0, [1], [0])
                            difference_mask = np.asarray(difference_mask, dtype=np.uint8)
                            
                   #         for zz in range(difference_mask.shape[2]):
                   #       
                   #             fig = plt.figure()
                   #             ax1 = fig.add_subplot(221)
                   #             ax1.set_axis_off()
                   #             ax1.imshow(img_arrs[:,:,zz])
                   #             ax2 = fig.add_subplot(222)
                   #             ax2.set_axis_off()
                   #             ax2.imshow(mask_arrs[:,:,zz])
                   #             ax3 = fig.add_subplot(223)
                   #             ax3.set_axis_off()
                   #             ax3.imshow(prediction_arr[:,:,zz])
                   #             ax1.title.set_text('a')
                   #             ax2.title.set_text('b')
                   #             ax3.title.set_text('c')
                   #             ax4 = fig.add_subplot(224)
                   #             ax4.set_axis_off()
                   #             ax4.imshow(difference_mask[:,:,zz], cmap=plt.cm.gnuplot)
                   #             ax1.title.set_text('a')
                   #             ax2.title.set_text('b')
                   #             ax3.title.set_text('c')
                   #             ax4.title.set_text('d')
                   #             plt.gray()
                   #             plt.show()
                            
                        
                            for zz in range(difference_mask.shape[2]):
                                plt.imshow(img_arrs[:,:,zz])
                                plt.gray()
                                plt.axis('off')
                                plt.show()
                                plt.imshow(mask_arrs[:,:,zz])
                                plt.gray()
                                plt.axis('off')
                                plt.show()
                                plt.imshow(prediction_arr[:,:,zz])
                                plt.gray()
                                plt.axis('off')
                                plt.show()
                                print('...')

                            diff_file_name = os.path.join(output_folder,
                                                          'difference',
                                                          'patient' + patient_id + frame_suffix + '.nii.gz')
                            logging.info('saving to: %s' % diff_file_name)
                            utils.save_nii(diff_file_name, difference_mask, out_affine, out_header)


        logging.info('Average time per volume: %f' % (total_time/total_volumes))

    return init_iteration


if __name__ == '__main__':

    base_path = config.project_root
    logging.info(base_path)
    model_path = config.weights_root
    logging.info(model_path)

    logging.warning('EVALUATING ON TEST SET')
    input_path = config.test_data_root
    output_path = os.path.join(model_path, 'predictions')

    path_pred = os.path.join(output_path, 'prediction')
    utils.makefolder(path_pred)
    path_eval = os.path.join(output_path, 'eval')
        
    gt_exists = config.gt_exists      #True if it exists the ground_truth images, otherwise set False.
                                      #if True it will be defined evalutation (eval)
   

    init_iteration = score_data(input_path,
                                output_path,
                                model_path,
                                config=config,
                                do_postprocessing=True,
                                gt_exists)


    if gt_exists:
        path_gt = os.path.join(config.test_data_root , 'mask')
        metrics.main(path_gt, path_pred, path_eval)
