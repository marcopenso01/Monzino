import os
import glob
import numpy as np
import logging
import nibabel as nib
import gc
import h5py
from skimage import transform
import cv2
from PIL import Image

import utils
import image_utils
import configuration as config

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def crop_or_pad_slice_to_size(slice, nx, ny):

    x, y = slice.shape

    x_s = (x - nx) // 2
    y_s = (y - ny) // 2
    x_c = (nx - x) // 2
    y_c = (ny - y) // 2

    if x > nx and y > ny:
        slice_cropped = slice[x_s:x_s + nx, y_s:y_s + ny]
    else:
        slice_cropped = np.zeros((nx, ny))
        if x <= nx and y > ny:
            slice_cropped[x_c:x_c + x, :] = slice[:, y_s:y_s + ny]
        elif x > nx and y <= ny:
            slice_cropped[:, y_c:y_c + y] = slice[x_s:x_s + nx, :]
        else:
            slice_cropped[x_c:x_c + x, y_c:y_c + y] = slice[:, :]

    return slice_cropped


def prepare_data(input_folder, output_file, mode, size, target_resolution):

    '''
    Main function that prepares a dataset from the raw challenge data to an hdf5 dataset
    '''

    assert (mode in ['2D', '3D']), 'Unknown mode: %s' % mode
    if mode == '2D' and not len(size) == 2:
        raise AssertionError('Inadequate number of size parameters')
    if mode == '3D' and not len(size) == 3:
        raise AssertionError('Inadequate number of size parameters')
    if mode == '2D' and not len(target_resolution) == 2:
        raise AssertionError('Inadequate number of target resolution parameters')
    if mode == '3D' and not len(target_resolution) == 3:
        raise AssertionError('Inadequate number of target resolution parameters')

    hdf5_file = h5py.File(output_file, "w")

    nx, ny = size
    scale_vector = [config.pixel_size[0] / target_resolution[0], config.pixel_size[1] / target_resolution[1]]
    count = 1
    train_addrs = []
    val_addrs = []
    
    if config.split_test_train:
        split = config.split
    else:
        split = 99999
        
    for folder in os.listdir(input_folder):
        folder_path = os.path.join(input_folder, folder)
        if count % split == 0:
            #validation
            path = os.path.join(folder_path, '*.png')
            for file in glob.glob(path):
                val_addrs.append(file)
        else:
            #training
            path = os.path.join(folder_path, '*.png')
            for file in glob.glob(path):
                train_addrs.append(file)
        
        count = count + 1
    
    train_shape = (len(train_addrs), nx, ny)
    val_shape = (len(val_addrs), nx, ny)
    hdf5_file.create_dataset("images_train", train_shape, np.uint8)
    if config.split_test_train:
        hdf5_file.create_dataset("images_val", val_shape, np.uint8)
    
    for i in range(len(train_addrs)):
        addr = train_addrs[i]
        img = cv2.imread(addr,0)
        img = transform.rescale(img,
                                scale_vector,
                                order=1,
                                preserve_range=True,
                                multichannel=False,
                                mode = 'constant')
        #img = cv2.resize(img, (nx, ny), interpolation=cv2.INTER_CUBIC)
        img = crop_or_pad_slice_to_size(img, nx, ny)
        hdf5_file["images_train"][i, ...] = img[None]
    
    if config.split_test_train:
        for i in range(len(val_addrs)):
            addr = val_addrs[i]
            img = cv2.imread(addr,0)
            img = transform.rescale(img,
                                    scale_vector,
                                    order=1,
                                    preserve_range=True,
                                    multichannel=False,
                                    mode = 'constant')
            #img = cv2.resize(img, (nx, ny), interpolation=cv2.INTER_CUBIC)
            img = crop_or_pad_slice_to_size(img, nx, ny)
            hdf5_file["images_val"][i, ...] = img[None]   

            
    # After test train loop:
    hdf5_file.close()
    
  
def load_and_maybe_process_data(input_folder,
                                preprocessing_folder,
                                mode,
                                size,
                                target_resolution,
                                force_overwrite=True):

    
    size_str = '_'.join([str(i) for i in size])
    res_str = '_'.join([str(i) for i in target_resolution])

    data_file_name = 'data_%s_size_%s_res_%s.hdf5' % (mode, size_str, res_str)

    data_file_path = os.path.join(preprocessing_folder, data_file_name)

    utils.makefolder(preprocessing_folder)

    if not os.path.exists(data_file_path) or force_overwrite:

        logging.info('This configuration of mode, size and target resolution has not yet been preprocessed')
        logging.info('Preprocessing now!')
        prepare_data(input_folder, data_file_path, mode, size, target_resolution)

    else:

        logging.info('Already preprocessed this configuration. Loading now!')

    return h5py.File(data_file_path, 'r')


if __name__ == '__main__':

    input_folder = '/content/drive/My Drive/train'
    preprocessing_folder = '/content/drive/My Drive/preproc_data'

    d=load_and_maybe_process_data(input_folder, preprocessing_folder, '2D', config.size, config.target_resolution)
