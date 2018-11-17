import config as conf
import numpy as np
from tqdm import tqdm
import glob
from get_image_size import get_image_size
from scipy import ndimage
import copy
import cv2
'''
An utility to read all filepaths of images and their masks
'''

def dataset_filepath(root, get_masks=True):
    '''
    input: a full path to root of dataset
    output: a list of filepath of images and their masks
    '''
    root += '/*'
    dir_list  = []
    path_list = glob.glob(root)
    for subdir in tqdm(path_list, total=len(path_list)):
        image = {'image': str(glob.glob(subdir+'/images/*.png')[0])}
        width, height = get_image_size(image['image'])
        image['width'] = width
        image['height']= height

        if get_masks:
            image['masks'] = []
            for mask_path in glob.glob(subdir+'/masks/*.png'):
                image['masks'].append(mask_path) # get bounding box

        dir_list.append(image)
    return dir_list

def dir_reader(img_meta_wo_markers, height, width, return_original=False):
    l = len(img_meta_wo_markers)
    input_images = np.zeros((l,height, width, 3))
    image_filenames  = []
    image_original = []
    size = []
    for i, img_meta in tqdm(enumerate(img_meta_wo_markers), total=l):
        img = cv2.imread(img_meta['image'], cv2.IMREAD_COLOR)[...,:3] # only BGR channels
        img = img[...,::-1] # BGR -> RGB
        if return_original:
            image_original.append(img.astype(np.uint8))
        img_input = cv2.resize(img, (width, height))
        img_input = img_input / 255.
        input_images[i,...] = img_input
        image_filenames.append(img_meta['image'])
        size.append((img_meta['width'], img_meta['height']))
    return (input_images, image_filenames, size, image_original) if return_original else (input_images, image_filenames, size) # return resized images and their original shapes, and orignal images

if __name__ == '__main__':
    l = dataset_filepath(conf.DATA_PATH)
    print(l)

