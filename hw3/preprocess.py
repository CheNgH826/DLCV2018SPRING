import numpy as np
import scipy.misc
import argparse
import os
import sys
# import pandas as pd

num_labels = 7
# ABS_PATH = './drive/colab/DLCV2018SPRING/hw3/'
ABS_PATH = ''

def read_img(filepath):
    print(filepath)
    file_list = [file for file in os.listdir(filepath) if file.endswith('.jpg')]
    file_list.sort()
    n_imgs = len(file_list)
    imgs = np.zeros((n_imgs, 512, 512, 3), dtype=np.uint8)
    for i, file in enumerate(file_list):
        print(file)
        img = scipy.misc.imread(os.path.join(filepath, file))
        imgs[i] = img
    print(imgs.shape)
    return imgs

def read_masks(filepath):
    '''
    Read masks from directory and tranform to categorical
    '''
    print(filepath)
    file_list = [file for file in os.listdir(filepath) if file.endswith('.png')]
    file_list.sort()
    n_masks = len(file_list)
    masks = np.zeros((n_masks, 512, 512, num_labels), dtype=np.uint8)

    for i, file in enumerate(file_list):
        print(file)
        mask = scipy.misc.imread(os.path.join(filepath, file))
        mask = (mask >= 128).astype(int)
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        # masks[i, mask == 3][0] = 1  # (Cyan: 011) Urban land 
        # masks[i, mask == 6][1] = 1  # (Yellow: 110) Agriculture land 
        # masks[i, mask == 5][2] = 1  # (Purple: 101) Rangeland 
        # masks[i, mask == 2][3] = 1  # (Green: 010) Forest land 
        # masks[i, mask == 1][4] = 1  # (Blue: 001) Water 
        # masks[i, mask == 7][5] = 1  # (White: 111) Barren land 
        # masks[i, mask == 0][6] = 1  # (Black: 000) Unknown 
        masks[i, mask == 3] = [1,0,0,0,0,0,0]  # (Cyan: 011) Urban land 
        masks[i, mask == 6] = [0,1,0,0,0,0,0]  # (Yellow: 110) Agriculture land 
        masks[i, mask == 5] = [0,0,1,0,0,0,0]  # (Purple: 101) Rangeland 
        masks[i, mask == 2] = [0,0,0,1,0,0,0]  # (Green: 010) Forest land 
        masks[i, mask == 1] = [0,0,0,0,1,0,0]  # (Blue: 001) Water 
        masks[i, mask == 7] = [0,0,0,0,0,1,0]  # (White: 111) Barren land 
        masks[i, mask == 0] = [0,0,0,0,0,0,1]  # (Black: 000) Unknown 
    print(masks.shape)
    return masks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--labels', help='ground truth masks directory', type=str)
    parser.add_argument('-d', '--train', help='train data dir', type=str)
    # parser.add_argument()
    # parser.add_argument('-p', '--pred', help='prediction masks directory', type=str)
    args = parser.parse_args()

    labels = read_masks(args.labels)
    np.save('npy/train_masks_uint8_64.npy', labels[:,::8,::8,:])
    # train_data = read_img(args.train)
    # np.save('/mnt/label/_sat_uint8_64.npy', train_data[:,::8,::8,:])
    
