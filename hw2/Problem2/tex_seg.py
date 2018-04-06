import numpy as np
import scipy.misc
import scipy.io
import scipy.signal
from sklearn.cluster import KMeans
from RGB2Lab import *

# color_bank = [[i, j, k] for i in [0, 128, 255] for j in [0, 128, 255] for k in [0, 128, 255]]

# def tex_filter(img, myfilter):
#     offset = myfilter.shape[0]/2
#     for i, j in np.ndindex((img.shape[0]-offset*2, img.shape[1]-offset*2)):

def tex_seg(jpg_file):
    img = scipy.misc.imread(jpg_file)
    gray_img = rgb2gray(img)
    # scipy.misc.imsave('gray_'+jpg_file, gray_img)
    filter_bank = scipy.io.loadmat('filterBank.mat')['F']
    feature_result = np.zeros(gray_img.shape + (filter_bank.shape[-1],))
    # padded_img = symmetric_padding(gray_img, filter_bank.shape[0]/2)
    # padded_img = np.pad(img, filter_bank.shape[0]/2)
    for i in range(filter_bank.shape[-1]):
        feature_result[:,:,i] = scipy.signal.convolve2d(gray_img, filter_bank[:,:,i], mode='same', boundary='symm')
        # feature_result[:,:,i] = tex_filter(padded_img, filter_bank[i])

    img_data = feature_result.reshape(-1, feature_result.shape[-1])
    np.save('tex_feature_{}.npy'.format(jpg_file[:-4]), img_data)
    kmeans = KMeans(n_clusters=6, max_iter=1000).fit(img_data)
    print(kmeans.labels_)
    color_img = np.array([[0, 0, 0] for i in range(len(kmeans.labels_))])
    for i, label in enumerate(kmeans.labels_):
        color_img[i] = color_bank[label]
    color_img = color_img.reshape(img.shape)
    
    outfile = 'tex_seg_'+jpg_file
    scipy.misc.imsave(outfile, color_img)

def rgb2gray(img):
    gray_img = np.zeros((img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            gray_img[i,j] = 0.3*img[i,j,0] + 0.6*img[i,j,1] + 0.1*img[i,j,2]
    return gray_img

if __name__ == '__main__':
    tex_seg('zebra.jpg')
    tex_seg('mountain.jpg')