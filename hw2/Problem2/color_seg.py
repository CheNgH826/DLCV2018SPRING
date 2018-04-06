import scipy.misc
import numpy as np
from sklearn.cluster import KMeans
from RGB2Lab import *


# color_bank = [[i, j, k] for i in [0, 128, 255] for j in [0, 128, 255] for k in [0, 128, 255]]
Lab = 1

def color_seg(jpg_file):
    img = scipy.misc.imread(jpg_file)

    img_data = img.reshape(-1,3)

    if Lab:
        Lab_data = np.array([[0, 0, 0] for i in range(img_data.shape[0])])
        for i, pixel in enumerate(img_data):
            Lab_data[i] = RGB2Lab(pixel)
        img_data = Lab_data
    
    np.save('color_feature_{}.npy'.format(jpg_file[:-4]), img_data)

    kmeans = KMeans(n_clusters=10, max_iter=1000).fit(np.array(img_data))
    print(kmeans.labels_)
    color_img = np.array([[0, 0, 0] for i in range(len(kmeans.labels_))])
    for i, label in enumerate(kmeans.labels_):
        color_img[i] = color_bank[label]
    color_img = color_img.reshape(img.shape)

    if Lab:
        outfile = 'lab_seg_' + jpg_file
    else:
        outfile = 'rgb_seg_' + jpg_file
    scipy.misc.imsave(outfile, color_img)

if __name__ == '__main__':
    color_seg('mountain.jpg')
    color_seg('zebra.jpg')
