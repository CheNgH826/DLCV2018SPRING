import numpy as np
import scipy.misc
from sklearn.cluster import KMeans
from RGB2Lab import *

# color_bank = [[i, j, k] for i in [0, 128, 255] for j in [0, 128, 255] for k in [0, 128, 255]]

jpg_file = 'mountain.jpg'
img = scipy.misc.imread(jpg_file)
img_name = jpg_file[:-4]
color_feature = np.load('color_feature_{}.npy'.format(img_name))
tex_feature = np.load('tex_feature_{}.npy'.format(img_name))
img_data = np.concatenate((color_feature, tex_feature), axis=1)#.reshape(-1, color_feature.shape[-1]+tex_feature.shape[-1])

kmeans = KMeans(n_clusters=6, max_iter=1000).fit(img_data)
print(kmeans.labels_)
color_img = np.array([[0, 0, 0] for i in range(len(kmeans.labels_))])
for i, label in enumerate(kmeans.labels_):
    color_img[i] = color_bank[label]
color_img = color_img.reshape(img.shape)

outfile = 'combine_seg_'+jpg_file
scipy.misc.imsave(outfile, color_img)