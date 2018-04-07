import cv2
import scipy.misc as misc
import os
import numpy as np
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


category_list = ['Coast', 'Forest', 'Highway', 'Mountain', 'Suburb']
img_name_list = []

for category in category_list:
    img_name_list += [category+'/'+img_name for img_name in os.listdir(category)]
print(img_name_list)

visual_word_dict = []
for img_name in img_name_list:
    img = misc.imread(img_name)
    surf = cv2.xfeatures2d.SURF_create(1000)
    kp, des = surf.detectAndCompute(img, None)
    print(len(kp), des.shape)
    for des_i in des:
        visual_word_dict.append(des_i)
visual_word_dict = np.array(visual_word_dict)
print(visual_word_dict.shape)

### KMeans and PCA

kmeans = KMeans(n_clusters=50, max_iter=5000).fit(visual_word_dict)
# np.save('centroids.npy', kmeans.cluster_centers_)
# exit(-1)
# cluster_idx = [47, 7, 27, 30, 31, 35]
cluster_idx = range(6)
pca_pool = []
sub_label_list = []
for i, label in enumerate(kmeans.labels_):
    if label in cluster_idx:
        pca_pool.append(visual_word_dict[i])
        sub_label_list.append(label)

for i in cluster_idx:
    pca_pool.append(kmeans.cluster_centers_[i])
sub_label_list += range(10,16)

pca = PCA(n_components=3).fit_transform(pca_pool)
print(pca.shape)
### Plot
fignum = 1
fig = plt.figure(fignum, figsize=(4, 3))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
sub_label_list = np.array(sub_label_list).astype(np.float)
# ax.scatter(pca[:-6,0], pca[:-6,1], pca[:-6,2], c=sub_label_list)
# ax.scatter(pca[-6:,0], pca[-6:,1], pca[-6:,2], c='r', edgecolor='k', marker='x')
ax.scatter(pca[:,0], pca[:,1], pca[:,2], c=sub_label_list)
plt.savefig('3b_vv5.jpg')
plt.close()