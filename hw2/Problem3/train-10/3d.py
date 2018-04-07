import numpy as np
from scipy.misc import imread
import cv2
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import os
import pickle

centroids = np.load('centroids.npy')
img_dir_list = ['Coast', 'Forest', 'Highway', 'Mountain', 'Suburb']
img_path_list = []
for img_dir in img_dir_list:
    for img_path in os.listdir(img_dir):
        img_path_list.append(img_dir+'/'+img_path)

def hard_sum(distance_table):
    hist_data = np.zeros((distance_table.shape[0]))
    for i, dis in enumerate(np.transpose(distance_table)):
        hist_data[np.argmin(dis)] += 1
    return hist_data
def soft_sum(distance_table):
    hist_data = np.zeros((distance_table.shape[0]))
    for dis in np.transpose(distance_table):
        normalized_dis = np.reciprocal(dis)/np.sum(np.reciprocal(dis))
        hist_data += normalized_dis
    return hist_data
def soft_max(distance_table):
    hist_data = np.zeros((distance_table.shape[0]))
    normalized_dis_table = np.copy(distance_table)
    for i, dis in enumerate(np.transpose(normalized_dis_table)):
        normalized_dis_table[:,i] = np.reciprocal(dis)/np.sum(np.reciprocal(dis))
    for i, centroid_i in enumerate(normalized_dis_table):
        hist_data[i] = np.max(centroid_i)
    return hist_data


train_data_hardsum = []
train_data_softsum = []
train_data_softmax = []
train_size = 10
label = [0]*train_size + [1]*train_size + [2]*train_size + [3]*train_size + [4]*train_size
for img_path in img_path_list:
    print(img_path)
    img = imread(img_path)
    surf = cv2.xfeatures2d.SURF_create(1000)
    kp, des = surf.detectAndCompute(img, None)
    distance_table = np.zeros((centroids.shape[0], des.shape[0]))
    for i, centroid_i in enumerate(centroids):
        for j, des_j in enumerate(des):
            distance_table[i, j] = np.linalg.norm(centroid_i-des_j)
    train_data_hardsum.append(hard_sum(distance_table))
    train_data_softsum.append(soft_sum(distance_table))
    train_data_softmax.append(soft_max(distance_table))

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(train_data_hardsum, label)
with open('../hardsum_neigh_10.pickle', 'wb') as f:
    pickle.dump(neigh, f)

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(train_data_softsum, label)
with open('../softsum_neigh_10.pickle', 'wb') as f:
    pickle.dump(neigh, f)

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(train_data_softmax, label)
with open('../softmax_neigh_10.pickle', 'wb') as f:
    pickle.dump(neigh, f)