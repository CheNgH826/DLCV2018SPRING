import numpy as np
from scipy.misc import imread
import cv2
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import os
import pickle
from util import *
from sklearn.metrics import accuracy_score

train_size = 100
centroids = np.load('centroids_{}_300_iter10000.npy'.format(train_size))
img_dir_list = ['Coast', 'Forest', 'Highway', 'Mountain', 'Suburb']
img_path_list = []
for img_dir in img_dir_list:
    for img_path in os.listdir('test-100/'+img_dir):
        img_path_list.append('test-100/'+img_dir+'/'+img_path)

test_data_hardsum = []
test_data_softsum = []
test_data_softmax = []
test_size = 100
label = [0]*test_size + [1]*test_size + [2]*test_size + [3]*test_size + [4]*test_size
for img_path in img_path_list:
    img = imread(img_path)
    surf = cv2.xfeatures2d.SURF_create(1000)
    kp, des = surf.detectAndCompute(img, None)
    distance_table = np.zeros((centroids.shape[0], des.shape[0]))
    for i, centroid_i in enumerate(centroids):
        for j, des_j in enumerate(des):
            distance_table[i, j] = np.linalg.norm(centroid_i-des_j)
    test_data_hardsum.append(hard_sum(distance_table))
    test_data_softsum.append(soft_sum(distance_table))
    test_data_softmax.append(soft_max(distance_table))

# neigh = KNeighborsClassifier(n_neighbors=5)
# neigh.fit(train_data_hardsum, label)
for test_type in ['hardsum', 'softsum', 'softmax']:
    with open('{}_neigh_{}_10000.pickle'.format(test_type, 300), 'rb') as f:
        test_model = pickle.load(f)
    if test_type == 'hardsum':
        prediction = test_model.predict(test_data_hardsum)
        print('{} accuray: {}'.format(test_type, accuracy_score(label, prediction)))
    elif test_type == 'softsum':
        prediction = test_model.predict(test_data_softsum)
        print('{} accuray: {}'.format(test_type, accuracy_score(label, prediction)))
    else:   # softmax
        prediction = test_model.predict(test_data_softmax)
        print('{} accuray: {}'.format(test_type, accuracy_score(label, prediction)))
