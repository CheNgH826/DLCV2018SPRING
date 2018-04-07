import numpy as np
from scipy.misc import imread
import cv2
import matplotlib.pyplot as plt

centroids = np.load('centroids.npy')
# print(centroids.shape)

img_list = ['Coast/image_0032.jpg',
            'Forest/image_0003.jpg',
            'Highway/image_0010.jpg',
            'Mountain/image_0044.jpg',
            'Suburb/image_0029.jpg']
# img_list = ['Coast/image_0032.jpg']

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
    # print(distance_table)
    for i, dis in enumerate(np.transpose(normalized_dis_table)):
        normalized_dis_table[:,i] = np.reciprocal(dis)/np.sum(np.reciprocal(dis))
    # print(distance_table)
    # print(normalized_dis_table)
    for i, centroid_i in enumerate(normalized_dis_table):
        hist_data[i] = np.max(centroid_i)
    return hist_data

for img_path in img_list:
    img = imread(img_path)
    surf = cv2.xfeatures2d.SURF_create(1000)
    kp, des = surf.detectAndCompute(img, None)
    distance_table = np.zeros((centroids.shape[0], des.shape[0]))
    for i, centroid_i in enumerate(centroids):
        for j, des_j in enumerate(des):
            distance_table[i, j] = np.linalg.norm(centroid_i-des_j)
    # print(distance_table)
    # print(distance_table.shape)
    
    plt.bar(range(distance_table.shape[0]), hard_sum(distance_table))
    plt.savefig('hardsum_hist_'+img_path.replace('/', '_')+'.jpg')
    plt.clf()
    plt.bar(range(distance_table.shape[0]), soft_sum(distance_table))
    plt.savefig('softsum_hist_'+img_path.replace('/', '_')+'.jpg')
    plt.clf()
    plt.bar(range(distance_table.shape[0]), soft_max(distance_table))
    plt.savefig('softmax_hist_'+img_path.replace('/', '_')+'.jpg')
    plt.clf()