import numpy as np

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

