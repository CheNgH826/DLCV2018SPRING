from keras.models import load_model
import numpy as np
from util import *
from scipy.misc import imsave

model_path = './drive/colab/DLCV2018SPRING/hw3/my_uint8.h5'
val_path = '.drive/colab/DLCV2018SPRING/hw3/npy/val_sat_uint8.npy'
outimg_path = './drive/colab/DLCV2018SPRING/hw3/pred-mask/'

model = load_model(model_path)
val_data = np.load(val_path)

val_result = model.predict(val_data)
print(val_result.shape)

sample_num = val_result.shape[0]
for i in range(sample_num):
    out_img = []
    for pixel in val_result[i]:
        out_img.append(color_list[pixel.index(1)])
    out_img = np.array(out_img).reshape(512, 512, 3)
    imsave(out_img, outimg_path+'{:4d}_pred_mask.png'.format(i))