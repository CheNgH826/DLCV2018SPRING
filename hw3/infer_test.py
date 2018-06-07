from keras.models import load_model
import numpy as np
from util import *
from scipy.misc import imsave
import sys
from mean_iou_evaluate import *
from preprocess import *


model_path = sys.argv[1]
data_path = sys.argv[2]
outimg_path = sys.argv[3]

model = load_model(model_path)
val_data = read_img(data_path)
# val_data = np.load('npy/val_sat_uint8.npy')
sample_size = val_data.shape[0]
batch_size = 128
val_result = []
for idx in range(0, sample_size, batch_size):
    val_result_part = model.predict(val_data[idx: idx+batch_size])
    for x in val_result_part:
        val_result.append(x)

val_result = np.array(val_result)
sample_num = val_result.shape[0]
out_img_list = []
for sample in val_result:
    out_img = []
    for row in sample:
        for pixel in row:
            out_img.append(color_list[np.argmax(pixel)])
    out_img = np.array(out_img, dtype=np.uint8).reshape(512, 512, 3)
    out_img_list.append(out_img)
for i, out_img in enumerate(out_img_list):
    imsave(outimg_path+'{:04d}_mask.png'.format(i), out_img)
