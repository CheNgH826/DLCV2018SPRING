from keras.models import load_model
import numpy as np
from util import *
from scipy.misc import imsave
import sys
from mean_iou_evaluate import *


model_path = sys.argv[1]
# model_path = '/data/model/'+model_name
# model_path = 'model/vggfixed_model_final.h5'
val_path = 'npy/val_sat_uint8.npy'
val_ans = 'npy/val_masks_uint8.npy'
outimg_path = '/mnt/dilation_pred-mask/'

model = load_model(model_path)
val_data = np.load(val_path)

val_result = model.predict(val_data)
print(val_result.shape)

sample_num = val_result.shape[0]
out_img_list = []
for sample in val_result:
    out_img = []
    for row in sample:
        for pixel in row:
            out_img.append(color_list[np.argmax(pixel)])
    out_img = np.array(out_img, dtype=np.uint8).reshape(64,64,3)
    print(out_img.shape)
    label_margin = 224
    out_img = np.pad(out_img, ((label_margin, label_margin),(label_margin,label_margin),(0,0)) , 'reflect')
    print(out_img.shape)
    out_img_list.append(out_img)
for i, out_img in enumerate(out_img_list):
    imsave(outimg_path+'{:04d}_pred_mask.png'.format(i), out_img)
