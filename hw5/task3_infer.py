from reader import readShortVideo
import pandas as pd
import numpy as np
import torch
from model import Task2Model_GRU, Task2Model_LSTM, resnet50, transform
import os
from skimage import io
import sys

# frame_path = '/mnt/DLCV_hw5/HW5_data/FullLengthVideos/videos/valid/'
# label_path = '/mnt/DLCV_hw5/HW5_data/FullLengthVideos/labels/valid/'
# model_path = '/mnt/DLCV_hw5/models/task2_LSTM_epoch100.pth'
model_path = sys.argv[1]
frame_path = sys.argv[2]
output_path = sys.argv[3]
video_namelist = os.listdir(frame_path)

hidden_size = 512
model = Task2Model_LSTM(2048*4*2, hidden_size, 11)
model.load_state_dict(torch.load(model_path))
model = model.cuda()

for video_name in video_namelist:
    #with open(label_path+video_name+'.txt') as f:
    #    labels = [int(i[:-1]) for i in list(f)]
    img_namelist = os.listdir(frame_path+video_name)
    pred_seq = []

    hidden = model.init_hidden()
    for i, img in enumerate(([io.imread(frame_path+video_name+'/'+img_name) for img_name in img_namelist])):
        img_tensor = transform(img).cuda()
        resnet_out = resnet50(img_tensor.view((1,)+img_tensor.size()))
        # print(resnet_out)
        # print(hidden)
        rnn_out, hidden = model(resnet_out.view(1, 1, -1), hidden)
        # print(rnn_out)
        pred_seq.append(int(rnn_out.topk(1)[1]))
    
    with open(output_path+'/'+video_name+'.txt', 'w') as f:
        for i, pred in enumerate(pred_seq):
            f.write(str(pred))
            if i+1 != len(pred_seq):
                f.write('\n')
