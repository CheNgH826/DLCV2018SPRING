from reader import readShortVideo
import pandas as pd
import numpy as np
import torch
from model import Task2Model, resnet50, transform, Task2Model_LSTM
import sys

# model_path = '/mnt/DLCV_hw5/models/task2_LSTM_epoch100.pth'
# video_path = '/mnt/DLCV_hw5/HW5_data/TrimmedVideos/video/valid/'
# label_csv_path = '/mnt/DLCV_hw5/HW5_data/TrimmedVideos/label/gt_valid.csv'
model_path = sys.argv[1]
video_path = sys.argv[2]
label_csv_path = sys.argv[3]
output_path = sys.argv[4]
# feature_path = '/mnt/DLCV_hw5/feature/'
label_df = pd.read_csv(label_csv_path)

hidden_size = 512
model = Task2Model_LSTM(2048*4*2, hidden_size, 11)
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['state_dict'])
model = model.cuda()

truth_seq = np.array(label_df['Action_labels'])
pred_seq = []

for idx in range(len(label_df)):
    video_category = label_df['Video_category'][idx]
    video_name = label_df['Video_name'][idx]

    rescale_factor = 1
    frames = readShortVideo(video_path, video_category, video_name, downsample_factor=12, rescale_factor=rescale_factor)
    frame_tensor_list = []
    for frame in frames:
        frame_tensor_transform = transform(frame)
        frame_tensor_list.append(frame_tensor_transform)
    
    frames_tensor = torch.stack(frame_tensor_list).cuda()
    resnet_out = resnet50(frames_tensor)
    
    # feature_np = np.load(feature_path+video_name+'.npy')
    # resnet_out = torch.tensor(feature_np).cuda()

    hidden = model.init_hidden()

    for i in range(resnet_out.size()[0]):
        classifier_out, hidden = model(resnet_out[i].view(1, 1, -1), hidden)
    
    pred_seq.append(int(classifier_out.topk(1)[1]))

with open(output_path+'/p2_result.txt', 'w') as f:
    for i, pred in enumerate(pred_seq):
        f.write(str(pred))
        if i+1 != len(pred_seq):
            f.write('\n')
