from reader import readShortVideo
import pandas as pd
import numpy as np
import torch
from model import Task1Model, resnet50, transform, Task1ModelDeep
import sys

model_path = sys.argv[1]
video_path = sys.argv[2]
label_csv_path = sys.argv[3]
output_path = sys.argv[4]
# feature_path = '/mnt/DLCV_hw5/feature/'
label_df = pd.read_csv(label_csv_path)

classifier = Task1ModelDeep()
classifier.load_state_dict(torch.load(model_path))
classifier = classifier.cuda()
classifier.eval()
# print(classifier)

# criterion = nn.CrossEntropyLoss()

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
    out_mean = torch.mean(resnet_out, dim=0).cuda().view(1, -1)
   
    # feature_np = np.load(feature_path+video_name+'.npy')
    # out_mean = torch.tensor(np.mean(feature_np, axis=0).reshape(1, -1)).cuda()

    classifier_out = classifier(out_mean).cpu()
    pred_seq.append(int(classifier_out.topk(1)[1]))

    # print(target)
    # loss = criterion(classifier_out, target)

with open(output_path+'/p1_valid.txt', 'w') as f:
    for i, pred in enumerate(pred_seq):
        f.write(str(pred))
        if i+1 != len(pred_seq):
            f.write('\n')
