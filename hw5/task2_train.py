import pandas as pd
import numpy as np
# from tqdm import tqdm
import torch
import torch.nn as nn
from model import Task2Model, Task2Model_GRU, Task2Model_LSTM
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from tensorboardX import SummaryWriter 

writer = SummaryWriter('summary/task2')

video_path = 'HW5_data/TrimmedVideos/video/train/'
label_csv_path = 'HW5_data/TrimmedVideos/label/gt_train.csv'
eval_csv_path = 'HW5_data/TrimmedVideos/label/gt_valid.csv'
# model_path = 'models/task2_LSTM_epoch100.pth'
# model_path = 'models/task2_LSTM_epoch8_1024_conti.pth'
feature_path = 'feature/'
label_df = pd.read_csv(label_csv_path)
eval_df = pd.read_csv(eval_csv_path)

hidden_size = 512
model = Task2Model_LSTM(2048*4*2, hidden_size, 11)
model.train()
# checkpoint = torch.load(model_path)
# model.load_state_dict(checkpoint['state_dict'])
# model.load_state_dict(checkpoint)
model = model.cuda()
print(model)

optimizer = optim.Adam(model.parameters(), lr=1e-5)
# optimizer.load_state_dict(checkpoint['optimizer'])
criterion = nn.NLLLoss()

def train(label, frames_np):
    frame_tensor = torch.tensor(frames_np).cuda()
    model.zero_grad()
    hidden = model.init_hidden()

    for i in range(frames_np.shape[0]):
        output, hidden = model(frame_tensor[i].view(1, 1, -1), hidden)
    
    loss = criterion(output.cpu(), label)
    # print([p.data for p in model.parameters()])
    loss.backward()
    optimizer.step()

    return output, loss.item()

epoch_num = 100
running_loss = 0
loss_history = []
for epoch in range(1, epoch_num+1):
    running_loss = 0
    for idx in range(len(label_df)):
        # video_category = label_df['Video_category'][idx]
        video_name = label_df['Video_name'][idx]
        feature = np.load(feature_path+video_name+'.npy')
        # feature = feature.reshape(feature.shape[0], 1, -1)
        target = torch.empty(1, dtype=torch.long).fill_(float(label_df['Action_labels'][idx])).cpu()

        output, loss = train(target, feature)
        running_loss += loss
        if (idx+1)%200 == 0:
            print('[Epoch %d/%d][%d/%d] loss: %f'%(epoch, epoch_num, idx+1, len(label_df), running_loss/(idx+1)))
            loss_history.append(running_loss/(idx+1))
            # running_loss = 0
            # print([p.data for p in model.parameters()])
    writer.add_scalar('loss', running_loss, epoch+1)
    loss_history.append(running_loss)
    np.save('task2_loss.npy', np.array(loss_history))
    if epoch%2==0:
        torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, 'models/task2_LSTM_epoch%d_512.pth'%(epoch))
    # simple evaluation
    pred_seq = []
    valid_loss = 0
    for j in range(200):
        video_name = eval_df['Video_name'][j]
        feature = np.load(feature_path+video_name+'.npy')
        feature_tensor = torch.tensor(feature).cuda()
        eval_hidden = model.init_hidden()

        for k in range(feature.shape[0]):
            output, hidden = model(feature_tensor[k].view(1,1,-1), eval_hidden)
        
        label = torch.empty(1, dtype=torch.long).fill_(float(eval_df['Action_labels'][k])).cpu()
        loss = criterion(output.cpu(), label)
        valid_loss += loss
        pred_seq.append(int(output.topk(1)[1]))

    truth_seq = np.array(eval_df['Action_labels'][:200])
    acc = accuracy_score(truth_seq, pred_seq)
    print('[Epoch %d/%d] val_loss: %f acc: %f'%(epoch, epoch_num, valid_loss/200, acc))
