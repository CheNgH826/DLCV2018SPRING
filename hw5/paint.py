import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
from tqdm import tqdm
import matplotlib
from model import Task2Model_LSTM
import torch

# loss_history = np.load('task1_loss.npy')
# plt.title('loss')
# plt.plot(loss_history[::100])
# plt.savefig('task1_loss.jpg')

# feature_path = '/mnt/DLCV_hw5/feature/'
# label_csv_path = '/mnt/DLCV_hw5/HW5_data/TrimmedVideos/label/gt_valid.csv'
# label_df = pd.read_csv(label_csv_path)

# truth_seq = np.array(label_df['Action_labels'])
# feature_mean_list = []
# for idx in tqdm(range(len(label_df))):
#     video_category = label_df['Video_category'][idx]
#     video_name = label_df['Video_name'][idx]
#     feature_np = np.load(feature_path+video_name+'.npy')
#     out_mean = np.mean(feature_np, axis=0).reshape(-1)
#     feature_mean_list.append(out_mean)
# latent_code = np.array(feature_mean_list)
# latent_emmbedded = TSNE(random_state=2).fit_transform(latent_code)
# plt.scatter(latent_emmbedded[:,0], latent_emmbedded[:,1], c=truth_seq)
# plt.savefig('p2_tsne_scatter_cnn.jpg')

# latent_code = np.load('LSTM_hidden_val.npy')
# label_csv_path = '/mnt/DLCV_hw5/HW5_data/TrimmedVideos/label/gt_valid.csv'
# model_path = '/mnt/DLCV_hw5/models/task2_LSTM_epoch100.pth'
# feature_path = '/mnt/DLCV_hw5/feature/'
# label_df = pd.read_csv(label_csv_path)
# truth_seq = np.array(label_df['Action_labels'])
# hidden_size = 1024
# model = Task2Model_LSTM(2048*4*2, hidden_size, 11)
# model.load_state_dict(torch.load(model_path))
# model = model.cuda()
# hidden_list = []
# for idx in tqdm(range(len(label_df))):
#     video_name = label_df['Video_name'][idx]
#     feature_np = np.load(feature_path+video_name+'.npy')
#     resnet_out = torch.tensor(feature_np).cuda()
#     hidden = model.init_hidden()

#     for i in range(resnet_out.size()[0]):
#         classifier_out, hidden = model(resnet_out[i].view(1, 1, -1), hidden)
#     hidden_list.append(hidden[0].cpu().detach().numpy())
# latent_code = np.array(hidden_list).reshape(len(label_df), -1)
# latent_emmbedded = TSNE(random_state=2).fit_transform(latent_code)
# plt.scatter(latent_emmbedded[:,0], latent_emmbedded[:,1], c=truth_seq)
# plt.savefig('p2_tsne_scatter_rnn.jpg')

# task3_loss = np.load('task3_loss.npy')
# plt.title('loss')
# plt.plot(task3_loss)
# plt.savefig('task3_loss_large.jpg')

pred = np.load('OP03-R02-TurkeySandwich_pred.npy')[:300]
ans  = np.load('OP03-R02-TurkeySandwich_label.npy')[:300]
print(pred)
print(ans)
# plt.figure(figsize=(16,4))
# ax1 = plt.subplot(211)
# colors = plt.cm.get_cmap('Set3',11).colors
# cmap = matplotlib.colors.ListedColormap([colors[idx] for idx in pred])
# bounds = range(300)
# norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
# cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap=cmap,
#                                        norm=norm,
#                                        boundaries=bounds,
#                                        spacing='proportional',
#                                        orientation='horizontal',
#                                        ticks=range(544,844))
# ax1.set_ylabel('Prediction')

# ax2 = plt.subplot(212)
# cmap = matplotlib.colors.ListedColormap([colors[idx] for idx in ans])
# cb2 = matplotlib.colorbar.ColorbarBase(ax2, cmap=cmap,
#                                        norm=norm,
#                                        boundaries=bounds,
#                                        spacing='proportional',
#                                        orientation='horizontal',
#                                        ticks=range(544,844))
# ax2.set_ylabel('Ground Truth')

# plt.savefig('p3_video_segmentation.jpg')