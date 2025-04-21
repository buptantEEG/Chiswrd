import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
import scipy.io
import argparse
import os
import time
from sklearn.model_selection import StratifiedKFold
import wandb
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

import sys
sys.path.append('xxx')
from model import *
def model_select(args_model):

    if args_model == 'LMDA':
        model = LMDA(num_classes=10, chans=29, samples=1000, channel_depth1=16, channel_depth2=4)
    elif args_model == 'EEGNet':
        model = EEGNet(fs=250, channel=29, num_class=10, signal_length=1000)
    elif args_model == 'ShallowConvNet':
        model = ShallowConvNet(num_classes=10, time_series_size=1000, num_kernels=32, node_size=29)
    elif args_model == 'DeepConvNet':
        model = DeepConvNet(num_classes=10, time_series_size=1000, num_kernels=32, node_size=29)
    return model

def get_source_data(path, nSub):
    # train data
    total_data = scipy.io.loadmat(path + f'sub_{nSub}.mat')
    data = total_data['data']
    label = total_data['label']
    data = np.expand_dims(data, axis=1)
    label = np.squeeze(label)

    # standardize
    target_mean = np.mean(data)
    target_std = np.std(data)
    data = (data - target_mean) / target_std
    return data, label

def init_args():
    parser = argparse.ArgumentParser()
    global_group = parser.add_argument_group(title="global", description="")
    global_group.add_argument("--model", default="EEGNet", type=str, help="")

    return parser.parse_args()
def set_seed(seed_n):
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)
def mkdir_fold(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

if __name__ == "__main__":
    os.environ["WANDB_MODE"] = "offline"
    used_seed = 114514
    set_seed(used_seed)

    args = init_args()
    root = '/xxx'
    for i in range(1, 11):
        batch_size = 64
        n_epochs = 500
        lr = 0.001

        data, label = get_source_data(root, i)
        Tensor = torch.cuda.FloatTensor
        LongTensor = torch.cuda.LongTensor

        kfolder = StratifiedKFold(n_splits=5, shuffle=True, random_state=1014)
        for fold, (train_index, test_index) in enumerate(kfolder.split(data, label)):
            print(f'this is subject_{i}, folder_{fold} ')
            model = model_select(args.model).cuda()
            print(args.model)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
            criterion_cls = torch.nn.CrossEntropyLoss().cuda()
            train_data = torch.from_numpy(data[train_index])
            train_label = torch.from_numpy(label[train_index])
            train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
            train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
            test_data = torch.from_numpy(data[test_index])
            test_label = torch.from_numpy(label[test_index])
            test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
            test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
            test_data = Variable(test_data.type(Tensor))
            test_label = Variable(test_label.type(LongTensor))

            for e in range(n_epochs):
                model.train()
                for i, (img, labl) in enumerate(train_dataloader):

                    img = Variable(img.cuda().type(Tensor))
                    labl = Variable(labl.cuda().type(LongTensor))
                    outputs = model(img)
                    loss = criterion_cls(outputs, labl) 
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    Cls = model(test_data)
                    loss_test = criterion_cls(Cls, test_label)
                    y_pred = torch.max(Cls, 1)[1]
                    acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))
                    train_pred = torch.max(outputs, 1)[1]
                    train_acc = float((train_pred == labl).cpu().numpy().astype(int).sum()) / float(labl.size(0))

                    all_preds = y_pred.cpu().numpy()
                    all_labels = test_label.cpu().numpy()
                    all_outputs = Cls.cpu().numpy()
                    all_labels_tsne = all_labels
        
                print('Epoch:', e,
                        '  Train loss: %.6f' % loss.detach().cpu().numpy(),
                        '  Test loss: %.6f' % loss_test.detach().cpu().numpy(),
                        '  Train accuracy %.6f' % train_acc,
                        '  Test accuracy is %.6f' % acc)
