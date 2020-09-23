from joints_conv_model import PointSolver_conv

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import numpy as np
import matplotlib.pyplot as plt

import json

class Points(Dataset):
    def __init__(self, path_to_json, kpts_size=17, train=False, val=False, prob = 1./17, conv=False):
        with open(path_to_json) as f:
            self.files = json.load(f)
        if val:
            self.files = self.files[:400]
        else:
            self.files = self.files[400:]
        self.conv = conv
        self.train = train
        self.kpts_size = kpts_size
        self.prob = prob
    def __len__(self):
        return len(self.files)
    def normalize(self, keypoints):
        kpts_x, kpts_y = np.array([keypoints[0::3]]), np.array([keypoints[1::3]])
        kpts_x = kpts_x - np.min(kpts_x)
        kpts_y = kpts_y - np.min(kpts_y)
        kpts_x = kpts_x / np.max(kpts_x) + 0.001
        kpts_y = kpts_y / np.max(kpts_y) + 0.001
        return np.array([kpts_x, kpts_y])
    def get_elim_mask(self):
        #mask = torch.rand(1,self.kpts_size)
        number = torch.randint(0, self.kpts_size, (1,)).item()
        mask = 1. * torch.tensor([(i!=number) for i in range(self.kpts_size)])
        #mask = torch.cat([mask.t(),mask.t()], dim=0).reshape(-1)
        return mask, number
    def set_elim_prob(self, new_prob):
        self.prob = new_prob
    def __getitem__(self, idx):
        kpts_original = torch.tensor(self.normalize(self.files[idx]['keypoints']), 
                                     dtype=torch.float32)
        get_elim_mask, idx = self.get_elim_mask()
        #kpts_norm = kpts_original * get_elim_mask
        kpts_original_ = torch.zeros((2 * self.kpts_size), dtype=torch.float32)
        kpts_norm = torch.zeros_like(kpts_original_)
        kpts_original_[0::2] = kpts_original[0][0]
        kpts_original_[1::2] = kpts_original[1][0]
        kpts_norm[::2] = kpts_original_[0::2]*get_elim_mask
        kpts_norm[1::2] = kpts_original_[1::2]*get_elim_mask
        if self.conv:
            kpts_norm = kpts_norm.reshape(1, 2*self.kpts_size)
            kpts_original_ = kpts_original_.reshape(1, 2*self.kpts_size)
        if self.train:
            return kpts_norm, kpts_original_, idx
        return kpts_norm, kpts_original_, idx

if __name__ == "__main__":

    cuda = False
    dset_train = Points("kpts_valid.json", train=True, prob=1./14, conv=True)
    dset_val = Points("kpts_valid.json", val=True, conv=True)

    model = PointSolver_conv(torch.tensor(np.load("very_important_constant.npy")))
    if cuda:
        model.cuda()
        model.constant = model.constant.cuda()

    BATCH_SIZE = 64
    N_EPOCHS = 100
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    train_loader = DataLoader(dset_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(dset_val, batch_size=BATCH_SIZE, drop_last=True)

    for i in range(N_EPOCHS):
        loss_epoch = 0.
        joint_error_per_epoch = []
    
        model.train()
        for x, y, idx in train_loader:
            if cuda:
                x = x.cuda()
                y = y.cuda()
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
            y_pred = y_pred.reshape(-1, 34)
            y = y.reshape(-1,34)
            joint_error = torch.tensor([y_pred[i][2*idx[i]] - y[i][2*idx[i]] \
                                    for i in range(BATCH_SIZE)]).abs().mean().item()
        
            joint_error_per_epoch.append(joint_error)
    
        print(f"Epoch {i+1}, joint error {np.array(joint_error_per_epoch).mean()}")
        with torch.no_grad():
            model.eval()
            val_joint_error_per_epoch = []
            val_joints_5_per_epoch = []
            val_joints_10_per_epoch = []
            val_joints_125_per_epoch = []
            for x, y, idx in val_loader:
                if cuda:
                    x = x.cuda()
                    y = y.cuda()
                y_pred = model(x)
                loss = criterion(y_pred, y)
                
                y_pred = y_pred.reshape(-1, 34)
                y = y.reshape(-1,34)
                loss_epoch += loss.item()
                joint_error = torch.tensor([y_pred[i][2*idx[i]] - y[i][2*idx[i]] \
                                            for i in range(BATCH_SIZE)]).abs().mean().item()
                correct_joints_share_5 = ((torch.tensor([y_pred[i][2*idx[i]] - y[i][2*idx[i]] \
                                            for i in range(BATCH_SIZE)]).abs()) < 0.05).sum().item()*1./BATCH_SIZE
                correct_joints_share_10 = ((torch.tensor([y_pred[i][2*idx[i]] - y[i][2*idx[i]] \
                                            for i in range(BATCH_SIZE)]).abs()) < 0.1).sum().item()*1./BATCH_SIZE
                correct_joints_share_125 = ((torch.tensor([y_pred[i][2*idx[i]] - y[i][2*idx[i]] \
                                            for i in range(BATCH_SIZE)]).abs()) < 0.125).sum().item()*1./BATCH_SIZE
                val_joints_5_per_epoch.append(correct_joints_share_5)
                val_joints_10_per_epoch.append(correct_joints_share_10)
                val_joints_125_per_epoch.append(correct_joints_share_125)
                val_joint_error_per_epoch.append(joint_error)
            val_joint_error_per_epoch = np.array(val_joint_error_per_epoch).mean()
            val_joints_5_per_epoch = np.array(val_joints_5_per_epoch).mean()
            val_joints_10_per_epoch = np.array(val_joints_10_per_epoch).mean()
            val_joints_125_per_epoch = np.array(val_joints_125_per_epoch).mean()
            print(f"val_joint_error {val_joint_error_per_epoch},\n\
            jts_5: {val_joints_5_per_epoch}, jts_10: {val_joints_10_per_epoch},\
                jts_125: {val_joints_125_per_epoch}")
        torch.save(model.state_dict(), f"conv_jts5_{val_joints_5_per_epoch}_jts_125_{val_joints_125_per_epoch}_jts_error_{val_joint_error_per_epoch}.pt")