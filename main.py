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
    def __init__(self, path_to_json, kpts_size=17, train=False, val=False, conv=False):
        with open(path_to_json) as f:
            self.files = json.load(f)
        if val:
            self.files = self.files[:400]
        else:
            self.files = self.files[400:]
        self.conv = conv
        self.train = train
        self.kpts_size = kpts_size
    def __len__(self):
        return len(self.files)
    def normalize(self, keypoints):
        kpts_x, kpts_y = np.array([keypoints[0::3]]), np.array([keypoints[1::3]])
        kpts_x = kpts_x - np.min(kpts_x)
        kpts_y = kpts_y - np.min(kpts_y)
        kpts_x = kpts_x / np.max(kpts_x)
        kpts_y = kpts_y / np.max(kpts_y)
        return np.array([kpts_x, kpts_y])
    
    def __getitem__(self, idx):
        kpts_original = self.normalize(self.files[idx]['keypoints'])
        
        kpts_out = np.array([kpts_original[0][0],
                                kpts_original[1][0]], dtype=np.float32)
        return torch.tensor(kpts_out), torch.tensor(kpts_out.copy())

def get_elim_mask(val_thresh=0.6):
    mask = 1.*(torch.rand(x.shape[0], 1, x.shape[-1]) > val_thresh*torch.rand(size=(1,)).item())
    mask = torch.cat([mask, mask], dim=1)
    return mask

if __name__ == "__main__":

    cuda = False
    dset_train = Points("kpts_valid.json", train=True, conv=True)
    dset_val = Points("kpts_valid.json", val=True, conv=True)

    model = PointSolver_conv()
    if cuda:
        model.cuda()
        model.constant = model.constant.cuda()

    BATCH_SIZE = 64
    N_EPOCHS = 100
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    train_loader = DataLoader(dset_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(dset_val, batch_size=BATCH_SIZE, drop_last=True)

    print("Starting train....\n")

    for i in range(N_EPOCHS):

        loss_epoch = 0.
        joint_error_per_epoch = []

        model.train()
        for x, y in train_loader:
            mask = get_elim_mask()
            x = x*mask
            if cuda:
                x = x.cuda()
                y = y.cuda()
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
            joint_error = (torch.abs(y_pred - y)*mask).sum(dim=1).reshape(-1)
            joint_error = joint_error[joint_error != 0.]
            #print("type:",type(joint_error))

            joint_error_per_epoch.append(joint_error.mean().item())

        print("Epoch %d \t train joint error %.5f" %(i+1, np.array(joint_error_per_epoch).mean()))
        with torch.no_grad():
            model.eval()
            val_joint_error_per_epoch = []
            val_joints_5_per_epoch = []
            val_joints_10_per_epoch = []
            val_joints_125_per_epoch = []
            for x, y in val_loader:
                mask = get_elim_mask()
                x = x*mask
                if cuda:
                    x = x.cuda()
                    y = y.cuda()
                y_pred = model(x)
                loss = criterion(y_pred, y)
                loss_epoch += loss.item()
                joint_error = (torch.abs(y_pred - y)*mask).sum(dim=1).reshape(-1)
                shape_first = len(joint_error)
                joint_error = joint_error[joint_error != 0.]
                #print(len(joint_error), shape_first)
                #break
                correct_joints_share_5 = (joint_error < 0.05).sum().item()/len(joint_error)
                #print("len y is", len(y))
                correct_joints_share_10 = (joint_error < 0.1).sum().item()/len(joint_error)
                correct_joints_share_125 = (joint_error < 0.125).sum().item()/len(joint_error)
                val_joints_5_per_epoch.append(correct_joints_share_5)
                val_joints_10_per_epoch.append(correct_joints_share_10)
                val_joints_125_per_epoch.append(correct_joints_share_125)
                val_joint_error_per_epoch.append(joint_error.mean().item())
            #print(val_joint_error_per_epoch)
            val_joint_error_per_epoch = np.array(val_joint_error_per_epoch).mean()
            val_joints_5_per_epoch = np.array(val_joints_5_per_epoch).mean()
            val_joints_10_per_epoch = np.array(val_joints_10_per_epoch).mean()
            val_joints_125_per_epoch = np.array(val_joints_125_per_epoch).mean()
            print("val_joint_error %.2f,\njts_5: %.2f\tjts_10: %.2f\tjts_125: %.2f"%(val_joint_error_per_epoch, 
                                                                                    val_joints_5_per_epoch, 
                                                                                    val_joints_10_per_epoch, 
                                                                                    val_joints_125_per_epoch))
        model_name = "conv_missed_jts5_%.2f_jts_125_%.2f_jts_error_%.2f.pt"%(val_joints_5_per_epoch, val_joints_125_per_epoch, val_joint_error_per_epoch)
        torch.save(model.state_dict(), model_name)
