#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler


class log_dataset(Dataset):
    def __init__(self, logs, labels):
        self.logs = []
        for i in range(len(labels)):
            features = [torch.tensor(logs[i][0][0], dtype=torch.long)]
            for j in range(1, len(logs[i][0])):
                features.append(torch.tensor(logs[i][0][j], dtype=torch.float))
            self.logs.append({
                "features": features,
                "idx": logs[i][1]
            })
        self.labels = labels    

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.logs[idx], self.labels[idx]

class pairwise_log_dataset(Dataset):
    def __init__(self, logs, labels):
        self.logs = []
        for i in range(len(labels)):
            features = [torch.tensor(logs[i][0][0], dtype=torch.long)]
            for j in range(1, len(logs[i][0])):
                features.append(torch.tensor(logs[i][0][j], dtype=torch.float))
            self.logs.append({
                "features": features,
                "idx": logs[i][1]
            })
        self.labels = np.array(labels)
        self.unlabeled_id = np.where(self.labels == 0)[0]  
        self.known_anom_id = np.where(self.labels == 1)[0]
        # self.known_anom_id = self.known_anom_id [:1000]  # 保留多少异常日志序列

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if idx % 4 == 0 or idx % 4 == 1:
            sid = np.random.choice(self.unlabeled_id, 2, replace=False)
            x1 = self.logs[sid[0]]
            x2 = self.logs[sid[1]]
            y = 0.
        elif idx % 4 == 2:
            sid1 = np.random.choice(self.unlabeled_id, 1)
            sid2 = np.random.choice(self.known_anom_id, 1)
            x1 = self.logs[sid1[0]]
            x2 = self.logs[sid2[0]]
            y = 4.
        else:
            sid = np.random.choice(self.known_anom_id, 2, replace=False)
            x1 = self.logs[sid[0]]
            x2 = self.logs[sid[1]]
            y = 8.
                 
        return (x1, x2), torch.tensor(y, dtype=torch.float32)

class pairwise_log_dataset_for_test(Dataset):
    def __init__(self, test_logs, test_labels, train_logs, train_labels):
        self.logs_train = []
        for i in range(len(train_labels)):
            features = [torch.tensor(train_logs[i][0][0], dtype=torch.long)]
            for j in range(1, len(train_logs[i][0])):
                features.append(torch.tensor(train_logs[i][0][j], dtype=torch.float))
            self.logs_train.append({
                "features": features,
                "idx": train_logs[i][1]
            })
        train_labels = train_labels
        train_labels = np.array(train_labels)
        self.unlabeled_id =  np.where(train_labels == 0)[0]  
        self.known_anom_id = np.where(train_labels == 1)[0]  
        # self.known_anom_id = self.known_anom_id [:1000]  # 保留多少异常日志序列
        
        self.logs_test = []
        for i in range(len(test_labels)):
            features = [torch.tensor(test_logs[i][0][0], dtype=torch.long)]
            for j in range(1, len(test_logs[i][0])):
                features.append(torch.tensor(test_logs[i][0][j], dtype=torch.float))
            self.logs_test.append({
                "features": features,
                "idx": test_logs[i][1]
            })
        self.test_labels = test_labels
        self.a = 30   # 平均30 pairs
        
    def __len__(self):
        return len(self.test_labels)

    def __getitem__(self, idx):
        x = self.logs_test[idx]
        x2_a_list = []
        x2_u_list = []
        a_idx = np.random.choice(self.known_anom_id, self.a, replace=True)
        u_idx = np.random.choice(self.unlabeled_id, self.a, replace=True)
        for i in range(self.a):
            x2_a = self.logs_train[a_idx[i]]
            x2_u = self.logs_train[u_idx[i]]
            x2_a_list.append(x2_a)
            x2_u_list.append(x2_u)
        
        return (x, x2_a_list, x2_u_list), self.test_labels[idx]
        
if __name__ == '__main__':
    data_dir = '../../data/'
