import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_classification
import collections
from imblearn.under_sampling import RandomUnderSampler

class RRIDataset(Dataset):
    def __init__(self, files):
        self.sig = []
        self.annot = []
        
        for file in files:
            data = np.load(file, allow_pickle=True).item()
            # sig = torch.tensor(data['sig']).float()
            # annot = torch.tensor(data['annot']).long()
            self.sig.append(data["sig"])
            self.annot.append(data["annot"])
        
        self.sig = np.vstack(self.sig)
        print(self.sig.shape)

        self.annot = np.concatenate(self.annot)
        
        # 不均衡データを均衡データにする
        #c = collections.Counter(self.annot)
        #print(c)
        sampler = RandomUnderSampler(random_state=42)
        # downsampling
        self.sig, self.annot = sampler.fit_resample(self.sig, self.annot)
        print(collections.Counter(list(self.annot)))
        
        self.sig = np.vstack(self.sig)
        
        #データの正規化
        scaler = MinMaxScaler(feature_range=(0, 1))
        normalized_sig = scaler.fit_transform(self.sig.T).T
        self.sig = torch.tensor(normalized_sig, dtype=torch.float32)
        
        #次元追加
       # self.sig = torch.unsqueeze(self.sig, -1)
        self.annot = torch.tensor(self.annot, dtype=torch.int64)
        print(self.sig.shape, self.annot.shape)        
        
    def __getitem__(self, index):
        sig = self.sig[index]
        annot = self.annot[index]
        return sig, annot

    def __len__(self):
        return len(self.sig)

if __name__ == '__main__':
    # npyファイルが保存されているディレクトリへのパス
    DATA_DIR = 'dataset\\YSYW_seq\\ecg'
    TRAIN_RATIO = 0.7
    NUM_EPOCHS = 10
    BATCH_SIZE = 32

    # npyファイルのリストを取得
    data_files = [os.path.join(DATA_DIR, file) for file in os.listdir(DATA_DIR) if file.endswith('.npy')]

    # データセットをTrainとValidationに分割
    train_size = int(TRAIN_RATIO * len(data_files))
    val_size = len(data_files) - train_size
    train_files, val_files = random_split(data_files, [train_size, val_size])

    # カスタムデータセットインスタンスを作成
    train_dataset = RRIDataset(train_files)
    val_dataset = RRIDataset(val_files)

    # DataLoaderを使用してミニバッチ学習を実現
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    for epoch in range(NUM_EPOCHS):
        # データローダーからバッチごとにデータを取得
        for batch_x, label in train_loader:
           # print(batch_x, label)
            print(batch_x.shape, label.shape)
            plt.plot(batch_x[0, :])
            plt.show()
            
            print("")
            
