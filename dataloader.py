import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

class RRIDataset(Dataset):
    def __init__(self, files):
        self.sig = []
        self.annot = []
        
        for file in files:
            data = np.load(file, allow_pickle=True).item()
            sig = torch.tensor(data['sig'], dtype=torch.float32)
            annot = torch.tensor(data['annot'], dtype=torch.long)
            self.sig.append(sig)
            self.annot.append(annot)
        self.sig = torch.cat(self.sig, axis=0)
        self.sig = torch.unsqueeze(self.sig, -1)
        self.annot = torch.cat(self.annot, axis=0)
        print(self.sig.shape, self.annot.shape)
        
        # #標準化
        # tmp_sigs = []
        # for i in range(self.sig.shape[0]):
        #     tmp_data = self.sig[i, :, :].reshape(-1)
        #     mean_tmp = torch.mean(tmp_data)
        #     sd = torch.std(tmp_data, unbiased=False)
        #     tmp_data = ((tmp_data - mean_tmp) / sd).reshape(1, 30, 1)
        #     tmp_sigs.append(tmp_data)
        # self.sig = torch.cat(tmp_sigs, axis=0)
        # print(self.sig.shape, self.annot.shape)
        
    def __getitem__(self, index):
        sig = self.sig[index]
        annot = self.annot[index]
        return sig, annot

    def __len__(self):
        return len(self.sig)

if __name__ == '__main__':
    # npyファイルが保存されているディレクトリへのパス
    DATA_DIR = 'dataset\\YSYW_seq'
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
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    for epoch in range(NUM_EPOCHS):
        # データローダーからバッチごとにデータを取得
        for batch_x, label in train_loader:
           # print(batch_x, label)
            print(batch_x.shape)
            a = batch_x[0].reshape(-1)
            b = batch_x[1].reshape(-1)
            
            
            plt.plot(a)
            plt.plot(b)
            plt.show()
            print("")
            
