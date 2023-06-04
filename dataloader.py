import torch
from torchvision import transforms
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_classification
import collections
from imblearn.under_sampling import RandomUnderSampler
import librosa
from tqdm import tqdm
from PIL import Image

class RRIDataset(Dataset):
    def __init__(self, files, transform=None):
        self.sig = []
        self.annot = []
        self.transform = transform
        
        for file in tqdm(files):
            data = np.load(file, allow_pickle=True).item()
            # sig = torch.tensor(data['sig']).float()
            # annot = torch.tensor(data['annot']).long()
            self.sig.append(data["sig"])
            self.annot.append(data["annot"])
        
        self.sig = np.vstack(self.sig)
        print(self.sig.shape)

        self.annot = np.concatenate(self.annot)
        print()
        # # 不均衡データを均衡データにする
        # #c = collections.Counter(self.annot)
        # #print(c)
        # sampler = RandomUnderSampler(random_state=42)
        # # downsampling
        # self.sig, self.annot = sampler.fit_resample(self.sig, self.annot)
        print(collections.Counter(list(self.annot)))
        #print(self.sig.shape)
        #self.sig = np.vstack(self.sig)
        
        #データの正規化
        # scaler = MinMaxScaler(feature_range=(0, 1))
        # normalized_sig = scaler.fit_transform(self.sig.T).T
        # self.sig = torch.tensor(normalized_sig, dtype=torch.float32)
        
        #スペクトログラム
        # spec = []
        # for i in tqdm(range(self.sig.shape[0])):
        #     spec.append(wav2spec(self.sig[i]))
        
        # self.sig = np.stack(spec)
        # print(self.sig.shape)
        # #次元追加
        # self.sig = torch.unsqueeze(self.sig, -1)
        # self.sig = torch.tensor(self.sig).float()
        self.annot[self.annot==2] = 1
        self.annot[self.annot==3] = 1
        self.annot[self.annot==4] = 2
        # # 不均衡データを均衡データにする
        #c = collections.Counter(self.annot)
        #print(c)
        sampler = RandomUnderSampler(random_state=42)
        sig_flattened = self.sig.reshape(self.sig.shape[0], -1)
        # downsampling
        sig_resampled, self.annot = sampler.fit_resample(sig_flattened, self.annot)
        self.sig = sig_resampled.reshape(-1, self.sig.shape[1], self.sig.shape[2], self.sig.shape[3])
        print(collections.Counter(list(self.annot)))
        
        self.annot = torch.tensor(self.annot).long()      
        
        print(self.sig.shape, self.annot.shape)        
        
    def __getitem__(self, index):
        #sig = self.transform(self.sig[index].transpose(1, 2, 0).astype(np.float32))
        sig = self.sig[index]
        annot = self.annot[index]
        return self.transform(sig), annot

    def __len__(self):
        return len(self.sig)
    
def wav2spec(sig, n_fft=2048, hop_length=512, sr=200):
    # STFTを用いてスペクトログラムを計算
    D = np.abs(librosa.stft(sig, n_fft=n_fft, hop_length=hop_length))
    # 対数スペクトログラムを計算
    D = librosa.amplitude_to_db(D, ref=np.max)
    # 画像として表示
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
    ax.axis('off')  # 軸の数値を表示しない
    #余白削除
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    # フィギュアのRGBデータを取得しPillowのImageに変換
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = Image.frombytes('RGB', fig.canvas.get_width_height(), data)
    # リサイズ
    img_resized = image.resize((224, 224))
    plt.close()
    img_np = np.array(img_resized)
    
    return img_np

if __name__ == '__main__':
    # npyファイルが保存されているディレクトリへのパス
    DATA_DIR = 'dataset\\YSYW_seq\\spec'
    TRAIN_RATIO = 0.7
    NUM_EPOCHS = 10
    BATCH_SIZE = 32
    padding = (1025 - 1025, 1025 - 12)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Pad(padding),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    # npyファイルのリストを取得
    data_files = [os.path.join(DATA_DIR, file) for file in os.listdir(DATA_DIR) if file.endswith('.npy')]

    # データセットをTrainとValidationに分割
    train_size = int(TRAIN_RATIO * len(data_files))
    val_size = len(data_files) - train_size
    train_files, val_files = random_split(data_files, [train_size, val_size])

    # カスタムデータセットインスタンスを作成
    train_dataset = RRIDataset(train_files, transform=transform)
    val_dataset = RRIDataset(val_files)

    # DataLoaderを使用してミニバッチ学習を実現
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    for epoch in range(NUM_EPOCHS):
        # データローダーからバッチごとにデータを取得
        for batch_x, label in train_loader:
           # print(batch_x, label)
            print(batch_x.shape, label.shape)
            #img = 
            plt.plot(batch_x[0, :])
            plt.show()
            
            print("")
            
