import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import LSTM_Classifier, RNN_Classifier, MLP
from dataloader import RRIDataset
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
import os
import tqdm

# Constants
INPUT_DIM = 1 # 時系列データの次元数を設定してください
OUTPUT_DIM = 5 # カテゴリデータの次元数を設定してください
HIDDEN_DIM = 256  # Transformer内の隠れ層の次元数を設定してください
NUM_LAYERS = 1
# NHEAD = 8  # Transformerのマルチヘッド数を設定してください
# NLAYERS = 6  # Transformerのエンコーダーとデコーダーのレイヤー数を設定してください
# DROPOUT = 0.1# ドロップアウト率を設定してください

DATA_DIR = 'dataset\\YSYW_seq'
TRAIN_RATIO = 0.8
NUM_EPOCHS = 100000
BATCH_SIZE = 512
lr = 0.001

#GPUデバイス
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# npyファイルのリストを取得
data_files = [os.path.join(DATA_DIR, file) for file in os.listdir(DATA_DIR) if file.endswith('.npy')]

# データセットをTrainとValidationに分割
train_size = int(TRAIN_RATIO * len(data_files))
val_size = len(data_files) - train_size
train_files, val_files = random_split(data_files, [train_size, val_size])

print("train:", train_size, "val:", val_size)

# カスタムデータセットインスタンスを作成
train_dataset = RRIDataset(train_files)
val_dataset = RRIDataset(val_files)

# DataLoaderを使用してミニバッチ学習を実現
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

# # 変換方法の指定
# transform = transforms.Compose([
#     transforms.ToTensor()
#     ])

# # MNISTデータの取得
# # https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST
# # 学習用
# train_dataset = datasets.MNIST(
#     'dataset\\MNIST',               # データの保存先
#     train = True,           # 学習用データを取得する
#     download = True,        # データが無い時にダウンロードする
#     transform = transform   # テンソルへの変換など
#     )
# # 評価用
# val_dataset = datasets.MNIST(
#     'dataset\\MNIST', 
#     train = False,
#     transform = transform
#     )

# # データローダー
# train_loader = torch.utils.data.DataLoader(
#     train_dataset,
#     batch_size = BATCH_SIZE,
#     shuffle = True)

# val_loader = torch.utils.data.DataLoader(
#     val_dataset,     
#     batch_size = BATCH_SIZE,
#     shuffle = True)


# モデルのインスタンスを作成
model = LSTM_Classifier(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS).to(device)
#model = MLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)
print(model)



# for parameter in iter(model.parameters()):
#     print(parameter)
    
# オプティマイザと損失関数を設定
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

train_loss_per_epoch_list = []
val_loss_per_epoch_list = []
train_acc_per_epoch_list = []
val_acc_per_epoch_list = []

# 学習ループ
for epoch in range(NUM_EPOCHS):
    train_loss_list = []
    
    num_train = 0
    train_acc = 0
    
    model.train()
    # データローダーからバッチごとにデータを取得
    for seq_rris, labels in train_loader:
        #入力をGPUに転送
        seq_rris, labels = seq_rris.to(device), labels.to(device)
        #print(seq_rris[0].reshape(-1))
        #print(seq_rris[1].reshape(-1))
        #print(labels.data)
        #seq_rris = seq_rris.view(-1, INPUT_DIM) # 画像データ部分を一次元へ並び変える
        #print(seq_rris.shape, labels.shape)
        # 勾配をクリア
        optimizer.zero_grad()
        # 推論
        outputs = model(seq_rris)
        #print(outputs, outputs.shape)
        #予測ラベル
        predicted = torch.max(outputs, 1)[1]
        #print(predicted)
        #loss calc.
        loss = criterion(outputs, labels)
        #back prop.
        loss.backward()
        #パラメータ更新
        optimizer.step()
        #学習回数更新
        num_train+=len(labels)
        #lossの保存
        train_loss_list.append(loss.item())
        #精度の保存
        train_acc += (predicted == labels.data).sum()

    train_loss = np.sum(train_loss_list) / num_train
    train_loss_per_epoch_list.append(train_loss)
    train_acc = train_acc / num_train
    train_acc_per_epoch_list.append(train_acc)
    
    # Validation loop
    model.eval()
    with torch.no_grad():
        val_loss_list = []
        num_val = 0
        val_acc = 0
        for seq_rris, labels in val_loader:
            #入力をGPUに転送
            seq_rris, labels = seq_rris.to(device), labels.to(device)
            #seq_rris = seq_rris.view(-1, INPUT_DIM) # 画像データ部分を一次元へ並び変える
            # 推論
            outputs = model(seq_rris)
            #予測ラベル
            predicted = torch.max(outputs, 1)[1]
            #print(predicted)
            # loss calc.
            loss = criterion(outputs, labels)
            #学習回数更新
            num_val+=len(labels)
            # lossの保存
            val_loss_list.append(loss.item())
            #精度の保存
            val_acc += (predicted == labels).sum()
        
    val_loss = np.sum(val_loss_list) / num_val
    val_loss_per_epoch_list.append(val_loss)
    val_acc = val_acc / num_val
    val_acc_per_epoch_list.append(val_acc)
    #コンソール表示
    print(f"Epoch:{epoch}, Train loss:{train_loss}, Train acc:{train_acc}, Valid loss:{val_loss}, Valid acc:{val_acc}")
            