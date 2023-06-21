import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import numpy as np

from dataloader import RRIDataset
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
import os
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import optuna
import gc
from model import *
from util import *

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def train_model(model, optimizer,dataloader_train, dataloader_valid, n_epochs, device):
    epoch_losses_train = []
    epoch_losses_valid = []
    epoch_accs_train = []
    epoch_accs_valid = []
    wait = 0
    best_val_loss = np.inf
    #weights = torch.tensor([1.8, 1.0, 1.1, 3.6, 6.4]).cuda()
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(n_epochs):
        losses_train = []
        losses_valid = []
        train_num = 0
        train_true_num = 0
        valid_num = 0
        valid_true_num = 0

        #train
        model.train()  # 訓練時には勾配を計算するtrainモードにする
        acc = 0.0
        for x, t in dataloader_train:
            # WRITE ME
            #勾配初期化
            optimizer.zero_grad()
            
            #データをGPU
            x = x.to(device)
            t = t.to(device)
            #t_hot = torch.eye(10)[t].to(device)  # 正解ラベルをone-hot vector化
            
            #順伝播
            y = model(x)
            #loss calc.
            loss = criterion(y, t)
            
            #逆伝播
            loss.backward()
            
            #パラメータ更新
            optimizer.step()
            #scheduler.step()
            
            #loss append
            losses_train.append(loss.item())
            
            #確率からラベルに変換
            pred = y.argmax(dim=1)
            
            #精度計算
            acc = torch.where(t.to("cpu") - pred.to("cpu") == 0, torch.ones_like(t.to("cpu")), torch.zeros_like(t.to("cpu")))
            train_num += acc.size()[0]
            train_true_num += acc.sum().item()

        #eval
        model.eval()  # 評価時には勾配を計算しないevalモードにする
        acc = 0.0
        for x, t in dataloader_valid:
            # WRITE ME
            x = x.to(device)
            t = t.to(device)
            #t_hot = torch.eye(10)[t].to(device)  # 正解ラベルをone-hot vector化
            
            # 順伝播
            y = model(x)
            #loss calc.
            loss = criterion(y, t)
            losses_valid.append(loss.item())
            
            #確率からラベルに変換
            #pred = y.argmax(dim=1)
            pred = torch.max(y, 1)[1]
            #精度計算
            acc += pred.eq(t.data).sum().to("cpu").detach().numpy().copy()


        loss_train = np.mean(losses_train)
        loss_valid = np.mean(losses_valid)
        acc_train = train_true_num/train_num
        #acc_valid = valid_true_num/valid_num
        acc_valid = acc / len(dataloader_valid.dataset)
        print('EPOCH: {}, Train [Loss: {:.4f}, Accuracy: {:.4f}], Valid [Loss: {:.4f}, Accuracy: {:.4f}], wait: {}'.format(
            epoch,
            loss_train,
            acc_train,
            loss_valid,
            acc_valid,
            wait
        ))
        
        epoch_losses_train.append(loss_train)
        epoch_losses_valid.append(loss_valid)
        epoch_accs_train.append(acc_train)
        epoch_accs_valid.append(acc_valid)
        
        #Early stopping
        # wait, best_val_loss, stop_flag = early_stop(loss_valid, best_val_loss, wait, patience=10, min_delta=0.00001)
        # if stop_flag:
        #     break
        
    return epoch_losses_train, epoch_losses_valid, epoch_accs_train, epoch_accs_valid

def objective(trial):
    # ハイパーパラメータの範囲を指定
    output_dim = 5
    
    lr = trial.suggest_float("lr", 1e-7, 1e-2, log=True)
    num_epochs = 20#trial.suggest_int("num_epochs", 500, 1000)
    batch_size = 32#trial.suggest_int("batch_size", 32, 128)
    #n_layers = trial.suggest_int("n_layers", 5, 15)
    weight_decay = trial.suggest_float("weight_decay",1e-7, 1e-4, log=True)
    #hidden_dims = []
    
    # for i in range(n_layers):
    #     hidden_dims.append(trial.suggest_int(f"n_units_layer_{i}", 16, 512))
    
    # print("Trial: {}\nlr: {}\nnum_epochs: {}\nbatch_size: {}\nn_layers: {}\nweight_decay: {}".format(
    #     trial.number, lr, num_epochs, batch_size, n_layers, weight_decay))

    print("Trial: {}\nlr: {}\nnum_epochs: {}\nbatch_size: {}\nweight_decay: {}".format(
        trial.number, lr, num_epochs, batch_size, weight_decay))
    #デバイス指定
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device, torch.cuda.is_available())
    
    # model = MLP(input_dim=input_dim, 
    #             hidden_dims=hidden_dims, 
    #             output_dim=output_dim).to(device)
    #model = CNN(output_dim=output_dim, hidden_dim=64).to(device)
    model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    #model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    # for param in model.parameters():
    #     param.requires_grad = False
    #model.heads[0] = torch.nn.Linear(model.heads[0].in_features, output_dim)
    model.fc = torch.nn.Linear(model.fc.in_features, output_dim)
    model = model.to(device)
    print(model)
    
    
    DATA_DIR = 'dataset\\YSYW_seq\\spec_RRI_16-1'
    TRAIN_RATIO = 0.8
    # npyファイルのリストを取得
    data_files = [os.path.join(DATA_DIR, file) for file in os.listdir(DATA_DIR) if file.endswith('.npy')]

    # データセットをTrainとValidationに分割
    train_size = int(TRAIN_RATIO * len(data_files))
    val_size = len(data_files) - train_size
    train_files, val_files = random_split(data_files, [train_size, val_size])

    print("train:", train_size, "val:", val_size)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # カスタムデータセットインスタンスを作成
    train_dataset = RRIDataset(train_files, transform=transform)
    val_dataset = RRIDataset(val_files, transform=transform)

    # DataLoaderを使用してミニバッチ学習を実現
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    # optimizer = optim.Adam([{'params': model.heads[0].parameters()}],
    #                        lr=lr,
    #                        weight_decay=weight_decay, 
    #                        amsgrad=True)
    
    optimizer = optim.SGD(model.parameters(),
                        lr=lr,
                        momentum=0.9, 
                        weight_decay=weight_decay)
    loss_train, loss_valid, acc_train, acc_valid = train_model(model, optimizer, train_loader, val_loader, num_epochs, device=device)
    
    #保存
    output_dir = f'{base_dir}\\trial_{trial.number}'
    os.makedirs(output_dir)
    save_running_code("train.py", f"{output_dir}\\train.py")#実行コード保存
    save_running_code("model.py", f"{output_dir}\\model.py")#実行コード保存
    save_running_code("util.py", f"{output_dir}\\util.py")#実行コード保存
    save_running_code("make_dataset.py", f"{output_dir}\\make_dataset.py")#実行コード保存
    with open(f"{output_dir}\\params.pkl", "wb") as f:
        pickle.dump(trial.params, f)
    
    #Lossのグラフ保存
    fig, ax = plt.subplots()
    ax.plot(np.array(loss_train), label="Train")
    ax.plot(np.array(loss_valid), label="Valid")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("loss")
    ax.set_title(f"Trial: {trial.number}, Accuracy: {acc_valid[-1]}")
    ax.legend()
    fig.savefig(f"{output_dir}\\loss.png")
    model_path = f'{output_dir}\\model.pth'
    torch.save(model.state_dict(), model_path)
    
    #メモリ開放
    del train_dataset
    del val_dataset
    del train_loader
    del val_loader
    gc.collect()
    
    return loss_valid[-1]

if __name__ == '__main__':

    base_dir = f"result\\{get_now_time()}"
    os.makedirs(base_dir)

    # Optunaで最適化のセッションを作成
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=200)

    # 最適化されたモデルのパラメータを表示
    print("Best trial:")
    trial = study.best_trial
    print("  Loss: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    # Studyオブジェクトをファイルに保存
    now_time = get_now_time()
    with open(f"{base_dir}\\optuna_study.pkl", "wb") as f:
        pickle.dump(study, f)