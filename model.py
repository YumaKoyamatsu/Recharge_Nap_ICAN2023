import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F


# Constants
# INPUT_DIM = 30 # 時系列データの次元数を設定してください
# OUTPUT_DIM = 5  # カテゴリデータの次元数を設定してください
# HIDDEN_DIM = 128  # Transformer内の隠れ層の次元数を設定してください
# NHEAD = 8  # Transformerのマルチヘッド数を設定してください
# NLAYERS = 6  # Transformerのエンコーダーとデコーダーのレイヤー数を設定してください
# DROPOUT = 0.1# ドロップアウト率を設定してください

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        out = self.linear2(x)
        out = F.log_softmax(out, dim=1)
        return out
        
class RNN_Classifier(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        h_out, h_0 = self.rnn(x, None)
        print(h_out.shape)
        h_out = h_out[:, -1, :]
        out = self.fc(h_out)
        out = self.softmax(out)
        return out

class LSTM_Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTM_Classifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, output_dim)

        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        lstm_out, (h0, c0) = self.lstm(x, (h0, c0))
        out = h0[-1, :, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        #out = self.softmax(out)
        return out

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, nhead, nlayers, dropout):
        super(TimeSeriesTransformer, self).__init__()

        # 入力層
        self.input_fc = nn.Linear(input_dim, hidden_dim)

        # Transformer層
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=nhead,
            num_encoder_layers=nlayers,
            num_decoder_layers=0,
            dropout=dropout,
        )

        # 出力層
        self.output_fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 入力データの次元を調整
        x = self.input_fc(x)
        print(x.shape)
        x = x.permute(1, 0, 2)  # Transformerは [sequence length, batch size, hidden_dim] を入力として期待しています

        # Transformerを通してデータを渡す
        x = self.transformer(x)

        # 出力データの次元を調整
        x = x.permute(1, 0, 2)
        x = self.output_fc(x)

        return x