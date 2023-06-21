import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        
        self.net = nn.Sequential()

        #入力層
        self.net.add_module("Input", nn.Linear(input_dim, hidden_dims[0]))
        self.net.add_module(f"ReLU0", nn.LeakyReLU())
        #中間層
        for i in range(len(hidden_dims)-1):            
            self.net.add_module(f"Hidden{i}", nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.net.add_module(f"ReLU{i+1}", nn.LeakyReLU())
            #self.net.add_module(f"Dropout{i}", nn.Dropout(p=0.3))
        #出力層
        self.net.add_module("Output", nn.Linear(hidden_dims[-1], output_dim))
    
    def forward(self, x):
        x = self.net(x)
        return x

class CNN(nn.Module):
    def __init__(self, output_dim, hidden_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 254 * 1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape)
        x = self.flatten(x)
        #x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

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

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.5)
        self.linear = nn.Linear(self.hidden_dim, output_dim)
        
    def forward(self, x):
        self.h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        self.c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        lstm_out, (h0, c0) = self.lstm(x, (self.h0, self.c0))
        out = h0[-1, :, :]
        out = self.linear(out)
        return out

class TimeSeriesClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super(TimeSeriesClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (batch_size, seq_length, input_dim)
        x = self.embedding(x)
        # x: (batch_size, seq_length, d_model)
        x = x.permute(1, 0, 2)
        # x: (seq_length, batch_size, d_model)
        x = self.transformer_encoder(x)
        # x: (seq_length, batch_size, d_model)
        x = x.mean(dim=0)
        # x: (batch_size, d_model)
        x = self.classifier(x)
        # x: (batch_size, num_classes)
        return x

class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.0):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, inputs, targets):
        one_hot = torch.full_like(inputs, self.smoothing / (self.num_classes - 1))
        one_hot.scatter_(1, targets.unsqueeze(1), self.confidence)

        return nn.CrossEntropyLoss()(inputs, targets)