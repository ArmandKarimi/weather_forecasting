import torch
import torch.nn as nn

class model_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first = True, num_layers = 2, dropout = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            dropout=dropout, 
                            batch_first=batch_first)        
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x, (h,c) = self.lstm(x)
        x = x[:,-1,:]
        out = self.fc(x)

        return out