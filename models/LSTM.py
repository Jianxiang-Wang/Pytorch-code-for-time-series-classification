'''LSTM in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)  
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        self.lstm.flatten_parameters()
        out, _ = self.lstm(x)  
        out = out[:, -1, :]
        out = self.classifier(out)
        return out
