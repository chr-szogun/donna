import torch
import torch.nn as nn

class LSTMWW(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, dropout, device="cpu"):
        super(LSTMWW, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.layernorm = nn.LayerNorm(feature_size)
        self.model = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def create_h0c0(self, batch_size):
        num_layers, hidden_size = self.num_layers, self.hidden_size
        h0, c0 = torch.zeros(num_layers, batch_size, hidden_size).to(self.device),
                torch.zeros(num_layers, batch_size, hidden_size).to(self.device)
        return (h0, c0)

    def forward(self, x):
        x = self.layernorm(x)
        h0c0 = self.create_h0c0(x.size()[1])
        out, (hn, cn) = self.model(x, h0c0)
        out = self.classifier(hn)
        return out