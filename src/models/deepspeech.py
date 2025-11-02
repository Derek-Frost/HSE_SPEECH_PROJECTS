import torch
import torch.nn as nn

class DS2(nn.Module):
    def __init__(self, n_mels=80, rnn_layers=5, hidden=768, num_classes=29):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, (11,41), stride=(2,2), padding=(5,20)), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32,32,(11,21), stride=(2,2), padding=(5,10)), nn.BatchNorm2d(32), nn.ReLU(),
        )
        rnn_in = (n_mels//4)*32  # после двух conv stride=2
        self.rnn = nn.GRU(rnn_in, hidden, num_layers=rnn_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden*2, num_classes)
        self.num_classes = num_classes

    def forward(self, x, x_lens):
        x = x.unsqueeze(1)
        x = self.conv(x)
        B,C,Mp,Tp = x.shape
        x = x.permute(0,3,1,2).contiguous().view(B,Tp,C*Mp)
        lengths = (x_lens//4).cpu()
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x,_ = self.rnn(x)
        x,_ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        logits = self.fc(x)
        return logits