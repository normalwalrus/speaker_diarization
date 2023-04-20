import torch.nn as nn

class FeedForwardNN(nn.Module):

    def __init__(self,in_channel, output, dropout):
        super(FeedForwardNN, self).__init__()
        self.ln1 = nn.Linear(in_channel, 512)
        self.ln2 = nn.Linear(512, 512)
        self.ln3 = nn.Linear(512, output)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.ln1(x)
        x = self.dropout(x)
        x = self.ln2(x)
        x = self.dropout(x)
        x = self.ln3(x)

        return x
