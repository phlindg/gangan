
import torch
import torch.nn as nn
import torch.nn.functional as F

class Predictor(nn.Module):
    """
    Ska predicta sequences beroende av tidigare data. Ska bytas ut mot ett GAN
    """
    def __init__(self, input_size, batch_size,output_size=1):
        """
        input_size = ()
        """
        super(Predictor, self).__init__()
        self.input_size = input_size #features
        self.hidden_size = 128 #noder i layer
        self.batch_size = batch_size #antal tidsstep
        self.num_layers = 1 #ska va 1
        self.output_size = output_size #forecast horizon

        self.lstm1 = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, dropout=1, batch_first=True)
        self.lstm2 = nn.LSTM(self.input_size, self.hidden_size//4, self.num_layers, dropout=1, batch_first=True)
        self.linear = nn.Linear(self.hidden_size//4, self.output_size)
        self.hidden = self.init_hidden()
        self.state = self.init_hidden()
    def init_hidden(self):
        z = torch.randn(self.num_layers, self.batch_size, self.hidden_size)
        return (z,z)
    def forward(self, input):
        output, self.hidden = self.lstm1(input, self.hidden)
        output = F.dropout(output, p=0.5, training=True)
        output, self.state = self.lstm2(output, self.state)
        output = F.dropout(output, p=0.5, training=True)
        output = self.linear(self.state[0].squeeze(0))
        return output

