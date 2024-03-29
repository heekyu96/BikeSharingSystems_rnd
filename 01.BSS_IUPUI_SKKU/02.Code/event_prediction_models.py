import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.manual_seed(2024)
np.random.seed(2024)


class Prediction_Model_v1_LSTM(nn.Module):
    def __init__(
        self,
        unit_input_dim,
        unit_hidden_dim,
        unit_output_dim,
        dropout=0.0,
        bidirectional=False,
    ):
        nn.Module.__init__(self)
        self.model_name = "v1_LSTM"
        self.input_dim = unit_input_dim
        self.hidden_dim = unit_hidden_dim
        self.num_direction = int(bidirectional) + 1
        self.rnn = torch.nn.LSTM(
            unit_input_dim,
            unit_hidden_dim,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.fc = torch.nn.Linear(unit_hidden_dim, unit_output_dim)

    def forward(self, input):
        # input shape: batch, seq, dim
        # output = input
        rnn_output, _ = self.rnn(input)
        output = self.fc(rnn_output[:, -1])
        return output.unsqueeze(1)

# 
class Prediction_Model_v2_LSTM(nn.Module):
    def __init__(
        self,
        unit_input_dim,
        unit_hidden_dim,
        unit_output_dim,
        dropout=0.0,
        bidirectional=False,
        station_cnt=None
    ):
        nn.Module.__init__(self)
        self.model_name = "v2_LSTM"
        self.input_dim = unit_input_dim
        self.hidden_dim = unit_hidden_dim
        self.output_dim = unit_output_dim
        self.num_direction = int(bidirectional) + 1
        self.rnn = torch.nn.LSTM(
            unit_input_dim,
            unit_hidden_dim,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.fc = torch.nn.Linear(unit_hidden_dim, unit_output_dim)
        self.fc_cate = torch.nn.Linear(unit_hidden_dim, (unit_output_dim * 3))
        self.station = station_cnt

    def forward(self, input):
        # input shape: batch, seq, dim
        # output = input
        rnn_output, _ = self.rnn(input)
        output = self.fc(rnn_output[:, -1])
        output = output.unsqueeze(1)
        output_cate = self.fc_cate(rnn_output[:, -1])
        output_cate = output_cate.reshape(-1,self.output_dim,3)
        # output_cate = output_cate.reshape(-1, self.output_dim, 3)
        # return output, F.softmax(output_cate, dim=2)
        return output, output_cate


class Prediction_Model_v3_LSTM(nn.Module):
    def __init__(
        self,
        unit_input_dim,
        unit_hidden_dim,
        unit_output_dim,
        unit_cate_output_dim,
        dropout=0.0,
        bidirectional=False
    ):
        nn.Module.__init__(self)
        self.model_name = "v3_LSTM"
        self.input_dim = unit_input_dim
        self.hidden_dim = unit_hidden_dim
        self.output_dim = unit_output_dim
        self.cate_output_dim = unit_cate_output_dim
        self.num_direction = int(bidirectional) + 1
        self.rnn = torch.nn.LSTM(
            unit_input_dim,
            unit_hidden_dim,
            num_layers=1,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.fc = torch.nn.Linear(unit_hidden_dim, unit_output_dim)
        self.fc_cate = torch.nn.Linear(unit_output_dim, (unit_output_dim * unit_cate_output_dim))

    def forward(self, input):
        # input shape: batch, seq, dim(internal vector dim)
        rnn_output, _ = self.rnn(input)
        output = self.fc(rnn_output[:, -1])
        output_cate = self.fc_cate(output)
        
        output = output.unsqueeze(1)
        output_cate = output_cate.reshape(-1,self.output_dim,self.cate_output_dim) # 3 refers # of event classes
        return output, output_cate

class Prediction_Model_v4_GRU(nn.Module):
    def __init__(
        self,
        unit_input_dim,
        unit_hidden_dim,
        unit_output_dim,
        dropout=0.0,
        bidirectional=False
    ):
        nn.Module.__init__(self)
        self.model_name = "v4_GRU"
        self.input_dim = unit_input_dim
        self.hidden_dim = unit_hidden_dim
        self.output_dim = unit_output_dim
        self.num_direction = int(bidirectional) + 1
        self.rnn = torch.nn.GRU(
            unit_input_dim,
            unit_hidden_dim,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.fc = torch.nn.Linear(unit_hidden_dim, unit_output_dim)
        self.fc_cate = torch.nn.Linear(unit_output_dim, (unit_output_dim * 3))

    def forward(self, input):
        # input shape: batch, seq, dim(internal vector dim)
        self.fc(input)
        rnn_output, _ = self.rnn(input)
        
        output = self.fc(rnn_output[:, -1])
        output_cate = self.fc_cate(output)
        
        output = output.unsqueeze(1)
        output_cate = output_cate.reshape(-1,self.output_dim,3) # 3 refers # of event classes
        return output, output_cate

class Prediction_Model_v5_LSTM_FFT(nn.Module):
    def __init__(
        self,
        unit_input_dim,
        unit_hidden_dim,
        unit_output_dim,
        unit_cate_output_dim,
        dropout=0.0,
        bidirectional=False
    ):
        nn.Module.__init__(self)
        self.model_name = "v5_LSTM_FFT"
        self.input_dim = unit_input_dim
        self.hidden_dim = unit_hidden_dim
        self.output_dim = unit_output_dim
        self.cate_output_dim = unit_cate_output_dim
        self.fft_feature_n = 10
        self.num_direction = int(bidirectional) + 1
        self.rnn = torch.nn.LSTM(
            unit_input_dim,
            unit_hidden_dim,
            num_layers=1,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.fc_fft = torch.nn.Linear(self.input_dim*self.fft_feature_n,self.hidden_dim)
        self.fc_cat = torch.nn.Linear(self.hidden_dim*2,self.hidden_dim)
        self.fc = torch.nn.Linear(unit_hidden_dim, unit_output_dim)
        self.fc_cate = torch.nn.Linear(unit_output_dim, (unit_output_dim * unit_cate_output_dim))
        self.fc_out = torch.nn.Linear(unit_hidden_dim, unit_output_dim)
    def forward(self, input, fft):
        # input shape: batch, seq, dim(internal vector dim)
        rnn_output, _ = self.rnn(input)
        fft_output = self.fc_fft(fft.reshape(-1,self.input_dim*self.fft_feature_n))
        
        # output = self.fc(rnn_output[:, -1])
        # output = self.fc(output)
        cat_output = self.fc_cat(torch.cat((fft_output,rnn_output[:, -1]),dim=-1))
        
        # output = self.fc(output)
        output = self.fc_out(cat_output)
        
        output_cate = self.fc_cate(output)
        
        output = output.unsqueeze(1)
        output_cate = output_cate.reshape(-1,self.output_dim,self.cate_output_dim) # 3 refers # of event classes
        return output, output_cate