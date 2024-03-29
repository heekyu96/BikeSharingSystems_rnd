import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import math
import numpy as np
torch.manual_seed(2024)
np.random.seed(2024)

# %matplotlib inline
import matplotlib.pyplot as plt

# Confusion matrix
#https://tutorials.pytorch.kr/intermediate/char_rnn_classification_tutorial

class SinusDataset(Dataset):
    def __init__(self,input_seq_len, output_seq_len, dataset_size, train_or_test):
        self.input_seq = input_seq_len
        self.output_seq = output_seq_len
        self.dataset_size = dataset_size
        self.train_or_test = train_or_test
        self.division_ratio = 0.8
        self.seq_x, self.seq_y,self.plotx = self.generate_sinus_wave_seq(self.input_seq+self.output_seq,self.dataset_size)
        self.re_seq_x,self.re_seq_y,self.re_plotx = self.train_test_division()
        print(self.re_seq_x.shape)
        print(self.re_seq_y.shape)
        
    def generate_sinus_wave_seq(self,seq_len, num_total):
        time_steps = np.linspace(0, 8*np.pi, num_total)
        data = np.sin(time_steps)

        seqs_x = []
        seqs_y = []
        plot_x = []
        for i in range(len(data)-seq_len):
            seqs_x.append(data[i:i+self.input_seq])
            seqs_y.append(data[i+self.input_seq])
            plot_x.append(time_steps[i:i+seq_len])
        
        return seqs_x,seqs_y,plot_x
    
    def train_test_division(self):
        train_len = int(self.dataset_size*self.division_ratio)
        
        if self.train_or_test =="train":
            seqs_x = self.seq_x[0:train_len]
            seqs_y = self.seq_y[0:train_len]
            plotx = self.plotx[0:train_len]
            return torch.FloatTensor(np.array(seqs_x)).unsqueeze(2),torch.FloatTensor(np.array(seqs_y)).unsqueeze(1),plotx
        else:
            self.re_seq_x = torch.FloatTensor(np.array(self.seq_x[train_len:-1])).unsqueeze(2)
            self.re_seq_y = torch.FloatTensor(np.array(self.seq_y[train_len:-1])).unsqueeze(1)
            self.re_plotx = self.plotx[train_len:-1]
            return None
    
    def train_test_transfer(self):
        if self.train_or_test =="train":
            self.train_or_test = "test"
            print("Time to Test")
        elif self.train_or_test =="test":
            self.train_or_test = "train"
            print("Time to Train")
        self.train_test_division()

    def __len__(self):
        return len(self.re_seq_x)

    def __getitem__(self, index):
        return self.re_seq_x[index],self.re_seq_y[index]
    
    def getD4Plot(self):
        return self.re_seq_x.cpu().data.numpy(),self.re_seq_y.cpu().data.numpy(),self.re_plotx


class RNNs(nn.Module):
    def __init__(self, input_size, hidden_size,output_size, dropout=0, bidirectional=False):
        nn.Module.__init__(self)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1
        self.rnn = torch.nn.LSTM(input_size, hidden_size, num_layers=1,dropout=dropout, batch_first=True, bidirectional=bidirectional)
        self.fc = torch.nn.Linear(hidden_size,output_size)
        
    def forward(self, input):
        # input shape: batch, seq, dim
        # output = input
        rnn_output, _ = self.rnn(input)
        # print(rnn_output.shape)
        # rnn_output.view(-1,1)
        # print(rnn_output.shape)
        # print(len(rnn_output))
        # print(len(rnn_output[0]))
        # test = rnn_output[:,-1]
        # print(len(test))
        # print(len(test[0]))
        output = self.fc(rnn_output[:,-1])
        # output = self.fc(rnn_output)
        return output
    
input_seq_len = 15
output_seq_len = 1
batch_size = 128
epoch =100
total_size = 3000

train_dataset = SinusDataset(input_seq_len,output_seq_len,total_size,"train")
train_loader = DataLoader(train_dataset,batch_size)

input_, output_, plotx_ = train_dataset.getD4Plot()
# print(input_[0])
# print(len(input_[0]))
# print(plotx_[0])
# print(len(plotx_[0]))

#plot our data
# fig, ax = plt.subplots(figsize=(20,5))
# plt.scatter(plotx_[0][0:input_seq_len], input_.tolist()[0], s=90, label='input')
# plt.scatter([x[-1] for x in plotx_], output_.tolist(), label='output')
# # plt.scatter(, data[train_len:], label='valid')
# ax.legend()
# plt.show()

# model = RNNs(input_seq_len,256,output_seq_len)
model = RNNs(1,256,output_seq_len)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
for i in range(epoch):
    losses = 0
    for batch_idx, data in enumerate(train_loader):
        tr_x, tr_y= data
        optimizer.zero_grad()
        outputs = model(tr_x)
        # print(len(outputs))
        # print(len(tr_y))
        # outputs = model(train_x_tensor,train_xdt_tensor)
        loss = loss_function(outputs, tr_y)
        # loss = loss_function(outputs,train_y_tensor)
        loss.backward()
        optimizer.step()
        losses += loss.item()
    print(i, losses / len(train_loader))
    
train_dataset.train_test_transfer()
test_loader = DataLoader(train_dataset,1)

model.eval()
predictions = []

for batch_idx, data in enumerate(test_loader):
    tr_x, tr_y= data
    # optimizer.zero_grad()
    outputs = model(tr_x)
    # outputs = model(train_x_tensor,train_xdt_tensor)
    # loss = loss_function(outputs, tr_y)
    # loss = loss_function(outputs,train_y_tensor)
    # loss.backward()
    # optimizer.step()
    # losses += loss.item()
    predictions+=outputs.data.numpy().tolist()
# print(i, losses / len(train_loader))

print(len(predictions))
print(len(test_loader))

input_, output_, plotx_ = train_dataset.getD4Plot()
# print(input_[0])
# print(len(input_[0]))
# print(plotx_[0])
# print(len(plotx_[0]))

#plot our data
fig, ax = plt.subplots(figsize=(20,5))
# plt.scatter(plotx_[0][0:input_seq_len], input_.tolist()[0], s=90, label='input')
plt.scatter([x[-1] for x in plotx_], output_.tolist(), label='target')
plt.scatter([x[-1] for x in plotx_], predictions, s=90,label='output')
# plt.scatter(, data[train_len:], label='valid')
ax.legend()
plt.show()