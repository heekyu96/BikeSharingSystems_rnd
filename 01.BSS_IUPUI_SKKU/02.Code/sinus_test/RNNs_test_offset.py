import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import math
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(2024)
np.random.seed(2024)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\n" + str(DEVICE) + " is working...\n")


# Confusion matrix
#https://tutorials.pytorch.kr/intermediate/char_rnn_classification_tutorial

MAX_OFFSET = 10
class SinusDataset(Dataset):
    def __init__(self,input_seq_len, output_seq_len, offset,dataset_size, train_or_test):
        self.input_seq = input_seq_len
        self.output_seq = output_seq_len
        self.offset = offset-1
        self.dataset_size = dataset_size
        self.train_or_test = train_or_test
        self.division_ratio = 0.8
        self.seq_x, self.seq_dt, self.seq_y,self.plotx = self.generate_sinus_wave_seq(self.input_seq+self.output_seq,self.dataset_size)
        self.re_seq_x,self.re_seq_dt,self.re_seq_y,self.re_plotx = self.train_test_division()
        print(self.re_seq_x.shape)
        print(self.re_seq_y.shape)
        
    def generate_sinus_wave_seq(self,seq_len, num_total):
        time_steps = np.linspace(0, 8*np.pi, num_total)
        data = np.sin(time_steps)

        seqs_x = []
        seqs_dt = []
        seqs_y = []
        plot_x = []
        for i in range(len(data)-seq_len-self.offset):
            seqs_x.append(data[i:i+self.input_seq])
            seqs_dt.append(self.offset+1/MAX_OFFSET)
            seqs_y.append(data[i+self.input_seq+self.offset])
            plot_x.append(time_steps[i:i+self.input_seq+self.offset])
            # plot_x[-1].append(time_steps[i+self.input_seq+self.offset])
        
        return seqs_x,seqs_dt,seqs_y,plot_x
    
    def train_test_division(self):
        train_len = int(self.dataset_size*self.division_ratio)
        
        if self.train_or_test =="train":
            seqs_x = self.seq_x[0:train_len]
            seqs_dt = self.seq_dt[0:train_len]
            seqs_y = self.seq_y[0:train_len]
            plotx = self.plotx[0:train_len]
            return torch.FloatTensor(np.array(seqs_x)).unsqueeze(2).to(DEVICE),torch.FloatTensor(np.array(seqs_dt)).unsqueeze(1).to(DEVICE),torch.FloatTensor(np.array(seqs_y)).unsqueeze(1).to(DEVICE),plotx
        else:
            self.re_seq_x = torch.FloatTensor(np.array(self.seq_x[train_len:-1])).unsqueeze(2).to(DEVICE)
            self.re_seq_y = torch.FloatTensor(np.array(self.seq_dt[train_len:-1])).unsqueeze(1).to(DEVICE)
            self.re_seq_y = torch.FloatTensor(np.array(self.seq_y[train_len:-1])).unsqueeze(1).to(DEVICE)
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
        return self.re_seq_x[index],self.re_seq_dt[index],self.re_seq_y[index]
    
    def getD4Plot(self):
        return self.re_seq_x.cpu().data.numpy(),self.re_seq_dt.cpu().data.numpy(),self.re_seq_y.cpu().data.numpy(),self.re_plotx


class RNNs(nn.Module):
    def __init__(self, input_size, hidden_size,output_size, dropout=0, bidirectional=False):
        nn.Module.__init__(self)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1
        self.rnn = torch.nn.LSTM(input_size, hidden_size, num_layers=1,dropout=dropout, batch_first=True, bidirectional=bidirectional)
        self.fc = torch.nn.Linear(hidden_size,hidden_size)
        self.fc_out = torch.nn.Linear(hidden_size,output_size)
        self.cat = torch.nn.Linear(hidden_size+1,hidden_size)
        self.dt = torch.nn.Linear(1,1)
        # self.cat = torch.cat()
        
    def forward(self, input,dt):
        # input shape: batch, seq, dim
        # output = input
        rnn_output, _ = self.rnn(input)
        # rnn_output = self.fc(rnn_output[:,-1])
        
        dt = self.dt(dt)
        
        output = torch.cat((rnn_output[:,-1],dt),dim=1)
        output = self.cat(output)
        output = self.fc_out(output)
        # print(rnn_output.shape)
        # rnn_output.view(-1,1)
        # print(rnn_output.shape)
        # print(len(rnn_output))
        # print(len(rnn_output[0]))
        # test = rnn_output[:,-1]
        # print(len(test))
        # print(len(test[0]))
        # output = self.fc(rnn_output)
        return output
    
input_seq_len = 15
output_seq_len = 1
batch_size = 128
epoch =100
total_size = 3000

datasets = []
loaders = []
for i in range(MAX_OFFSET):
    datasets.append(SinusDataset(input_seq_len,output_seq_len,i+1,total_size,"train"))
    loaders.append(DataLoader(datasets[-1],batch_size))
# train_dataset = SinusDataset(input_seq_len,output_seq_len,2,total_size,"train")
# train_loader = DataLoader(train_dataset,batch_size)

for dset in datasets:
    input_, dt_,output_, plotx_ = dset.getD4Plot()
    
# input_, dt_,output_, plotx_ = train_dataset.getD4Plot()
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
model.cuda()

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
for train_loader in loaders:
    for i in range(epoch):
        losses = 0
        for batch_idx, data in enumerate(train_loader):
            tr_x, tr_dt, tr_y= data
            optimizer.zero_grad()
            outputs = model(tr_x,tr_dt)
            # print(len(outputs))
            # print(len(tr_y))
            # outputs = model(train_x_tensor,train_xdt_tensor)
            loss = loss_function(outputs, tr_y)
            # loss = loss_function(outputs,train_y_tensor)
            loss.backward()
            optimizer.step()
            losses += loss.item()
        print(i, losses / len(train_loader))

test_loaders = []
for test_data in datasets:
    test_data.train_test_transfer()
    test_loaders.append(DataLoader(test_data,1))
    
# train_dataset.train_test_transfer()
# test_loader = DataLoader(train_dataset,1)

model.eval()
predictions = []

for test_loader in test_loaders:
    pred = []
    for batch_idx, data in enumerate(test_loader):
        tr_x, tr_dt, tr_y= data
        # optimizer.zero_grad()
        outputs = model(tr_x,tr_dt)
        # outputs = model(train_x_tensor,train_xdt_tensor)
        # loss = loss_function(outputs, tr_y)
        # loss = loss_function(outputs,train_y_tensor)
        # loss.backward()
        # optimizer.step()
        # losses += loss.item()
        pred+=outputs.cpu().data.numpy().tolist()
    predictions.append(pred)
# print(i, losses / len(train_loader))

print(len(predictions))
print(len(test_loader))

offset_plotx = []
offset_inp = []
offset_outp = []
offset_pred = []
path_ = "./04.event_prediction/sinus_test/results/"
for offset,test_dataset in enumerate(datasets):
    input_, dt_, output_, plotx_ = test_dataset.getD4Plot()
    # print(input_[0])
    # print(len(input_[0]))
    # print(plotx_[0])
    # print(len(plotx_[0]))

    #plot our data
    fig, ax = plt.subplots(figsize=(20,5))
    # plt.scatter(plotx_[0][0:input_seq_len], input_.tolist()[0], s=90, label='input')
    plot4x = [x[-1] for x in plotx_]
    plt.scatter(plot4x, output_.tolist(), s=90,label='target')
    plt.scatter(plot4x, predictions[offset], label='output')
    offset_inp = (plotx_[1][0:input_seq_len], input_.tolist()[1])
    offset_plotx.append(plot4x[1])
    offset_outp.append(output_.tolist()[1])
    offset_pred.append(predictions[offset][1])
    # plt.scatter(, data[train_len:], label='valid')
    ax.legend()
    # plt.show()
    plt.savefig(
        path_ + str(offset)+".png",
        dpi=500,
        edgecolor="white",
        bbox_inches="tight",
        pad_inches=0.2,
    )

fig, ax = plt.subplots(figsize=(20,5))
plt.scatter(offset_inp[0], offset_inp[1], s=90,label='input')
plt.scatter(offset_plotx, offset_outp, s=90,label='target')
plt.scatter(offset_plotx, offset_pred, label='output')
# plt.scatter(, data[train_len:], label='valid')
ax.legend()
# plt.show()
plt.savefig(
    path_ + "offset_test.png",
    dpi=500,
    edgecolor="white",
    bbox_inches="tight",
    pad_inches=0.2,
)