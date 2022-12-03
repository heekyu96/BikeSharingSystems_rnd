from cmath import e
import os
import csv
import glob
from turtle import forward
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
from timeit import default_timer as timer

# https://eda-ai-lab.tistory.com/429?category=764743

# Utils
class _progressbar_printer:
    def __init__(self, runner_name, iterations, fill_length=45, fillout="█", CR=True):
        self.runner_name = runner_name
        self.iters = iterations + 1
        self.fill_len = fill_length
        self.fill = fillout
        self.line_end = {True: "\r", False: "\r\n"}
        self.print_end = self.line_end[CR]
        self.iter_time = 0.0
        self._progressBar(0)
        pass

    def _progressBar(self, cur, iter_t=0.0):
        cur = cur+1
        time_left = ("{0:.2f}").format(0)
        self.iter_time += iter_t
        time_spend = ("{0:.2f}").format(self.iter_time)
        if not iter_t == 0.0:
            time_left = ("{0:.2f}").format(self.iter_time / cur * self.iters)
            # time_spend = ("{0:.2f}").format(timer()-self.time_stamp)
        percent = ("{0:." + str(1) + "f}").format(100 * (cur / float(self.iters)))
        filledLength = int(self.fill_len * cur // self.iters)
        bar = self.fill * filledLength + "-" * (self.fill_len - filledLength)
        print(
            f"\r{self.runner_name}\t|{bar}| {percent}% {time_spend}/{time_left}s",
            end=self.print_end,
        )
        # Print New Line on Complete
        if cur == self.iters:
            print("\nCompleted\n")

torch.manual_seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\n" + str(DEVICE) + " is working...\n")

file_count = 10

# Historical station dataset processing
dir_path = "./00.datasets/chicago_bike_dataset/use_historical/*.csv"
station_list_hitorical = []
data_dict = {}
data_sub_dict = {}
max_val = 0
for i,file in enumerate(sorted(glob.glob(dir_path))):
    # print(file)
    if i>file_count:
        break
    file = open(file, "r", encoding="utf-8")
    csv_reader = csv.reader(file)
    # next(csv_reader)
    for j, line in enumerate(csv_reader):
        """
        [0] ID,
        [1] Timestamp,
        [2] Station Name,
        [3] Address,
        [4] Total Docks,
        [5] Docks in Service,
        [6] Available Docks,
        [7] Available Bikes,
        [8] Percent Full,
        [9] Status,
        [10-12] Latitude,Longitude,Location,
        [13] Record
        """
        if max_val < int(line[7]):
            max_val = int(line[7])
        
        station = line[2].replace(" ","").lower()
        if not station_list_hitorical.__contains__(station):
            station_list_hitorical.append(station)
            data_sub_dict[station] = int(line[4])
        
        if not data_dict.__contains__(station):
            # data_dict[station]=[float(int(line[8])/100)]
            data_dict[station]=[int(line[7])]
            # data_sub_dict[station] = [int(line[4])]
        else:
            # data_dict[station].append(float(int(line[8])/100))
            data_dict[station].append(int(line[7]))
            # data_sub_dict[station] = [int(line[4])]
    
    list_max = max([len(data_dict[d]) for d in data_dict])
    for d in data_dict:
        len_ = len(data_dict[d])
        if len_<list_max:
            for i in range(list_max-len_):
                # add padding
                data_dict[d].insert(0, 0.0001)
print(len(station_list_hitorical)) # the number of stations : (10files)

# User trip dataset processing
dir_path = "./00.datasets/chicago_bike_dataset/use_trip/*.csv"

trip_data_load_progress = _progressbar_printer("Trip data", iterations=file_count)
# trip_data_load_progress = _progressbar_printer("Trip data", iterations=len(glob.glob(dir_path)))

station_count = len(station_list_hitorical)
trip_maps  = []
cur_map = [[0 for x in range(station_count)] for x in range(station_count)]
cur_time = 0
for i,file in enumerate(sorted(glob.glob(dir_path))):
    if i>file_count:
        break
    file = open(file, "r", encoding="utf-8")
    csv_reader = csv.reader(file)
    # next(csv_reader)
    start_time = timer()
    for j, line in enumerate(csv_reader):
        """
        [0] TRIP ID,
        [1] START TIME,
        [2] STOP TIME,
        [3] BIKE ID,
        [4] TRIP DURATION,
        [5] FROM STATION ID,
        [6] FROM STATION NAME,
        [7] TO STATION ID,
        [8] TO STATION NAME,
        [9-11] USER TYPE,GENDER,BIRTH YEAR,
        [12-14] FROM LATITUDE,FROM LONGITUDE,FROM LOCATION,
        [15-17] TO LATITUDE,TO LONGITUDE,TO LOCATION
        """
        station_s = line[6].replace(" ","").lower()
        station_e = line[8].replace(" ","").lower()
        
        if not station_list_hitorical.__contains__(station_s):
            continue
        elif not station_list_hitorical.__contains__(station_e):
            continue
        
        s_idx = station_list_hitorical.index(station_s)
        e_idx = station_list_hitorical.index(station_e)
        
        s_time_stamp = datetime.datetime.strptime(line[1],'%m/%d/%Y %I:%M:%S %p')
        # e_time_stamp = datetime.datetime.strptime(line[2],'%m/%d/%Y %I:%M:%S %p')
        if not cur_time == s_time_stamp.hour:
            trip_maps.append(cur_map)
            cur_map = [[0 for x in range(station_count)] for x in range(station_count)]
            cur_time = int(s_time_stamp.hour)
        else:
            cur_map[s_idx][e_idx] +=1

    end_time = timer()
    trip_data_load_progress._progressBar(i, end_time - start_time)
print(len(trip_maps))


# Dataset GEN
data_list = []
for d in data_dict:
    data_dict[d] = [x/max_val for x in data_dict[d]]
    data_list.append(data_dict[d])

data_np_arr = numpy.asarray(data_list)
# print(data_np_arr.shape)
# print(len(data_np_arr[0]))
# print(data_np_arr[0])
data_np_arr=numpy.transpose(data_np_arr)
# print(len(data_np_arr[0]))
# print(data_np_arr[0])
# print(data_np_arr.shape)
data_list = data_np_arr.tolist()

station_count = len(station_list_hitorical)


def dataset_(list_, input_len, output_len, train_rat=0.8):
    train_x_ = []
    train_y_ = []
    test_x_ = []
    test_y_ = []
    
    x_ = []
    y_ = []
    
    for i in range(0, len(list_)-input_len):
        x_.append(list_[i:i+input_len,:])
        y_.append(list_[i+input_len:i+input_len+output_len,:])
        # x_.__add__(np_arr[i:i+input_len])
        # y_.__add__(np_arr[i+input_len:i+input_len+output_len])
    
    # print(len(list_))
    # print(len(x_))
    # print(len(y_))
    # print(len(x_[0]))
    # print(len(y_[0]))
    
    train_len = int(len(x_)*train_rat)
    train_x_= x_[0:train_len]
    train_y_ = y_[0:train_len]
    test_x_= x_[train_len:-1]
    test_y_= y_[train_len:-1]
    
    # Categorical Output
    # under  [1,0,0]
    # normal [0,1,0]
    # upper  [0,0,1]
    
    under_threshold = 0.05
    upper_threshold = 0.95

    target_label  = numpy.array(train_y_)[:,0,:]
    target_label = numpy.transpose(target_label)
    label4result = [[0]*(len(target_label)) for i in range(station_count)]
    under_count = 0
    upper_count = 0

    target_label = target_label.tolist()
    for i,station in enumerate(target_label):
        for j,label in enumerate(station):
            percentage = label*max_val/data_sub_dict[station_list_hitorical[i]]
            # if percentage<=under_threshold or percentage >=upper_threshold:
            #     eval_binary_label[i][j] = True
            if percentage<=under_threshold:
                label4result[i][j] = [1.,0.,0.]
                under_count+=1
            elif percentage >=upper_threshold:
                label4result[i][j] = [0.,0.,1.]
                upper_count+=1
            else:
                label4result[i][j] = [0.,1.,0.]
    
    # return numpy.array(train_x_),numpy.array(train_y_),numpy.array(test_x_),numpy.array(test_y_)
    return numpy.array(train_x_),numpy.array(label4result),numpy.array(test_x_),numpy.array(test_y_)

# Model Construction
input_day = 7

data_dim = station_count
input_length = input_day*24
target_length = 1

hidden_dim = 256
output_dim = (data_dim,3)
# input_unit = input_day*24
# Data Loading Test log
# print(len(data_list))
# for d in data_list:
#     print(d+" "+str(len(data_list[d])))


# input_size = input_day*24 # 입력의 크기, hours
# hidden_size = station_count # 은닉 상태의 크기, 입력별 들어가는 데이터의 수
# # output_size = 1 # hours
# output_size = station_count
# target_length = 1 # hours

train_x, train_y, test_x, test_y = dataset_(data_np_arr,input_length,target_length)

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

train_x_tensor = torch.FloatTensor(train_x).to(DEVICE)
train_y_tensor = torch.FloatTensor(train_y).to(DEVICE)
train_y_tensor = torch.FloatTensor(train_y[:,0,:]).to(DEVICE)
test_x_tensor = torch.FloatTensor(test_x).to(DEVICE)
test_y_tensor = torch.FloatTensor(test_y[:,0,:]).to(DEVICE)

class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers=1):
        super(Net, self).__init__()
        self.cells = nn.LSTM(input_dim, 512, num_layers = layers, batch_first = True)
        self.fc1= nn.Linear(512, 256, bias=True)
        self.fc2= nn.Linear(256, 128, bias=True)
        self.fc3= nn.Linear(128, output_dim, bias=True)
        
    def forward(self, x):
        x, _status = self.cells(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x[:,-1])
        return x
        
model = Net(data_dim,hidden_dim,output_dim,1)
model.cuda()

# class CNN(nn.Module):
#       def __init__(self):
#     super(CNN, self).__init__()
#     self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5, stride=1)
#     self.conv2 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5, stride=1)
#     self.fc1 = nn.Linear(10 * 12 * 12, 50)
#     self.fc2 = nn.Linear(50, 10)
  
#   def forward(self, x):
#     print("연산 전", x.size())
#     x = F.relu(self.conv1(x))
#     print("conv1 연산 후", x.size())
#     x = F.relu(self.conv2(x))
#     print("conv2 연산 후",x.size())
#     x = x.view(-1, 10 * 12 * 12)
#     print("차원 감소 후", x.size())
#     x = F.relu(self.fc1(x))
#     print("fc1 연산 후", x.size())
#     x = self.fc2(x)
#     print("fc2 연산 후", x.size())
#     return x

# cnn = CNN()
# output = cnn(torch.randn(10, 1, 20, 20))  # Input Size: (10, 1, 20, 20)

loss_function = nn.CrossEntropyLoss()
# loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



iterations = 500
for i in range(iterations):
    optimizer.zero_grad()
    outputs = model(train_x_tensor)
    # print(outputs.data.shape)k
    loss = loss_function(outputs,train_y_tensor)
    loss.backward()
    optimizer.step()
    print(i, loss.item())


# Evaluation
under_threshold = 0.05
upper_threshold = 0.95

target_label  = test_y[:,0,:]
target_label = numpy.transpose(target_label)
label4result = [[0]*(len(test_x)) for i in range(station_count)]
under_count = 0
upper_count = 0

target_label = target_label.tolist()
for i,station in enumerate(target_label):
    for j,label in enumerate(station):
        percentage = label*max_val/data_sub_dict[station_list_hitorical[i]]
        # if percentage<=under_threshold or percentage >=upper_threshold:
        #     eval_binary_label[i][j] = True
        if percentage<=under_threshold:
            label4result[i][j] = -1
            under_count+=1
        elif percentage >=upper_threshold:
            label4result[i][j] = 1
            upper_count+=1

predicted = model(test_x_tensor).cpu().data.numpy()
predicted = numpy.transpose(predicted)
pred4result = [[0]*(len(test_x)) for i in range(station_count)]

predicted = predicted.tolist()
for i,station in enumerate(predicted):
    # print(station)
    # print(data_sub_dict[station])
    # data_sub_dict[station]
    for j,pred in enumerate(station):
        percentage = pred*max_val/data_sub_dict[station_list_hitorical[i]]
        # if percentage<=under_threshold or percentage >=upper_threshold:
        #     eval_binary_pred[i][j] = True
        if percentage<=under_threshold:
            pred4result[i][j] = -1
        if percentage >=upper_threshold:
            pred4result[i][j] = 1

total = 0
correct = 0
under_correct = 0
upper_correct = 0

for i,station in enumerate(predicted):
    for j,label in enumerate(station):
        if label4result[i][j]==pred4result[i][j]:
            if label4result[i][j]==1:
                upper_correct+=1
            elif label4result[i][j]==-1:
                under_correct+=1
            correct+=1
        total+=1
        
print(correct," / ", total)
print(round((correct/total)*100,2))
print(round((under_correct/under_count)*100,2))
print(round((upper_correct/upper_count)*100,2))

#TODO
'''
True/False ratio ? => Precison, recall, F1-score
'''