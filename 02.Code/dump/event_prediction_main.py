from cProfile import label
from cmath import inf
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import math
import numpy as np
import glob
import csv
from timeit import default_timer as timer
import matplotlib.pyplot as plt

from event_prediction_utils import _progressbar_printer
from event_prediction_utils import model_weight_path, plot_path
from event_prediction_data import *
from event_prediction_models import Prediction_Model_v4_GRU, Prediction_Model_v3_LSTM

torch.manual_seed(2024)
np.random.seed(2024)

# torch.manual_seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\n" + str(DEVICE) + " is working...\n")

file_count = 10


# Train Model or Use Trained Model Weight
train= False
# train = True

# Hyper-Params
epoch = 100
batch_size = 500
dropout_rate = 0.5

# Model Input and Output Shape
input_day = 7
output_day = 1
input_seq_len = input_day * 24 * 6
output_seq_len = output_day * 24 * 6
station_cnt_lock = -1

# Event Definition
# upper_bound = 0.90
# lower_bound = 0.10
# boundary = [0.00,0.10,0.25,0.50,0.75,0.90,1.00]
# boundary = [0.00,0.10,0.90,sys.maxsize/100]
boundary = [0.00,0.10,0.25,0.50,0.75,0.90,sys.maxsize/100]


# Dataset Generation
historical_station_dataset = StationDataset(
    file_count, input_seq_len, output_seq_len, boundary, DEVICE, station_cnt_lock
)

# Model Construction
model = Prediction_Model_v3_LSTM(
# model = Prediction_Model_v4_GRU(
    unit_input_dim=historical_station_dataset.station_count,
    unit_hidden_dim=256,
    unit_output_dim=historical_station_dataset.station_count,
    unit_cate_output_dim=len(boundary)-1,
    dropout=dropout_rate,
)
if DEVICE == torch.device("cuda"):  # Cuda Setup
    model.cuda()

# Loss func
loss_function = nn.MSELoss() # for Bike amount
loss_function_cate = nn.CrossEntropyLoss() # for Event
# Optimizer
max_norm = 5
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# Train or Load weights
if not train:
    model_weight = torch.load(
        model_weight_path
        + model.model_name
        + str(input_day * 24)
        + "_"
        + str(output_day * 24)
        + "_"
        # + str(101)
        + str(epoch)
        + ".pt",
        map_location=DEVICE,
    )
    model.load_state_dict(model_weight)
else: # Train
    historical_station_dataset.set_train()
    train_batch_size = batch_size
    # train_batch_size = batch_size * output_seq_len
    train_loader = DataLoader(historical_station_dataset, train_batch_size)
    
    model.train() # setup train mode
    
    train_progress = _progressbar_printer("Model Training", iterations=epoch) # status bar
    for i in range(epoch):
        losses = 0
        losses_cate = 0
        re = 0
        # output_cnt = 1
        start_time = timer()
        for batch_idx, data in enumerate(train_loader):
            # tr_x, tr_y = data
            tr_x, tr_y, tr_y_cate = data
            # print(tr_x.shape)
            optimizer.zero_grad()
            # outputs = model(tr_x)
            
            # bike amount output branch
            outputs, outputs_cate = model(tr_x)
            loss = loss_function(outputs, tr_y) # loss calculation

            # event classes output branch
            output_cate_dim = outputs_cate.shape[-1]
            outputs_cate = outputs_cate[:].view(-1, output_cate_dim)
            tr_y_cate = tr_y_cate[:].view(-1)
            loss_cate = loss_function_cate(outputs_cate, tr_y_cate) # loss calculation
            
            # Backpropagation
            loss.backward(retain_graph= True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            loss_cate.backward()
            
            optimizer.step()
            
            # Drop back-propagation tree and save only numerical loss value
            losses += loss.item()
            losses_cate += loss_cate.item()

        # Current status Printer
        print(
            "Epoch",
            i + 1,
            "MSE:",
            round(losses / len(train_loader), 8),
            "CE:",
            round(losses_cate / len(train_loader), 8),
            " " * 60,
        )
        if (i+1)%10==0:
            torch.save(
                model.state_dict(),
                model_weight_path
                + model.model_name
                + str(input_day * 24)
                + "_"
                + str(output_day * 24)
                + "_"
                + str(i+1)
                + ".pt",
            )
        end_time = timer()
        train_progress._progressBar(i+1, end_time - start_time)

    # Trained Weight Save
    # torch.save(
    #     model.state_dict(),
    #     model_weight_path
    #     + model.model_name
    #     + str(input_day * 24)
    #     + "_"
    #     + str(output_day * 24)
    #     + "_"
    #     + str(epoch)
    #     + ".pt",
    # )

# Test
test_batch_size = 1
historical_station_dataset.set_test()
test_loader = DataLoader(historical_station_dataset, test_batch_size)

# if train:
model.eval()

predictions = []  # 0:stations, 1:time-steps
predictions_cate = []  # 0:stations, 1:time-steps
labels = []  # 0:stations, 1:time-steps
labels_cate = []  # 0:stations, 1:time-steps
losses = 0
delta_losses = [0 for x in range(output_seq_len)]

test_progress = _progressbar_printer("Model Testing", iterations=len(test_loader))
for batch_idx, data in enumerate(test_loader):
    start_time = timer()

    tr_x, tr_y, tr_y_cate = data

    if batch_idx % output_seq_len == 0:
        outputs = []
        outputs_cate = []
        targets = []
        targets_cate = []
        cur = tr_x
        delta_loss = 0

    output_tensor,output_cate_tensor = model(tr_x)
    # 출력저장
    if DEVICE == torch.device("cuda"):
        output = output_tensor.cpu().squeeze().data
        output_cate = torch.argmax(output_cate_tensor.cpu().squeeze().data,dim=-1)
        # output = np.transpose(output) # 0:time-steps, 1:stations
        target = tr_y.cpu().squeeze().data  #
        target = np.transpose(target)  # 0:time-steps, 1:stations
        target_cate = tr_y_cate.cpu().squeeze()
    else:
        output = output_tensor.squeeze().data
        output_cate = torch.argmax(output_cate_tensor.squeeze().data,dim=-1)
        target = tr_y.squeeze().data
        target = np.transpose(target)  # 0:time-steps, 1:stations
        target_cate = tr_y_cate.squeeze()

    outputs.append(output)
    outputs_cate.append(output_cate)
    targets.append(target)
    targets_cate.append(target_cate)

    # get loss according to delta
    delta_loss = loss_function(output_tensor, tr_y)
    delta_losses[batch_idx % output_seq_len] += delta_loss.item()

    cur = torch.cat([cur[:, 1:, :], output_tensor.data], dim=1)

    if batch_idx % output_seq_len == output_seq_len - 1:
        predictions.append(outputs)
        predictions_cate.append(outputs_cate)
        labels.append(targets)
        labels_cate.append(targets_cate)

    loss = loss_function(output_tensor, tr_y)
    # loss = loss_function(outputs,train_y_tensor)
    losses += loss.item()
    end_time = timer()
    test_progress._progressBar(batch_idx+1, end_time - start_time)
# delta loss normalize
# divdr = len(delta_losses[0])
delta_losses = [x / (len(test_loader) / output_seq_len) for x in delta_losses]


print("Predicted sequences Count: ", len(predictions), " and labels: ", len(labels))
print(
    "Avg. Loss: ",
    round(losses / len(test_loader), 4),
    "\tconverted:",
    round(losses / len(test_loader) * historical_station_dataset.max_val, 4),
)

def root_mean_squared_error(y, t):
    l2 = []
    for x in zip(y,t):
        l2.append(((x[0]-x[1])**2)**0.5)
    return l2

# Ploting
predictions_plot = []  # tensor list
labels_plot = []  # tensor list
for data in zip(predictions, labels):
    if predictions_plot == []:
        predictions_plot = data[0]
    else:
        predictions_plot.append(data[0][-1])
    if labels_plot == []:
        labels_plot = data[1]
    else:
        labels_plot.append(data[1][-1])

predictions_plot2 = []  # tensor list
labels_plot2 = [] 
for data in zip(predictions, labels):
    predictions_plot2.append(data[0])
    labels_plot2.append(data[1])

predictions_plot = torch.cat(predictions_plot, dim=0).reshape(
    -1, historical_station_dataset.station_count
)  # time-step X station
labels_plot = torch.cat(labels_plot, dim=0).reshape(
    -1, historical_station_dataset.station_count
)

predictions_plot_byStation = np.transpose(predictions_plot.numpy())  # station x time-step
labels_plot_byStation = np.transpose(labels_plot.numpy()).tolist()

#FFT##################################


#loss calculation
losses_rmse = []
for xy in zip(predictions_plot_byStation,labels_plot_byStation):
    losses_rmse.append(root_mean_squared_error(xy[1],xy[0]))

losses_rmse_np_byStation = np.array(losses_rmse)
losses_rmse_np = np.transpose(losses_rmse_np_byStation)
losses_rmse = losses_rmse_np.tolist()
# input_, output_, plotx_ = train_dataset.getD4Plot()
# print(input_[0])
# print(len(input_[0]))
# print(plotx_[0])
# print(len(plotx_[0]))

# plot predicted bike amounts in all of stations
# upper_bound = [95.00 for x in range(len(predictions_plot_byStation[0]))]
# lower_bound = [5.00 for x in range(len(predictions_plot_byStation[0]))]
for i, data in enumerate(zip(predictions_plot_byStation, labels_plot_byStation)):
    if i == 2:
        break
    fig, ax1 = plt.subplots(figsize=(20, 5))

    # ax1.set_ylabel("Nomalized Bike Amount by Max Amount of All stations")
    # ax1.set_ylim([0.0, 1.0])
    plt.ylabel("Available Bike Amount (%)")
    plt.xlabel("Time-step (10 mins)")
    plt.text(
        0,
        -3,
        "Avg.loss: "
        + str(round(losses / len(test_loader), 4))
        + " converted:"
        + str(round(losses / len(test_loader) * historical_station_dataset.max_val, 4))
        + "(bikes)",
    )
    # plt.scatter(plotx_[0][0:input_seq_len], input_.tolist()[0], s=90, label='input')
    x_ = [x for x in range(len(data[0]))]

    max_val_ = historical_station_dataset.max_val
    total_dock_ = historical_station_dataset.station_total_dock[
        historical_station_dataset.station_list[i]
    ]
    data0_ = [round(x * max_val_ / total_dock_ * 100, 2) for x in data[0]]
    for i,x in enumerate(data0_): 
        if x<0: 
            data0_[i] = 0
    plt.scatter(
        x_,
        [round(x * max_val_ / total_dock_ * 100, 2) for x in data[1]],
        s=80,
        label="target",
    )
    plt.scatter(
        x_,
        # [round(x * max_val_ / total_dock_ * 100, 2) for x in data0_],
        data0_,
        label="output",
    )
    #boundary plots
    bound_plot_color = ["red","orange","gray","green","blue"]
    for l, bound in enumerate(boundary[1:-1]):
        plt.plot(x_, [bound for x in range(len(x_))],linestyle="--", color=bound_plot_color[l], label=str(bound))
    plt.legend()

    plt.savefig(
        "./04.event_prediction/event_prediction_results/"
        + str(historical_station_dataset.station_list[i])
        + "_"
        + str(input_day * 24)
        + "_"
        + str(output_day * 24)
        + "_"
        + str(epoch)
        + ".png",
        dpi=500,
        edgecolor="white",
        bbox_inches="tight",
        pad_inches=0.2,
    )

#LOSS############################################
fig, ax = plt.subplots(figsize=(20, 5))
ax.set_ylabel("Nomalized Bike Amount")
ax.set_ylim([0.0, 0.01])
ax.set_xlabel("Offset (10 mins interval)")
plt.grid(linestyle="--")

ax.scatter([x for x in range(len(delta_losses))], delta_losses,label="normalized loss")

ax.legend()

ax2 = ax.twinx()
ax2.set_ylabel("Bike Amount")
ax2.scatter([x for x in range(len(delta_losses))], [y*max_val_ for y in delta_losses],label="bike amount difference")
ax2.set_ylim([0.0, 1])
plt.savefig(
    "./04.event_prediction/event_prediction_results/"
    + "00.delta_loss"
    + "_"
    + str(input_day * 24)
    + "_"
    + str(output_day * 24)
    + "_"
    + str(epoch)
    + ".png",
    dpi=500,
    edgecolor="white",
    bbox_inches="tight",
    pad_inches=0.2,
)

#LOSS#scatter###########################################
fig, ax = plt.subplots(figsize=(20, 5))
plt.ylabel("Bike Amount Difference", fontsize = 12)

def mean_stddev_(error_list):
    stddev_list = []
    mean_list = []
    for errors in error_list:
        mean = sum(errors) / len(errors)
        variance = sum([((x - mean) ** 2) for x in errors]) / len(errors)
        res = variance ** 0.5
        
        mean_list.append(mean)
        stddev_list.append(res)
    return mean_list,stddev_list
plt.xlabel("Offset (10 mins interval)",fontsize = 12)

mean_,std_dev_ = mean_stddev_(losses_rmse)
plt.scatter([i for i in range(len(mean_))], mean_,s=50)
plt.errorbar([i for i in range(len(mean_))], mean_,yerr =std_dev_)

ax.legend()
plt.savefig(
    "./04.event_prediction/event_prediction_results/"
    + "00.delta_loss"
    + "_station_wise"
    + str(input_day * 24)
    + "_"
    + str(output_day * 24)
    + "_"
    + str(epoch)
    + ".png",
    dpi=500,
    edgecolor="white",
    bbox_inches="tight",
    pad_inches=0.2,
)

#Precision and Accuracy Preperation############################################
predictions_plot_cate = []  # tensor list
labels_plot_cate = []  # tensor list
for data in zip(predictions_cate, labels_cate):
    if predictions_plot_cate == []:
        predictions_plot_cate = data[0]
    else:
        predictions_plot_cate.append(data[0][-1])
    if labels_plot_cate == []:
        labels_plot_cate = data[1]
    else:
        labels_plot_cate.append(data[1][-1])
        
def acc_cal(pred, tar,num_labels = 3):
    cnt = [0 for x in range(num_labels)]
    correct = [0 for x in range(num_labels)]
    for xy in zip(pred,tar):
        cnt[xy[1]]+=1
        if xy[0]==xy[1]:
            correct[xy[1]]+=1
    # total acc, precision list
    acc = round(sum(correct)/sum(cnt)*100,2)
    prec = []
    for x in zip(correct, cnt):
        if x[1] != 0:
            prec.append(round(x[0]/x[1]*100,2))
        elif x[1] == 0:
            prec.append(round(x[0]/(x[1]+1)*100,2))
    return acc, prec
    
cate_acc = []
cate_precision_classes = [[] for x in range(len(boundary)-1)]
# cate_c1 = []
# cate_c2 = []
# cate_c3 = []
for set_ in  zip(predictions_plot_cate, labels_plot_cate):
    acc, prec = acc_cal(set_[0],set_[1],len(boundary)-1)
    cate_acc.append(acc)
    for c, prec_class in enumerate(cate_precision_classes):
        prec_class.append(prec[c])
    # cate_c1.append(prec[0])
    # cate_c2.append(prec[1])
    # cate_c3.append(prec[2])
for cate_c in cate_precision_classes:
    print(len(cate_c))
# print(len(cate_c1))
# print(len(cate_c2))
# print(len(cate_c3))

#ACCURACY############################################
fig, ax = plt.subplots(figsize=(20, 5))
plt.ylabel("Accuracy(%)", fontsize = 12)

plt.ylim([80, 105])
plt.xlabel("Offset (10 mins interval)",fontsize = 12)
# plt.scatter(plotx_[0][0:input_seq_len], input_.tolist()[0], s=90, label='input')
plt.plot([x for x in range(len(cate_acc))], cate_acc, label="overall acc")
plt.text(
        len(cate_acc),
        cate_acc[-1]+1,
        "Avg.Acc.: "
        + str(round(sum(cate_acc) / len(cate_acc),2)),
    )
plt.legend()
# plt.show()
plt.savefig(
    plot_path
    + "00.acc"
    + "_"
    + str(input_day * 24)
    + "_"
    + str(output_day * 24)
    + "_"
    + str(epoch)
    + ".png",
    dpi=500,
    edgecolor="white",
    bbox_inches="tight",
    pad_inches=0.2,
)

# Precisions
fig, ax = plt.subplots(figsize=(20, 5))
plt.ylabel("Precision(%)", fontsize = 12)

plt.ylim([0, 100])
plt.xlabel("Offset (10 mins interval)",fontsize = 12)
# plt.scatter(plotx_[0][0:input_seq_len], input_.tolist()[0], s=90, label='input')
labels = ["Lower","Normal", "Upper"]
for c, cate_c in enumerate(cate_precision_classes):
    plt.plot([x for x in range(len(cate_c))], cate_c, label=labels[c])
    plt.text(
            len(cate_c),
            cate_c[-1]+1,
            "Avg.Acc.: "
            + str(round(sum(cate_c) / len(cate_c),2)),
        )
plt.legend()
# plt.show()
plt.savefig(
    "./04.event_prediction/event_prediction_results/"
    + "00.precisions"
    + "_"
    + str(input_day * 24)
    + "_"
    + str(output_day * 24)
    + "_"
    + str(epoch)
    + ".png",
    dpi=500,
    edgecolor="white",
    bbox_inches="tight",
    pad_inches=0.2,
)