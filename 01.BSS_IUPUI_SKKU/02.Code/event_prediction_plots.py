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
from event_prediction_data import StationDataset
from event_prediction_models import Prediction_Model_v4_GRU, Prediction_Model_v3_LSTM

torch.manual_seed(2024)
np.random.seed(2024)


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

def root_mean_squared_error(y, t):
    l2 = []
    for x in zip(y,t):
        l2.append(((x[0]-x[1])**2)**0.5)
    return l2


# Ploting Preperation
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

predictions_plot = torch.cat(predictions_plot, dim=0).reshape(
    -1, historical_station_dataset.station_count
)  # time-step X station
labels_plot = torch.cat(labels_plot, dim=0).reshape(
    -1, historical_station_dataset.station_count
)

predictions_plot_byStation = np.transpose(predictions_plot.numpy())  # station x time-step
labels_plot_byStation = np.transpose(labels_plot.numpy()).tolist()



#loss calculation
def 
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
upper_bound = [95.00 for x in range(len(predictions_plot_byStation[0]))]
lower_bound = [5.00 for x in range(len(predictions_plot_byStation[0]))]
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
        [round(x * max_val_ / total_dock_ * 100, 2) for x in data0_],
        label="output",
    )
    plt.plot(x_, upper_bound, linestyle="--", color="red", label="upper boundary")
    plt.plot(x_, lower_bound, linestyle="--", color="blue", label="lower boundary")
    plt.legend()

    plt.savefig(
        plot_path
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

fig, ax = plt.subplots(figsize=(20, 5))
plt.ylabel("Nomalized Bike Amount")
plt.ylim([0.0, 0.01])
plt.xlabel("Offset (10 mins interval)")
plt.grid(linestyle="--")
# plt.scatter(plotx_[0][0:input_seq_len], input_.tolist()[0], s=90, label='input')
plt.scatter([x for x in range(len(delta_losses))], delta_losses)
# plt.scatter([x for x in range(len(data[0]))], data[0], s=80, label="output")
# plt.scatter([x[-1] for x in plotx_], predictions, s=90,label='output')
# plt.scatter(, data[train_len:], label='valid')
ax.legend()
# plt.show()
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

fig, ax = plt.subplots(figsize=(20, 5))
plt.ylabel("Bike Amount Difference", fontsize = 12)


# plt.ylim([0.0, 0.01])
plt.xlabel("Offset (10 mins interval)",fontsize = 12)
# plt.scatter(plotx_[0][0:input_seq_len], input_.tolist()[0], s=90, label='input')

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
cate_c1 = []
cate_c2 = []
cate_c3 = []
for set_ in  zip(predictions_plot_cate, labels_plot_cate):
    acc, prec = acc_cal(set_[0],set_[1])
    cate_acc.append(acc)
    cate_c1.append(prec[0])
    cate_c2.append(prec[1])
    cate_c3.append(prec[2])
    
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
# plt.scatter([x for x in range(len(cate_c2))], cate_c2, label="Normal case Precision")
# plt.scatter([x for x in range(len(cate_c1))], cate_c1, label="Lower case Precision")
# plt.scatter([x for x in range(len(cate_c3))], cate_c3, label="Upper case Precision")
# plt.scatter([x for x in range(len(data[0]))], data[0], s=80, label="output")
# plt.scatter([x[-1] for x in plotx_], predictions, s=90,label='output')
# plt.scatter(, data[train_len:], label='valid')
ax.legend()
# plt.show()
plt.savefig(
    "./04.event_prediction/event_prediction_results/"
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

# precision
fig, ax = plt.subplots(figsize=(20, 5))
plt.ylabel("Precision(%)", fontsize = 12)

plt.ylim([0, 100])
plt.xlabel("Offset (10 mins interval)",fontsize = 12)
# plt.scatter(plotx_[0][0:input_seq_len], input_.tolist()[0], s=90, label='input')
plt.plot([x for x in range(len(cate_c1))], cate_c1, label="Lower case Precion")
plt.text(
        len(cate_c1),
        cate_c1[-1]+1,
        "Avg.Acc.: "
        + str(round(sum(cate_c1) / len(cate_c1),2)),
    )
plt.legend()
# plt.show()
plt.savefig(
    "./04.event_prediction/event_prediction_results/"
    + "00.precision_c1"
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

fig, ax = plt.subplots(figsize=(20, 5))
plt.ylabel("Precision (%)", fontsize = 12)

plt.ylim([0, 100])
plt.xlabel("Offset (10 mins interval)",fontsize = 12)
# plt.scatter(plotx_[0][0:input_seq_len], input_.tolist()[0], s=90, label='input')
plt.plot([x for x in range(len(cate_c2))], cate_c2, label="Normal case Precision")
plt.text(
        len(cate_c2),
        cate_c2[-1]+1,
        "Avg.Acc.: "
        + str(round(sum(cate_c2) / len(cate_c2),2)),
    )
plt.legend()
# plt.show()
plt.savefig(
    "./04.event_prediction/event_prediction_results/"
    + "00.precision_c2"
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

fig, ax = plt.subplots(figsize=(20, 5))
plt.ylabel("Precision(%)", fontsize = 12)

plt.ylim([0, 100])
plt.xlabel("Offset (10 mins interval)",fontsize = 12)
# plt.scatter(plotx_[0][0:input_seq_len], input_.tolist()[0], s=90, label='input')
plt.plot([x for x in range(len(cate_c3))], cate_c3, label="Upper case Precision")
plt.text(
        len(cate_c3),
        cate_c3[-1]+1,
        "Avg.Acc.: "
        + str(round(sum(cate_c3) / len(cate_c3),2)),
    )
plt.legend()
# plt.show()
plt.savefig(
    "./04.event_prediction/event_prediction_results/"
    + "00.precision_c3"
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