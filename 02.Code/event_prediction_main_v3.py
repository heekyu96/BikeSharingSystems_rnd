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
import scipy

from event_prediction_utils import _progressbar_printer
from event_prediction_utils import model_weight_path, plot_path
from event_prediction_data import *
from event_prediction_models import Prediction_Model_v4_GRU, Prediction_Model_v3_LSTM,Prediction_Model_v5_LSTM_FFT

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
# boundary = [0.00,0.20,0.40,0.60,0.80,sys.maxsize/100]
boundary = [0.00,0.20,0.40,0.60,0.80,sys.maxsize/100]
# boundary = [0.00,0.25,0.50,0.75,sys.maxsize/100]


# Dataset Generation
# historical_station_dataset = StationDataset_beta(
#     file_count, input_seq_len, output_seq_len, boundary, DEVICE, station_cnt_lock,train=train,fft_regen=False
# )
historical_station_dataset = StationDataset(
    file_count, input_seq_len, output_seq_len, boundary, DEVICE, station_cnt_lock
)
# Model Construction
# model = Prediction_Model_v5_LSTM_FFT(
model = Prediction_Model_v3_LSTM(
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
        "boundary_"+str(len(boundary)-1)
        + "_"+ str(epoch)
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
            if model.model_name.__contains__("FFT"):
                tr_x, tr_y, tr_x_fft, tr_y_cate = data
            else:
                tr_x, tr_y, tr_y_cate = data

            # print(tr_x.shape)
            optimizer.zero_grad()
            # outputs = model(tr_x)
            
            # bike amount output branch
            if model.model_name.__contains__("FFT"):
                outputs, outputs_cate = model(tr_x,tr_x_fft)
            else:
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
                + "boundary_"+str(len(boundary)-1)
                + "_"
                + str(i+1)
                + ".pt",
            )
        end_time = timer()
        train_progress._progressBar(i+1, end_time - start_time)

# Test
test_batch_size = 1
historical_station_dataset.set_test()
test_loader = DataLoader(historical_station_dataset, test_batch_size)
input_for_per_station_plot = []
output_for_per_station_plot = []
target_for_per_station_plot = []
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

    if model.model_name.__contains__("FFT"):
        tr_x, tr_y, tr_x_fft, tr_y_cate = data
    else:
        tr_x, tr_y, tr_y_cate = data
    # tr_x, tr_y, tr_x_fft, tr_y_cate = data

    if batch_idx % output_seq_len == 0 or batch_idx==0:
        outputs = []
        outputs_p = []
        outputs_cate = []
        targets = []
        targets_p = []
        targets_cate = []
        cur = tr_x
        delta_loss = 0
        input_for_per_station_plot.append(np.transpose(tr_x.cpu().squeeze().data))

    if model.model_name.__contains__("FFT"):
        output_tensor,output_cate_tensor = model(tr_x,tr_x_fft)
    else:
        output_tensor,output_cate_tensor = model(tr_x)
    # output_tensor,output_cate_tensor = model(tr_x,tr_x_fft)
    # 출력저장
    if DEVICE == torch.device("cuda"):
        output = output_tensor.cpu().squeeze().data
        output_cate = torch.argmax(output_cate_tensor.cpu().squeeze().data,dim=-1)
        # output = np.transpose(output) # 0:time-steps, 1:stations
        output_p = np.array(output)
        target = tr_y.cpu().squeeze().data  #
        target = np.transpose(target)  # 0:time-steps, 1:stations
        target_p = np.array(target)
        target_cate = tr_y_cate.cpu().squeeze()
    else:
        output = output_tensor.squeeze().data
        output_p = np.array(output)
        output_cate = torch.argmax(output_cate_tensor.squeeze().data,dim=-1)
        
        target = tr_y.squeeze().data
        target = np.transpose(target)  # 0:time-steps, 1:stations
        target_p = np.array(target)
        target_cate = tr_y_cate.squeeze()

    outputs.append(output)
    outputs_p.append(output_p)
    outputs_cate.append(output_cate)
    targets.append(target)
    targets_p.append(target_p)
    targets_cate.append(target_cate)

    # get loss according to delta
    delta_loss = loss_function(output_tensor, tr_y)
    delta_losses[batch_idx % output_seq_len] += delta_loss.item()

    cur = torch.cat([cur[:, 1:, :], output_tensor.data], dim=1)

    if batch_idx % output_seq_len == output_seq_len - 1:
        predictions.append(outputs)
        output_for_per_station_plot.append(np.transpose(np.array(outputs_p)))
        predictions_cate.append(outputs_cate)
        labels.append(targets)
        target_for_per_station_plot.append(np.transpose(np.array(targets_p)))
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


max_val_ = historical_station_dataset.max_val
    
## main_v2

##RMSE
rmse_list = []
rmse_cate_pred = []
rmse_per_station_list = []
for pred_label in zip(predictions, labels):
    rmse_per_station = [[] for x in range(611)]
    rmse = [0 for x in range(0,144)]
    rmse_cate = []
    for d, pred_label_day in enumerate(zip(pred_label[0],pred_label[1])):
        loss_list_per_station = root_mean_squared_error(pred_label_day[0],pred_label_day[1])
        for ps in range(len(loss_list_per_station)):
            rmse_per_station[ps].append(loss_list_per_station[ps])
        rmse[d] = sum(loss_list_per_station)/len(loss_list_per_station)
    rmse_list.append(rmse)
    rmse_per_station_list.append(rmse_per_station)


rmse_day = [0 for x in range(0,144)]
for rmse in rmse_list:
    rmse_day = [x+y for x,y in zip(rmse_day,rmse)]
rmse_day = [float(x)/len(rmse_list) for x in rmse_day]

## RMSE_Confidential_Interval
rmse_list_for_CI = np.array(rmse_list).transpose()
# rmse_ci_95 = []
# rmse_ci_99 = []
# for rm in rmse_list_for_CI:
#     rmse_ci_95.append(st.t.interval(alpha=0.95, df=len(rm), loc=np.mean(rm), scale=st.sem(rm)))
#     rmse_ci_99.append(st.t.interval(alpha=0.99, df=len(rm), loc=np.mean(rm), scale=st.sem(rm)))
# rmse_ci_95 = [[c*max_val_ for c in list(np.array(rmse_ci_95)[:,0])],[c*max_val_ for c in list(np.array(rmse_ci_95)[:,0])]]
# rmse_ci_99 = [[c*max_val_ for c in list(np.array(rmse_ci_99)[:,0])],[c*max_val_ for c in list(np.array(rmse_ci_99)[:,0])]]

def mean_confidence_interval_rmse(points, max_val, confidence=0.95):
    M = []
    ci_h =[]
    ci_l=[]
    for data in points:
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
        M.append(m*max_val)
        ci_h.append((m+h)*max_val)
        ci_l.append((m-h)*max_val)
    
    return M, ci_h, ci_l

rmse_m, rmse_ci_95_h,rmse_ci_95_l = mean_confidence_interval_rmse(rmse_list_for_CI, max_val=max_val_,confidence=0.925)
rmse_m, rmse_ci_99_h,rmse_ci_99_l = mean_confidence_interval_rmse(rmse_list_for_CI, max_val=max_val_,confidence=0.99)
# rmse_ci_99 = []

##loss plot
fig, ax = plt.subplots(figsize=(20, 5))
ax.set_ylabel("RMSE of normalized bike counts",fontsize=13)
ax.set_ylim([0.0, 1])
ax.set_xlabel("Prediction Time (hour)", fontsize=14)
x_tick = [6*x for x in range(24)]+[144]
ax.set_xticks(x_tick)
ax.set_xticklabels([int(x/6) for x in x_tick])
plt.grid(linestyle="--")
# ax.scatter([x+1 for x in range(len(rmse_day))], rmse_day, label="Normalized difference in the number of bikes")
# ax.scatter(color="orange",label="difference in the number of bikes")

ax.bar(0, height=0, width=0, color="tan",label="92.5% Confidential Interval")
ax.bar(0, height=0, width=0, color="wheat",label="99% Confidential Interval")
ax.plot([x+1 for x in range(len(rmse_day))], rmse_m,color="orange",label="Bike counts")
ax.scatter([x+1 for x in range(len(rmse_day))], rmse_day, label="Normalized bike counts")
ax.legend(loc="upper left", ncol=2)
ax2 = ax.twinx()
ax2.set_ylabel("RMSE of bike counts", fontsize=13)
# for rmse in rmse_list:
#     ax2.scatter([x+1 for x in range(len(rmse))], [y*max_val_ for y in rmse],color="wheat")
# for t, ci in enumerate(rmse_ci_95):

# for t, ci in enumerate(rmse_ci_99):
ax2.fill_between([x+1 for x in range(len(rmse_day))],rmse_ci_99_h,rmse_ci_99_l,color="wheat",alpha=0.5)
ax2.fill_between([x+1 for x in range(len(rmse_day))],rmse_ci_95_h,rmse_ci_95_l,color="tan",alpha=0.5)
ax2.plot([x+1 for x in range(len(rmse_day))], rmse_m,color="orange",label="difference in the number of bikes")
# ax2.plot([x+1 for x in range(len(rmse_day))], [y*max_val_ for y in rmse_day],color="orange",label="bike amount difference")
ax2.set_ylim([0.0, 6.5])
# ax2.legend(loc="upper left", ncol=2)
# ax.text(
#         0,
#         0.90,
#         "Avg. RMSE in the number of normalized bikes: "
#         + str(round(sum(rmse_m) / (len(rmse_m)*max_val_),2)),
#     )
# ax.text(
#         0,
#         0.95,
#         "Avg. RMSE in the number of bikes: "
#         + str(round(sum(rmse_m) / len(rmse_m),2)),
#     )
plt.savefig(
    "./02.Code/event_prediction_results/"
    + "00.day_loss"
    + "_"
    + str(input_day * 24)
    + "_"
    + str(output_day * 24)
    + "_"
    + str(epoch)
    + "_"+model.model_name
    +"_"+str(len(boundary)-1)
    + ".png",
    dpi=500,
    edgecolor="white",
    bbox_inches="tight",
    pad_inches=0.2,
)

def acc_cal(pred, tar,num_labels = len(boundary)-1):
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

## ACC
acc_list = []
prec_list = []
for pred_label in zip(predictions_cate, labels_cate):
    acc = []
    prec = []
    for d, pred_label_cate_day in enumerate(zip(pred_label[0],pred_label[1])):
        acc_, prec_ = acc_cal(pred_label_cate_day[0],pred_label_cate_day[1])
        acc.append(acc_)
        prec.append(prec_)
    acc_list.append(acc)
    prec_list.append(prec)

acc_day = [0 for x in range(0,144)]
for acc in acc_list:
    acc_day = [x+y for x,y in zip(acc_day,acc)]
acc_day = [float(x)/len(acc_list) for x in acc_day]


def acc_cal_from_rmse(rmse_pred, tar,num_labels = len(boundary)-1):
    slot = 1.0/(len(boundary)-1.0)
    cnt = [0 for x in range(num_labels)]
    correct = [0 for x in range(num_labels)]
    for xy in zip(rmse_pred,tar):
        xy_ = int(xy[0]*max_val_/historical_station_dataset.station_total_dock[historical_station_dataset.station_list[d]]/slot)
        cnt[xy[1]]+=1
        if xy_==xy[1]:
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

## ACC
rmse_acc_list = []
rmse_prec_list = []
for pred_label in zip(predictions, labels_cate):
    acc = []
    prec = []
    for d, pred_label_cate_day in enumerate(zip(pred_label[0],pred_label[1])):
        acc_, prec_ = acc_cal_from_rmse(pred_label_cate_day[0],pred_label_cate_day[1])
        acc.append(acc_)
        prec.append(prec_)
    rmse_acc_list.append(acc)
    rmse_prec_list.append(prec)

rmse_acc_day = [0 for x in range(0,144)]
for acc in rmse_acc_list:
    rmse_acc_day = [x+y for x,y in zip(rmse_acc_day,acc)]
rmse_acc_day = [float(x)/len(rmse_acc_list) for x in rmse_acc_day]
# # # prec_day = [[0 for x in range(len(boundary)-1)] for b in range(144)]
# # prec_day = [[0 for x in range(144)] for b in range(len(boundary)-1)]
# # for prec in prec_list:
# #     for d,prec_ in enumerate(prec):
# #         for i in range(len(boundary)-1):
# #             prec_day[i][d]+=prec_[i]
            
#     prec_day = [[a+b for a,b in zip(x,y)] for x,y in zip(prec_day,prec)]
# # acc_day = [[ float(a)/len(prec_list) for a in x] for x in acc_day]
# for day in prec_day:
#     for cls in day:
#         cls = cls/len(prec_list)

## ACC_Confidential_Interval
acc_list_for_CI = np.array(acc_list).transpose()
rmse_acc_list_for_CI = np.array(rmse_acc_list).transpose()
def mean_confidence_interval(points, confidence=0.95):
    M = []
    ci_h =[]
    ci_l=[]
    for data in points:
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
        M.append(m)
        ci_h.append(m+h)
        ci_l.append(m-h)
    
    return M, ci_h, ci_l

acc_m, acc_ci_95_h,acc_ci_95_l = mean_confidence_interval(acc_list_for_CI,confidence=0.925)
acc_m, acc_ci_99_h,acc_ci_99_l = mean_confidence_interval(acc_list_for_CI, confidence=0.99)
# rmse_acc_m, rmse_acc_ci_95_h,rmse_acc_ci_95_l = mean_confidence_interval(rmse_acc_list_for_CI,confidence=0.925)
# rmse_acc_m, rmse_acc_ci_99_h,rmse_acc_ci_99_l = mean_confidence_interval(rmse_acc_list_for_CI, confidence=0.99)

fig, ax = plt.subplots(figsize=(20, 5))
ax.set_ylabel("Prediction Accuracy (%)", fontsize=13)
ax.set_ylim([0, 104])
ax.set_xlabel("Prediction Time (hour)", fontsize=14)
x_tick = [6*x for x in range(24)]+[144]
ax.set_xticks(x_tick)
ax.set_xticklabels([int(x/6) for x in x_tick])
plt.grid(linestyle="--")

ax.fill_between([x+1 for x in range(len(acc_day))],acc_ci_95_h,acc_ci_95_l,color="deepskyblue",alpha=0.5)
ax.fill_between([x+1 for x in range(len(acc_day))],acc_ci_99_h,acc_ci_99_l,color="lightblue",alpha=0.5)

ax.bar(0, height=0, width=0, color="deepskyblue",label="92.5% Confidential Interval")
ax.bar(0, height=0, width=0, color="lightblue",label="99% Confidential Interval")

# for t, ci in enumerate(rmse_ci_99):

ax.plot([x+1 for x in range(len(acc_m))], acc_m, color = "royalblue", label="Accuracy")
# for t, ci in enumerate(rmse_ci_99):
# ax.fill_between([x+1 for x in range(len(rmse_acc_day))],rmse_acc_ci_99_h,rmse_acc_ci_99_l,color="lightgray",alpha=0.5)
# ax.fill_between([x+1 for x in range(len(rmse_acc_day))],rmse_acc_ci_95_h,rmse_acc_ci_95_l,color="lightsteelblue",alpha=0.5)

# ax.plot([x+1 for x in range(len(rmse_acc_m))], rmse_acc_m, color = "dimgray", label="Converted accuracy from predicted number of bikes")
ax.legend(loc="lower right")
# ax2 = ax.twinx()
# ax2.set_ylabel("Bike Amount")
# for rmse in p:
#     ax2.scatter([x+1 for x in range(len(rmse))], [y*max_val_ for y in rmse],color="wheat")
# # ax2.scatter([x+1 for x in range(len(rmse_day))], [y*max_val_ for y in rmse_day],color="orange",label="bike amount difference")
# ax2.set_ylim([0, 104])
# ax2.legend(loc="upper right")
# ax.text(
#         0,
#         57,
#         "Avg. Accuracy for 1 day: "
#         + str(round(sum(acc_m) / len(acc_m),2)),
#     )
# ax.text(
#         0,
#         50,
#         "Converted Avg. accuracy from predicted number of bikes: "
#         + str(round(sum(rmse_acc_m) / len(rmse_acc_m),2)),
#     )
# ax.text(
#         0,
#         0.95,
#         "Avg. RMSE of bike amount: "
#         + str(round(sum(rmse_day) / len(rmse_day)*max_val_,2)),
#     )
plt.savefig(
    "./02.Code/event_prediction_results/"
    + "00.day_acc"
    + "_"
    + str(input_day * 24)
    + "_"
    + str(output_day * 24)
    + "_"
    + str(epoch)
    + "_"+model.model_name
    +"_"+str(len(boundary)-1)
    + ".png",
    dpi=500,
    edgecolor="white",
    bbox_inches="tight",
    pad_inches=0.2,
)


# ##  Plottings per station
# # for in_out in zip(input_for_per_station_plot,rmse_per_station_list[0]):
# for dir_, data_points in enumerate(zip(input_for_per_station_plot,output_for_per_station_plot, target_for_per_station_plot)):
#     dir_path = "./02.Code/event_prediction_results/station_wise/"+str(dir_+1)+"/"
#     file=0
#     print(file)
#     for inputt, outputt, targett in zip(data_points[0],data_points[1],data_points[2]):
#         fig, ax = plt.subplots(figsize=(20, 5))
#         ax.set_ylabel("Predicted shared bikes")
#         # ax.set_ylim([0.0, 1])
#         ax.set_xlabel("Prediction Time (Days)")
#         x_tick = [24*6*x for x in range(8)]+[24*6*8]
#         ax.set_xticks(x_tick)
#         ax.set_xticklabels([int(x/(6*24)) for x in x_tick])
#         plt.grid(linestyle="--")
#         ax.plot([x+1 for x in range(len(inputt))], [in_*max_val_ for in_ in inputt],color = "forestgreen",label="The number of bikes in historical sequence")
#         ax.plot([x+1+len(inputt) for x in range(len(outputt))], [out_*max_val_ for out_ in outputt],color = "goldenrod",label="The predicted number of bikes")
#         ax.plot([x+1+len(inputt) for x in range(len(targett))], [tar_*max_val_ for tar_ in targett],color = "olivedrab",label="The number of bikes (Ground-Truth)")
#         ax.legend(loc="lower left")

#         plt.savefig(
#             dir_path
#             + "00.station_loss"
#             + "_"
#             + str(input_day * 24)
#             + "_"
#             + str(output_day * 24)
#             + "_"
#             + str(epoch)
#             + "_"+model.model_name
#             +"_"+str(len(boundary)-1)
#             +"_"+str(file)
#             + ".png",
#             dpi=500,
#             edgecolor="white",
#             bbox_inches="tight",
#         )
#         file+=1
        
        
