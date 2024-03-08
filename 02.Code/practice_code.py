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
import folium
from folium.plugins import HeatMap
import pandas as pd
import geopandas as gpd

from event_prediction_utils import _progressbar_printer
from event_prediction_utils import model_weight_path, plot_path, data_path
from event_prediction_data_ import *
from event_prediction_models import Prediction_Model_v4_GRU, Prediction_Model_v3_LSTM,Prediction_Model_v5_LSTM_FFT

torch.manual_seed(2024)
np.random.seed(2024)

# torch.manual_seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\n" + str(DEVICE) + " is working...\n")


# Train Model or Use Trained Model Weight
#train= False
train = True

# Hyper-Params
epoch = 10
batch_size = 500 
dropout_rate = 0.5

# Model Input and Output Shape
input_day = 7
output_day = 1
input_seq_len = input_day * 24 * 6
output_seq_len = output_day * 24 * 6
station_cnt_lock = -1


boundary = [0.00,0.20,0.40,0.60,0.80,sys.maxsize/100]

file_count = 10
#dir_path_ = "."+data_path+"use_historical/*.csv"
dir_path_ = data_path+"use_historical/*.csv"


# Dataset Generation
# historical_station_dataset = StationDataset_beta(
#     file_count, input_seq_len, output_seq_len, boundary, DEVICE, station_cnt_lock,train=train,fft_regen=False
# )
historical_station_dataset = StationDataset(
    file_count, dir_path_, input_seq_len, output_seq_len, boundary, DEVICE, station_cnt_lock
)
# Model Construction
# model = Prediction_Model_v3_LSTM(
# model = Prediction_Model_v5_LSTM_FFT(
    
# LSTM 모델 model 선언
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
    ) # 해당 파일 경로의 weight 불러오기
    model.load_state_dict(model_weight) # 모델에 weight 적용
    
else: # Train
    historical_station_dataset.set_train() # mode = "train"
    train_batch_size = batch_size
    # train_batch_size = batch_size * output_seq_len
    train_loader = DataLoader(historical_station_dataset, train_batch_size) # 데이터 올리기
    
    model.train() # setup train mode
    
    train_progress = _progressbar_printer("Model Training", iterations=epoch) # status bar
    
    for i in range(epoch):
        losses = 0      # pred
        losses_cate = 0 # class
        re = 0
        # output_cnt = 1
        start_time = timer()
        
        for batch_idx, data in enumerate(train_loader):
            # tr_x, tr_y = data
            
            # Fast Fourier Transform한 LSTM이면
            if model.model_name.__contains__("FFT"):
                tr_x, tr_y, tr_x_fft, tr_y_cate = data

            else:
                tr_x, tr_y, tr_y_cate = data

            # print(tr_x.shape)
            optimizer.zero_grad() # 옵티마이저 초기화
            # outputs = model(tr_x)
            
            
            # bike amount output branch
            if model.model_name.__contains__("FFT"):
                outputs, outputs_cate = model(tr_x,tr_x_fft)
            else:
                outputs, outputs_cate = model(tr_x)
                
            # loss calculation 1
            loss = loss_function(outputs, tr_y)


            # event classes output branch
            output_cate_dim = outputs_cate.shape[-1]
            outputs_cate = outputs_cate[:].view(-1, output_cate_dim)
            tr_y_cate = tr_y_cate[:].view(-1)
            
            # loss calculation 2
            loss_cate = loss_function_cate(outputs_cate, tr_y_cate)
            
            
            # Backpropagation
            loss.backward(retain_graph= True) # loss 1에 대해 bp
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            loss_cate.backward() # loss 2에 대해 bp
            
            optimizer.step() # 위에서 계산된 그라디언트로 모델의 가중치 업데이트 (학습)
            
            # Drop back-propagation tree and save only numerical loss value
            # loss 1과 2에 대해 loss값 기록
            losses += loss.item()
            losses_cate += loss_cate.item()
            
            
            # Backpropagation
            '''loss_cate.backward(retain_graph= True) # loss 1에 대해 bp
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm) # gradient exploding 방지 위해 clipping
            loss.backward() # loss 2에 대해 bp
            
            optimizer.step() # 위에서 계산된 그라디언트로 모델의 가중치 업데이트 (학습)
            
            # Drop back-propagation tree and save only numerical loss value
            # loss 1과 2에 대해 loss값 기록
            losses_cate += loss_cate.item()
            losses += loss.item()'''
            

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
historical_station_dataset.set_test() # mode = "test"
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
    '''
    if batch_idx == 3:
        print("batch_idx = 0, 1, 2, 3, 4")
        break'''
    
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
        # tr_x.cpu().squeeze() : cpu로 이동 후 squeeze (차원이 1이면 해당 차원 제거)
        
    if model.model_name.__contains__("FFT"):
        output_tensor,output_cate_tensor = model(tr_x,tr_x_fft)
    else:
        output_tensor,output_cate_tensor = model(tr_x)
    # output_tensor,output_cate_tensor = model(tr_x,tr_x_fft)
    # 출력저장
    if DEVICE == torch.device("cuda"):
        output = output_tensor.cpu().squeeze().data
        output_cate = torch.argmax(output_cate_tensor.cpu().squeeze().data,dim=-1) # output_cate: 가장 확률이 높은 클래스 선택
        # output = np.transpose(output) # 0:time-steps, 1:stations
        output_p = np.array(output)
        # output_p: output을 squeeze 후 tanspose
        
        target = tr_y.cpu().squeeze().data
        target = np.transpose(target)  # 0:time-steps, 1:stations 
        target_p = np.array(target)
        # target_p: tr_y를 squeeze 후 transpose
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

'''
print(len(predictions)) # batch size 7200 
# output seq len 배수 배치만 저장하면 50 (7200/144)
# batch 144, batch 288, batch 432, batch 576, ...
print(len(predictions[0])) # 144
print(len(predictions[0][0])) # 611
# print(b_c) # 총 배치 개수: 7200

#print(predictions[0][0][0])
'''

#############

rmse_list = []
rmse_cate_pred = []
rmse_per_station_list = []
avg_rmse_per_station_list = []

for pred_label in zip(predictions, labels): # pred_label = (predictions[0], labels[0]) 50번 iter
    rmse_per_station = [[] for x in range(611)]
    rmse = [0 for x in range(0,144)]
    rmse_cate = []
    for d, pred_label_day in enumerate(zip(pred_label[0],pred_label[1])): # pred_label_day = (predictions[0][0], labels[0][0]) ~ (predictions[0][143], labels[0][143])
        # 144번 실행
        loss_list_per_station = root_mean_squared_error(pred_label_day[0],pred_label_day[1]) # loss_list_per_station = 1 timestep의 rmse값 611개
        for ps in range(len(loss_list_per_station)): #ps=611
            rmse_per_station[ps].append(loss_list_per_station[ps])
            # rmse_per_station = [s1: [t1, t2, t3, ..., t144], s2: [t1, t2, t3, ..., t144], .. , s611: [t1, t2, t3, ..., t144]]
        rmse[d] = sum(loss_list_per_station)/len(loss_list_per_station) # rmse = [d1, d2, ..., d144] : 각 time step의 611개 정거장의 rmse '평균' 값 144개
    
    ###############
    avg_rmse_per_station = list(map(lambda station_rmse: sum(station_rmse) / len(station_rmse), rmse_per_station))
    # avg_rmse_per_station = [s1: [하루 rmse 평균값], s2: [하루 rmse 평균값], .. , s611: [하루 rmse 평균값]]
    avg_rmse_per_station_list.append(avg_rmse_per_station)
    # avg_rmse_per_station_list = [1:[611개의 평균값], 2:[611개의 평균값], 3:[611개의 평균값], ... , 50:[611개의 평균값]]
    ###############

rmse_per_station_array = np.array(avg_rmse_per_station_list)
avg_per_station = rmse_per_station_array.mean(axis=0)


### RMSE Plot ###
'''location_df = pd.read_csv('C:/Users/huigyu/workspace/BikeSharingSystems_rnd/02.Code/loc_zip_.csv', dtype={'ZIP_Code': str})
location_df['rmse_avg'] = avg_per_station


gdf = gpd.read_file('C:/Users/huigyu/workspace/BikeSharingSystems_rnd/02.Code/Practice_code/geo_export_dd9da705-66ae-428c-9857-158c713facfe.shp')
gdf1 = gpd.GeoDataFrame(location_df, geometry=gpd.points_from_xy(location_df['LONGITUDE'], location_df['LATITUDE']))

ax = gdf.plot(figsize=(10, 10), color='whitesmoke', edgecolor='black', linewidth=1)
gdf1.plot(ax=ax, column='rmse_avg', cmap='OrRd', markersize=1, legend=True)
ax.axis('off')

plt.savefig('Average_RMSE.png', dpi = 300)
'''

### ACC Plot ###

def acc_cal(pred, tar,num_labels = len(boundary)-1): # num_labels = 5 (분류 범주 개수)
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

'''acc_list = []
prec_list = []
for pred_label in zip(predictions_cate, labels_cate): # 50번
    acc = []
    prec = []
    for d, pred_label_cate_day in enumerate(zip(pred_label[0],pred_label[1])):
        # 144번 실행
        acc_, prec_ = acc_cal(pred_label_cate_day[0],pred_label_cate_day[1])
        acc.append(acc_)
        prec.append(prec_)
    acc_list.append(acc)
    prec_list.append(prec)

acc_day = [0 for x in range(0,144)]
for acc in acc_list:
    acc_day = [x+y for x,y in zip(acc_day,acc)]
acc_day = [float(x)/len(acc_list) for x in acc_day]
'''
'''
pred_label_cate_day[0] = [timestep 1에서 611개 station의 cate pred]
pred_label_cate_day[1] = [timestep 1에서 611개 station의 cate label]

acc_ = timestep 1에서의 accuracy
prec_ = timestep 1에서의 precision

acc = [t1 acc, t2 acc, t3 acc, ... , t144 acc]
prec = [t1 prec, t2 prec, t3 prec, ... , t144 prec]
====================================================
pred_label_cate_day[0] = [timestep 1에서 611개 station의 cate pred]
pred_label_cate_day[1] = [timestep 1에서 611개 station의 cate label]

pred_label[0]= [1:[611], 2:[611], 3:[611], ... , 144:[611]] -> pred
pred_label[1]= [1:[611], 2:[611], 3:[611], ... , 144:[611]] -> label

pred_per_station = [1:[144], 2:[144], ... , 611:[144]]
label_per_station = [1:[144], 2:[144], ... , 611:[144]]

acc_list = [1: [s1 acc, s2 acc, s3 acc, ... , s611 acc],
            2: [s1 acc, s2 acc, s3 acc, ... , s611 acc]
            , ... ,
            50: [s1 acc, s2 acc, s3 acc, ... , s611 acc]]
'''


acc_list = []
prec_list = []
for pred_label in zip(predictions_cate, labels_cate): # 50번
    pred_per_station = [list(group) for group in zip(*pred_label[0])]
    label_per_station = [list(group) for group in zip(*pred_label[1])]
    acc = []
    prec = []
    for st, pred_label_cate_day in enumerate(zip(pred_per_station, label_per_station)):
        #611번 실행
        acc_, prec_ = acc_cal(pred_label_cate_day[0], pred_label_cate_day[1])
        acc.append(acc_)
        prec.append(prec_)
    acc_list.append(acc)
    prec_list.append(prec)
    
print(len(acc_list))
print(len(prec_list))


acc_list_ = np.array(acc_list)
acc_per_station = np.mean(acc_list_, axis=0)

#######
'''
acc_list = []
prec_list = []
for pred_label in zip(predictions_cate, labels_cate): # 50번
    pred_per_station = np.array(pred_label[0]).transpose().tolist()
    label_per_station = np.array(pred_label[1]).transpose().tolist()
    acc = []
    prec = []
    for st, pred_label_cate_day in enumerate(zip(pred_per_station, label_per_station)):
        #611번 실행
        acc_, prec_ = acc_cal(pred_label_cate_day[0], pred_label_cate_day[1])
        acc.append(acc_)
        prec.append(prec_)
    acc_list.append(acc)
    prec_list.append(prec)

print(len(acc_list))
print(len(prec_list))

acc_list_ = np.array(acc_list)
acc_per_station = np.mean(acc_list_, axis=0)

print(acc_per_station[0])
print(len(acc_per_station))'''


'''
location_df = pd.read_csv('C:/Users/huigyu/workspace/BikeSharingSystems_rnd/02.Code/loc_zip_.csv', dtype={'ZIP_Code': str})

location_df['acc_avg'] = acc_per_station

gdf = gpd.read_file('C:/Users/huigyu/workspace/BikeSharingSystems_rnd/02.Code/Practice_code/geo_export_dd9da705-66ae-428c-9857-158c713facfe.shp')
gdf1 = gpd.GeoDataFrame(location_df, geometry=gpd.points_from_xy(location_df['LONGITUDE'], location_df['LATITUDE']))

ax = gdf.plot(figsize=(10, 10), color='whitesmoke', edgecolor='black', linewidth=1)
gdf1.plot(ax=ax, column='acc_avg', cmap='OrRd', markersize=1, legend=True)
ax.axis('off')

plt.savefig('Average_ACC.png', dpi = 300)'''