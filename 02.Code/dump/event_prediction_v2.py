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

torch.manual_seed(2024)
np.random.seed(2024)

# Confusion matrix
# https://tutorials.pytorch.kr/intermediate/char_rnn_classification_tutorial

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
        cur = cur + 1
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
            flush=True,
        )
        # Print New Line on Complete
        if cur >= self.iters:
            print("\nCompleted\n")


# torch.manual_seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\n" + str(DEVICE) + " is working...\n")

file_count = 10

# Historical station dataset processing
def historical_station_logs(file_count):
    dir_path = "./00.datasets/chicago_bike_dataset/use_historical/*.csv"
    historical_data_load_progress = _progressbar_printer(
        "Station historical data", iterations=file_count
    )
    station_list = []
    station_total_dock_dict = {}  # for denomalization
    data_dict = {}

    max_val = 0
    for i, file in enumerate(sorted(glob.glob(dir_path))):
        # print(file)
        if i > file_count:
            break
        file = open(file, "r", encoding="utf-8")
        csv_reader = csv.reader(file)
        # next(csv_reader)
        start_time = timer()
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

            station = line[2].replace(" ", "").lower()
            if not station_list.__contains__(station):
                station_list.append(station)
                station_total_dock_dict[station] = int(line[4])

            if not data_dict.__contains__(station):
                data_dict[station] = [int(line[7])]
            else:
                data_dict[station].append(int(line[7]))

        list_max = max([len(data_dict[d]) for d in data_dict])

        # padding ver
        # for d in data_dict:
        #     len_ = len(data_dict[d])
        #     if len_ < list_max:
        #         for i in range(list_max - len_):
        #             # add padding
        #             data_dict[d].insert(0, data_dict[d][0])
        # removing ver
        for d in data_dict:
            len_ = len(data_dict[d])
            if len_ < list_max:
                data_dict.pop(d)
                station_list.remove(d)
                station_total_dock_dict.pop(d)

        end_time = timer()
        historical_data_load_progress._progressBar(i+1, end_time - start_time)
    print(
        "Number of Stations: ", len(station_list)
    )  # the number of stations : (10files)
    print("Number of time-steps: ", len(data_dict[station_list[0]]))
    return data_dict, station_list, station_total_dock_dict, max_val


class StationDataset(Dataset):
    def __init__(self, file_cnt, input_seq_len, output_seq_len, upper_bound, lower_bound, station_cnt_lock=-1):
        self.input_len = input_seq_len
        self.output_len = output_seq_len
        self.upper_boundary = upper_bound
        self.lower_boundary = lower_bound
        self.file_count = file_cnt
        (
            self.data_dict,  # station data:list
            self.station_list,
            self.station_total_dock,
            self.max_val,
        ) = historical_station_logs(file_count=self.file_count)
        self.station_count = len(self.station_list)
        (
            self.normalized_data,
            self.sized_seqs,
            self.sized_seqs_categorical,
        ) = self.generate_bike_amount_seq(
            station_cnt_lock
            # 10
        )  # 0:time_step, 1:all values of stations
        self.dataset_total_size = len(self.sized_seqs)
        # self.dataset_size = dataset_size
        # self.train_or_test = train_or_test
        self.division_ratio = 0.8
        self.train_seqs_cnt = 0
        self.test_seqs_cnt = 0
        self.mode = ""
        (
            self.train_input,
            self.train_output,
            self.train_output_categorical,
            self.test_input,
            self.test_output,
            self.test_output_categorical,
        ) = self.dataset_gen()

        print(
            "Train I/O shape: ", len(self.train_input),"X",len(self.train_input[0]), "\t", len(self.train_output),"X",len(self.train_output[0])
        )
        print(
            "Test I/O shape: ", len(self.test_input),"X",len(self.test_input[0]), "\t", len(self.test_output),"X",len(self.test_output[0])
        )

    def generate_bike_amount_seq(self, station_cnt_lock):
        data_list = []
        data_list_categorical = []
        for lock, station_name in enumerate(self.data_dict):
            if lock == station_cnt_lock:
                self.station_count = station_cnt_lock
                break
            # normalization with overall max_val
            data_list.append([x / self.max_val for x in self.data_dict[station_name]])
            # create a label list
            cate_temp = []
            # [x / self.station_total_dock[station_name] for x in self.data_dict[station_name]]
            for x in self.data_dict[station_name]:
                per = x / self.station_total_dock[station_name]
                if per >= self.upper_boundary:
                    cate_temp.append(2)
                elif per < self.upper_boundary and per > self.lower_boundary:
                    cate_temp.append(1)
                else:
                    cate_temp.append(0)
            data_list_categorical.append(cate_temp)
            # data_list_categorical.append([x / self.station_total_dock[station_name] for x in self.data_dict[station_name]])

        data_list = np.asarray(data_list)
        data_list_categorical = np.asarray(data_list_categorical)
        print(data_list.shape)
        data_list = np.transpose(data_list)
        data_list_categorical = np.transpose(data_list_categorical)
        print(data_list.shape)
        data_list = data_list.tolist()
        data_list_categorical = data_list_categorical.tolist()
        combined_seqs = []
        combined_seqs_cate = []
        for i in range(0, len(data_list) - self.input_len - self.output_len, 1):
            # print(i)
            combined_seqs.append(data_list[i : i + self.input_len + self.output_len])
            combined_seqs_cate.append(
                data_list_categorical[i : i + self.input_len + self.output_len]
            )
            # i += 143

        return data_list, combined_seqs, combined_seqs_cate

    def dataset_gen(self):
        train_len = int(self.dataset_total_size * self.division_ratio)

        train_seqs = self.sized_seqs[:train_len]
        test_seqs = self.sized_seqs[train_len:]
        train_seqs_cate = self.sized_seqs_categorical[:train_len]
        test_seqs_cate = self.sized_seqs_categorical[train_len:]
        self.train_seqs_cnt = len(train_seqs)
        self.test_seqs_cnt = len(test_seqs)

        train_data_load_progress = _progressbar_printer(
            "Train data sequence Sizing", iterations=len(train_seqs)
        )
        train_input = []
        train_output = []
        train_output_cate = []
        for j, seq in enumerate(zip(train_seqs, train_seqs_cate)):
            start_time = timer()
            for i in range(self.output_len):
                train_input.append(seq[0][i : i + self.input_len])
                train_output.append(seq[0][i + self.input_len : i + self.input_len + 1])
                train_output_cate.append(
                    # F.one_hot(
                        torch.tensor(seq[1][i + self.input_len : i + self.input_len + 1])
                        # , num_classes=3
                    # ).unsqueeze(0)
                )
            end_time = timer()
            train_data_load_progress._progressBar(j+1, end_time - start_time)

        test_data_load_progress = _progressbar_printer(
            "Test data sequence Sizing", iterations=len(test_seqs)
        )
        test_input = []
        test_output = []
        test_output_cate = []
        for j, seq in enumerate(zip(test_seqs, test_seqs_cate)):
            start_time = timer()
            for i in range(self.output_len):
                test_input.append(seq[0][i : i + self.input_len])
                test_output.append(seq[0][i + self.input_len : i + self.input_len + 1])
                test_output_cate.append(
                    # F.one_hot(
                        torch.tensor(seq[1][i + self.input_len : i + self.input_len + 1])
                        # , num_classes=3
                    # ).unsqueeze(0)
                )
            end_time = timer()
            test_data_load_progress._progressBar(j+1, end_time - start_time)

        return (
            train_input,
            train_output,
            train_output_cate,
            test_input,
            test_output,
            test_output_cate,
        )

    def set_train(self):
        self.mode = "train"

    def set_test(self):
        self.mode = "test"

    def __len__(self):
        if self.mode == "":
            print("Mode is not set")
            return None
        elif self.mode == "train":
            return len(self.train_input)
        elif self.mode == "test":
            return len(self.test_input)

    def __getitem__(self, index):
        if self.mode == "":
            print("Mode is not set")
            return None
        elif self.mode == "train":
            return (
                torch.FloatTensor(self.train_input[index]).to(DEVICE),
                torch.FloatTensor(self.train_output[index]).to(DEVICE),
                self.train_output_categorical[index].type(torch.long).to(DEVICE),
            )
        elif self.mode == "test":
            return (
                torch.FloatTensor(self.test_input[index]).to(DEVICE),
                torch.FloatTensor(self.test_output[index]).to(DEVICE),
                self.test_output_categorical[index].type(torch.long).to(DEVICE),
            )

# 
class Prediction_Model_v1(nn.Module):
    def __init__(
        self,
        unit_input_dim,
        unit_hidden_dim,
        unit_output_dim,
        dropout=0.0,
        bidirectional=False,
    ):
        nn.Module.__init__(self)
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
class Prediction_Model_v2(nn.Module):
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
        dropout=0.0,
        bidirectional=False
    ):
        nn.Module.__init__(self)
        self.model_name = "v3_LSTM"
        self.input_dim = unit_input_dim
        self.hidden_dim = unit_hidden_dim
        self.output_dim = unit_output_dim
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
        self.fc_cate = torch.nn.Linear(unit_output_dim, (unit_output_dim * 3))

    def forward(self, input):
        # input shape: batch, seq, dim(internal vector dim)
        rnn_output, _ = self.rnn(input)
        output = self.fc(rnn_output[:, -1])
        output_cate = self.fc_cate(output)
        
        output = output.unsqueeze(1)
        output_cate = output_cate.reshape(-1,self.output_dim,3) # 3 refers # of event classes
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
        rnn_output, _ = self.rnn(input)
        output = self.fc(rnn_output[:, -1])
        output_cate = self.fc_cate(output)
        
        output = output.unsqueeze(1)
        output_cate = output_cate.reshape(-1,self.output_dim,3) # 3 refers # of event classes
        return output, output_cate


# Train Model or Use Trained Model Weight
train= False
# train = True

# Hyper-Params
epoch = 100
batch_size = 500
dropout_rate = 0.2

# Model Input and Output Shape
input_day = 7
output_day = 1
input_seq_len = input_day * 24 * 6
output_seq_len = output_day * 24 * 6

# Event Definition
upper_bound = 0.90
lower_bound = 0.10
station_cnt_lock = -1


# Dataset Generation
historical_station_dataset = StationDataset(
    file_count, input_seq_len, output_seq_len, upper_bound, lower_bound, station_cnt_lock
)

# Model Construction
model = Prediction_Model_v4_GRU(

    unit_input_dim=historical_station_dataset.station_count,
    unit_hidden_dim=256,
    unit_output_dim=historical_station_dataset.station_count,
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
        "./04.event_prediction/trained_model_weights/"
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
                "./04.event_prediction/trained_model_weights/"
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
    torch.save(
        model.state_dict(),
        "./04.event_prediction/trained_model_weights/"
        + model.model_name
        + str(input_day * 24)
        + "_"
        + str(output_day * 24)
        + "_"
        + str(epoch)
        + ".pt",
    )

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

predictions_plot = torch.cat(predictions_plot, dim=0).reshape(
    -1, historical_station_dataset.station_count
)  # time-step X station
labels_plot = torch.cat(labels_plot, dim=0).reshape(
    -1, historical_station_dataset.station_count
)

predictions_plot_byStation = np.transpose(predictions_plot.numpy())  # station x time-step
labels_plot_byStation = np.transpose(labels_plot.numpy()).tolist()


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
# plt.ylim([0.0, 0.01])
plt.xlabel("Offset (10 mins interval)",fontsize = 12)
# plt.scatter(plotx_[0][0:input_seq_len], input_.tolist()[0], s=90, label='input')

mean_,std_dev_ = mean_stddev_(losses_rmse)
plt.scatter([i for i in range(len(mean_))], mean_,s=50)
plt.errorbar([i for i in range(len(mean_))], mean_,yerr =std_dev_)

# for x in range(len(losses_rmse)):
#     plt.scatter([x for i in range(len(losses_rmse[x]))], losses_rmse[x],s=50)
# plt.scatter([x for x in range(len(data[0]))], data[0], s=80, label="output")
# plt.scatter([x[-1] for x in plotx_], predictions, s=90,label='output')
# plt.scatter(, data[train_len:], label='valid')
ax.legend()
# plt.show()
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