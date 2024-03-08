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
import os

torch.manual_seed(2024)
np.random.seed(2024)

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
        )
        # Print New Line on Complete
        if cur == self.iters:
            print("\nCompleted\n")

file_count = 10

# Historical station dataset processing
def historical_station_logs():
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
                # data_dict[station] = []

            if not data_dict.__contains__(station):
                # if not temp_data_dict.__contains__(station):
                # data_dict[station]=[float(int(line[8])/100)]
                data_dict[station] = [int(line[7])]
                # temp_data_dict[station] = [int(line[7])]
            else:
                # data_dict[station].append(float(int(line[8])/100))
                data_dict[station].append(int(line[7]))
                # temp_data_dict[station].append(int(line[7]))
        list_max = max([len(data_dict[d]) for d in data_dict])
        for d in data_dict:
            len_ = len(data_dict[d])
            if len_ < list_max:
                for i in range(list_max - len_):
                    # add padding
                    data_dict[d].insert(0, 0.5)
        end_time = timer()
        historical_data_load_progress._progressBar(i, end_time - start_time)
    print(
        "Number of Stations: ", len(station_list)
    )  # the number of stations : (10files)
    print("Number of time-stps: ", len(data_dict[station_list[0]]))
    return data_dict, station_list, station_total_dock_dict, max_val

class StationDataset(Dataset):
    def __init__(self, input_seq_len, output_seq_len):
        self.input_len = input_seq_len
        self.output_len = output_seq_len
        (
            self.data_dict,  # station data:list
            self.station_list,
            self.station_total_dock,
            self.max_val,
        ) = historical_station_logs()
        self.station_count = len(self.station_list)
        (self.normalized_data, self.sized_seqs,) = self.generate_bike_amount_seq()  # 0:time_step, 1:all values of stations
        self.dataset_total_size = len(self.sized_seqs)
        # self.dataset_size = dataset_size
        # self.train_or_test = train_or_test
        self.division_ratio = 0.8
        self.train_seqs_cnt = 0
        self.test_seqs_cnt = 0
        self.mode = ""
        # (
        #     self.train_input,
        #     self.train_output,
        #     self.test_input,
        #     self.test_output,
        # ) = 
        self.dataset_gen()

        # print(
        #     "Train I/O shape: ", self.train_input.shape, "\t", self.train_output.shape
        # )
        # print("Test I/O shape: ", self.test_input.shape, "\t", self.test_output.shape)

    def generate_bike_amount_seq(self, station_cnt_lock=-1):
        data_list = []
        for lock, station_name in enumerate(self.data_dict):
            if lock == station_cnt_lock:
                self.station_count = station_cnt_lock
                break
            # normalization with overall max_val
            data_list.append([x / self.max_val for x in self.data_dict[station_name]])
        data_list = np.asarray(data_list)
        print(data_list.shape)
        data_list = np.transpose(data_list)
        print(data_list.shape)
        data_list = data_list.tolist()
        combined_seqs = []
        for i in range(len(data_list) - self.input_len - self.output_len):
            combined_seqs.append(data_list[i : i + self.input_len + self.output_len])

        return data_list, combined_seqs

    def dataset_gen(self):
        # file exportation setup
        path = "./04.event_prediction/processed_datasets/"+str(self.input_len)+"_"+str(self.output_len)+"/"
        if not os.path.isdir(path):
            os.mkdir(path)
        
        file_meta_data = open(path + "/meta_data.csv", "w", newline="", encoding="utf-8",)
        meta_csv_writer = csv.writer(file_meta_data)
        meta_csv_writer.writerow([self.data_dict,  # station data:list
            self.station_list,
            self.station_total_dock,
            self.max_val])
        file_meta_data.close()
        
        train_len = int(self.dataset_total_size * self.division_ratio)

        train_seqs = self.sized_seqs[:train_len]
        test_seqs = self.sized_seqs[train_len:]
        self.train_seqs_cnt = len(train_seqs)
        self.test_seqs_cnt = len(test_seqs)

        # Train
        file_input_train = open(
            path + "/input_train.csv", "w", newline="", encoding="utf-8",
        )
        file_target_train = open(
            path + "/target_train.csv", "w", newline="", encoding="utf-8",
        )
        train_input_csv_writer = csv.writer(file_input_train)
        train_target_csv_writer = csv.writer(file_target_train)
        train_data_load_progress = _progressbar_printer(
            "Train data sequence Sizing", iterations=len(train_seqs)
        )
        train_input = []
        train_output = []
        for j,seq in enumerate(train_seqs):
            start_time = timer()
            for i in range(self.output_len):
                train_input.append(seq[i : i + self.input_len])
                train_input_csv_writer.writerow(train_input[-1])
                train_output.append(seq[i + self.input_len : i + self.input_len + 1])
                train_target_csv_writer.writerow(train_output[-1])
            end_time = timer()
            train_data_load_progress._progressBar(j, end_time - start_time)
        file_input_train.close()
        file_target_train.close()
        # Test
        file_input_test = open(
            path + "/input_test.csv", "w", newline="", encoding="utf-8",
        )
        file_target_test = open(
            path + "/target_test.csv", "w", newline="", encoding="utf-8",
        )
        test_input_csv_writer = csv.writer(file_input_test)
        test_target_csv_writer = csv.writer(file_target_test)
        test_data_load_progress = _progressbar_printer(
            "Test data sequence Sizing", iterations=len(test_seqs)
        )
        test_input = []
        test_output = []
        for j,seq in enumerate(test_seqs):
            start_time = timer()
            for i in range(self.output_len):
                test_input.append(seq[i : i + self.input_len])
                test_input_csv_writer.writerow(test_input[-1])
                test_output.append(seq[i + self.input_len : i + self.input_len + 1])
                test_target_csv_writer.writerow(test_output[-1])
            end_time = timer()
            test_data_load_progress._progressBar(j, end_time - start_time)
        file_input_test.close()
        file_target_test.close()

    # def set_train(self):
    #     self.mode = "train"

    # def set_test(self):
    #     self.mode = "test"

    # def __len__(self):
    #     if self.mode == "":
    #         print("Mode is not set")
    #         return None
    #     elif self.mode == "train":
    #         return len(self.train_input)
    #     elif self.mode == "test":
    #         return len(self.test_input)

    # def __getitem__(self, index):
    #     if self.mode == "":
    #         print("Mode is not set")
    #         return None
    #     elif self.mode == "train":
    #         return self.train_input[index], self.train_output[index]
    #     elif self.mode == "test":
    #         return self.test_input[index], self.test_output[index]
input_day = 7
output_day = 1
input_seq_len = input_day * 24 * 6
output_seq_len = output_day * 24 * 6

StationDataset(input_seq_len,output_seq_len)
print("\n","Completed")