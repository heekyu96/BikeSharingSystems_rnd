#!/usr/bin/env python3
# from selectors import EpollSelector
import torch
from   torch.utils.data import Dataset
import numpy as np
import glob
import csv
import matplotlib.pyplot as plt
from datetime import datetime
import time
import math
from timeit import default_timer as timer
from event_prediction_utils import _progressbar_printer
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import find_peaks, peak_prominences

torch.manual_seed(2024)
np.random.seed(2024)

INTERVAL = 600
SLOTS_PER_DAY = 24 * 60 * 60 / INTERVAL

class StationDataset(Dataset):
    def __init__(self, datapath, maxfiles, ilen, olen, boundary, device, station_selected=None):
        self.path = datapath
        self.maxfiles = maxfiles
        self.input_len = ilen
        self.output_len = olen
        self.boundary = boundary
        self.device = device
        self.station_selected = station_selected
        self.filenames = sorted(glob.glob(self.path+"use_historical/*.csv"))
        if self.maxfiles > 0: 
            self.filenames = self.filenames[:self.maxfiles]

        self.normalize_station_data()

        if len(self.bikes) < self.input_len + self.output_len: 
            print("Too little data. It needs at least %d points, but the data contains only %d points" % (
                 self.input_len + self.output_len, len(self.bikes)))
            sys.exit()

        self.max_val = np.amax(self.bikes)
        self.station_count = len(self.stations)

        self.generate_bike_amount_seq()

        self.dataset_total_size = len(self.sized_seqs)
        # print("self.dataset_total_size =", self.dataset_total_size)
        # self.dataset_size = dataset_size
        # self.train_or_test = train_or_test
        self.division_ratio = 0.8
        self.mode = ""
        # to generate the sequence for train and test
        train_len = int(self.dataset_total_size * self.division_ratio)
        self.train_input, self.train_output, self.train_output_categorical = self.dataset_sequence(0, train_len)
        self.test_input,  self.test_output,  self.test_output_categorical  = self.dataset_sequence(train_len, self.dataset_total_size)
        # print("total data size:   ", self.dataset_total_size)
        # print("train len:         ", train_len)        
        # print("train input size:  ", len(self.train_input))
        # print("output length:     ", self.output_len)
        format = "%s IO shape. input: %d x %s, output %d x %s\n"
        print(format % ("Train", len(self.train_input), self.train_input[0].shape, 
                        len(self.train_output), self.train_output[0].shape))
        print(format % ("Test", len(self.test_input), self.test_input[0].shape, 
                        len(self.test_output), self.test_output[0].shape))
        return 
    
    # normaliz the historical station data
    def normalize_station_data(self):
        sid = 0 
        stations_dict = {}
        tm_first = datetime.now().timestamp()
        tm_last  = 0
        print("reading ", end="", flush=True)
        for fname in self.filenames: 
            print(".", end='', flush=True)
            with open(fname, "r", encoding="utf-8") as file:
                lines = csv.reader(file)
                for item in lines:
                    skey = item[2].replace(" ", "").lower()
                    dateobj = datetime.strptime(item[1], "%m/%d/%Y %I:%M:%S %p")
                    tm = int(dateobj.timestamp())
                    tm_first = min(tm_first, tm)    # to find the earliest time
                    tm_last  = max(tm_last, tm)     # to find the latest timestamp
                    if not skey in stations_dict.keys(): 
                        stations_dict[skey] = {}
                        stations_dict[skey]['id'] = sid
                        stations_dict[skey]['docks'] = int(item[4])
                        stations_dict[skey]['latit'] = float(item[10])
                        stations_dict[skey]['longi'] = float(item[11])
                        sid += 1
        num_stations = len(stations_dict)
        self.offset = (tm_first // INTERVAL) * INTERVAL
        slots = (tm_last - self.offset) // INTERVAL
        max_slots = int(math.ceil(slots / SLOTS_PER_DAY)*SLOTS_PER_DAY)

        # to collect the bikes count for each station for every INTERVAL seconds
        # to use numpy 2d array to take advantage of the library
        # If data is imperfect with missing data, the corresponding slots contains zero.
        # The k-th column, or bikes[][k], is the number of bike on the timestamp of
        # (k * INTERVAL + self.offset)
        self.bikes = np.zeros((max_slots, num_stations), dtype=np.int32)
        for fname in self.filenames: 
            print(".", end='', flush=True)
            with open(fname, "r", encoding="utf-8") as file:
                lines = csv.reader(file)
                for item in lines:
                    skey = item[2].replace(" ", "").lower()
                    sid = stations_dict[skey]['id']
                    dateobj = datetime.strptime(item[1], "%m/%d/%Y %I:%M:%S %p")
                    tm = int(dateobj.timestamp())
                    tag = (tm - self.offset) // INTERVAL
                    self.bikes[tag][sid] = int(item[7])
        print(" done.", flush=True)
        # to convert station dict info to a list. 
        # The index of the list is the station ID, identical to the row index of bikes array
        self.stations = [None] * num_stations
        for k, x in stations_dict.items():
            self.stations[x['id']] = {'name': k, 'docks': x['docks'], 
                                'latit':x['latit'], 'longi':x['longi']}
            
        # print("num_stations =", num_stations)    
        # print("the first time ", tm_first, time.ctime(tm_first))
        # print("the last  time ", tm_last,  time.ctime(tm_last))
        # print("offset   = %d"%self.offset)
        # print("max_slots= %d"%max_slots)
        # print("shape of bikes =", self.bikes.shape)
        # print("stations\n", self.stations[:4])
        # print("bikes\n", self.bikes)
        # np.save("bikes", self.bikes)
        # np.save("stations", self.stations)
        return

    def generate_bike_amount_seq(self):
        scaler = MinMaxScaler()
        self.normalized_data = scaler.fit_transform(self.bikes)
        cap_docks = np.array([x['docks'] for x in self.stations])
        prate = self.bikes / cap_docks
        dl_category = np.digitize(prate, self.boundary)
        self.sized_seqs = []
        self.sized_seqs_categorical = []
        seq_length = self.input_len + self.output_len

        for i in range( len(self.normalized_data) -  seq_length):
            self.sized_seqs.append(self.normalized_data[i : i+seq_length])
            self.sized_seqs_categorical.append(dl_category[i : i+seq_length])

        # print("shape of sized_seqs = %d x %s" % (len(self.sized_seqs), self.sized_seqs[0].shape))
        # print("shape of cap_docks=", cap_docks.shape)
        # print("bikes=\n", self.bikes[:4])
        # print("cap_docks=\n", cap_docks)
        # print("shape of prate =", prate.shape)
        # print(prate)
        # print("shape of dl_categorical=", dl_category.shape)
        # print(dl_category)
        # print("input_length=%d, output_len=%d"%(self.input_len, self.output_len))
        # print("length of normalized_data =", len(self.normalized_data))
        # print(self.normalized_data[:4])
        # print("len of sized_seq =", len(self.sized_seqs))
        # print(self.sized_seqs[:4])
        return 
    
    def dataset_sequence(self, start, end):
        input  = []
        output = []
        categ  = []
        for j, seq in enumerate(zip(self.sized_seqs[start:end], self.sized_seqs_categorical[start:end])):
            # start_time = timer()
            for i in range(self.output_len):
                input.append(seq[0][i : i+self.input_len])
                output.append(seq[0][i+self.input_len : i+self.input_len+1])
                categ.append(torch.tensor(seq[1][i+self.input_len : i+self.input_len+1]))
            # end_time = timer()
            # test_data_load_progress._progressBar(j+1, end_time - start_time)        
        return input, output, categ

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
                torch.FloatTensor(self.train_input[index]).to(self.device),
                torch.FloatTensor(self.train_output[index]).to(self.device),
                # torch.LongTensor(self.train_output_categorical[index]).to(self.device),
                self.train_output_categorical[index].type(torch.long).to(self.device),
            )
        elif self.mode == "test":
            return (
                torch.FloatTensor(self.test_input[index]).to(self.device),
                torch.FloatTensor(self.test_output[index]).to(self.device),
                # torch.LongTensor(self.test_output_categorical[index]).to(self.device),
                self.test_output_categorical[index].type(torch.long).to(self.device),
            )

if __name__ == "__main__":
    import sys
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = "../03.Data/"
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
    station_selected = {'larrabeest&kingsburyst', 'orleansst&merchandisemartplaza'}
    boundary = [.0, .2, .4, .6, .8, 1.0]
    print(str(DEVICE) + " is working.")

    ds = StationDataset(data_path, file_count, input_seq_len, output_seq_len, boundary, DEVICE)
    # ds = StationDataset(data_path, file_count, input_seq_len, output_seq_len, boundary, DEVICE, station_selected)    
