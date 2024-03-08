# from selectors import EpollSelector
import torch
from torch.utils.data import Dataset
import numpy as np
import glob
import csv
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from event_prediction_utils import _progressbar_printer
from event_prediction_utils import data_path
from scipy.signal import find_peaks, peak_prominences
torch.manual_seed(2024)
np.random.seed(2024)

dir_path = data_path+"use_historical/*.csv"

# Historical station dataset processing
def historical_station_logs(file_count, dir_path):
    #dir_path = data_path+"use_historical/*.csv"
    historical_data_load_progress = _progressbar_printer(
        "Station historical data", iterations=file_count
    )
    station_list = []
    station_total_dock_dict = {}  # for denomalization
    data_dict = {}
    location = {}
    max_val = 0
    
    for i, file in enumerate(sorted(glob.glob(dir_path))):
        #print(file)
        if i > file_count:
            break
        file = open(file, "r", encoding="utf-8")
        print(file)
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
            [6] Available Docks (이용 중인 자전거),
            [7] Available Bikes (이용 가능한 자전거),
            [8] Percent Full,
            [9] Status,
            [10-12] Latitude,Longitude,Location,
            [13] Record
            """
            # max_val: 데이터 내의 bike 최대 개수
            if max_val < int(line[7]):
                max_val = int(line[7])
            
            # station: 공백 제거
            station = line[2].replace(" ", "").lower()
            lat, lon = line[10], line[11]
            
            # station_list: Station Name (역 명) append
            if not station_list.__contains__(station):
                station_list.append(station)
                # station_total_dock_dict: {(STATION) : (TOTAL DOCKS)}
                station_total_dock_dict[station] = int(line[4])

            # data_dict: {(STATION) : (BIKES1, BIKES2, ...)}
            if not data_dict.__contains__(station):
                data_dict[station] = [int(line[7])]
            else:
                data_dict[station].append(int(line[7]))
                # time stamp 간격이 10분이니까 -> t, t+1, t+2, ...에 중복 존재하는 station key에 대해 bike counts value append
            
            if not location.__contains__(station):
                location[station] = [float(lat), float(lon)]
                
        # list_max: 가장 자주 STATION이 기록된 개수
        list_max = max([len(data_dict[d]) for d in data_dict])

        # padding ver.
        # for d in data_dict:
        #     len_ = len(data_dict[d])
        #     if len_ < list_max:
        #         for i in range(list_max - len_):
        #             # add padding
        #             data_dict[d].insert(0, data_dict[d][0])
        # removing ver.
        
        for d in data_dict:
            len_ = len(data_dict[d])
            if len_ < list_max:
                data_dict.pop(d)
                station_list.remove(d)
                station_total_dock_dict.pop(d)
                location.pop(d)

        end_time = timer()
    
        historical_data_load_progress._progressBar(i+1, end_time - start_time)
        
    print(        "Number of Stations: ", len(station_list)    )
    print("Number of time-steps: ", len(data_dict[station_list[0]]))
    
    return data_dict, station_list, station_total_dock_dict, location, max_val
    # historical_station_logs(file_count)
    # data_dict, station_list, station_total_dock_dict, max_val 반환

class StationDataset(Dataset):
    def __init__(self, file_cnt, dir_path, input_seq_len, output_seq_len, boundary, device, station_cnt_lock=-1):
        self.input_len = input_seq_len
        self.output_len = output_seq_len
        # self.upper_boundary = upper_bound
        # self.lower_boundary = lower_bound
        self.boundary = boundary
        self.device = device
        self.file_count = file_cnt
        self.dir_path = dir_path
        (
            self.data_dict,  # station data:list
            self.station_list,
            self.station_total_dock,
            self.location,
            self.max_val,
        ) = historical_station_logs(file_count=self.file_count, dir_path = self.dir_path)
        
        self.station_count = len(self.station_list)
        
        (
            self.normalized_data,
            self.sized_seqs,
            self.sized_seqs_categorical,
        ) = self.generate_bike_amount_seq(station_cnt_lock)  # idx 0:time_step, idx 1:all values of stations
        
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
            # lock: 인덱스
            # station_name: data_dict의 key (station name)
            
            if lock == station_cnt_lock:
                self.station_count = station_cnt_lock
                break
            
            # normalization with overall max_val 정규화
            data_list.append([x / self.max_val for x in self.data_dict[station_name]])
            # create a label list
            cate_temp = []
            # [x / self.station_total_dock[station_name] for x in self.data_dict[station_name]]
            for x in self.data_dict[station_name]:
                # per: 각 역의 BIKE1, BIKE2, ... 를 비율로 바꿈
                per = x / self.station_total_dock[station_name]

                # boundary: 범주에 대한 list
                # ex) [0,20,40,60,80,100] 
                for ca in range(1,len(self.boundary)):
                    if int(per*100) in range(int(self.boundary[ca-1]*100),int(self.boundary[ca]*100)):
                        cate_temp.append(ca-1)
                        break

                # print(cate_temp)
                # if per >= self.upper_boundary:
                #     cate_temp.append(2)
                # elif per < self.upper_boundary and per > self.lower_boundary:
                #     cate_temp.append(1)
                # else:
                #     cate_temp.append(0)
            data_list_categorical.append(cate_temp)
            # print(len(data_list_categorical[-1]))
            # data_list_categorical.append([x / self.station_total_dock[station_name] for x in self.data_dict[station_name]])
        
        # data_list: station별 정규화한 bike 개수 list
        # data_list_categorical: station별 범주화한 bike 개수 list
        
        # data_list, data_list_categorical transpose 후 list화
        data_list = np.asarray(data_list)
        print(data_list.shape)
        data_list = np.transpose(data_list)
        print(data_list.shape)
        data_list = data_list.tolist()
        
        data_list_categorical = np.asarray(data_list_categorical)
        data_list_categorical = np.transpose(data_list_categorical)
        data_list_categorical = data_list_categorical.tolist()
        
        # 시퀀스 생성: self.input_len + self.output_len 길이만큼의 sequence 생성 (bike cnt와 cate 둘 다 적용)
        # self.input_len = 과거 데이터 길이
        # self.output_len = 예측 데이터 길이
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
        # combined_seqs = [[모든 station의 t1동안 수집된 bike 수 list], [모든 station의 t2동안 수집된 bike 수 list], ...]

    def dataset_gen(self):
        train_len = int(self.dataset_total_size * self.division_ratio)

        train_seqs = self.sized_seqs[:train_len] # sized_seqs: combined_seqs
        test_seqs = self.sized_seqs[train_len:]
        train_seqs_cate = self.sized_seqs_categorical[:train_len] # sized_seqs_categorical: combined_seqs_cate
        test_seqs_cate = self.sized_seqs_categorical[train_len:]
        self.train_seqs_cnt = len(train_seqs)
        self.test_seqs_cnt = len(test_seqs)

        train_data_load_progress = _progressbar_printer(
            "Train data sequence Sizing", iterations=len(train_seqs)
        )
        # input, output 분리
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
                        # torch.tensor(seq[1])
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
        for j, seq in enumerate(zip(self.sized_seqs[train_len:], self.sized_seqs_categorical[train_len:])):
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


class StationDataset_beta(Dataset):
    def __init__(self, file_cnt, input_seq_len, output_seq_len, boundary, device, station_cnt_lock=-1, train=False, fft_regen=False):
        self.input_len = input_seq_len
        self.output_len = output_seq_len
        # self.upper_boundary = upper_bound
        # self.lower_boundary = lower_bound
        self.boundary = boundary
        self.device = device
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
        )  # idx 0:time_step, idx 1:all values of stations
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
        
        if fft_regen:
            if train:
                print("Training input FFT Processing...")
                self.train_input_fft = self.fft_feature_gen(self.train_input, "Train")
            print("Testing input FFT Processing...")
            self.test_input_fft = self.fft_feature_gen(self.test_input, "Test")
        else:
            if train:
                print("Training input FFT Loading...")
                self.train_input_fft = self.fft_result_loader("Train")
            print("Testing input FFT Loading...")
            self.test_input_fft = self.fft_result_loader("Test")

        print(
            "Train I/O shape: ", len(self.train_input),"X",len(self.train_input[0]), "\t", len(self.train_output),"X",len(self.train_output[0])
        )
        print(
            "Test I/O shape: ", len(self.test_input),"X",len(self.test_input[0]), "\t", len(self.test_output),"X",len(self.test_output[0])
        )
        
    def fft_feature_gen(self, seqs, train_test):
        n = 1008 
        k = np.arange(n)
        Fs = 1/144
        T = n/Fs
        freq = k/T 
        freq = freq[range(int(n/2))]
        fft_results =[]
        
        f = open("./00.datasets/chicago_bike_dataset/"+train_test+"_fft.csv", "w",encoding="UTF-8")
        writer = csv.writer(f)
        fft_progress = _progressbar_printer(
            "FFT data sequence Sizing", iterations=len(seqs)
        )
        
        for i,seq in enumerate(seqs):
            start_time = timer()
            cvt = np.array(seq)[:,:].transpose()
            fft_result =[]
            for j,fft_ in enumerate(cvt):
                Y = np.fft.fft(fft_)/n
                Y = Y[range(int(n/2))]
                peaks, _ = find_peaks(Y)
                if len(peaks)==0:
                    fft_result.append([0.0 for x in range(10)])
                else:
                    prominences = peak_prominences(Y, peaks)[0]
                    prominence_new = np.percentile(prominences, [0, 25, 50, 75, 100], interpolation='nearest')[3]
                    # 새로 찾은 prominence를 이용해 다시 피크를 찾는다
                    new_peaks, _ = find_peaks(Y, prominence = prominence_new)
                    temp_ = abs(Y[new_peaks][:10]).tolist()

                    # self.fft_related_plotting(j, fft_, freq, Y, peaks, new_peaks)

                    if(len(temp_)<10):
                        for x in range(10-len(temp_)):
                            temp_.append(0)
                    fft_result.append(temp_)
            writer.writerow(fft_result) 
            fft_results.append(fft_result)
            end_time = timer()
            fft_progress._progressBar(i+1, end_time - start_time)
        
        f.close()
        return fft_results
    
    def fft_result_loader(self, train_test):
        csvfile_ = open("./00.datasets/chicago_bike_dataset/"+train_test+"_fft.csv", "r",encoding="UTF-8")
        csv_reader = csv.reader(csvfile_)
        return_ = []
        for row in csv_reader:
            if len(row)==0:
                continue
            ttemp_ = []
            for x in row:
                x = x.replace("[",'')
                x = x.replace("]",'')
                ttemp_.append([float(i) for i in x.split(',')])
            return_.append(ttemp_)
            # print(row)
            # print(len(row))
            # print(type(row))
        return torch.tensor(return_)
        
                
    def fft_related_plotting(self, i, fft_, freq, Y, peaks, new_peaks):
        fig, ax = plt.subplots(4, 1, figsize=(12,8))
        ax[0].plot(range(len(fft_)), fft_)
        ax[0].set_xlabel('Time')
        ax[0].set_ylabel('Amplitude'); ax[0].grid(True)
        ax[1].plot(freq, abs(Y), 'r', linestyle=' ', marker='^') 
        ax[1].set_xlabel('Freq (Hz)')
        ax[1].set_ylabel('Power')
        ax[1].vlines(freq, [0], abs(Y))
        ax[1].grid(True)
        ax[2].set_xlabel('Freq (Hz)')
        ax[2].set_ylabel('Power-Peaks')
        ax[2].vlines(freq[peaks], [0], abs(Y[peaks]))
        ax[2].grid(True)
        ax[3].set_xlabel('Freq (Hz)')
        ax[3].set_ylabel('Power-Prominence Peaks')
        ax[3].vlines(freq[new_peaks], [0], abs(Y[new_peaks]))
        ax[3].grid(True)
        plt.savefig(
            "./04.event_prediction/event_prediction_results/fft_results/"
            + "fft_"+str(self.station_list[i][:5])+".png",
            dpi=500,
            edgecolor="white",
            bbox_inches="tight",
            pad_inches=0.2,
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
                for ca in range(1,len(self.boundary)):
                    if int(per*100) in range(int(self.boundary[ca-1]*100),int(self.boundary[ca]*100)):
                        cate_temp.append(ca-1)
                        break
            data_list_categorical.append(cate_temp)

        data_list = np.asarray(data_list)
        print(data_list.shape)
        data_list = np.transpose(data_list)
        print(data_list.shape)
        data_list = data_list.tolist()
        data_list_categorical = np.asarray(data_list_categorical)
        data_list_categorical = np.transpose(data_list_categorical)
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
                        # torch.tensor(seq[1])
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
        for j, seq in enumerate(zip(self.sized_seqs[train_len:], self.sized_seqs_categorical[train_len:])):
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
                torch.FloatTensor(self.train_input[index]).to(self.device),
                torch.FloatTensor(self.train_output[index]).to(self.device),
                torch.FloatTensor(self.train_input_fft[index]).to(self.device),
                
                # torch.LongTensor(self.train_output_categorical[index]).to(self.device),
                self.train_output_categorical[index].type(torch.long).to(self.device),
            )
        elif self.mode == "test":
            return (
                torch.FloatTensor(self.test_input[index]).to(self.device),
                torch.FloatTensor(self.test_output[index]).to(self.device),
                torch.FloatTensor(self.test_input_fft[index]).to(self.device),
                # torch.LongTensor(self.test_output_categorical[index]).to(self.device),
                self.test_output_categorical[index].type(torch.long).to(self.device),
            )
