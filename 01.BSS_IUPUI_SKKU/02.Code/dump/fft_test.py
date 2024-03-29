import numpy as np
import matplotlib.pyplot as plt
from event_prediction_data import StationDataset, StationDataset_beta
import sys
import torch

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
boundary = [0.00,0.10,0.25,0.50,0.75,0.90,sys.maxsize/100]



# Dataset Generation
historical_station_dataset = StationDataset_beta(
    file_count, input_seq_len, output_seq_len, boundary, DEVICE, station_cnt_lock
)




#################################################
def sin_wave(amp, freq, time):
    return amp * np.sin(2*np.pi*freq*time)

time = np.arange(0, 10, 0.001)
sin1 = sin_wave(1, 10, time)
sin2 = sin_wave(2, 5, time)
sin3 = sin_wave(4, 1, time)

sin_sum = sin1 + sin2 + sin3

n = len(sin_sum) 
k = np.arange(n)
Fs = 1/0.001
T = n/Fs
freq = k/T 
freq = freq[range(int(n/2))]

Y = np.fft.fft(sin_sum)/n 
Y = Y[range(int(n/2))]

fig, ax = plt.subplots(2, 1, figsize=(12,8))
ax[0].plot(time, sin_sum)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Amplitude'); ax[0].grid(True)
ax[1].plot(freq, abs(Y), 'r', linestyle=' ', marker='^') 
ax[1].set_xlabel('Freq (Hz)')
ax[1].set_ylabel('|Y(freq)|')
ax[1].vlines(freq, [0], abs(Y))
ax[1].set_xlim([0, 20]); ax[1].grid(True)
plt.savefig(
    "./04.event_prediction/event_prediction_results/"
    + "fft_test.png",
    dpi=500,
    edgecolor="white",
    bbox_inches="tight",
    pad_inches=0.2,
)