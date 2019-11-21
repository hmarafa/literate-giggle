# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy.signal import butter, lfilter, freqz , find_peaks


def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_bandstop_filter(data, lowcut1, highcut1, fs, order):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        i, u = butter(order, [low, high], btype='bandstop')
        y = lfilter(i, u, data)
        return y

path_to_file="/Users/hanyarafa/Box/Northwestern/Rogers/Mechanoacoustic/Swallow/DRS Abstract/TV_F_1316_LV/TV_dev_left.tsv"
sensordata= pd.read_csv( path_to_file , delimiter = '\t')
sd = pd.DataFrame(sensordata)
sd = sd.drop(['gyro x' , 'gyro y' , 'gyro z'], axis=1)

path_to_event="/Users/hanyarafa/Box/Northwestern/Rogers/Mechanoacoustic/Swallow/DRS Abstract/TV_F_1316_LV/event.tsv"
eventmarker= pd.read_csv( path_to_event , delimiter = '\t')
ev = pd.DataFrame(eventmarker)
ev = ev.drop(['event'], axis=1)
print(ev.keys())

#time = sd.loc[: , 'local time' ]
#z = sd.loc[: , 'accel z'] 
#plt.plot(time, z) #plot raw data 


fs_z = 1600 #sampling frequency in the z axis
fs_xy = 200 # sampling frequency in x/y axes
b_conv = 4/(2**16)
t = sd.loc[: , 'local time' ] #Used for event markers
a_x = sd.loc[:, 'accel x' ]
a_y = sd.loc[:, 'accel y' ]
a_z = sd.loc[:, 'accel z' ]

t_z = np.arange(len(a_z)) / fs_z
a_x= a_x * b_conv
a_y= -a_y * b_conv
a_zc= -a_z * b_conv
evmark = ev.loc[:, 'local time'] / 1000
#ev_mark = np.arange(len(a_z)) / fs_z
#evmark = (1:length(az_c))/fs; 

lowcut = 0.1
highcut = 150
lowcut1 = 0.8
highcut1 = 90

plt.figure(1)   
plt.ioff()
az_bp = butter_bandpass_filter(a_zc, lowcut, highcut , fs_z, order=3)
plt.plot(t/1000, az_bp, label='Filtered signal (%g Hz)')
#plt.xlabel('time (seconds)')
#plt.hlines([-a, a], 0, t_z, linestyles='--')
#plt.grid(True)
plt.legend(loc='upper left')
plt.show()

plt.figure(2)
plt.ioff()
az_bp_bs = butter_bandstop_filter(az_bp, lowcut1, highcut1 , fs_z, order=3)
plt.plot(t/1000, az_bp_bs, label='Filtered signal (%g Hz)')
plt.xlabel('Time (seconds)')
#plt.hlines([-a, a], 0, t/1000, linestyles='--')
#plt.grid(True)
plt.legend(loc='upper left')
plt.show()


#compute standard deviation and ignore any NaN values along the axis for threshold
tlv = np.nanstd(az_bp_bs)
print(tlv)
#peaks, _ = find_peaks(az_bp_bs, distance=15000)
#Interval between swallows cannot be less than 1 second
peaks4, _ = find_peaks(az_bp_bs, threshold=tlv, distance=2400)
#peaks4 = np.true_divide(peaks4, 1600)
plt.ion()
plt.rcParams['interactive']
fig, ax = plt.subplots()

print(peaks4)
plt.plot(t_z, az_bp_bs)
plt.plot(peaks4/1600, az_bp_bs[peaks4], "x")
#plt.plot( az_bp_bs[peaks4], "x")
ax.vlines(evmark, -1, 1, color='r')

