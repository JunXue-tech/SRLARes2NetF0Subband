import librosa
import numpy as np
from scipy import signal
#import matplotlib.pyplot as plt
# import pyworld as pw
import torch
import torch.nn.functional as F
import torch.nn as nn
sample_rate = 8000
win_length = 1728
hop_length = 130

def extract(wav_path, feature_type):
    if feature_type == "f0":
        return extract_F0_subband(wav_path)

def extract_F0_subband(wav_path):
    p_preemphasis = 0.97
    num_freq = 1728
    def preemphasis(x):
        return signal.lfilter([1, -p_preemphasis], [1], x)
    def _stft(y):
        return librosa.stft(y=y, n_fft=num_freq, hop_length=hop_length, win_length=win_length, window=signal.windows.blackman)
    y, _ = librosa.load(wav_path, sr=None)
    if(len(y)>80000):
        y=y[0:80000]
    #print(y.shape)
    D = _stft(preemphasis(y))
    #print(D.shape)
    S = np.log(abs(D)+np.exp(-80))
    total=S.shape[1]
    if total<600:
        for j in range(0,600//total):
            if j==0:
                S=np.hstack((S,np.fliplr(S)))
            else:
                S=np.hstack((S,S))
                if S.shape[1]>=600:
                    break
    feature=S[0:45,0:600]
    return np.reshape(np.array(feature), (-1, 45, 600))

