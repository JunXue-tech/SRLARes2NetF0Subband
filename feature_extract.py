import librosa
import numpy as np
from scipy import signal

# 全局变量设置
sample_rate = 8000
win_length = 1728
hop_length = 130
num_freq = 1728
max_length = 80000
min_frames = 600

def extract(wav_path, feature_type):
    if feature_type == "f0":
        return extract_F0_subband(wav_path)

def extract_F0_subband(wav_path):
    """
    从音频文件中提取 F0 子带特征。
    1. 加载音频
    2. 预加重
    3. 短时傅里叶变换 (STFT)
    4. 特征矩阵调整
    """
    # 预加重系数
    p_preemphasis = 0.97
    
    # 加载音频文件并限制最大长度
    y, _ = librosa.load(wav_path, sr=None)
    y = y[:max_length]

    # 预加重函数
    def preemphasis(x):
        return signal.lfilter([1, -p_preemphasis], [1], x)

    # STFT变换
    def _stft(y):
        return librosa.stft(y=y, n_fft=num_freq, hop_length=hop_length, win_length=win_length, window=signal.windows.blackman)

    # 应用预加重和STFT
    D = _stft(preemphasis(y))
    S = np.log(np.abs(D) + np.exp(-80))  # 对数幅度谱

    # 填充特征矩阵至至少600帧
    total_frames = S.shape[1]
    if total_frames < min_frames:
        repeat_count = (min_frames // total_frames) + 1
        S = np.hstack([S] * repeat_count)  # 通过重复填充
    S = S[:, :min_frames]  # 裁剪到600帧

    # 提取前45个频率成分并返回
    feature = S[:45, :]
    return feature.reshape(1, 45, 600)  # 1个通道，45个频率，600帧
