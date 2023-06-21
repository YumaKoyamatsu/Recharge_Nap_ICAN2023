import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import serial
import time
import logging
import coloredlogs
import numpy as np
import pandas as pd
import scipy as sp
from scipy.integrate import simps
from scipy import interpolate, signal
import matplotlib.pyplot as plt
import threading

logger = logging.getLogger('app')
coloredlogs.install(level='INFO')

class estimSleepStage:
    def __init__(self, window_time=60, step_time=10, max_time=35):
        #Init Serial connection
        self.ser = serial.Serial("COM4", 115200, timeout = 0.1)
        self.ser.reset_input_buffer() #シリアルバッファ初期化
        
        # Constant
        self.window_time = window_time # 解析する窓時間
        self.step_time = step_time # 窓のずらし時間時間
        self.fs = 50 #Hz, 生データのサンプリング周波数
        self.lag_win = self.fs * self.window_time #15000, ラグ特徴量のウィンドウ幅
        self.max_time = max_time + 5 #min., 最大計測時間.余分に取っている．
        self.max_elements = self.lag_win*self.max_time #450000, 最大データ数
        
        # variable init.
        self.data = np.zeros([self.max_elements, 3])
        self.vlf_hi_list = []
        self.lf_list = []
        self.hf_list = []
        self.hf_per_all_freq_list = []
        self.thread_stop_flag = False
        self.estim_count = 0
        
        logger.info("Initialization of the model finished.")
        
    def start(self):
        logger.info("Start sensing.")
        #Threadの開始処理
        self.thread = threading.Thread(target=self.get_sensor_data)
        self.thread.daemon = True
        self.thread.start()
        return 0
        
    def close(self):
        logger.info("Finish sensing.")
        #Threadの終了処理
        self.thread_stop_flag = True
        self.thread.join()
        #シリアル通信の終了処理
        self.ser.close()
        return 0

    def estim(self):
        #過去60秒分のPPG,accを読み込み
        ppg_win = self.data[self.estim_count:self.estim_count+self.lag_win, 1].copy()
        acc_win = self.data[self.estim_count:self.estim_count+self.lag_win, 2].copy()
        
        #ノイズデータ削除処理
        thresh_noise = 10 # G
        noise_boolean = acc_win > thresh_noise
        ppg_win[ppg_win==noise_boolean] = 300
        #Lowpass-filter
        ppg_lowpass = self.lowpass(ppg_win, samplerate=self.fs, fp=2, fs=10, gpass=3, gstop=40)
        # find peaks
        peaks_time = signal.find_peaks(ppg_lowpass)[0] / self.fs * 1000
        # Calc raw RRI
        rri_raw = np.diff(peaks_time)
        # リサンプリング
        x, y = peaks_time[:-1], rri_raw
        func = interpolate.interp1d(x, y, kind="cubic")
        x_interp = np.linspace(x[0], x[-1], len(y))
        y_interp = func(x_interp)
        # FFT
        fs_interp = 1000 / np.diff(x_interp)[0]
        freq, ps = self.FFT_PSD(y_interp, fs_interp)
        # 各指標計算
        vlf_hi = self.cal_VLF_hi(ps, freq)
        lf = self.cal_LF(ps, freq)
        hf = self.cal_HF(ps, freq)
        hf_per_all_freq = hf / (vlf_hi + lf + hf)
        
        #保存
        self.vlf_hi_list.append(vlf_hi)
        self.lf_list.append(lf)
        self.hf_list.append(hf)
        self.hf_per_all_freq_list.append(hf_per_all_freq)

        #睡眠段階判定
        sleep_stage = 0 if hf_per_all_freq < 0.6 else 1
        
        #推論回数カウンタ更新
        self.estim_count += self.step_time * self.fs
        return sleep_stage
    
    def get_sensor_data(self):
        for i in range(self.max_elements):
            
            if self.thread_stop_flag:
                break
            
            payload = self.ser.readline().decode('utf-8').replace('\n', '')
            t_sensor, ppg, acc = float(payload.split(",")[0]), float(payload.split(",")[1]), float(payload.split(",")[2])
            self.data[i, 0], self.data[i, 1], self.data[i, 2] = t_sensor, ppg, acc
    
    def lowpass(self, x, samplerate, fp, fs, gpass, gstop):
        fn = samplerate / 2                           #ナイキスト周波数
        wp = fp / fn                                  #ナイキスト周波数で通過域端周波数を正規化
        ws = fs / fn                                  #ナイキスト周波数で阻止域端周波数を正規化
        N, Wn = signal.buttord(wp, ws, gpass, gstop)  #オーダーとバターワースの正規化周波数を計算
        b, a = signal.butter(N, Wn, "low")            #フィルタ伝達関数の分子と分母を計算
        y = signal.filtfilt(b, a, x)                  #信号に対してフィルタをかける
        return y
    
    def FFT_PSD(self, y, fs):
        N = len(y)
        delta_f = fs / N
        y_fft = np.fft.fft(y) # 離散フーリエ変換
        freq = np.fft.fftfreq(N, d=1/fs) # 周波数を割り当てる（※後述）
        psd = abs(y_fft/(N/2)) ** 2 / delta_f # 音の大きさ（振幅の大きさ）
        return freq[1:int(N/2)], psd[1:int(N/2)]
    
    # VLF-hi
    def cal_VLF_hi(self, y, x):
        VLF_hi_th = (0.016666666666666666666666, 0.04)
        idx = (x >= VLF_hi_th[0]) & (x <= VLF_hi_th[1])
        return simps(y[idx], x[idx])
    
    # LF
    def cal_LF(self, y, x):
        lf_th = (0.04, 0.15)
        idx = (x >= lf_th[0]) & (x <= lf_th[1])
        return simps(y[idx], x[idx])
    
    # HF
    def cal_HF(self, y, x):
        hf_th = (0.15, 0.4)
        idx = (x >= hf_th[0]) & (x <= hf_th[1])
        return simps(y[idx], x[idx])

if __name__ == '__main__':
    window_time = 30
    step_time = 5
    
    model = estimSleepStage(window_time, step_time)
    
    elapsed_time = 0
    initial_time = time.perf_counter_ns()# 計測開始時間
    
    # センサデータ取得開始
    model.start()
    
    #最初の5分はデータが溜まるまで待つ
    while elapsed_time < window_time:
        elapsed_time = (time.perf_counter_ns() - initial_time) / 1000000000
        print(elapsed_time)
    
    for i in range(5):
        sleep_stage = model.estim()
        print(sleep_stage)
        time.sleep(step_time)
    
    #センサデータ取得終了
    model.close()
    
    print(model.data)
    plt.plot(model.data[:, 1])
    plt.show()