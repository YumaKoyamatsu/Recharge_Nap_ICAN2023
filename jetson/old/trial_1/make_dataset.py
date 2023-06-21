import wfdb
import numpy as np
from biosppy.signals import ecg
import scipy
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import tarfile

#ANNOT_CONVERT_DICT = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4} #数値ラベル定義
ANNOT_CONVERT_DICT = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4} #数値ラベル定義
FREQ = 200
NUM_INPUT = 60
EXTRA_TIME = 3

def load_all_data(path):
    sig = wfdb.rdsamp(path) #signalデータ
    annot = wfdb.rdann(path, extension="arousal") #Annotationデータ

    ecg_sig = sig[0][:, -1] #ECGデータ
    
    annot_idx = annot.sample[annot.chan == 0] #睡眠ステージアノテーションのインデックス番号
    annot_stage = [x for x, b in zip(annot.aux_note, annot.chan==0) if b]
    annot_stage = [ANNOT_CONVERT_DICT[x] for x in annot_stage] #睡眠ステージアノテーションデータ

    return ecg_sig, annot_idx, annot_stage

def make_RRI_dataset(mode, subject, ecg_sig, annot_idx, annot_stage):
    seq_sig_list = []
    stage_list = []
    
    #annot_idxの数分ループする

    try:        
        for idx, stage in tqdm(zip(annot_idx, annot_stage)):
            start_idx = idx - NUM_INPUT*FREQ - EXTRA_TIME*FREQ
            end_idx = idx + EXTRA_TIME*FREQ
            
            # アノテーションインデックスの前30秒間のデータを抽出．
            # この時，補完処理用に前後に3秒間余分に収集．
            ecg_element = ecg.ecg(
                                    signal=ecg_sig[start_idx:end_idx], 
                                    sampling_rate=FREQ, 
                                    show=False, 
                                    interactive=False
                                )
            
            (ts, filtered, rpeaks, templates_ts, templates, heart_rate_ts, heart_rate) = ecg_element
            
            #RRI検出
            #rpeaks_christov, = ecg.christov_segmenter(filtered, FREQ) #一番検出精度がいいらしい
            rpeaks_engzee, = ecg.engzee_segmenter(filtered, FREQ) #こっちの方がロバストだった
            ts_peaks = ts[rpeaks_engzee]
            rri = np.diff(ts_peaks) * 1000 #[ms]

            spline_func = scipy.interpolate.interp1d(ts_peaks[:-1], rri, kind='linear', bounds_error=False, fill_value='extrapolate')

            ts_1sec = np.arange(0, NUM_INPUT+2*EXTRA_TIME, 1) #1秒毎のタイムシーケンス
            rri_tmp = spline_func(ts_1sec) #スプライン補完
            
            #データを前後に3秒間余分にとっていたので削除
            rri = rri_tmp[EXTRA_TIME:-EXTRA_TIME]
            
            #欠損判定
            isnan_result = np.isnan(rri)
            contains_nan = np.any(isnan_result)
            if contains_nan:
                continue
            
            #RRI値,Stage格納
            seq_sig_list.append(rri)
            stage_list.append(stage)
        
        #ndarray形式に変換
        seq_sig_list = np.array(seq_sig_list)
        stage_list = np.array(stage_list)
        
        dataset = {"sig":seq_sig_list, "annot":stage_list}
        
        #保存
        np.save(f"dataset\\YSYW_seq\\{mode}\\processed_{subject}.npy", dataset)
    
    except ValueError:
        pass

def make_ECGWave_dataset(mode, subject, ecg_sig, annot_idx, annot_stage):
    seq_sig_list = []
    stage_list = []
    for idx, stage in tqdm(zip(annot_idx, annot_stage)):
        start_idx = idx - NUM_INPUT*FREQ
        end_idx = idx
        try:
            (_, ecg_elem, _, _, _, _, _) = ecg.ecg(
                                        signal=ecg_sig[start_idx:end_idx], 
                                        sampling_rate=FREQ, 
                                        show=False, 
                                        interactive=False
                                    )
        except ValueError:
            continue
        
        #print(ecg_elem.shape)
        
        #mean = np.mean(ecg_elem, axis=0)  # 平均値を計算
        #std = np.std(ecg_elem, axis=0)    # 標準偏差を計算
        #ecg_elem = (ecg_elem - mean) / std
        
        #欠損判定
        isnan_result = np.isnan(ecg_elem)
        contains_nan = np.any(isnan_result)
        if contains_nan:
            print("nan detected")
            continue
        
        seq_sig_list.append(ecg_elem)
        stage_list.append(stage)
        
    seq_sig_list = np.array(seq_sig_list)
    stage_list = np.array(stage_list)
        
    dataset = {"sig":seq_sig_list, "annot":stage_list}
    
    #保存
    np.save(f"dataset\\YSYW_seq\\{mode}\\processed_{subject}.npy", dataset)
    

if __name__ == "__main__":
    
    subjects = os.listdir("dataset\\YSYW")
    mode = "RRI"
    
    if not os.path.exists(f"dataset\\YSYW_seq\\{mode}"):
        os.makedirs(f"dataset\\YSYW_seq\\{mode}")
    
    for subNo, subject in tqdm(enumerate(subjects)):
        print(subNo, subject)
        
        #ファイルがあればスキップ
        if os.path.exists(f"dataset\\YSYW_seq\\{mode}\\processed_{subject}.npy"):
            continue
        
        #ECGデータ，アノテーションインデックスデータ，アノテーションデータの読み込み
        ecg_sig, annot_idx, annot_stage = load_all_data(f"dataset\\YSYW\\{subject}\\{subject}")
        
        if mode == "RRI":
            make_RRI_dataset(mode, subject, ecg_sig, annot_idx, annot_stage)
        elif mode == "ECG":
            make_ECGWave_dataset(mode, subject, ecg_sig, annot_idx, annot_stage)
            
    with tarfile.open(f'dataset\\{mode}.tar.gz', 'w:gz') as tar:
        tar.add('dataset\\YSYW_seq\\ecg')