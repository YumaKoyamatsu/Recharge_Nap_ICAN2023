import time
import pygame
from mymodels import estimSleepStage
from random import randint
import numpy as np
import os
import pickle as pk
import datetime

#init params
i = -1
j = 0
sleep_stage_Es = []
sleep_stage_Su = 0
sleep_stage_Av = []
Deep_sleep = 0
Score = 30
Deep_Sleep_Co = 3
finish = 0

sleep_stage = 0         # Wake = 0, NonRem = 1
start_a_nap = 1         # リチャージナップスタート
initialize = 0          # 初期化

def startanap(rain_sound):        #昼寝を始める
    ch_rain = rain_sound.play(loops=-1)
    ch_rain.set_volume(1)
    return 0

def SleepStage(i,j,sleep_stage_Es,sleep_stage_Su,sleep_stage_Av,Score,Deep_sleep,Deep_Sleep_Co,finish,model):
    # 20230618 Accが取れるように追加．dtype=ndarray, shape=(5*60*50)
    sleep_stage = model.estim() # Get Sleep Stage
    i = i + 1
    
    sleep_stage_Es.append(sleep_stage)
    sleep_stage_Su = sleep_stage_Su + sleep_stage_Es[i]

    if i == count - 1 :

        if sleep_stage_Su / count >= 0.5:
            ssav = 1
        else:
            ssav = 0
        sleep_stage_Av.append(ssav)
        print(sleep_stage_Av)
        
        i = -1
        sleep_stage_Es = []

        if Deep_sleep == 1:
            if np.round(sleep_stage_Su / count ,decimals=0) == 1:
                Score = Score + 5
            if Deep_Sleep_Co == 9:
                finish = 1
            Deep_Sleep_Co = Deep_Sleep_Co + 1

        sleep_stage_Su = 0

        if Deep_sleep == 0:
            if len(sleep_stage_Av) >= 3:
                for j in range(len(sleep_stage_Av)-2+5):
                    if sleep_stage_Av[j+5] == sleep_stage_Av[j+1+5] == sleep_stage_Av[j+2+5] == 1:
                        Deep_sleep = 1
                        Deep_Sleep_Co = 3
                        Score = 65
                        print("DEEPSLEEP")

    return i,j,sleep_stage_Es,sleep_stage_Su,sleep_stage_Av,Score,Deep_sleep,Deep_Sleep_Co,finish
                
def wakeup(Score,wake_sound):  
    wake_sound.play(loops=-1)
    print(Score)
    return 0

def get_now_time():
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))) # 日本時刻
    now = now.strftime('%Y-%m-%d-%H-%M-%S')
    return now

if __name__ == "__main__":
    #データ保存用ディレクトリ作成
    base_dir = f"result/{get_now_time()}"
    os.makedirs(base_dir, exist_ok=True)
    
    # Pygame init
    pygame.mixer.init(frequency = 44100)
    rain_sound = pygame.mixer.Sound("sound/rain.mp3")
    wake_sound = pygame.mixer.Sound("sound/mezamashi.wav")


    # init sleep stage estimation model
    window_time = 5*60 # s. 解析窓
    step_time = 10 # s. 解析窓のステップ数
    count = 60/step_time
    max_time = 30 # min.
    model = estimSleepStage(window_time=window_time,
                            step_time=step_time,
                            max_time=max_time)
    
    while(1):
        if start_a_nap == 1:      # リチャージナップ スタート

            if initialize == 0:   # 初期化
                # startanap(rain_sound)
                start_time = time.time()
                elapsed_time = 0

                # センサデータ取得開始
                model.start()
                
                #最初の窓数分のデータが溜まるまで待つ
                while elapsed_time < window_time:
                    elapsed_time = time.time() - start_time
                    if elapsed_time % 10:
                        print(elapsed_time)
                    
                initialize = 1
            
            i,j,sleep_stage_Es,sleep_stage_Su,sleep_stage_Av,Score,Deep_sleep,Deep_Sleep_Co,finish = \
            SleepStage( i,j,sleep_stage_Es,sleep_stage_Su,sleep_stage_Av,Score,Deep_sleep,Deep_Sleep_Co,finish,
                        model=model)
        
            print(Score)  
            
            #経過時間表示
            elapsed_time = time.time() - start_time
            if elapsed_time % 10:
                print(elapsed_time)
                
            time.sleep(step_time)
            
            if time.time() - start_time >= max_time*60 or finish == 1:  # 30分経過したら起こす
                rain_sound.stop()
                wakeup(Score,wake_sound)
                model.close()
                break
    

    #データ保存処理
    
    ## Score保存
    with open(f"{base_dir}/Score.pkl", "wb") as f:
        pk.dump(Score, f)
    ## モデル保存
    with open(f"{base_dir}/metrics.pkl", "wb") as f:
        metrics = {}
        metrics["data"] = model.data
        metrics["vlf"] = model.vlf_hi_list
        metrics["lf"] = model.lf_list
        metrics["hf"] = model.hf_list
        metrics["hf_per_all"] = model.hf_per_all_freq_list
        
        pk.dump(metrics, f)
    
    time.sleep(10)

