import time
import pygame
from mymodels import estimSleepStage
from random import randint
import numpy as np
import threading
from util import get_now_time
import os
import pickle as pk

#init params
Deep_Sleep_Co = 0

sleep_stage = 0         # Rem = 0, NonRem = 1
sleep_stage_Es = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
sleep_stage_Su = 0
sleep_stage_Av = []      # available
i = -1
j = 0
k = 0
finish = 0

Deep_sleep = 0

start_a_nap = 1         # GPIO set suru.     teikou hitsuyou

Score = 30

initialize = 0          # hirune kaishi shokika
finish = 0              # hirune shuryouji

def startanap(rain_sound):        #昼寝を始める
    ch_rain = rain_sound.play(loops=-1)
    ch_rain.set_volume(1)
    return 0

def SleepStage(model, time_interval, num_inference):
    global i
    global j
    global k
    global sleep_stage_Su
    global sleep_stage_Es
    global sleep_stage_Av
    global Deep_sleep
    global finish
    global Score
    global Deep_Sleep_Co

    sleep_stage = model.estim() # Get Sleep Stage

    i = i + 1
    
    print(sleep_stage)
    sleep_stage_Es[i] = sleep_stage
    sleep_stage_Su = sleep_stage_Su + sleep_stage_Es[i]

    if i == num_inference - 1 :
        if sleep_stage_Su / num_inference >= 0.5:
            ssav = 1
        else:
            ssav = 0
        sleep_stage_Av.append(ssav)
        print(sleep_stage_Av)
        
        if Deep_sleep == 1 and np.round(sleep_stage_Su / num_inference ,decimals=0) == 1:
            Score = Score + 5
        
        i = -1
        k = k + 1
        sleep_stage_Su = 0
        
        if Deep_sleep == 1:
            Deep_Sleep_Co = Deep_Sleep_Co + 1
            if Deep_Sleep_Co == 10:
                finish = 1

        if Deep_sleep == 0:
            if len(sleep_stage_Av) >= 8:
                for j in range(len(sleep_stage_Av)-8):
                    if sleep_stage_Av[j+5] == sleep_stage_Av[j+6] == sleep_stage_Av[j+7] == 1:
                        Deep_sleep = 1
                        Deep_Sleep_Co = 3
                        Score = 65
                        print("DEEPSLEEP")

def wakeup(wake_sound):  
    global Score
    wake_sound.play(loops=-1)
    print(Score)
    return 0

if __name__ == "__main__":
    #データ保存用ディレクトリ作成
    base_dir = f"result/{get_now_time()}"
    os.makedirs(base_dir, exist_ok=True)
    
    # Pygame init
    pygame.mixer.init(frequency = 44100)
    rain_sound = pygame.mixer.Sound("sound/rain.mp3")
    wake_sound = pygame.mixer.Sound("sound/bird_tweet.mp3")

    # init sleep stage estimation model
    window_time = 30 # s. 解析窓
    step_time = 10 # s. 解析窓のステップ数
    num_inference = int(window_time / step_time)
    max_time = 2 # min.
    model = estimSleepStage(window_time=window_time,
                            step_time=step_time,
                            max_time=max_time)
    
    while(1):
        if start_a_nap == 1:      # switch  ga  osaretara  1 ni naru

            if initialize == 0:   # shokika
                
                startanap(rain_sound)
                start_time = time.time()
                elapsed_time = 0

                # センサデータ取得開始
                model.start()
                
                #最初の窓数分のデータが溜まるまで待つ
                while elapsed_time < window_time:
                    elapsed_time = time.time() - start_time
                    print(elapsed_time)
                    
                initialize = 1

            SleepStage(model=model, 
                    time_interval=step_time, 
                    num_inference=num_inference,
                    )          
            print(time.time() - start_time)
            time.sleep(step_time)
            
            if time.time() - start_time >= max_time*60 or finish == 1:  # 30fun tattara okosu 
                rain_sound.stop()
                wakeup(wake_sound)
                model.close()
                break
    
    
    #データ保存処理
    
    ## Score保存
    with open(f"{base_dir}/score.pkl", "wb") as f:
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
    
    
    
    
    


