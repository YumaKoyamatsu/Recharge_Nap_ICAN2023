import time
import pygame
import mymodels
from random import randint
import numpy as np
import threading

pygame.mixer.init(frequency = 44100)

#estim = mymodels.estimSleepStage()

Deep_Sleep_Co = 0
now_time = 0

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

def startanap():        #昼寝を始める
    pygame.mixer.music.load("sound/rain.mp3")     # 音楽ファイルの読み込み
    pygame.mixer.music.play(1)              # 音楽の再生回数(1回)
    time.sleep(3)
    pygame.mixer.music.stop()               # 再生の終了
    print("startanap")

def SleepStage():
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
    global now_time
    global start_time

    time_interval = 10               # suiron no kankaku
    kaisu = 6                        # suiron 1 set no kaisu 

    if now_time - start_time >= time_interval*(i+2) + k*time_interval*kaisu :
        ###############################
        sleep_stage = randint(0, 1)#estim.forward()           # Get Sleep Stage

        i = i + 1
        
        print(sleep_stage)
        sleep_stage_Es[i] = sleep_stage
        sleep_stage_Su = sleep_stage_Su + sleep_stage_Es[i]

        if i == kaisu - 1 :
            if sleep_stage_Su / kaisu >= 0.5:
                ssav = 1
            else:
                ssav = 0
            sleep_stage_Av.append(ssav)
            print(sleep_stage_Av)
           
            if Deep_sleep == 1 and np.round(sleep_stage_Su / kaisu ,decimals=0) == 1:
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

def wakeup():  
    global Score
    pygame.mixer.music.load("sound/bird_tweet.mp3")
    pygame.mixer.music.play(1)
    time.sleep(3)
    pygame.mixer.music.stop()               # 再生の終了    
    print(Score)

if __name__ == "__main__":

    while(1):
        if start_a_nap == 1:      # switch  ga  osaretara  1 ni naru

            if initialize == 0:   # shokika
                startanap()
                start_time = time.time()
                now_time = time.time()
                initialize = 1

            SleepStage()          

            if now_time - start_time >= 30*60 or finish ==1:  # 30fun tattara okosu 
                wakeup()
                break

            now_time = time.time()
    
    #sleep_stage_Av.append(np.round(sleep_stage_Su / kaisu ,decimals=0))
    #print(sleep_stage_Av)
    #sleep_stage = randint(0,1)