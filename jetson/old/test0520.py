print("hello world")

import time
import pygame
import mymodels
from random import randint
import numpy as np
import threading

pygame.mixer.init(frequency = 44100)

# estim = mymodels.estimSleepStage()

start_time_Sl = time.time()
start_time_Es = time.time()
start_time_Av = time.time()

now_time_Sl = time.time()    # suimin zenbu jikan
now_time_Es = time.time()    # inference interval
now_time_Av = time.time()    # average interval

sleep_stage = 0         # Rem = 0, NonRem = 1
sleep_stage_Es = [0,0,0,0,0,0,0,0,0,0]
sleep_stage_Su = 0
sleep_stage_Av = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]      # available
i = -1
j = 0

start_a_nap = 1         # GPIO set suru.     teikou hitsuyou

initialize = 0          # hirune kaishi shokika
finish = 0              # hirune shuryouji


def startanap():        #昼寝を始める
    pygame.mixer.music.load("kurae.mp3")     # 音楽ファイルの読み込み
    pygame.mixer.music.play(1)              # 音楽の再生回数(1回)

    pygame.mixer.music.stop()               # 再生の終了
    print("startanap")


def SleepStage():
    global i
    global j
    global now_time_Es
    global now_time_Av
    global start_time_Es
    global start_time_Av
    global sleep_stage_Su
    global sleep_stage_Es
    global sleep_stage_Av

    now_time_Es = time.time()
    now_time_Av = time.time()

    if now_time_Es - start_time_Es >= 3 :
        #sleep_stage = estim.forward()           # Get Sleep Stage
        i = i + 1
        sleep_stage = randint(0,1)
        print(sleep_stage)
        sleep_stage_Es[i] = sleep_stage
        sleep_stage_Su = sleep_stage_Su + sleep_stage_Es[i]
        start_time_Es = time.time()

        if i == 2:
            sleep_stage_Av[j] = np.round(sleep_stage_Su / 3,decimals=0)
            print(sleep_stage_Av[j])
            start_time_Av = time.time()
            i = -1
            j = j + 1
            sleep_stage_Su = 0

def check_consecutive(lst):
    for i in range(len(lst) - 2):
        if lst[i] == lst[i + 1] == lst[i + 2]:
            return True
    return False

#
my_list = [1, 2, 2, 2, 3, 4, 5]

if check_consecutive(my_list):
    i = 1
    print("oo")
else:
    print("3")





def wakeup():  
    pygame.mixer.music.load("amai.mp3")
    pygame.mixer.music.play(1)




if __name__ == "__main__":

    while(1):
        if start_a_nap == 1:      # switch  ga  osaretara  1 ni naru

            if initialize == 0:   # shokika
                startanap()
                initialize = 1
            
            SleepStage()          

            if now_time_Sl - start_time_Sl >= 30*60:  # 30fun tattara okosu 
                wakeup()


            now_time_Sl = time.time()