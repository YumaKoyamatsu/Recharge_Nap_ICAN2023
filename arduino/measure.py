import serial
import numpy as np
import matplotlib.pyplot as plt

def getData(ser):
    payload = ser.readline().decode().replace('\n', '')
    log_time, ppg, acc = float(payload.split(",")[0]), float(payload.split(",")[1]), float(payload.split(",")[2])
    return log_time, ppg, acc

ser = serial.Serial("COM3", 115200, timeout=0.5)
data_seq = []

for i in range(90000):
    data = getData(ser)
    data_seq.append(data)

data_seq_array = np.array(data_seq)
print(data_seq_array)

np.save("data_koyamatsu.npy", data_seq_array)
plt.plot(data_seq_array[:, 1])
plt.show()
ser.close()