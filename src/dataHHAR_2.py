import hickle as hkl
import numpy as np
import os
import pandas as pd
from subprocess import call
import requests
np.random.seed(0)
import urllib.request
import zipfile
from scipy import signal
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


#%%
def load_file_acc():
    dataframe = pd.read_csv('Phones_accelerometer.csv', sep=',')#, nrows=2000000)
    dataframe.drop('Index', inplace=True, axis=1)
    return dataframe

def load_file_gyro():
    dataframe = pd.read_csv('Phones_gyroscope.csv', sep=',')#, nrows=20)
    dataframe.drop('Index', inplace=True, axis=1)
    return dataframe
#[[col for col in dataframe.columns if col not in ['Index']]]


def consecutive(data, treshHoldSplit, stepsize=1):
    splittedData = np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
    returnResults = [newArray for newArray in splittedData if len(newArray) >= treshHoldSplit]
    return returnResults


def segmentData(accData, time_step, step):
    step = int(step)
    segmentAccData = []
    for i in range(0, accData.shape[0] - time_step, step):
        segmentAccData.append(accData[i:i+time_step, :])
    return np.asarray(segmentAccData)


def downSampleLowPass(motionData,factor):
    accX = signal.decimate(motionData[:,:,0],factor)
    accY = signal.decimate(motionData[:,:,1],factor)
    accZ = signal.decimate(motionData[:,:,2],factor)
    gyroX = signal.decimate(motionData[:, :, 3], factor)
    gyroY = signal.decimate(motionData[:, :, 4], factor)
    gyroZ = signal.decimate(motionData[:, :, 5], factor)
    return np.dstack((accX, accY, accZ, gyroX, gyroY, gyroZ))



AccData = load_file_acc()
unprocessedAccData = AccData.values
GyroData = load_file_gyro()
unprocessedGyroData = GyroData.values

AccData['Arrival_Time'] = pd.to_datetime(AccData['Arrival_Time'], unit='ms')
AccData['Creation_Time'] = pd.to_datetime(AccData['Creation_Time'], unit='ns')
GyroData['Arrival_Time'] = pd.to_datetime(GyroData['Arrival_Time'], unit='ms')
GyroData['Creation_Time'] = pd.to_datetime(GyroData['Creation_Time'], unit='ns')
df = pd.DataFrame({'colonna1': AccData['Arrival_Time'], 'colonna2': GyroData['Arrival_Time']})
dg = pd.DataFrame({'colonna1': AccData['Creation_Time'], 'colonna2': GyroData['Creation_Time']})

print('ok')

classCounts = ['stairsdown','stairsup','bike','sit','stand','walk']
deviceCounts = ['nexus4','s3', 's3mini','samsungold']
userCounts = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
downSamplingRate = [4,3,2,1]
deviceWindowFrame = [512,384,256,128]
indexOffset = 0
allProcessedData = {}
allProcessedLabel = {}
deviceIndex = {}

clientCount = len(deviceCounts) * len(userCounts)
deviceIndexes = {new_list: [] for new_list in range(len(deviceCounts))}
clientIndexes = {new_list: [] for new_list in range(len(userCounts))}


def create_window_dict(start, end, window_size_seconds, sliding_milliseconds):
    window_dict = {}
    start_window = start
    end_window = start + pd.Timedelta(seconds=window_size_seconds)
    window_index = 0
    while end_window <= end:
        window_dict[window_index] = (start_window, end_window)
        start_window = start_window + pd.Timedelta(milliseconds=sliding_milliseconds)
        end_window = end_window + pd.Timedelta(milliseconds=sliding_milliseconds)
        window_index += 1
    return window_dict

def interpolate_signal(df, window_size_seconds, size_output_signal):
    t_original = np.linspace(0, window_size_seconds, len(df), endpoint=False)
    t_target = np.linspace(0, window_size_seconds, size_output_signal, endpoint=False)

    interpolator_x = interp1d(t_original, df['x'].to_numpy(), kind='linear')
    seg_interpolated_x = pd.Series(interpolator_x(t_target))

    interpolator_y = interp1d(t_original, df['y'].to_numpy(), kind='linear')
    seg_interpolated_y = pd.Series(interpolator_y(t_target))

    interpolator_z = interp1d(t_original, df['z'].to_numpy(), kind='linear')
    seg_interpolated_z = pd.Series(interpolator_z(t_target))
    return np.column_stack((seg_interpolated_x, seg_interpolated_y, seg_interpolated_z))



def plot_signal_before_and_after_interpolation(df, window_size_seconds, size_output_signal):
    t_original = np.linspace(0, window_size_seconds, len(df), endpoint=False)
    t_target = np.linspace(0, window_size_seconds, size_output_signal, endpoint=False)  # 100 campioni uniformemente spaziati

    # Interpolazione lineare del segnale originale ai tempi target
    interpolator = interp1d(t_original, df['x'].to_numpy(), kind='linear')
    seg_interpolated = pd.Series(interpolator(t_target))

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    ax[0].plot(pd.Series(t_target), seg_interpolated, marker='o', linestyle='-', )
    ax[0].set_ylabel('Valore di X')
    ax[0].legend()
    ax[0].grid(True)
    ax[0].tick_params(axis='x', rotation=45)

    # Plot per la seconda window
    ax[1].plot(df['Arrival_Time'], df['x'], marker='o', linestyle='-', color='green')
    ax[1].set_ylabel('Valore di X')
    ax[1].legend()
    ax[1].grid(True)
    ax[1].tick_params(axis='x', rotation=45)

    # Ajusta il layout per evitare sovrapposizioni
    plt.tight_layout()
    plt.show()


map_activities = {
    'stairsdown': 0,
    'stairsup': 1,
    'bike': 2,
    'sit': 3,
    'stand': 4,
    'walk': 5
}

list_x = []
list_y = []
for clientDeviceIndex, deviceName in enumerate(deviceCounts):
    print('device: ' + str(deviceName))
    for clientIDIndex, clientIDName in enumerate(userCounts):
        print("Processing device: " + str(deviceName) + ", client: " + str(clientIDName))
        df_acc = AccData.loc[(AccData['User'] == clientIDName) & (AccData['Model'] == deviceName)].sort_values(by='Arrival_Time')
        df_gyro = GyroData.loc[(GyroData['User'] == clientIDName) & (GyroData['Model'] == deviceName)].sort_values(by='Arrival_Time')

        start_acc = df_acc['Arrival_Time'].min()
        end_acc = df_acc['Arrival_Time'].max()

        start_gyro = df_gyro['Arrival_Time'].min()
        end_gyro = df_gyro['Arrival_Time'].max()

        start = min(start_acc, start_gyro)
        end = max(end_acc, end_gyro)

        window_dict = create_window_dict(start=start, end=end, window_size_seconds=1, sliding_milliseconds=0)

        for i in list(window_dict.keys()):
            #list_errors = []
            size_output_signal = 128
            start, end = window_dict[i]
            df1 = df_acc[(df_acc['Arrival_Time'] >= start) & (df_acc['Arrival_Time'] <= end)]
            df2 = df_gyro[(df_gyro['Arrival_Time'] >= start) & (df_gyro['Arrival_Time'] <= end)]
            ######### controllo molto importante
            if len(df1) < size_output_signal or len(df2) < size_output_signal:
                continue
            if df1['gt'].iloc[-1] != df2['gt'].iloc[-1]:
                #list_errors.append(i)
                continue
            s_acc_np = interpolate_signal(df1, 1, size_output_signal=size_output_signal)
            s_gyro_np = interpolate_signal(df2, 1, size_output_signal=size_output_signal)
            x = np.hstack((s_acc_np, s_gyro_np))
            list_x.append(x)
            activity = map_activities[df1['gt'].iloc[-1]]
            list_y.append(np.array([activity]))
            #sistemare codifica
x = np.stack(list_x)
np.save('x', x)
y = np.stack(list_y)
np.save('y', y)
print('ok')







'''
        unprocessedAccData = df_acc.values



        unprocessedGyroData = df_gyro.values
        df_acc = AccData
        df_acc['diff_milliseconds'] = df_acc['Arrival_Time'].diff()
        gni = df_acc['diff_milliseconds'].value_counts()

        processedClassData = []
        processedClassLabel = []
        IndexDataAcc = (unprocessedAccData[:,5] == clientIDName) & (unprocessedAccData[:,6] == deviceName)
        userDeviceDataAcc = unprocessedAccData[IndexDataAcc]


        print('ok')
        iii = (unprocessedGyroData[:,5] == clientIDName) & (unprocessedGyroData[:,6] == deviceName)
        print(str(np.sum(IndexDataAcc)) + ' / ' + str(len(IndexDataAcc)))
        print(str(np.sum(iii)) + ' / ' + str(len(iii)))
        intersezione = np.logical_and(IndexDataAcc, iii)
        print(np.sum(intersezione))
        userDeviceDataAcc = unprocessedAccData[IndexDataAcc]
        if (len(userDeviceDataAcc) == 0):
            print("No acc data found")
            print("Skipping device :" + str(deviceName) + " Client: " + str(clientIDName))
            indexOffset += 1
            continue
        userDeviceDataGyro = unprocessedGyroData[(unprocessedGyroData[:,5] == clientIDName) & (unprocessedGyroData[:,6] == deviceName)]
        if (len(userDeviceDataGyro) == 0):
            userDeviceDataGyro = unprocessedGyroData[IndexDataAcc]
        for classIndex, className in enumerate(classCounts):
            if (len(userDeviceDataAcc) <= len(userDeviceDataGyro)):
                classData = np.where(userDeviceDataAcc[:, 8] == className)[0]
            else:
                classData = np.where(userDeviceDataGyro[:, 8] == className)[0]
            segmentedClass = consecutive(classData,deviceWindowFrame[int(clientDeviceIndex/2)])
            for segmentedClassRange in (segmentedClass):
                combinedData = np.dstack((
                    segmentData(userDeviceDataAcc[segmentedClassRange][:,2:5],
                                deviceWindowFrame[clientDeviceIndex],
                                deviceWindowFrame[clientDeviceIndex]/2),
                    segmentData(userDeviceDataGyro[segmentedClassRange][:, 2:5],
                                deviceWindowFrame[clientDeviceIndex],
                                deviceWindowFrame[clientDeviceIndex]/2)
                ))
                processedClassData.append(combinedData)
                processedClassLabel.append(np.full(combinedData.shape[0], classIndex, dtype=int))
        deviceCheckIndex = clientDeviceIndex % 2
        tempProcessedData = np.vstack((processedClassData))
        allProcessedData[(deviceName, clientIDName)] = tempProcessedData
        allProcessedLabel[(deviceName, clientIDName)] = np.hstack((processedClassLabel))

        dataIndex = (len(userCounts) * clientDeviceIndex) + clientIDIndex - indexOffset
        print("Index is at " + str(dataIndex))
        allProcessedData[dataIndex] = tempProcessedData
        allProcessedLabel[dataIndex] = np.hstack((processedClassLabel))
        deviceIndex[dataIndex] = np.full(allProcessedLabel[dataIndex].shape[0], clientDeviceIndex)
        deviceIndexes[clientDeviceIndex].append(dataIndex)
        clientIndexes[clientIDIndex].append(dataIndex)
'''
processedDataX = np.concatenate(list(allProcessedData.values()), axis=0)
processedDataY = np.concatenate(list(allProcessedLabel.values()), axis=0)

'''
allProcessedData = np.asarray(list(allProcessedData.items()))[:,1]
allProcessedLabel = np.asarray(list(allProcessedLabel.items()))[:,1]
deviceIndex =  np.asarray(list(deviceIndex.items()))[:,1]
clientIndexes =  np.asarray(list(clientIndexes.items()))[:,1]
'''
