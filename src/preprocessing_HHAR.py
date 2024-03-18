import numpy as np
import pandas as pd
np.random.seed(0)
from scipy import signal
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def load_file_acc():
    dataframe = pd.read_csv('..\data\Phones_accelerometer.csv', sep=',')#, nrows=200000)
    dataframe.drop('Index', inplace=True, axis=1)
    return dataframe

def load_file_gyro():
    dataframe = pd.read_csv('..\data\Phones_gyroscope.csv', sep=',')#, nrows=2000)
    dataframe.drop('Index', inplace=True, axis=1)
    return dataframe



AccData = load_file_acc()
GyroData = load_file_gyro()

AccData['Arrival_Time'] = pd.to_datetime(AccData['Arrival_Time'], unit='ms')
#AccData['Creation_Time'] = pd.to_datetime(AccData['Creation_Time'], unit='ns')
GyroData['Arrival_Time'] = pd.to_datetime(GyroData['Arrival_Time'], unit='ms')
#GyroData['Creation_Time'] = pd.to_datetime(GyroData['Creation_Time'], unit='ns')

classCounts = ['stairsdown','stairsup','bike','sit','stand','walk']
deviceCounts = ['nexus4','s3', 's3mini','samsungold']
userCounts = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']


def create_window_dict(start, end, window_size_seconds, sliding_milliseconds):
    window_dict = {}
    start_window = start
    end_window = start + pd.Timedelta(seconds=window_size_seconds)
    window_index = 0
    while end_window < end:
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


list_x = []
list_y = []
list_user = []

for clientDeviceIndex, deviceName in enumerate(deviceCounts):
    for clientIDIndex, clientIDName in enumerate(userCounts):
        for activityIndex, activityName in enumerate(classCounts):
            print("Processing -- device: " + str(deviceName) + ", client: " + str(clientIDName) + ", activity: " + str(activityName))
            df_acc = AccData.loc[(AccData['User'] == clientIDName) &
                                 (AccData['Model'] == deviceName) &
                                 (AccData['gt'] == activityName)].sort_values(by='Arrival_Time')
            df_gyro = GyroData.loc[(GyroData['User'] == clientIDName) &
                                   (GyroData['Model'] == deviceName) &
                                   (AccData['gt'] == activityName)].sort_values(by='Arrival_Time')

            start_acc = df_acc['Arrival_Time'].min()
            end_acc = df_acc['Arrival_Time'].max()

            start_gyro = df_gyro['Arrival_Time'].min()
            end_gyro = df_gyro['Arrival_Time'].max()

            start = min(start_acc, start_gyro)
            end = max(end_acc, end_gyro)

            window_dict = create_window_dict(start=start, end=end, window_size_seconds=1, sliding_milliseconds=1000)

            for i in list(window_dict.keys()):
                size_output_signal = 128
                start, end = window_dict[i]
                df1 = df_acc[(df_acc['Arrival_Time'] >= start) & (df_acc['Arrival_Time'] <= end)]
                df2 = df_gyro[(df_gyro['Arrival_Time'] >= start) & (df_gyro['Arrival_Time'] <= end)]
                if len(df1) < size_output_signal or len(df2) < size_output_signal:
                    continue
                s_acc_np = interpolate_signal(df1, 1, size_output_signal=size_output_signal)
                s_gyro_np = interpolate_signal(df2, 1, size_output_signal=size_output_signal)
                x = np.hstack((s_acc_np, s_gyro_np))
                list_x.append(x)
                list_y.append(np.array([activityIndex]))
                list_user.append(np.array([clientIDName]))

x = np.stack(list_x)
np.save('x', x)
y = np.stack(list_y)
np.save('y', y)
user = np.stack(list_user)
np.save('user', user)