# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 14:44:08 2025

@author: admin
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# CSVファイル読み込み
df = pd.read_csv("C:/Users/admin/Desktop/FFT/Sensor__4_readings.csv")

# タイムスタンプを日時形式に変換してインデックスに設定
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# 欠損データの補完（前値で埋める）
df['temperature'] = df['temperature'].interpolate()
df['humidity'] = df['humidity'].interpolate()

# 時間領域のプロット
plt.figure(figsize=(14, 6))

plt.subplot(2, 1, 1)
plt.plot(df['temperature'], label='Temperature (°C)', color='red')
plt.ylabel('Temperature (°C)')
plt.title('Temperature Over Time')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(df['humidity'], label='Humidity (%)', color='blue')
plt.ylabel('Humidity (%)')
plt.title('Humidity Over Time')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


# FFTによる周波数解析
# サンプリング間隔（秒）を推定（例：10分ごと = 600秒）
sampling_interval = (df.index[1] - df.index[0]).total_seconds()
fs = 1 / sampling_interval  # サンプリング周波数 [Hz]

# 信号（温度）の準備
signal = df['temperature'] - df['temperature'].mean()  # DC成分除去
N = len(signal)

# FFT実行
fft_result = np.fft.fft(signal)
freqs = np.fft.fftfreq(N, d=sampling_interval)

# 正の周波数成分だけ抽出
pos_mask = freqs > 0
freqs_pos = freqs[pos_mask]
fft_magnitude = np.abs(fft_result)[pos_mask]

# 周波数スペクトルのプロット
plt.figure(figsize=(10, 4))
plt.plot(freqs_pos, fft_magnitude, color='green')
plt.title('Frequency Spectrum of Temperature')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.show()

# 主要周波数の検出
peak_index = np.argmax(fft_magnitude)
peak_freq = freqs_pos[peak_index]
period_sec = 1 / peak_freq
period_hr = period_sec / 3600
print(f"温度の最大周波数ピーク: {peak_freq:.6f} Hz ≒ {period_hr:.2f} 時間周期")

# 湿度信号のFFT処理
humidity_signal = df['humidity'] - df['humidity'].mean()  # DC成分除去
N_humidity = len(humidity_signal)

fft_result_humidity = np.fft.fft(humidity_signal)
freqs_humidity = np.fft.fftfreq(N_humidity, d=sampling_interval)

# 正の周波数成分のみ抽出
pos_mask_humidity = freqs_humidity > 0
freqs_pos_humidity = freqs_humidity[pos_mask_humidity]
fft_magnitude_humidity = np.abs(fft_result_humidity)[pos_mask_humidity]

# 温度と湿度の周波数スペクトルを同時プロット
plt.figure(figsize=(10, 5))
plt.plot(freqs_pos, fft_magnitude, label='Temperature', color='red')
plt.plot(freqs_pos_humidity, fft_magnitude_humidity, label='Humidity', color='blue')
plt.title('Frequency Spectrum Comparison: Temperature vs Humidity')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 湿度の主要周波数
peak_idx_humidity = np.argmax(fft_magnitude_humidity)
peak_freq_humidity = freqs_pos_humidity[peak_idx_humidity]
period_hr_humidity = 1 / peak_freq_humidity / 3600

print(f"湿度の最大周波数ピーク: {peak_freq_humidity:.6f} Hz ≒ {period_hr_humidity:.2f} 時間周期")



