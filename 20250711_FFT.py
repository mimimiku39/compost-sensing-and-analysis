# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 14:38:09 2025

@author: admin
"""

import pandas as pd
import matplotlib.pyplot as plt

# 正しいファイルパスで読み込み
df = pd.read_csv("C:/Users/admin/Desktop/FFT/Sensor__4_readings.csv")

# タイムスタンプを時系列データに変換
df['timestamp'] = pd.to_datetime(df['timestamp'])

# タイムスタンプをインデックスに
df.set_index('timestamp', inplace=True)

# 温度と湿度の時系列グラフを作成
plt.figure(figsize=(14, 6))

plt.subplot(2, 1, 1)
plt.plot(df['temperature'], label='Temperature (°C)', color='tomato')
plt.ylabel('Temperature (°C)')
plt.title('Temperature Over Time')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(df['humidity'], label='Humidity (%)', color='skyblue')
plt.ylabel('Humidity (%)')
plt.title('Humidity Over Time')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
