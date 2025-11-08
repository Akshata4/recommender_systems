
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 2 10:21:43 2024
@author: chvuppal
"""
import pandas as pd
import numpy as np
# Create a sample time series dataframe
data = {
'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='h'),
'value': np.random.randint(1, 10, 100)
}
df = pd.DataFrame(data)
# Set the timestamp as the index
df.set_index('timestamp', inplace=True)
# Define hopping window parameters
window_size = '3h' # Window duration
hop_size = '1h' # Hop interval
# Create hopping windows
df['hopping_mean'] = df['value'].rolling(window=window_size,min_periods=1).mean().shift(-1)
print(df)
df = df.reset_index()
import matplotlib.pyplot as plt
plt.plot(df['timestamp'], df['value'], label='Original Data')
plt.plot(df['timestamp'], df['hopping_mean'], label='Hopping Mean')
plt.legend()
plt.show()
