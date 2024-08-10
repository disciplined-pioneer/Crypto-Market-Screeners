import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import json
from tqdm import tqdm
from data_processing import CryptoDataFetcher

class ATR:

    def __init__(self, period=10, multiplier=1):
        self.period = period
        self.multiplier = multiplier
        self.values = []

    def calculate_true_range(self, high, low, close, prev_close):
        high_low = high - low
        high_close = abs(high - prev_close)
        low_close = abs(low - prev_close)
        true_range = max(high_low, high_close, low_close)
        return true_range

    def calculate_atr(self, df):
        self.values = [0] * len(df)
        for bar in range(len(df)):
            high = df.iloc[bar]['high']
            low = df.iloc[bar]['low']
            close = df.iloc[bar]['close']
            prev_close = df.iloc[bar - 1]['close'] if bar > 0 else close

            if bar == 0:
                self.values[bar] = high - low
            else:
                true_range = self.calculate_true_range(high, low, close, prev_close)
                self.values[bar] = ((min(bar + 1, self.period) - 1) * self.values[bar - 1] + true_range) / min(bar + 1, self.period)

        df['atr'] = [self.multiplier * val for val in self.values]
        return df

    def mark_volatility_periods(self, df, threshold):
        df['volatility'] = np.where(df['atr'] > threshold, 'High', 'Low')
        return df

    def plot_atr(self, df, threshold):
        sns.set(style='darkgrid')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10), gridspec_kw={'height_ratios': [3, 1]})
        ax1.plot(df.index, df['close'], label='Close')
        ax2.plot(df.index, df['atr'], label='ATR')
        ax2.fill_between(df.index, df['atr'], threshold, where=(df['atr'] > threshold), facecolor='red', alpha=0.3, interpolate=True, label='High Volatility')
        ax2.fill_between(df.index, df['atr'], threshold, where=(df['atr'] <= threshold), facecolor='green', alpha=0.3, interpolate=True, label='Low Volatility')
        ax2.axhline(threshold, color='black', linestyle='--', label='Threshold')
        plt.show()
