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

api_key = ''
api_secret = ''
fetcher = CryptoDataFetcher(api_key, api_secret)
list_futures = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'BCHUSDT',
                 'LTCUSDT', 'ADAUSDT', 'ETCUSDT', 'LINKUSDT', 'TRXUSDT',
                   'DOTUSDT', 'DOGEUSDT', 'SOLUSDT', 'MATICUSDT', 'BNBUSDT', 'UNIUSDT', 'ICPUSDT', 'AAVEUSDT', 'FILUSDT', 'XLMUSDT', 'ATOMUSDT', 'XTZUSDT', 'SUSHIUSDT', 'AXSUSDT', 'THETAUSDT', 'AVAXUSDT', 'MANAUSDT', 'SANDUSDT', 'DYDXUSDT', 'NEARUSDT', 'EGLDUSDT', 'KSMUSDT', 'ARUSDT', 'FTMUSDT', 'PEOPLEUSDT', 'LRCUSDT', 'NEOUSDT', 'ALGOUSDT', 'IOTAUSDT', 'ENJUSDT', 'GMTUSDT', 'ZILUSDT', 'IOSTUSDT', 'APEUSDT', 'RUNEUSDT', 'KNCUSDT', 'APTUSDT', 'CHZUSDT', 'ROSEUSDT', 'ZRXUSDT', 'KAVAUSDT', 'ENSUSDT', 'SXPUSDT', 'OPUSDT', 'RSRUSDT', 'SNXUSDT', 'STORJUSDT', 'COMPUSDT', 'IMXUSDT', 'FLOWUSDT', 'QTUMUSDT', 'MASKUSDT', 'WOOUSDT', 'GRTUSDT', 'BANDUSDT', 'STGUSDT', 'ONEUSDT', 'JASMYUSDT', 'MKRUSDT', 'BATUSDT', 'MAGICUSDT', 'LDOUSDT', 'BLURUSDT', 'MINAUSDT', 'CFXUSDT', 'ASTRUSDT', 'GMXUSDT', 'ANKRUSDT', 'ACHUSDT', 'FETUSDT', 'FXSUSDT', 'HOOKUSDT', 'BNXUSDT', 'SSVUSDT', 'LQTYUSDT', 'STXUSDT', 'TRUUSDT', 'HBARUSDT', 'INJUSDT', 'BELUSDT', 'COTIUSDT', 'VETUSDT', 'ARBUSDT', 'KLAYUSDT', 'FLMUSDT', 'OMGUSDT', 'CKBUSDT', 'IDUSDT', 'LITUSDT', 'JOEUSDT', 'TLMUSDT', 'HOTUSDT', 'CHRUSDT', 'RDNTUSDT', 'ICXUSDT', 'ONTUSDT', 'UNFIUSDT', 'NKNUSDT', 'ARPAUSDT', 'DARUSDT', 'SFPUSDT', 'SKLUSDT', 'RVNUSDT', 'CELRUSDT', 'SPELLUSDT', 'SUIUSDT', 'IOTXUSDT', 'STMXUSDT', 'BSVUSDT', 'TONUSDT', 'GTCUSDT', 'DENTUSDT', 'ORDIUSDT', 'KEYUSDT', 'LEVERUSDT', 'QNTUSDT', 'MAVUSDT', 'XVGUSDT', 'AGLDUSDT', 'WLDUSDT', 'PENDLEUSDT', 'ARKMUSDT', 'YGGUSDT', 'OGNUSDT', 'LPTUSDT', 'BNTUSDT', 'BAKEUSDT', 'LOOMUSDT', 'BIGTIMEUSDT', 'ORBSUSDT', 'WAXPUSDT', 'POLYXUSDT', 'TIAUSDT', 'MEMEUSDT', 'PYTHUSDT', 'JTOUSDT', 'ACEUSDT', 'XAIUSDT', 'MANTAUSDT', 'ALTUSDT', 'JUPUSDT', 'ZETAUSDT', 'STRKUSDT', 'PIXELUSDT', 'DYMUSDT', 'WIFUSDT', 'AXLUSDT', 'BOMEUSDT', 'METISUSDT', 'NFPUSDT', 'VANRYUSDT', 'AEVOUSDT', 'ETHFIUSDT', 'OMUSDT', 'ONDOUSDT', 'CAKEUSDT', 'PORTALUSDT', 'NTRNUSDT', 'KASUSDT', 'AIUSDT', 'ENAUSDT', 'WUSDT', 'TNSRUSDT', 'SAGAUSDT', 'TAOUSDT', 'FRONTUSDT', 'ATAUSDT', 'SUPERUSDT', 'ONGUSDT', 'LSKUSDT', 'GLMUSDT', 'REZUSDT', 'XVSUSDT', 'MOVRUSDT', 'BBUSDT', 'NOTUSDT', 'BICOUSDT', 'HIFIUSDT']
print('\nРаботает скринер Volatility_ATR...')

high_vol = []
low_vol = []

for fut in tqdm(list_futures, desc="Загрузка цен фьючерсов"):

    df = fetcher.price(fut, '1h', 168)
    # df = fetcher.price(fut, '15m', 96)  # можно раскомментировать
    atr_instance = ATR(period=10, multiplier=1)
    df = atr_instance.calculate_atr(df)

    # расчёт волатильности
    threshold = df['atr'].mean()
    df = atr_instance.mark_volatility_periods(df, threshold)
    result = df.iloc[-1]['volatility']
    #atr_instance.plot_atr(df)
    
    # определение волатильности
    if result == 'High':
        high_vol.append(fut)
    else:
        low_vol.append(fut)

# Сохранение результатов в JSON
results = {
    "high_volatility": high_vol,
    "low_volatility": low_vol
}
with open('C:\\Programs\\Screeners\\data\\ATR\\volatility_results.json', 'w') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

# Красивый вывод результатов
print(f"\n{'Высокая волатильность':-^50}")
print(", ".join(high_vol))

print(f"\n{'Низкая волатильность':-^50}")
print(", ".join(low_vol))
